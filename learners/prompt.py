from __future__ import print_function
import torch
import models
from utils.metric import accuracy, AverageMeter, Timer
from .default import NormalNN, weight_reset, accumulate_acc
from utils.schedulers import CosineSchedule
import torch.nn.functional as F
class Prompt_Learner(NormalNN):
    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        self.ema_coeff = learner_config['ema_coeff']
        super(Prompt_Learner, self).__init__(learner_config)

    def update_model(self, inputs, targets):
        # logits
        logits = self.model(inputs, train=True)
        
        logits = logits[:,:self.valid_out_dim]
        logits[:,:self.last_valid_out_dim] = -float('inf')
        total_loss = self.criterion(logits, targets.long())       
        
        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.detach(), logits

    def get_attn_heatmap(self, inputs):
        return 

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
        print('*****************************************')
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

class APT_Learner(Prompt_Learner):

    def __init__(self, learner_config):
        super(APT_Learner, self).__init__(learner_config)
        # {class_idx: prototype_vector (768,)}
        self.class_prototypes = {}

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](
            out_dim=self.out_dim,
            ema_coeff=self.ema_coeff,
            prompt_flag='apt',
            prompt_param=self.prompt_param,
            tasks=self.tasks
        )
        return model

    def compute_prototypes(self, dataloader, task_classes):
        """
        Tính prototype = mean CLS embedding cho từng class trong task.
        Dùng global_merged_prompt (inference mode).
        """
        self.model.eval()
        proto_sum = {}
        proto_count = {}

        for cls in task_classes:
            proto_sum[cls] = torch.zeros(768).cuda()
            proto_count[cls] = 0

        with torch.no_grad():
            for x, y, _ in dataloader:
                x, y = x.cuda(), y.cuda()
                # Forward inference mode — dùng global_merged_prompt
                feat = self.model.feat(
                    x, prompt=self.model.prompt, train=False
                )[:, 0, :]  # (B, 768)
                feat = self.model.clf_norm(feat)  # normalize giống training

                for i, cls_idx in enumerate(y.tolist()):
                    if cls_idx in proto_sum:
                        proto_sum[cls_idx] += feat[i]
                        proto_count[cls_idx] += 1

        prototypes = {}
        for cls in task_classes:
            if proto_count[cls] > 0:
                prototypes[cls] = proto_sum[cls] / proto_count[cls]

        self.model.train()
        return prototypes

    def compute_drift_and_correct(self, dataloader, prompt_old, prompt_new):
        """
        Tính drift vector = mean(f(x; p_new) - f(x; p_old)) trên current data.
        Sau đó correct tất cả prototypes cũ.

        Trong APT: backbone frozen → drift gần đồng nhất với mọi input
        → correction chính xác, không phải ước lượng.
        """
        self.model.eval()
        drift_sum = torch.zeros(768).cuda()
        n = 0

        with torch.no_grad():
            for x, _, _ in dataloader:
                x = x.cuda()

                # Feature với prompt mới (đã merge)
                self.model.prompt.global_merged_prompt = prompt_new
                feat_new = self.model.feat(
                    x, prompt=self.model.prompt, train=False
                )[:, 0, :]
                feat_new = self.model.clf_norm(feat_new)

                # Feature với prompt cũ (trước merge)
                self.model.prompt.global_merged_prompt = prompt_old
                feat_old = self.model.feat(
                    x, prompt=self.model.prompt, train=False
                )[:, 0, :]
                feat_old = self.model.clf_norm(feat_old)

                drift_sum += (feat_new - feat_old).mean(dim=0)
                n += 1

        drift = drift_sum / max(n, 1)

        # Restore prompt mới
        self.model.prompt.global_merged_prompt = prompt_new

        # Correct tất cả prototypes cũ
        self.log(f'Drift magnitude: {drift.norm().item():.4f}')
        for cls in self.class_prototypes:
            self.class_prototypes[cls] = self.class_prototypes[cls] + drift

        self.model.train()

    def ncm_classify(self, features):
        """
        Nearest Class Mean classification bằng cosine similarity.
        features: (B, 768)
        Returns: (B,) predicted class indices
        """
        if not self.class_prototypes:
            raise ValueError("No prototypes stored yet")

        classes = sorted(self.class_prototypes.keys())
        proto_matrix = torch.stack(
            [self.class_prototypes[c] for c in classes]
        )  # (n_classes, 768)

        # Cosine similarity
        feat_norm = F.normalize(features, dim=1)
        proto_norm = F.normalize(proto_matrix, dim=1)
        sims = feat_norm @ proto_norm.t()  # (B, n_classes)

        pred_local = sims.argmax(dim=1)  # local index
        pred_global = torch.tensor(
            [classes[i] for i in pred_local.tolist()]
        ).to(features.device)

        return pred_global, sims