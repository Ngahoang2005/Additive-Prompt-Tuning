from __future__ import print_function
import torch
import models
from utils.metric import accuracy, AverageMeter, Timer
from .default import NormalNN, weight_reset, accumulate_acc
from utils.schedulers import CosineSchedule

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
        # Frozen snapshot của task trước để distillation
        self.model_frozen = None
        # Lambda điều chỉnh weight của distillation loss
        # Đọc từ config nếu có, mặc định 1.0
        self.distill_lambda = learner_config.get('distill_lambda', 1.0)

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
    def update_model(self, inputs, targets):
        # ── Standard CE loss ──────────────────────────────────────────
        logits = self.model(inputs, train=True)
        logits = logits[:, :self.valid_out_dim]
        logits[:, :self.last_valid_out_dim] = -float('inf')
        total_loss = self.criterion(logits, targets.long())

        # ── CLS Distillation loss (chỉ từ task 2 trở đi) ─────────────
        if self.model_frozen is not None:
            # CLS embedding của model hiện tại — TRƯỚC clf_norm
            # feat() trả về full sequence (B, N+1, 768), lấy token 0
            cls_new = self.model.feat(
                inputs, prompt=self.model.prompt, train=True
            )[:, 0, :]   # (B, 768)

            # CLS embedding của frozen model — no_grad, inference mode
            with torch.no_grad():
                cls_frozen = self.model_frozen.feat(
                    inputs, prompt=self.model_frozen.prompt, train=False
                )[:, 0, :]   # (B, 768)

            # Cosine distillation: minimize (1 - cosine_similarity)
            # cosine_similarity trả về (B,), mean over batch
            cos_sim = torch.nn.functional.cosine_similarity(
                cls_new, cls_frozen, dim=1
            )   # (B,)
            loss_distill = (1.0 - cos_sim).mean()

            total_loss = total_loss + self.distill_lambda * loss_distill

        # ── Backward ──────────────────────────────────────────────────
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits
    def save_frozen_model(self):
        """
        Snapshot model hiện tại thành frozen copy để dùng làm teacher.
        Chỉ copy feat + prompt (phần ảnh hưởng CLS output).
        model_frozen hoàn toàn frozen — không có gradient.
        """
        import copy
        self.model_frozen = copy.deepcopy(self.model)
        # Tắt hoàn toàn gradient của frozen model
        for param in self.model_frozen.parameters():
            param.requires_grad = False
        self.model_frozen.eval()
        self.log('Frozen model saved for CLS distillation.')

