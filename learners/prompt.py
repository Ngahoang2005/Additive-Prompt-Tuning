from __future__ import print_function
import torch
import torch.nn.functional as F
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
        logits = self.model(inputs, train=True)
        logits = logits[:, :self.valid_out_dim]
        logits[:, :self.last_valid_out_dim] = -float('inf')
        total_loss = self.criterion(logits, targets.long())

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    def init_optimizer(self):
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
        print('*****************************************')
        optimizer_arg = {'params': params_to_opt,
                         'lr': self.config['lr'],
                         'weight_decay': self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD', 'RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'], 0.999)

        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)

        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.schedule, gamma=0.1)

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=self.config['gpuid'],
                output_device=self.config['gpuid'][0])
        return self


class APT_Learner(Prompt_Learner):

    def __init__(self, learner_config):
        super(APT_Learner, self).__init__(learner_config)

        # Ridge regression components
        # G = Φ^T Φ (Gram matrix), C = Φ^T Y (cross matrix)
        self.ridge_lambda = learner_config.get('ridge_lambda', 1.0)
        self.rp_dim = learner_config.get('rp_dim', 768)  # dim sau random projection

        # Random projection matrix — frozen, init once
        # None cho đến khi biết emb_d
        self.W_rand = None

        # Gram và cross matrices — tích lũy qua tasks
        self.G = None  # [rp_dim, rp_dim]
        self.C = None  # [rp_dim, num_classes]

    def _init_random_projection(self, emb_d, device):
        """Khởi tạo W_rand một lần duy nhất."""
        if self.W_rand is None:
            W = torch.randn(emb_d, self.rp_dim, device=device)
            # Normalize columns
            W = W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.W_rand = W  # frozen, không train
            self.log(f'Random projection: {emb_d} -> {self.rp_dim}')

    def _random_project(self, feat):
        """
        feat: [B, emb_d]
        return: [B, rp_dim] sau ReLU(feat @ W_rand)
        """
        return F.relu(feat @ self.W_rand)

    @torch.no_grad()
    def update_gram_matrices(self, train_loader, task_classes):
        """
        Tích lũy Gram matrix G và cross matrix C từ task hiện tại.
        Dùng merged prompt (inference mode).
        """
        self.model.eval()
        num_total_classes = self.valid_out_dim

        # Init C nếu chưa có hoặc cần expand
        if self.C is None:
            self.C = torch.zeros(self.rp_dim, num_total_classes)
        elif self.C.shape[1] < num_total_classes:
            # Expand C để accommodate new classes
            pad = torch.zeros(self.rp_dim, num_total_classes - self.C.shape[1])
            self.C = torch.cat([self.C, pad], dim=1)

        if self.G is None:
            self.G = torch.zeros(self.rp_dim, self.rp_dim)

        for x, y, task in train_loader:
            if self.gpu:
                x = x.cuda()
                y = y.cuda()

            # CLS embedding với merged prompt
            feat = self.model.feat(
                x, prompt=self.model.prompt, train=False
            )[:, 0, :]
            feat = self.model.clf_norm(feat)  # [B, 768]

            # Init random projection nếu chưa có
            self._init_random_projection(feat.shape[1], feat.device)

            # Random project
            phi = self._random_project(feat)  # [B, rp_dim]

            # One-hot labels
            B = y.shape[0]
            Y_onehot = torch.zeros(B, num_total_classes, device=y.device)
            Y_onehot.scatter_(1, y.long().unsqueeze(1), 1.0)

            # Cập nhật Gram và cross matrix (incremental)
            self.G += (phi.T @ phi).cpu()           # [rp_dim, rp_dim]
            self.C += (phi.T @ Y_onehot).cpu()      # [rp_dim, num_classes]

        self.log(f'Updated Gram matrices for {len(task_classes)} new classes')
        self.model.train()

    def solve_classifier_ridge(self):
        """
        Giải ridge regression: W = (G + λI)^{-1} C
        Cập nhật classifier weights của model.
        """
        if self.G is None or self.C is None:
            return

        self.log('Solving ridge regression for classifier...')
        device = self.model.last.weight.device

        G = self.G.to(device)
        C = self.C.to(device)

        # W = (G + λI)^{-1} C
        reg = self.ridge_lambda * torch.eye(self.rp_dim, device=device)
        try:
            W = torch.linalg.solve(G + reg, C)  # [rp_dim, num_classes]
        except Exception:
            # Fallback nếu singular
            W = torch.linalg.lstsq(G + reg, C).solution

        # W.T: [num_classes, rp_dim]
        # Cần update last layer — nhưng last layer expect [num_classes, 768]
        # Nên ta lưu W riêng và dùng trong inference
        self.W_ridge = W.T  # [num_classes, rp_dim]
        self.log('Ridge regression solved.')

    def validation(self, dataloader, model=None, task_in=None,
                   task_metric='acc', verbal=True, task_global=False):
        """Override để dùng ridge classifier nếu đã có."""
        if model is None:
            model = self.model

        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()
        orig_mode = model.training
        model.eval()

        use_ridge = hasattr(self, 'W_ridge') and self.W_ridge is not None

        for i, (input, target, task) in enumerate(dataloader):
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()

            if task_in is None:
                with torch.no_grad():
                    if use_ridge and self.W_rand is not None:
                        # Ridge inference
                        feat = model.feat(
                            input, prompt=model.prompt, train=False
                        )[:, 0, :]
                        feat = model.clf_norm(feat)
                        phi = self._random_project(feat)  # [B, rp_dim]
                        # logits = phi @ W_ridge.T
                        output = phi @ self.W_ridge[:self.valid_out_dim].T
                    else:
                        # Fallback về APT gốc
                        output = model.forward(input)[:, :self.valid_out_dim]

                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))

        model.train(orig_mode)
        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'.format(
                acc=acc, time=batch_timer.toc()))
        return acc.avg

    def learn_batch(self, train_loader, train_dataset, model_save_dir):
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        if self.reset_optimizer:
            self.log('Optimizer is reset!')
            self.init_optimizer()

        if need_train:
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()

            for epoch in range(self.config['schedule'][-1]):
                self.epoch = epoch
                if epoch > 0:
                    self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()

                for i, (x, y, task) in enumerate(train_loader):
                    self.model.train()
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                    loss, output = self.update_model(x, y)
                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss, y.size(0))
                    batch_timer.tic()

                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(
                    epoch=self.epoch+1, total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(
                    loss=losses, acc=acc))
                losses = AverageMeter()
                acc = AverageMeter()

        self.model.train()

        # PPF merge — giữ nguyên
        merge_flag = self.model.prompt.merge_flag
        if merge_flag:
            if self.last_valid_out_dim == 0:
                self.model.prompt.global_merged_prompt = \
                    self.model.prompt.prompt_tokens.clone().detach()
            else:
                now_task_p = self.model.prompt.prompt_tokens.clone().detach()
                global_p = self.model.prompt.global_merged_prompt
                merged_p = self.model.prompt.merge_prompt(global_p, now_task_p)
                self.model.prompt.global_merged_prompt.data = merged_p

        # Ridge regression — thay CA hoàn toàn
        task_classes = list(range(self.last_valid_out_dim, self.valid_out_dim))
        self.update_gram_matrices(train_loader, task_classes)
        self.solve_classifier_ridge()

        self.model.eval()
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False
        self.task_count += 1

        if self.memory_size > 0:
            train_dataset.update_coreset(
                self.memory_size, np.arange(self.last_valid_out_dim))

        try:
            return batch_time.avg
        except:
            return None

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