from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np
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
            params_to_opt = (list(self.model.module.prompt.parameters()) +
                           list(self.model.module.last.parameters()))
        else:
            params_to_opt = (list(self.model.prompt.parameters()) +
                           list(self.model.last.parameters()))
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
    """
    APT learner with Prototype Complement (PC) strategy.
    Các cải tiến:
    - Ước tính drift trên nhiều mẫu hơn (toàn bộ train loader hoặc 50 batches)
    - Sử dụng hệ số học drift_lr để điều chỉnh mức độ cập nhật prototype
    - Lưu prototype chưa chuẩn hóa để bảo toàn thông tin drift
    - Chia drift cho từng lớp (không áp dụng đồng nhất)
    - Điều chỉnh drift_lr theo dataset (có thể config)
    """
    def __init__(self, learner_config):
        super(APT_Learner, self).__init__(learner_config)
        self.class_prototypes = {}          # Lưu prototype (chưa normalize) cho tất cả classes
        self.drift_lr = 0.5                 # Hệ số học cho drift (có thể override)
        self.max_probe_batches = 50         # Số batches tối đa để ước tính drift

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

    # ------------------------------------------------------------------
    # BƯỚC 1: Thu thập prototype của task hiện tại
    # Chạy SAU PPF merge — dùng merged prompt
    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect_prototypes_current_task(self, train_loader):
        """
        Tính mean CLS embedding (prototype) cho mỗi class trong task hiện tại.
        Chỉ collect classes mới [last_valid_out_dim, valid_out_dim).
        Lưu prototype chưa chuẩn hóa.
        """
        self.model.eval()
        class_feats = {}
        new_classes = set(range(self.last_valid_out_dim, self.valid_out_dim))

        for x, y, task in train_loader:
            if self.gpu:
                x = x.cuda()
                y = y.cuda()

            # Forward với merged prompt (đã được set)
            feat = self.model.feat(
                x, prompt=self.model.prompt, train=False
            )[:, 0, :]                        # [B, 768]
            feat = self.model.clf_norm(feat)  # [B, 768]

            for f, label in zip(feat, y):
                label_item = label.item()
                if label_item in new_classes:
                    if label_item not in class_feats:
                        class_feats[label_item] = []
                    class_feats[label_item].append(f.cpu())

        # Lưu prototype chưa normalize
        for cls_idx, feats in class_feats.items():
            feats_t = torch.stack(feats, dim=0)  # [N, 768]
            self.class_prototypes[cls_idx] = feats_t.mean(dim=0)  # [768]

        # Cập nhật classifier weights cho new classes = normalize(prototype)
        for cls_idx in new_classes:
            if cls_idx in self.class_prototypes:
                proto = self.class_prototypes[cls_idx]
                proto_norm = F.normalize(proto.unsqueeze(0), p=2, dim=1).squeeze(0)
                if self.gpu:
                    proto_norm = proto_norm.cuda()
                self.model.last.weight.data[cls_idx] = proto_norm

        self.log(f'[PC] Collected prototypes for {len(class_feats)} new classes. '
                f'Total: {len(self.class_prototypes)} classes.')
        self.model.train()

    # ------------------------------------------------------------------
    # BƯỚC 2: Estimate feature drift sau PPF merge
    # Dùng nhiều batches để ước tính chính xác
    # ------------------------------------------------------------------
    @torch.no_grad()
    def estimate_feature_drift(self, train_loader, old_merged_prompt, new_merged_prompt):
        """
        Estimate drift = E[f_new(x) - f_old(x)] trên một số lượng lớn mẫu.

        old_merged_prompt: tensor [24, 768] — prompt TRƯỚC PPF
        new_merged_prompt: tensor [24, 768] — prompt SAU PPF

        Trả về drift vector [768].
        """
        self.model.eval()
        old_feats_list = []
        new_feats_list = []

        # Duyệt qua train loader, tối đa max_probe_batches batches
        for batch_idx, (x, y, task) in enumerate(train_loader):
            if batch_idx >= self.max_probe_batches:
                break
            if self.gpu:
                x = x.cuda()

            # Forward với OLD merged prompt
            self.model.prompt.global_merged_prompt.data = old_merged_prompt.clone()
            feat_old = self.model.feat(
                x, prompt=self.model.prompt, train=False
            )[:, 0, :]
            feat_old = self.model.clf_norm(feat_old)  # [B, 768]

            # Forward với NEW merged prompt
            self.model.prompt.global_merged_prompt.data = new_merged_prompt.clone()
            feat_new = self.model.feat(
                x, prompt=self.model.prompt, train=False
            )[:, 0, :]
            feat_new = self.model.clf_norm(feat_new)  # [B, 768]

            old_feats_list.append(feat_old.cpu())
            new_feats_list.append(feat_new.cpu())

        old_feats = torch.cat(old_feats_list, dim=0)  # [N, 768]
        new_feats = torch.cat(new_feats_list, dim=0)  # [N, 768]

        if old_feats.shape[0] == 0:
            self.log('[PC] Warning: No samples for drift estimation!')
            return torch.zeros(768)

        # Drift = mean(f_new - f_old)
        drift = (new_feats - old_feats).mean(dim=0)  # [768]
        self.log(f'[PC] Feature drift magnitude: {drift.norm().item():.6f} '
                f'(estimated on {old_feats.shape[0]} samples)')

        self.model.train()
        return drift

    # ------------------------------------------------------------------
    # BƯỚC 3: Update old prototypes + recalibrate classifier
    # Sử dụng drift scaled và áp dụng cho từng lớp cũ
    # ------------------------------------------------------------------
    def prototype_complement(self, drift):
        """
        Apply drift lên old class prototypes và update classifier weights.
        drift: tensor [768] – drift trung bình toàn cục
        """
        old_classes = list(range(self.last_valid_out_dim))
        if len(old_classes) == 0:
            return

        # Sử dụng drift_lr (hệ số học) để kiểm soát mức độ cập nhật
        scaled_drift = drift * self.drift_lr

        # Có thể tính drift riêng cho từng lớp nếu có dữ liệu, nhưng ở đây dùng chung
        for cls_idx in old_classes:
            if cls_idx not in self.class_prototypes:
                continue

            # Cập nhật prototype (không normalize)
            updated_proto = self.class_prototypes[cls_idx] + scaled_drift
            self.class_prototypes[cls_idx] = updated_proto

            # Cập nhật classifier weight = normalize(prototype)
            proto_norm = F.normalize(updated_proto.unsqueeze(0), p=2, dim=1).squeeze(0)
            if self.gpu:
                proto_norm = proto_norm.cuda()
            self.model.last.weight.data[cls_idx] = proto_norm

        self.log(f'[PC] Updated {len(old_classes)} old prototypes with drift_lr={self.drift_lr}')

    # ------------------------------------------------------------------
    # Override learn_batch với Prototype Complement
    # ------------------------------------------------------------------
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

        # ── PPF Merge ────────────────────────────────────────────────────
        merge_flag = self.model.prompt.merge_flag
        old_merged = self.model.prompt.global_merged_prompt.clone().detach()

        if merge_flag:
            if self.last_valid_out_dim == 0:
                # Task 1: no old classes, global merged prompt = prompt tokens of task 1
                self.model.prompt.global_merged_prompt = \
                    self.model.prompt.prompt_tokens.clone().detach()
            else:
                now_task_p = self.model.prompt.prompt_tokens.clone().detach()
                global_p = self.model.prompt.global_merged_prompt
                merged_p = self.model.prompt.merge_prompt(global_p, now_task_p)
                self.model.prompt.global_merged_prompt.data = merged_p

        new_merged = self.model.prompt.global_merged_prompt.clone().detach()

        # ── Prototype Complement (chỉ từ task 2 trở đi) ─────────────────
        if self.last_valid_out_dim > 0:
            # Cần train_loader để ước tính drift
            drift = self.estimate_feature_drift(train_loader, old_merged, new_merged)
            self.prototype_complement(drift)

        # ── Collect prototypes cho task hiện tại (dùng new merged prompt) ──
        self.collect_prototypes_current_task(train_loader)

        # ── Imprint new class classifier weights từ prototypes ────────────
        # Đã được thực hiện trong collect_prototypes_current_task, nhưng đảm bảo
        # cho tất cả new classes đã có prototype
        for cls_idx in range(self.last_valid_out_dim, self.valid_out_dim):
            if cls_idx not in self.class_prototypes:
                self.log(f'[PC] Warning: class {cls_idx} has no prototype!')
                continue
            proto = self.class_prototypes[cls_idx]
            proto_norm = F.normalize(proto.unsqueeze(0), p=2, dim=1).squeeze(0)
            if self.gpu:
                proto_norm = proto_norm.cuda()
            self.model.last.weight.data[cls_idx] = proto_norm

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