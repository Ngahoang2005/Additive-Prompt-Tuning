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
        
        # Lưu class statistics cho Classifier Alignment
        # key: class_idx, value: {'mean': tensor [768], 'std': tensor [768]}
        self.class_stats = {}
        
        # Số synthetic samples per class khi align
        self.ca_n_samples = learner_config.get('ca_n_samples', 256)
        
        # Số epochs để retrain classifier khi align
        self.ca_epochs = learner_config.get('ca_epochs', 5)

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

    # -------------------------------------------------------
    # BƯỚC 1: Thu thập CLS embeddings sau khi train xong task
    # -------------------------------------------------------
    @torch.no_grad()
    def collect_class_stats(self, train_loader):
        """
        Chạy toàn bộ train_loader qua model (với merged prompt),
        lưu mean và std của CLS embedding cho mỗi class.
        """
        self.model.eval()
        
        # Accumulate features per class
        class_feats = {}  # class_idx -> list of [768] tensors
        
        for i, (x, y, task) in enumerate(train_loader):
            if self.gpu:
                x = x.cuda()
                y = y.cuda()
            
            # Forward pass lấy CLS embedding (không qua classifier)
            # model.feat trả về [B, N, 768], lấy [:, 0, :] là CLS
            with torch.no_grad():
                cls_feat = self.model.feat(
                    x, prompt=self.model.prompt, train=False
                )[:, 0, :]  # [B, 768]
                cls_feat = self.model.clf_norm(cls_feat)  # normalize
            
            for feat, label in zip(cls_feat, y):
                label_item = label.item()
                if label_item not in class_feats:
                    class_feats[label_item] = []
                class_feats[label_item].append(feat.cpu())
        
        # Tính mean và std cho từng class
        for cls_idx, feats in class_feats.items():
            feats_tensor = torch.stack(feats, dim=0)  # [N, 768]
            self.class_stats[cls_idx] = {
                'mean': feats_tensor.mean(dim=0),   # [768]
                'std':  feats_tensor.std(dim=0) + 1e-6  # [768], tránh std=0
            }
        
        self.log(f'Collected stats for {len(self.class_stats)} classes')
        self.model.train()

    # -------------------------------------------------------
    # BƯỚC 2: Classifier Alignment dùng synthetic features
    # -------------------------------------------------------
    def classifier_alignment(self):
        """
        Sample synthetic features từ Gaussian(mean, std) của mỗi class đã học.
        Retrain chỉ classifier head trên balanced synthetic data.
        """
        if len(self.class_stats) == 0:
            return
        w_before = self.model.last.weight.data.clone()
        self.log(f'Running Classifier Alignment on {len(self.class_stats)} classes...')
        self.model.eval()
        
        # Chỉ optimize classifier (last layer + clf_norm)
        if len(self.config['gpuid']) > 1:
            ca_params = (list(self.model.module.last.parameters()) + 
                        list(self.model.module.clf_norm.parameters()))
        else:
            ca_params = (list(self.model.last.parameters()) + 
                        list(self.model.clf_norm.parameters()))
        
        ca_optimizer = torch.optim.Adam(ca_params, lr=0.01)
        
        # Build balanced synthetic dataset
        all_classes = sorted(self.class_stats.keys())
        
        for epoch in range(self.ca_epochs):
            # Sample synthetic features cho tất cả classes
            syn_feats_list = []
            syn_labels_list = []
            
            for cls_idx in all_classes:
                mean = self.class_stats[cls_idx]['mean']  # [768]
                std  = self.class_stats[cls_idx]['std']   # [768]
                
                if self.gpu:
                    mean = mean.cuda()
                    std  = std.cuda()
                
                # Sample từ Gaussian distribution
                noise = torch.randn(
                    self.ca_n_samples, mean.shape[0],
                    device=mean.device
                )
                syn_feat = mean.unsqueeze(0) + noise * std.unsqueeze(0)  # [N, 768]
                syn_label = torch.full(
                    (self.ca_n_samples,), cls_idx,
                    dtype=torch.long,
                    device=mean.device
                )
                
                syn_feats_list.append(syn_feat)
                syn_labels_list.append(syn_label)
            
            # Concat tất cả classes
            syn_feats  = torch.cat(syn_feats_list,  dim=0)  # [N*C, 768]
            syn_labels = torch.cat(syn_labels_list, dim=0)  # [N*C]
            
            # Shuffle
            perm = torch.randperm(syn_feats.shape[0])
            syn_feats  = syn_feats[perm]
            syn_labels = syn_labels[perm]
            
            # Mini-batch update chỉ trên classifier
            batch_size = 128
            total_ca_loss = 0.0
            n_batches = 0
            
            for start in range(0, syn_feats.shape[0], batch_size):
                end = min(start + batch_size, syn_feats.shape[0])
                feat_batch  = syn_feats[start:end]
                label_batch = syn_labels[start:end]
                
                # Forward qua classifier (không qua backbone/prompt)
                wt_norm = F.normalize(self.model.last.weight, p=2, dim=1)
                logits = torch.matmul(feat_batch, wt_norm.t())
                logits = logits[:, :self.valid_out_dim]
                
                loss = self.criterion(logits, label_batch)
                w_current = self.model.last.weight
                anchor_loss = F.mse_loss(w_current, w_before.detach())
                loss = loss + 0.1 * anchor_loss 
                ca_optimizer.zero_grad()
                loss.backward()
                ca_optimizer.step()

                total_ca_loss += loss.item()
                n_batches += 1
            
            self.log(f'  CA Epoch {epoch+1}/{self.ca_epochs} | Loss: {total_ca_loss/n_batches:.4f}')
        
        self.model.train()
        self.log('Classifier Alignment done.')

    # -------------------------------------------------------
    # Override learn_batch để thêm CA sau mỗi task
    # -------------------------------------------------------
    def learn_batch(self, train_loader, train_dataset, model_save_dir):
        
        # Train bình thường (giữ nguyên từ default.py)
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
                    epoch=self.epoch + 1, total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(
                    loss=losses, acc=acc))
                losses = AverageMeter()
                acc = AverageMeter()

        self.model.train()

        # PPF merge (giữ nguyên từ code gốc)
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

        # -----------------------------------------------
        # CLASSIFIER ALIGNMENT: chạy sau PPF merge
        # -----------------------------------------------
        # Bước 1: Thu thập stats của task hiện tại (dùng merged prompt)
        self.collect_class_stats(train_loader)
        
        # Bước 2: Align classifier trên tất cả classes đã học
        # Chỉ chạy từ task 2 trở đi (task 1 không có old classes)
        if self.last_valid_out_dim > 0:
            self.classifier_alignment()

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