#prompt.py
from __future__ import print_function
import torch
from torch import nn
import models
import torch.nn as nn
from utils.metric import accuracy, AverageMeter, Timer
from .default import NormalNN, weight_reset, accumulate_acc
from utils.schedulers import CosineSchedule

class Prompt_Learner(NormalNN):
    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        self.ema_coeff = learner_config['ema_coeff']
        super(Prompt_Learner, self).__init__(learner_config)
        self.task_anchors = nn.ParameterDict()

    # def update_model(self, inputs, targets):
    #     # logits
    #     logits = self.model(inputs, train=True)
        
    #     logits = logits[:,:self.valid_out_dim]
    #     logits[:,:self.last_valid_out_dim] = -float('inf')
    #     total_loss = self.criterion(logits, targets.long())       
        
    #     # step
    #     self.optimizer.zero_grad()
    #     total_loss.backward()
    #     self.optimizer.step()
        
    #     return total_loss.detach(), logits
    # ortho
    def update_model(self, inputs, targets):
        # 1. Khởi tạo Mỏ neo cho Task hiện tại nếu chưa có
        task_str = str(self.task_count)
        if task_str not in self.task_anchors:
            # Khởi tạo vector ngẫu nhiên 768 chiều (giả sử ViT-B/16 dùng 768)
            anchor = nn.Parameter(torch.randn(768).cuda() if self.gpu else torch.randn(768))
            self.task_anchors[task_str] = anchor
            self.init_optimizer() # Refresh optimizer để nó nhận diện tham số mới

        # 2. Lấy Feature (để định tuyến) và Logit (để phân loại)
        # Tùy kiến trúc ViT của bạn, hàm model() có thể trả về cả feature nếu ta chỉnh nhẹ, 
        # hoặc gọi hàm extract_cls_features như lúc Validation.
        features = self.model.extract_cls_features(inputs, use_merge=True)
        logits = self.model(inputs, train=True)[:, :self.valid_out_dim]
        logits[:, :self.last_valid_out_dim] = -float('inf')
        
        # 3. Tính Loss Phân loại (Nhiệm vụ chính)
        loss_ce = self.criterion(logits, targets.long())       
        
        # =================================================================
        # 4. HỌC THUYẾT MỎ NEO (ANCHOR GUIDANCE LOSS)
        # =================================================================
        curr_anchor = self.task_anchors[task_str]
        
        # Lực Kéo (Pull): Ép feature của ảnh tiến về gần Mỏ neo của Task hiện tại
        # Cosine tiến về 1 -> (1 - Cosine) tiến về 0
        sim_pull = torch.nn.functional.cosine_similarity(features, curr_anchor.unsqueeze(0), dim=1)
        loss_pull = (1.0 - sim_pull).mean()
        
        loss_push_ortho = 0.0
        if self.task_count > 0:
            for t in range(self.task_count):
                old_anchor = self.task_anchors[str(t)]
                
                # Lực Đẩy 1 (Push Features): Đẩy feature ảnh hiện tại ra xa các Mỏ neo cũ
                sim_push = torch.nn.functional.cosine_similarity(features, old_anchor.unsqueeze(0), dim=1)
                loss_push_ortho += torch.abs(sim_push).mean() # Ép góc về 90 độ
                
                # Lực Đẩy 2 (Ortho Anchors): Bản thân các Mỏ neo cũng phải vuông góc với nhau
                sim_anchors = torch.nn.functional.cosine_similarity(curr_anchor.unsqueeze(0), old_anchor.unsqueeze(0))
                loss_push_ortho += torch.abs(sim_anchors).squeeze()
        
        # Trọng số cân bằng (Cần tuning, thử 0.1)
        lambda_guide = 0.1
        total_loss = loss_ce + lambda_guide * (loss_pull + loss_push_ortho)
        
        print(f'\r[Anchor Guide] Pull: {loss_pull.item():.4f} | Push: {loss_push_ortho if isinstance(loss_push_ortho, float) else loss_push_ortho.item():.4f} | CE: {loss_ce.item():.4f}', end='')
        # =================================================================
        
        # 5. Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.detach(), logits
        
        return total_loss.detach(), logits
    def get_attn_heatmap(self, inputs):
        return 

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters()) + list(self.task_anchors.parameters())
        else:
            # ---> BỔ SUNG list(self.task_anchors.parameters()) VÀO ĐÂY <---
            params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters()) + list(self.task_anchors.parameters())
        if hasattr(self, 'task_anchors'):
            params_to_opt = params_to_opt + list(self.task_anchors.parameters())
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

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, ema_coeff=self.ema_coeff, prompt_flag = 'apt', prompt_param=self.prompt_param, tasks=self.tasks)
        return model

