# prompt.py

from __future__ import print_function
import torch
import models
from utils.metric import accuracy, AverageMeter, Timer
from .default import NormalNN, weight_reset, accumulate_acc
from utils.schedulers import CosineSchedule

# 1. Thêm công cụ AMP của PyTorch
from torch.cuda.amp import autocast, GradScaler

class Prompt_Learner(NormalNN):
    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        self.ema_coeff = learner_config['ema_coeff']
        super(Prompt_Learner, self).__init__(learner_config)
        
        # 2. Khởi tạo bộ Scaler để phóng to/thu nhỏ gradient trong môi trường 16-bit
        self.scaler = GradScaler()

    def update_model(self, inputs, targets):
        self.optimizer.zero_grad()
        
        # 3. Bật autocast() để ép GPU tính toán ma trận ở chế độ 16-bit (Siêu nhanh)
        with autocast():
            # logits
            logits = self.model(inputs, train=True)
            
            logits = logits[:,:self.valid_out_dim]
            logits[:,:self.last_valid_out_dim] = -float('inf')
            total_loss = self.criterion(logits, targets.long())       
        
        # 4. Backward và cập nhật trọng số an toàn qua Scaler
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
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
    # === [BECAME ADAPTION] HÀM TÍNH ĐỘ CONG CỦA LOSS (FISHER) ===
    def compute_fisher(self, dataloader):
        self.model.eval() # Bật chế độ eval để tránh update BN/Dropout
        
        # Xử lý an toàn cho chế độ Multi-GPU (DataParallel)
        prompt_module = self.model.module.prompt if hasattr(self.model, 'module') else self.model.prompt
        fisher = torch.zeros_like(prompt_module.prompt_tokens.data)
        
        for i, (input, target, _) in enumerate(dataloader):
            if self.gpu:
                input = input.cuda()
                target = target.cuda()
            
            self.optimizer.zero_grad()
            
            # Forward pass bình thường (không dùng autocast để gradient tính ra chính xác tuyệt đối)
            logits = self.model(input, train=True)[:, :self.valid_out_dim]
            logits[:, :self.last_valid_out_dim] = -float('inf')
            
            # Tính đạo hàm
            loss = self.criterion(logits, target.long())
            loss.backward()
            
            # Tích lũy bình phương đạo hàm vào Fisher: F = E[(nabla L)^2]
            fisher += (prompt_module.prompt_tokens.grad.data ** 2) / len(dataloader)
            
        self.optimizer.zero_grad()
        return fisher.detach()
    # ==============================================================

class APT_Learner(Prompt_Learner):

    def __init__(self, learner_config):
        super(APT_Learner, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, ema_coeff=self.ema_coeff, prompt_flag = 'apt', prompt_param=self.prompt_param, tasks=self.tasks)
        return model