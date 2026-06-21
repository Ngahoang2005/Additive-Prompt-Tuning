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
        # SCE hyperparameters - default từ paper, không cần tune
        self.sce_alpha = learner_config.get('sce_alpha', 1.0)
        self.sce_beta = learner_config.get('sce_beta', 1.0)

    def symmetric_cross_entropy(self, logits, targets):
        """
        Symmetric Cross-Entropy Loss (SLCA++)
        L_SCE = alpha * CE(p, q) + beta * CE(q, p)
        p = predicted softmax distribution
        q = one-hot label distribution
        """
        num_classes = logits.shape[1]
        
        # Forward CE: alpha * CE(p, q) — standard cross entropy
        ce_loss = self.criterion_fn(logits, targets.long()).mean()
        
        # Reverse CE: beta * CE(q, p) — penalize confident wrong predictions
        # CE(q, p) = -sum(q * log(p)) với q là one-hot
        # = -log(p[target]) đã được tính trong ce_loss
        # Nhưng reverse CE = -sum(p * log(q))
        # q là one-hot nên log(q) = 0 ở mọi nơi trừ target → undefined
        # Dùng cách của paper: clip p và dùng smoothed q
        
        p = F.softmax(logits, dim=1)  # [B, C]
        p = torch.clamp(p, min=1e-7, max=1.0)
        
        # One-hot q với label smoothing nhẹ để tránh log(0)
        q = torch.zeros_like(p)
        q.scatter_(1, targets.long().unsqueeze(1), 1.0)
        q = torch.clamp(q, min=1e-4, max=1.0)  # clip để tránh log(0)
        
        # Reverse CE = -mean(sum(p * log(q), dim=1))
        rce_loss = -torch.mean(torch.sum(p * torch.log(q), dim=1))
        
        return self.sce_alpha * ce_loss + self.sce_beta * rce_loss

    def update_model(self, inputs, targets):
        logits = self.model(inputs, train=True)
        logits = logits[:, :self.valid_out_dim]
        logits[:, :self.last_valid_out_dim] = -float('inf')
        
        # Dùng SCE thay vì CE thông thường
        total_loss = self.symmetric_cross_entropy(logits, targets.long())

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

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
