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
        self.ccl_alpha  = learner_config.get('ccl_alpha', 0.3)
        self.ccl_margin = learner_config.get('ccl_margin', 0.1)
        self.ccl_tau    = learner_config.get('ccl_tau', 1.15)
    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, ema_coeff=self.ema_coeff, prompt_flag = 'apt', prompt_param=self.prompt_param, tasks=self.tasks)
        return model
    def update_model(self, inputs, targets):
        logits = self.model(inputs, train=True)
        logits = logits[:, :self.valid_out_dim]

        # ----- Cross entropy chỉ trên class mới (giữ nguyên hành vi gốc của APT) -----
        ce_logits = logits.clone()
        ce_logits[:, :self.last_valid_out_dim] = -float('inf')
        loss_ce = self.criterion(ce_logits, targets.long())

        total_loss = loss_ce

        # ----- Classifier Consistency Learning (CCL), chỉ áp dụng khi đã có task cũ -----
        if self.last_valid_out_dim > 0:
            loss_ccl = self.compute_ccl_loss(logits)
            total_loss = total_loss + self.ccl_alpha * loss_ccl

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    def compute_ccl_loss(self, logits):
        """
        Classifier Consistency Learning, adapt từ CPrompt cho trường hợp
        APT chỉ có MỘT classifier dùng chung (không có classifier pool riêng từng task).
        Thay vì lặp qua từng classifier C_k, ta slice logits theo [old_range] vs [new_range].
        """
        old_logits = logits[:, :self.last_valid_out_dim]               # (B, n_old)
        new_logits = logits[:, self.last_valid_out_dim:self.valid_out_dim]  # (B, n_new)

        max_old = old_logits.max(dim=1)[0]          # (B,)
        max_new = new_logits.max(dim=1)[0].detach()  # block gradient từ new, giống CPrompt block ground

        # τ thấp (off) khi old đã thấp hơn new đủ margin, τ cao (on) khi old đang lấn át new
        bool_ok = max_new > (max_old + self.ccl_margin)
        tau = torch.ones_like(max_old) 
        tau[~bool_ok] = self.ccl_tau
        tau = tau.unsqueeze(1).expand_as(old_logits)

        # smooth target (no grad qua nhánh temperature-scaled, giống CPrompt .detach())
        ground = F.softmax(old_logits.detach() / tau, dim=1)
        log_p  = F.log_softmax(old_logits, dim=1)

        loss_ccl = -(ground * log_p).sum(dim=1).mean()
        return loss_ccl

