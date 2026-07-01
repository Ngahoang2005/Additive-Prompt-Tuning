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
        self.ccl_alpha  = learner_config.get('ccl_alpha', 0.0)
        self.ccl_tau1   = learner_config.get('ccl_tau1', 1.05)   # xem lưu ý scale bên dưới
        self.ccl_margin = learner_config.get('ccl_margin', 0.02) # xem lưu ý scale bên dưới
        super(Prompt_Learner, self).__init__(learner_config)

    def update_model(self, inputs, targets):
        # logits
        logits_full = self.model(inputs, train=True)
        logits = logits_full[:, :self.valid_out_dim]

        # CE loss như cũ, nhưng làm việc trên bản clone để giữ logits gốc cho CCL
        ce_logits = logits.clone()
        ce_logits[:, :self.last_valid_out_dim] = -float('inf')
        total_loss = self.criterion(ce_logits, targets.long())
        #total_loss = self.criterion(logits, targets.long())       
        if self.ccl_alpha > 0 and len(self.task_boundaries) > 1:
            ccl_loss = self._ccl_loss(logits)
            total_loss = total_loss + self.ccl_alpha * ccl_loss
        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.detach(), logits
    def _ccl_loss(self, logits):
        """
        logits: [B, valid_out_dim], CHƯA mask -inf.
        task_boundaries[-1] = block của task hiện tại; các phần tử trước là task cũ.
        """
        prev_boundaries = self.task_boundaries[:-1]
        cur_s, cur_e = self.task_boundaries[-1]
        ell_t = logits[:, cur_s:cur_e]

        with torch.no_grad():
            max_t = ell_t.max(dim=1, keepdim=True)[0]

        total = logits.new_zeros(())
        for (s, e) in prev_boundaries:
            ell_i = logits[:, s:e]
            with torch.no_grad():
                max_i = ell_i.max(dim=1, keepdim=True)[0]
                # Eq.4: chỉ bật smoothing khi logit max của task cũ áp sát/vượt task hiện tại
                fire = (max_i + self.ccl_margin) >= max_t
                tau = torch.where(fire,
                                   torch.full_like(max_i, self.ccl_tau1),
                                   torch.ones_like(max_i))
                p_soft = F.softmax(ell_i / tau, dim=1)   # target bị block gradient
            log_p = F.log_softmax(ell_i, dim=1)          # nhánh có gradient
            le_i = -(p_soft * log_p).sum(dim=1)
            total = total + le_i.mean()

        return total / len(prev_boundaries)
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

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, ema_coeff=self.ema_coeff, prompt_flag = 'apt', prompt_param=self.prompt_param, tasks=self.tasks)
        return model

