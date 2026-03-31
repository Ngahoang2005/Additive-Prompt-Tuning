#default.py
from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
import copy
from utils.schedulers import CosineSchedule
from timm.models.layers import trunc_normal_, DropPath
import random
import matplotlib.pyplot as plt


class NormalNN(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, learner_config):
        super(NormalNN, self).__init__()
        self.log = print
        self.config = learner_config
        self.out_dim = learner_config['out_dim']
        self.reset_optimizer = True
        self.overwrite = learner_config['overwrite']
        self.batch_size = learner_config['batch_size']
        self.tasks = learner_config['tasks']
        self.top_k = learner_config['top_k']
        self.model = self.create_model()
        # replay memory parameters
        self.memory_size = self.config['memory']
        self.task_count = 0

        # class balancing
        self.dw = self.config['DW']
        if self.memory_size <= 0:
            self.dw = False

        # supervised criterion
        self.criterion_fn = nn.CrossEntropyLoss(reduction='none')
        
        # cuda gpu
        if learner_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        
        # highest class index from past task
        self.last_valid_out_dim = 0 

        # highest class index from current task
        self.valid_out_dim = 0

        # TẠO KHO CHỨA DATALOADER CŨ CHỈ ĐỂ TEST LAMBDA===
        self.history_loaders = []
        # set up schedules
        self.schedule_type = self.config['schedule_type']
        self.schedule = self.config['schedule']

        # initialize optimizer
        self.init_optimizer()

    ##########################################
    #           MODEL TRAINING               #
    ##########################################
    def learn_batch(self, train_loader, train_dataset, model_save_dir):
        
        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                print("Overwriting ...")
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()
        if need_train:
            
            # data weighting
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()

            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                     
                for i, (x, y, task)  in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()

                    # model update  
                    loss, output= self.update_model(x, y)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())  
                    batch_timer.tic()
                    
                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss,  y.size(0)) 
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

                # reset
                losses = AverageMeter()
                acc = AverageMeter()

                                         
        self.model.train()
        prompt_module = self.model.module.prompt if hasattr(self.model, 'module') else self.model.prompt
        merge_flag = self.model.prompt.merge_flag

        if merge_flag:
            current_fisher = self.compute_fisher(train_loader)
            if self.last_valid_out_dim == 0:
                prompt_module.global_merged_prompt = prompt_module.prompt_tokens.clone().detach()
                prompt_module.global_fisher = current_fisher.clone().detach()
            else:
                now_task_p = prompt_module.prompt_tokens.clone().detach()
                global_p = prompt_module.global_merged_prompt
                global_f = prompt_module.global_fisher
                
                # 1. Tính Delta P
                delta_p = now_task_p - global_p
                
                # 2. Tính hệ số lambda* = (Delta_P^T * Ft * Delta_P) / (Delta_P^T * (Ft + Fold) * Delta_P)
                numerator = torch.sum((delta_p ** 2) * current_fisher)
                denominator = torch.sum((delta_p ** 2) * (current_fisher + global_f))
                
                # Xử lý an toàn nếu mẫu số bằng 0 (tránh ZeroDivisionError)
                if denominator.item() == 0:
                    lambda_star = 0.5
                else:
                    lambda_star = (numerator / denominator).item()
                    # Ép lambda_star phải nằm trong khoảng [0, 1]
                    lambda_star = max(0.0, min(1.0, lambda_star))
                self.log("-" * 40)
                self.log(f"   + Tử số (Sức kéo Task mới): {numerator.item():.6f}")
                self.log(f"   + Mẫu số (Tổng lực cản):   {denominator.item():.6f}")
                self.log(f"   => 🚀 LAMBDA TỐI ƯU (λ*):  {lambda_star:.4f}")
                # self.log("🔍 Đang quét thực nghiệm (Grid Search) để kiểm chứng λ*...")
                
                # # =========================================================
                # # 🛠️ [THỰC NGHIỆM: TRUE CUMULATIVE LOSS - FIX LỖI DATA CON TRỎ]
                # self.log("🔍 Đang quét thực nghiệm TRUE Cumulative Loss trên toàn bộ Data...")
                
                # backup_global = prompt_module.global_merged_prompt.data.clone()
                # self.model.eval()
                
                # test_lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                # if lambda_star not in test_lambdas:
                #     test_lambdas.append(lambda_star)
                # test_lambdas.sort()

                # # --- LÔI TOÀN BỘ DATA TỪ TRONG LƯU TRỮ (ARCHIVE) RA ---
                # # 1. Backup data hiện tại của Dataset để tí nữa trả lại
                # orig_data = train_dataset.data.copy()
                # orig_targets = train_dataset.targets.copy()

                # # 2. Ép dataset hiện tại nuốt toàn bộ data từ Task 0 đến Task t
                # train_dataset.data = np.concatenate([train_dataset.archive[s][0] for s in range(self.task_count + 1)], axis=0)
                # train_dataset.targets = np.concatenate([train_dataset.archive[s][1] for s in range(self.task_count + 1)], axis=0)

                # # 3. Tạo một cái Dataloader tạm thời chứa TẤT CẢ
                # cum_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

                # for test_lbd in test_lambdas:
                #     temp_merged = prompt_module.merge_prompt(global_p, now_task_p, test_lbd)
                #     prompt_module.global_merged_prompt.data = temp_merged
                    
                #     total_cumulative_loss = 0.0
                #     total_samples = 0
                    
                #     with torch.no_grad():
                #         for bx, by, _ in cum_loader:
                #             if self.gpu:
                #                 bx, by = bx.cuda(), by.cuda()
                            
                #             # Tính Logits trên toàn bộ class đã mở khóa
                #             logits = self.model(bx, train=False)[:, :self.valid_out_dim]
                            
                #             # Tính Loss sòng phẳng cho tất cả Data
                #             loss_val = self.criterion(logits, by.long())
                            
                #             total_cumulative_loss += loss_val.item() * bx.size(0)
                #             total_samples += bx.size(0)
                    
                #     avg_cumulative_loss = total_cumulative_loss / total_samples
                #     marker = " <=== (ĐÁY LÝ THUYẾT TOÁN HỌC)" if test_lbd == lambda_star else ""
                #     self.log(f"   * Thử λ = {test_lbd:.4f} | TRUE Cum. Loss: {avg_cumulative_loss:.5f} {marker}")
                
                # # --- DỌN DẸP & TRẢ LẠI HIỆN TRƯỜNG ---
                # train_dataset.data = orig_data
                # train_dataset.targets = orig_targets
       
                # # =========================================================
                # # Trả lại nguyên vẹn trạng thái cũ để đi tiếp
                # prompt_module.global_merged_prompt.data = backup_global
                # self.model.train()
                # =========================================================
                self.log("-" * 40)

                # 3. Trộn Prompt với hệ số vừa tìm được
                merged_p = prompt_module.merge_prompt(global_p, now_task_p, lambda_star)
                prompt_module.global_merged_prompt.data = merged_p
                
                # 4. Cập nhật Tri thức Prior: F_mới = F_cũ + F_hiệntại
                prompt_module.global_fisher.data = global_f + current_fisher
            # ====================================================

        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False
        
        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))
        self.history_loaders.append(train_loader)
        try:
            return batch_time.avg
        except:
            return None
        
    def criterion(self, logits, targets): # data_weights [32]
        loss_supervised = (self.criterion_fn(logits, targets.long())).mean()
        return loss_supervised 

    def update_model(self, inputs, targets):
        
        logits = self.forward(inputs)
        total_loss = self.criterion(logits, targets.long())

        self.optimizer.zero_grad()
       
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), logits

    def validation(self, dataloader, model=None, task_in = None, task_metric='acc',  verbal = True, task_global=False):
        if model is None:
            model = self.model

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()

        err_cnt = 0
        old_cnt = 0
        X = []
        Y = []
        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            if task_in is None:
                output = model.forward(input)[:, :self.valid_out_dim]
                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
            else:
                mask = target >= task_in[0]
                mask_ind = mask.nonzero().view(-1)
                input, target = input[mask_ind], target[mask_ind]

                mask = target < task_in[-1]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]
                if len(target) > 1:
                    if task_global:
                        output = model.forward(input,local_test=False)[:, :self.valid_out_dim]
                        acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                    else:
                        output = model.forward(input,local_test=True)[:, task_in]
                        acc = accumulate_acc(output, target-task_in[0], task, acc, topk=(self.top_k,))
        
        
        model.train(orig_mode)

        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    ##########################################
    #             MODEL UTILS                #
    ##########################################
    def save_model(self, filename):
        model_state = self.model.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving class model to:', filename)
        torch.save(model_state, filename + 'class.pth')
        self.log('=> Save Done')

    def load_model(self, filename):
        model_dict = torch.load(filename + 'class.pth')
        self.model.load_state_dict(model_dict, strict=False)

        self.log('=> Load Done')
        if self.gpu:
            self.model = self.model.cuda()
        self.model.eval()

    def load_pretrained(self, filename):
        model_dict = torch.load(filename + 'class.pth')
        new_state_dict = {}
        for key, value in model_dict.items():
            new_key = key.replace("module.", "")  # Remove all instances of "module."
            new_state_dict[new_key] = value
        
        new_state_dict["last.weight"] = self.model.last.weight.data
        new_state_dict["last.bias"] = self.model.last.bias.data
        new_state_dict["last2.weight"] = self.model.last2.weight.data
        new_state_dict["last2.bias"] = self.model.last2.bias.data
        
        self.model.load_state_dict(new_state_dict, strict=False)

        self.log('=> Load Done')
        if self.gpu:
            self.model = self.model.cuda()
        self.model.eval()

    def load_model_other(self, filename, model):
        model.load_state_dict(torch.load(filename + 'class.pth'))
        if self.gpu:
            model = model.cuda()
        return model.eval()

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        optimizer_arg = {'params':self.model.parameters(),
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
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, tasks=self.tasks)
        return model

    def print_model(self):
        self.log(self.model)
        self.log('#parameter of model:', self.count_parameter())
    
    def reset_model(self):
        self.model.apply(weight_reset)

    def forward(self, x):
        return self.model.forward(x)[:, :self.valid_out_dim]

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        return out
    
    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())   

    def count_memory(self, dataset_size):
        return self.count_parameter() + self.memory_size * dataset_size[0]*dataset_size[1]*dataset_size[2]

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log("Running on:", device)
        return device

    def pre_steps(self):
        pass

class FinetunePlus(NormalNN):

    def __init__(self, learner_config):
        super(FinetunePlus, self).__init__(learner_config)

    def update_model(self, inputs, targets, target_KD = None):

        # get output
        logits = self.forward(inputs)

        # standard ce
        logits[:,:self.last_valid_out_dim] = -float('inf')
        total_loss = self.criterion(logits, targets.long())

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), logits

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def accumulate_acc(output, target, task, meter, topk):
    meter.update(accuracy(output, target, topk), len(target))
    return meter