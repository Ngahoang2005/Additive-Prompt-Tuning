#dèualt.py
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

        merge_flag = self.model.prompt.merge_flag

        if merge_flag:
            if self.last_valid_out_dim == 0:
                self.model.prompt.global_merged_prompt = self.model.prompt.prompt_tokens.clone().detach()
            else:
                now_task_p = self.model.prompt.prompt_tokens.clone().detach()
                global_p = self.model.prompt.global_merged_prompt
                merged_p = self.model.prompt.merge_prompt(global_p, now_task_p)
                
                self.model.prompt.global_merged_prompt.data = merged_p
            
            self.log(f'Extracting Prototypes for Task {self.task_count}...')
            self.model.eval()
            
            curr_expert = nn.Parameter(self.model.prompt.prompt_tokens.clone().detach())
            curr_expert.requires_grad = False
            self.model.prompt.expert_prompts[str(self.task_count)] = curr_expert
            
            all_features = []
            all_labels = [] # Bổ sung list lưu nhãn
            
            with torch.no_grad():
                # CHÚ Ý: Lấy thêm y_train để chia class
                for x_train, y_train, _ in train_loader:
                    if self.gpu: x_train = x_train.cuda()
                    feat = self.model.extract_cls_features(x_train, use_merge=True)
                    all_features.append(feat.cpu())
                    all_labels.append(y_train.cpu()) 
            
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # CHIA THEO CLASS ĐỂ TÍNH PROTOTYPE
            unique_classes = torch.unique(all_labels)
            for c in unique_classes:
                c_idx = c.item()
                c_feats = all_features[all_labels == c] # Lọc feature của riêng class này
                
                mu = torch.mean(c_feats, dim=0)
                centered = c_feats - mu
                cov = torch.matmul(centered.t(), centered) / (c_feats.shape[0] - 1)
                
                # Dùng PINV (Nghịch đảo giả) để chống sập nếu class có quá ít ảnh
                cov_inv = torch.linalg.pinv(cov + 1e-4 * torch.eye(cov.shape[0])) 
                
                self.model.prompt.prototypes[c_idx] = {
                    'mean': mu.cuda(),
                    'cov_inv': cov_inv.cuda()
                }
            
            self.log(f'Extracted {len(unique_classes)} class prototypes!')
        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False
        
        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

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
                B = input.shape[0]
                K = 3
                query = model.extract_cls_features(input, use_merge=True)
                
                # BƯỚC 1: TẠO BẢNG ÁNH XẠ TỪ CLASS SANG TASK (Class -> Task)
                if not hasattr(self, 'class_to_task_tensor'):
                    mapping = torch.zeros(self.out_dim, dtype=torch.long)
                    for t_idx, class_list in enumerate(self.tasks):
                        for c in class_list:
                            mapping[c] = t_idx
                    self.class_to_task_tensor = mapping.cuda() # Đẩy lên GPU
                
                # BƯỚC 2: ĐO KHOẢNG CÁCH TỚI TẤT CẢ CÁC CLASS ĐÃ HỌC
                available_classes = list(getattr(model.prompt, 'prototypes', {}).keys())
                num_classes_seen = len(available_classes)
                
                # Khởi tạo ma trận khoảng cách bằng Vô Cực
                dist_matrix = torch.full((B, self.valid_out_dim), float('inf')).cuda()
                
                for c_idx in available_classes:
                    proto = model.prompt.prototypes[c_idx]
                    mu = proto['mean'] # Chỉ cần dùng Mean, bỏ qua cov_inv
                    
                    # TÍNH KHOẢNG CÁCH COSINE (Distance = 1 - Cosine_Sim)
                    # Hàm cosine_similarity trả về [-1, 1] (1 là giống hệt, -1 là ngược hướng)
                    # Lấy 1.0 trừ đi để biến nó thành "Khoảng cách" (Càng nhỏ càng tốt giống Mahalanobis)
                    sim = torch.nn.functional.cosine_similarity(query, mu.unsqueeze(0), dim=1)
                    dist_matrix[:, c_idx] = 1.0 - sim
                
                # BƯỚC 3: XỬ LÝ TOP-K VÀ TÍNH TRỌNG SỐ (WEIGHTED ENSEMBLE)
                actual_K = min(K, num_classes_seen)
                if actual_K == 0: 
                    actual_K = 1 
                    
                # Lấy K class gần nhất (khoảng cách nhỏ nhất)
                topk_dist, topk_classes = torch.topk(dist_matrix, k=actual_K, largest=False, dim=1)
                
                # Dùng Softmax nghịch đảo khoảng cách để ra Trọng số
                topk_dist_stable = topk_dist - topk_dist[:, 0:1] 
                weights = torch.nn.functional.softmax(-topk_dist_stable, dim=1) # Shape: [B, K]
                
                # Ánh xạ Top-K Classes về Top-K Tasks
                topk_tasks = self.class_to_task_tensor[topk_classes] # Shape: [B, K]
                
                # Gộp trọng số cho các Task (scatter_add_)
                task_weights = torch.zeros(B, len(self.tasks)).cuda()
                task_weights.scatter_add_(1, topk_tasks, weights)
                
                # Tracking Accuracy (Dựa vào Task của class Top 1)
                if not hasattr(self, 'routing_correct'):
                    self.routing_correct, self.routing_total = 0, 0
                
                best_task_preds = topk_tasks[:, 0]
                self.routing_correct += (best_task_preds == task.cuda()).sum().item()
                self.routing_total += B
                
                # BƯỚC 4: ENSEMBLE ĐIỂM SỐ TỪ CÁC EXPERT ĐƯỢC CHỌN
                logits_merge = model.forward(input, use_merge=True)[:, :self.valid_out_dim]
                logits_expert = torch.zeros_like(logits_merge)
                
                num_experts = len(getattr(model.prompt, 'expert_prompts', {}))
                for t in range(num_experts):
                    # CHỈ CHẠY expert_id=t cho những bức ảnh mà task t có trọng số > 0
                    sample_mask = task_weights[:, t] > 0
                    if sample_mask.any():
                        out_t = model.forward(input[sample_mask], expert_id=t)[:, :self.valid_out_dim]
                        
                        # Nhân điểm Logit với trọng số niềm tin (Weight)
                        w_t = task_weights[sample_mask, t].unsqueeze(1)
                        logits_expert[sample_mask] += out_t * w_t

                # Chốt hạ: Kết hợp Kiến thức Nền (Merge) và Hội đồng Chuyên gia (Experts)
                #output = logits_merge + logits_expert 
                output = logits_expert
                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                
                # In ra tỷ lệ Routing đúng lúc Test xong
                if i == len(dataloader) - 1 and self.routing_total > 0:
                    self.log(f'>>> Class-Based Routing Acc: {self.routing_correct/self.routing_total * 100:.2f}% (Weighted K={actual_K})')
                    self.routing_correct, self.routing_total = 0, 0
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
        
        # --- BỔ SUNG LƯU PROTOTYPES VÀ EXPERTS ---
        custom_data = {
            'prototypes': getattr(self.model.prompt, 'prototypes', {}),
            'expert_prompts': {k: v.cpu() for k, v in getattr(self.model.prompt, 'expert_prompts', {}).items()}
        }
        torch.save(custom_data, filename + 'custom_data.pth')
        # --- KẾT THÚC BỔ SUNG ---
        
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