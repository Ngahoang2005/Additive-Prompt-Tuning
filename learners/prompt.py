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
        self.ridge_gamma = learner_config.get('ridge_gamma', 1.0)
        
        # Lưu thông tin tích lũy qua tasks
        # Key: class_idx, Value: dict với 'Phi', 'mu', 'H', 'N'
        self.class_stats = {}
        
        # Tổng hợp để reconstruct classifier
        # Phi_sum = sum_{i<=t} sum_{c in Ci} Phi^{theta_t}_{i,c}
        self.Phi_sum = None   # (d, d)
        self.H_sum   = None   # (d, n_classes_total)

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

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Extract features dùng current merged prompt
    # ─────────────────────────────────────────────────────────────
    def extract_features(self, dataloader, use_train_prompt=False):
        """
        Extract CLS features + labels từ dataloader.
        use_train_prompt=False → dùng global_merged_prompt (inference)
        Returns: features (N, d), labels (N,)
        """
        self.model.eval()
        all_feats, all_labels = [], []

        with torch.no_grad():
            for x, y, _ in dataloader:
                x, y = x.cuda(), y.cuda()
                feat = self.model.feat(
                    x, prompt=self.model.prompt,
                    train=use_train_prompt
                )[:, 0, :]
                feat = self.model.clf_norm(feat)
                all_feats.append(feat)
                all_labels.append(y)

        self.model.train()
        return torch.cat(all_feats), torch.cat(all_labels)

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Tính TSSP — task-wise projection matrix
    # Closed-form: P = (X_old^T X_old + eps*I)^{-1} X_old^T X_new
    # ─────────────────────────────────────────────────────────────
    def compute_tssp(self, feats_old, feats_new, eps=1e-9):
        """
        feats_old: (N, d) — features với prompt cũ
        feats_new: (N, d) — features với prompt mới
        Returns P: (d, d) — task-wise projection
        """
        d = feats_old.shape[1]
        A = feats_old.t() @ feats_old + eps * torch.eye(d, device=feats_old.device)
        B = feats_old.t() @ feats_new
        # Solve A @ P = B → P = A^{-1} B
        P = torch.linalg.solve(A, B)
        return P   # (d, d)

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Tính CIP — category-specific projection
    # P_c = P @ U_c^r U_c^{r,T} (project vào row space của class c)
    # ─────────────────────────────────────────────────────────────
    def compute_cip(self, P_tssp, Phi_c, rank_ratio=0.9):
        """
        P_tssp: (d, d) — task-wise projection
        Phi_c:  (d, d) — uncentered covariance của class c
        rank_ratio: giữ lại eigenvectors giải thích rank_ratio variance
        Returns P_c: (d, d) — category-specific projection
        """
        # SVD của covariance
        U, S, Vh = torch.linalg.svd(Phi_c, full_matrices=False)
        
        # Chọn số rank để giữ lại rank_ratio variance
        total_var = S.sum()
        cumvar = S.cumsum(0)
        r = (cumvar < rank_ratio * total_var).sum().item() + 1
        r = max(1, min(r, len(S)))
        
        U_r = U[:, :r]  # (d, r) — top-r eigenvectors
        
        # P_c = P_tssp @ U_r @ U_r^T
        P_c = P_tssp @ U_r @ U_r.t()
        return P_c   # (d, d)

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Calibrate stats cũ với dual projection
    # Phi_new = P_c^T Phi_old P_c
    # mu_new  = mu_old @ P_c
    # ─────────────────────────────────────────────────────────────
    def calibrate_old_stats(self, P_tssp):
        """
        Cập nhật tất cả class stats cũ với dual projection.
        Thực hiện sau PPF merge, trước khi tính stats mới.
        """
        self.log("Calibrating old class stats with Dual Projection...")
        
        for cls_idx, stats in self.class_stats.items():
            Phi_c = stats['Phi']   # (d, d)
            mu_c  = stats['mu']    # (d,)
            N_c   = stats['N']
            
            # CIP: project P_tssp vào row space của class c
            P_c = self.compute_cip(P_tssp, Phi_c)
            
            # Calibrate covariance: Phi_new = P_c^T Phi_old P_c
            Phi_c_new = P_c.t() @ Phi_c @ P_c
            
            # Calibrate prototype: mu_new = mu_old @ P_c
            mu_c_new = mu_c @ P_c
            
            # Update
            self.class_stats[cls_idx]['Phi'] = Phi_c_new
            self.class_stats[cls_idx]['mu']  = mu_c_new
            
            # Recalculate H từ calibrated mu
            # H_c = N_c * mu_c^T * one_hot(cls)
            # (stored separately in H_sum, update below)

    # ─────────────────────────────────────────────────────────────
    # STEP 5: Tính stats cho task mới
    # ─────────────────────────────────────────────────────────────
    def compute_new_task_stats(self, feats_new, labels, task_classes):
        """
        Tính Phi_c, mu_c, H_c cho các classes mới.
        feats_new: (N, d) — features với merged prompt mới
        labels:    (N,)
        task_classes: list of class indices
        """
        d = feats_new.shape[1]
        n_total = self.valid_out_dim  # tổng số classes đến nay
        
        for cls_idx in task_classes:
            mask = (labels == cls_idx)
            if mask.sum() == 0:
                continue
            
            X_c = feats_new[mask]   # (N_c, d)
            N_c = X_c.shape[0]
            
            Phi_c = X_c.t() @ X_c          # (d, d)
            mu_c  = X_c.mean(dim=0)        # (d,)
            
            self.class_stats[cls_idx] = {
                'Phi': Phi_c,
                'mu':  mu_c,
                'N':   N_c,
            }

    # ─────────────────────────────────────────────────────────────
    # STEP 6: Reconstruct classifier bằng ridge regression
    # W* = (sum Phi_c + gamma*I)^{-1} (sum H_c)
    # ─────────────────────────────────────────────────────────────
    def reconstruct_classifier(self):
        """
        Ridge regression classifier reconstruction.
        W* = (Σ Phi_c + γI)^{-1} (Σ H_c)
        
        H_c = N_c * mu_c (outer product với one-hot → column của W)
        
        Category-wise normalization: normalize mỗi cột của W
        """
        d = next(iter(self.class_stats.values()))['Phi'].shape[0]
        n_classes = self.valid_out_dim
        
        Phi_sum = torch.zeros(d, d).cuda()
        W_cols  = []

        sorted_classes = sorted(self.class_stats.keys())
        
        for cls_idx in sorted_classes:
            stats = self.class_stats[cls_idx]
            Phi_sum += stats['Phi']
            # H_c = N_c * mu_c → cột tương ứng trong W
            W_cols.append(stats['N'] * stats['mu'])

        # W_cols: list of (d,) → stack thành (d, n_classes)
        H_sum = torch.stack(W_cols, dim=1)   # (d, n_classes)
        
        # Solve: (Phi_sum + gamma*I) @ W = H_sum
        A = Phi_sum + self.ridge_gamma * torch.eye(d, device=Phi_sum.device)
        W = torch.linalg.solve(A, H_sum)   # (d, n_classes)
        
        # Category-wise normalization (CN) — từ DPCR Eq.22
        W_norm = W / (W.norm(dim=0, keepdim=True) + 1e-10)
        
        self.log(f'Reconstructed classifier: W shape {W_norm.shape}')
        return W_norm.t()   # (n_classes, d) — same layout as self.model.last.weight