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
                feat = F.normalize(feat, dim=1)   # thêm dòng này
                all_feats.append(feat)
                all_labels.append(y)

        self.model.train()
        return torch.cat(all_feats), torch.cat(all_labels)
    # ─────────────────────────────────────────────────────────────
    # STEP 2: Tính TSSP — task-wise projection matrix
    # Closed-form: P = (X_old^T X_old + eps*I)^{-1} X_old^T X_new
    # ─────────────────────────────────────────────────────────────
    def compute_tssp(self, feats_old, feats_new, eps=1.0):
        """
        eps lớn hơn để tránh ill-conditioned matrix.
        Default eps=1e-9 quá nhỏ → P explode khi features không full rank.
        """
        feats_old_n = F.normalize(feats_old, dim=1)
        feats_new_n = F.normalize(feats_new, dim=1)

        d = feats_old_n.shape[1]
        # eps=1.0 tương đương ridge regression với gamma=1
        # Đảm bảo P bounded: ||P|| <= ||feats_old||/eps
        A = feats_old_n.t() @ feats_old_n \
            + eps * torch.eye(d, device=feats_old_n.device)
        B = feats_old_n.t() @ feats_new_n
        P = torch.linalg.solve(A, B)

        self.log(
            f'TSSP ||P - I||='
            f'{(P - torch.eye(d, device=P.device)).norm().item():.4f}'
        )
        return P    # ─────────────────────────────────────────────────────────────
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
        self.log("Calibrating old class stats with Dual Projection...")
        
        for cls_idx, stats in self.class_stats.items():
            Phi_c = stats['Phi']
            mu_c  = stats['mu']

            # CIP
            P_c = self.compute_cip(P_tssp, Phi_c)

            # Clamp P_c để tránh explode
            P_c_norm = P_c / (P_c.norm() + 1e-10)
            # Scale về đơn vị hợp lý
            scale = min(P_c.norm().item(), 2.0)
            P_c = P_c_norm * scale

            # Calibrate
            Phi_c_new = P_c.t() @ Phi_c @ P_c
            mu_c_new  = mu_c @ P_c

            # Re-normalize mu sau calibration
            mu_c_new = F.normalize(mu_c_new.unsqueeze(0), dim=1).squeeze(0)

            self.class_stats[cls_idx]['Phi'] = Phi_c_new
            self.class_stats[cls_idx]['mu']  = mu_c_new
    # ─────────────────────────────────────────────────────────────
    # STEP 5: Tính stats cho task mới
    # ─────────────────────────────────────────────────────────────
    def compute_new_task_stats(self, feats_new, labels, task_classes):
        """
        feats_new phải đã được normalize (L2) trước khi truyền vào.
        """
        for cls_idx in task_classes:
            mask = (labels == cls_idx)
            if mask.sum() == 0:
                self.log(f'WARNING: class {cls_idx} has 0 samples!')
                # Fallback: zero prototype
                d = feats_new.shape[1]
                self.class_stats[cls_idx] = {
                    'Phi': torch.eye(d, device=feats_new.device) * 1e-6,
                    'mu':  torch.zeros(d, device=feats_new.device),
                    'N':   0,
                }
                continue

            X_c = feats_new[mask]
            N_c = X_c.shape[0]
            Phi_c = X_c.t() @ X_c
            mu_c  = X_c.mean(dim=0)

            self.class_stats[cls_idx] = {
                'Phi': Phi_c,
                'mu':  mu_c,
                'N':   N_c,
            }    # ─────────────────────────────────────────────────────────────
    # STEP 6: Reconstruct classifier bằng ridge regression
    # W* = (sum Phi_c + gamma*I)^{-1} (sum H_c)
    # ─────────────────────────────────────────────────────────────
    def reconstruct_classifier(self):
        d = next(iter(self.class_stats.values()))['Phi'].shape[0]
        n_classes = self.valid_out_dim   # dùng valid_out_dim, không phải len(class_stats)

        Phi_sum = torch.zeros(d, d).cuda()
        # Khởi tạo W với đủ n_classes columns — zero cho classes chưa có stats
        H_mat = torch.zeros(d, n_classes).cuda()

        for cls_idx, stats in self.class_stats.items():
            if cls_idx >= n_classes:
                continue
            Phi_sum += stats['Phi']
            H_mat[:, cls_idx] = stats['N'] * stats['mu']

        A = Phi_sum + self.ridge_gamma * torch.eye(d, device=Phi_sum.device)
        W = torch.linalg.solve(A, H_mat)   # (d, n_classes)

        # Category-wise normalization
        W_norm = W / (W.norm(dim=0, keepdim=True) + 1e-10)

        self.log(f'Reconstructed classifier: W shape {W_norm.t().shape}, '
                 f'classes with stats: {len(self.class_stats)}/{n_classes}')
        return W_norm.t()   # (n_classes, d)