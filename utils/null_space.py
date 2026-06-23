# utils/null_space.py
import torch

class NullSpaceManager:
    def __init__(self, device='cuda', max_samples=2000):
        self.device = device
        self.max_samples = max_samples
        self.feature_bank = []
        self.U = None
        self.rank = 0
        self.D = None   # sẽ được cập nhật từ features

    def update(self, features):
        if features is None or features.shape[0] == 0:
            return
        features = features.cpu()

        # Cập nhật chiều D từ features
        if self.D is None:
            self.D = features.shape[1]
        else:
            assert features.shape[1] == self.D, \
                f"Feature dim mismatch: {features.shape[1]} vs {self.D}"

        # Thêm vào bank (giới hạn số mẫu)
        if len(self.feature_bank) == 0:
            self.feature_bank.append(features)
        else:
            all_features = torch.cat(self.feature_bank + [features], dim=0)
            if all_features.shape[0] > self.max_samples:
                idx = torch.randperm(all_features.shape[0])[:self.max_samples]
                all_features = all_features[idx]
            self.feature_bank = [all_features]

        X = self.feature_bank[0]
        mean = X.mean(dim=0, keepdim=True)
        X_centered = X - mean
        U, S, _ = torch.svd(X_centered)
        rank = torch.sum(S > 1e-6).item()
        self.rank = rank
        self.U = U[:, :rank].to(self.device)
        print(f"[NullSpace] Updated: D={self.D}, rank={rank}")

    def project_gradient(self, grad):
        if self.U is None or self.U.numel() == 0:
            return grad

        device = grad.device
        U = self.U.to(device)          # (D, rank)

        # Đảm bảo grad có shape (num_prompts, D)
        original_shape = grad.shape
        if len(original_shape) == 1:
            grad = grad.view(1, -1)
        elif len(original_shape) == 2:
            # Kiểm tra khớp chiều
            if grad.shape[1] != U.shape[0]:
                raise ValueError(
                    f"Gradient dim {grad.shape[1]} != NullSpace D={U.shape[0]}\n"
                    "Please ensure prompt_tokens have the same dimension as feature."
                )
        else:
            raise ValueError(f"Unexpected grad shape: {original_shape}")

        num_prompts = grad.shape[0]

        # Phép chiếu: g_proj = g - U (U^T g)
        U_g = torch.mm(grad, U)        # (num_prompts, rank)
        U_U_g = torch.mm(U_g, U.T)     # (num_prompts, D)
        grad_proj = grad - U_U_g

        return grad_proj.view(original_shape)