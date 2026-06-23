# null_space.py - fix đúng
import torch

class NullSpaceManager:
    def __init__(self, device='cuda', max_samples=2000):
        self.device = device
        self.max_samples = max_samples
        self.feature_bank = []
        self.V = None   # right singular vectors, shape [D, rank]
        self.rank = 0

    def update(self, features):
        if features is None or features.shape[0] == 0:
            return
        features = features.detach().cpu()

        # Accumulate feature bank
        if len(self.feature_bank) == 0:
            self.feature_bank.append(features)
        else:
            all_features = torch.cat(self.feature_bank + [features], dim=0)
            if all_features.shape[0] > self.max_samples:
                idx = torch.randperm(all_features.shape[0])[:self.max_samples]
                all_features = all_features[idx]
            self.feature_bank = [all_features]

        X = self.feature_bank[0]          # [N, D]
        mean = X.mean(dim=0, keepdim=True)
        X_centered = X - mean             # [N, D]

        # SVD đúng: cần right singular vectors V shape [D, D]
        # X = U S V^T  →  V[:,i] là directions trong feature space
        # Dùng torch.linalg.svd full_matrices=False để tiết kiệm memory
        # U: [N, K], S: [K], Vh: [K, D]  với K = min(N, D)
        try:
            _, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            # Vh shape: [K, D], mỗi row là một right singular vector
            # V shape: [D, K] = Vh.T
            V = Vh.T                      # [D, K]
        except Exception:
            # Fallback nếu linalg.svd fail
            _, S, V = torch.svd(X_centered)
            # torch.svd trả về V shape [D, K] đúng rồi

        rank = torch.sum(S > 1e-6).item()
        self.rank = rank
        self.V = V[:, :rank].to(self.device)   # [D, rank]
        print(f"[NullSpace] Updated: D={self.V.shape[0]}, rank={rank}")

    def project_gradient(self, grad):
        """
        Chiếu gradient lên null space của feature matrix.
        Loại bỏ component nằm trong range space (directions đã học).
        grad: [num_prompts, D] hoặc [D]
        """
        if self.V is None or self.V.numel() == 0:
            return grad

        device = grad.device
        V = self.V.to(device)              # [D, rank]
        original_shape = grad.shape

        # Reshape grad về [num_prompts, D]
        if len(original_shape) == 1:
            grad_2d = grad.view(1, -1)     # [1, D]
        elif len(original_shape) == 2:
            grad_2d = grad                 # [num_prompts, D]
        else:
            raise ValueError(f"Unexpected grad shape: {original_shape}")

        D = V.shape[0]
        if grad_2d.shape[1] != D:
            raise ValueError(
                f"Gradient dim {grad_2d.shape[1]} != NullSpace D={D}\n"
                "Please ensure prompt_tokens have the same dimension as features."
            )

        # Project ra khỏi range space: grad_null = grad - V V^T grad
        # V V^T: projection matrix onto range space
        # grad - V V^T grad: component orthogonal to range space (null space)
        VVT_g = grad_2d @ V @ V.T         # [num_prompts, D]
        grad_proj = grad_2d - VVT_g       # [num_prompts, D]

        return grad_proj.view(original_shape)