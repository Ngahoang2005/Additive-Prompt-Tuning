# utils/null_space.py
import torch
import torch.nn as nn

class NullSpaceManager:
    """
    Quản lý không gian Null cho Prompt Tuning trong Continual Learning.
    Dựa trên cơ chế của NSP2 (NeurIPS 2024) [reference:2].
    """
    def __init__(self, feature_dim, device='cuda', max_samples=2000):
        self.feature_dim = feature_dim      # Chiều của đặc trưng (ví dụ: 768)
        self.device = device
        self.max_samples = max_samples      # Giới hạn số mẫu lưu trữ để tránh tràn memory
        self.feature_bank = []              # Lưu đặc trưng của các task cũ
        self.U = None                       # Ma trận cơ sở trực chuẩn của không gian con đặc trưng
        self.rank = 0                       # Hạng (rank) của không gian con

    def update(self, features):
        """
        Cập nhật không gian Null dựa trên đặc trưng của task mới.
        features: Tensor shape (N, D) với N là số mẫu, D là feature_dim.
        """
        if features is None or features.shape[0] == 0:
            return

        # Chuyển features về CPU để tiết kiệm GPU memory (có thể để GPU nếu đủ)
        features = features.cpu()

        # Thêm features mới vào bank, giới hạn số mẫu
        if len(self.feature_bank) == 0:
            self.feature_bank.append(features)
        else:
            all_features = torch.cat(self.feature_bank + [features], dim=0)
            if all_features.shape[0] > self.max_samples:
                idx = torch.randperm(all_features.shape[0])[:self.max_samples]
                all_features = all_features[idx]
            self.feature_bank = [all_features]

        # Xây dựng không gian con từ features đã thu thập
        X = self.feature_bank[0]  # shape: (N, D)
        # Chuẩn hóa về mean = 0 (centering)
        mean = X.mean(dim=0, keepdim=True)
        X_centered = X - mean

        # Phân rã SVD: X_centered = U @ S @ V^T
        # U: ma trận trái (D, D), chứa các vector cơ sở của không gian con
        U, S, _ = torch.svd(X_centered)

        # Xác định rank dựa trên ngưỡng năng lượng (giữ các thành phần chính)
        # Giữ tất cả singular values > 1e-6 (có thể điều chỉnh)
        rank = torch.sum(S > 1e-6).item()
        self.rank = rank
        self.U = U[:, :rank].to(self.device)  # Lấy rank vector đầu tiên

        print(f"[NullSpace] Updated: rank={rank}, feature_dim={self.feature_dim}")

    def project_gradient(self, grad):
        """
        Chiếu gradient lên không gian Null.
        grad: Tensor gradient cần chiếu (shape phù hợp với prompt).
        Trả về gradient đã được chiếu.
        """
        if self.U is None or self.U.numel() == 0:
            # Chưa có không gian Null, giữ nguyên gradient
            return grad

        # Đảm bảo grad và U cùng device
        device = grad.device
        U = self.U.to(device)

        # grad có thể có shape (num_prompts, feature_dim) hoặc (num_prompts * feature_dim,)
        # Chúng ta sẽ làm việc với grad dạng phẳng (flatten) để dễ tính toán
        original_shape = grad.shape
        grad_flat = grad.view(-1)  # (num_prompts * feature_dim,)

        # Phép chiếu lên không gian Null: g_proj = g - U (U^T g)
        # U có shape (D, rank), với D = feature_dim
        # Tuy nhiên, grad_flat có chiều là (num_prompts * D)
        # Ta cần chiếu từng prompt một cách độc lập
        num_prompts = original_shape[0] if len(original_shape) == 2 else 1
        D = U.shape[0]

        # Reshape grad về (num_prompts, D)
        grad_reshaped = grad_flat.view(num_prompts, D)

        # Tính U^T * g cho từng prompt: (num_prompts, rank)
        U_g = torch.mm(grad_reshaped, U)  # (num_prompts, rank)

        # Tính U * (U^T * g): (num_prompts, D)
        U_U_g = torch.mm(U_g, U.T)  # (num_prompts, D)

        # Trừ đi thành phần trong không gian con: g_proj = g - U * U^T * g
        grad_proj = grad_reshaped - U_U_g

        # Trả về shape ban đầu
        return grad_proj.view(original_shape)