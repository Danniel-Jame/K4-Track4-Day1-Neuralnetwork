"""optimizer.py — Bản hoàn thiện.

Được dùng torch.optim.* và torch.nn.utils.clip_grad_norm_ (xem README mục 5).
File này gom việc chọn bộ tối ưu và cắt gradient để `train.py` gọn và mọi thí nghiệm công bằng.

Công thức cần hiểu (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8) -> torch.optim.Optimizer:
    """Trả về một torch.optim.Optimizer.

    Các bước:
      1. kiểm tra name nằm trong OPTIMIZERS, nếu không raise ValueError
      2. "sgd"          -> torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
         "sgd_momentum" -> torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
         "adam"         -> torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
         "adamw"        -> torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    """
    name_lower = name.lower()
    if name_lower not in OPTIMIZERS:
        raise ValueError(f"Bộ tối ưu '{name}' không hợp lệ. Chọn một trong: {OPTIMIZERS}")

    if name_lower == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif name_lower == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name_lower == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name_lower == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, name: str | None, total_steps: int, **kwargs):
    """Bộ lập lịch tốc độ học (LR Scheduler).

    Trả về None nếu name là None.
    Hỗ trợ:
      - "cosine": Cosine Annealing LR
      - "cosine_warmup": Warmup tuyến tính trong vài epoch đầu + Cosine Annealing
    """
    if name is None or name == "none":
        return None

    name_lower = name.lower()
    if name_lower == "cosine":
        eta_min = kwargs.get("eta_min", 1e-6)
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    elif name_lower == "cosine_warmup":
        warmup_steps = kwargs.get("warmup_steps", int(total_steps * 0.1))
        eta_min = kwargs.get("eta_min", 1e-6)
        
        scheduler1 = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        scheduler2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min)
        
        return SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])
    else:
        raise ValueError(f"Scheduler '{name}' không được hỗ trợ.")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Giá trị trả về chính là `grad_norm` đo trước khi clip (để theo dõi các gai gradient).
    """
    if max_norm is None or max_norm <= 0:
        # Khi max_norm là None hoặc <= 0, truyền math.inf để đo chuẩn L2 toàn cục mà KHÔNG cắt gradient
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=math.inf)
    else:
        # Cắt gradient nếu vượt quá max_norm. Hàm clip_grad_norm_ vẫn trả về chuẩn gradient BAN ĐẦU trước khi clip.
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)

    return float(total_norm)