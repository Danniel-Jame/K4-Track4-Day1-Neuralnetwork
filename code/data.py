"""data.py — Bản hoàn thiện xử lý dữ liệu cho Lab Day 1.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    train_path = os.path.join(processed_dir, "train.npz")
    eval_path = os.path.join(processed_dir, "eval.npz")

    if not os.path.exists(train_path) or not os.path.exists(eval_path):
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu tại '{processed_dir}'. "
            f"Vui lòng chạy 'python scripts/split_data.py' trước!"
        )

    train_data = np.load(train_path)
    eval_data = np.load(eval_path)

    X_train_full = train_data["X"].astype(np.float32)
    y_train_full = train_data["y"].astype(np.int64)

    X_eval = eval_data["X"].astype(np.float32)
    y_eval = eval_data["y"].astype(np.int64)
    eval_row_id = eval_data["row_id"]

    # Assert shape và dtype đúng quy ước
    assert X_train_full.ndim == 2 and X_train_full.shape[1] == 54, f"X_train_full shape sai: {X_train_full.shape}"
    assert y_train_full.ndim == 1 and len(y_train_full) == X_train_full.shape[0], "Kích thước y_train_full không khớp X_train_full"
    assert X_train_full.dtype == np.float32, f"X_train_full dtype sai: {X_train_full.dtype}"
    assert y_train_full.dtype == np.int64, f"y_train_full dtype sai: {y_train_full.dtype}"

    assert X_eval.ndim == 2 and X_eval.shape[1] == 54, f"X_eval shape sai: {X_eval.shape}"
    assert y_eval.ndim == 1 and len(y_eval) == X_eval.shape[0], "Kích thước y_eval không khớp X_eval"
    assert len(eval_row_id) == X_eval.shape[0], "Kích thước eval_row_id không khớp X_eval"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=val_fraction,
        stratify=y,
        random_state=seed,
        shuffle=True
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Lý do không được tính trên val/eval: để tránh rò rỉ thông tin (data leakage) từ tập kiểm thử.
    """
    numeric_data = X_tr[:, :N_NUMERIC]
    mean = np.mean(numeric_data, axis=0)
    std = np.std(numeric_data, axis=0)
    # Tránh chia cho 0 nếu có cột std = 0 bằng cách thay std = 0 thành 1.0
    std[std == 0.0] = 1.0
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ để tránh ghi đè dữ liệu gốc.
    """
    X_scaled = X.copy()
    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / std
    return X_scaled


def prepare_data(device: str | torch.device, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và mảng numpy: eval_row_id
    """
    # 1. Nạp và tách validation
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction=val_fraction, seed=seed)

    # 2. Tính mean/std chỉ trên X_tr và áp dụng cho cả 3 tập
    mean, std = fit_standardizer(X_tr)
    X_tr_scaled = apply_standardizer(X_tr, mean, std)
    X_val_scaled = apply_standardizer(X_val, mean, std)
    X_eval_scaled = apply_standardizer(X_eval, mean, std)

    # 3. Chuyển sang PyTorch Tensor và đưa lên thiết bị (device)
    device = torch.device(device)
    X_tr_t = torch.tensor(X_tr_scaled, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.int64, device=device)

    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.int64, device=device)

    X_eval_t = torch.tensor(X_eval_scaled, dtype=torch.float32, device=device)
    y_eval_t = torch.tensor(y_eval, dtype=torch.int64, device=device)

    # 4. In ra thông tin thống kê và Baseline Accuracy "đoán lớp đa số"
    counts = np.bincount(y_val)
    majority_class = np.argmax(counts)
    majority_acc = counts[majority_class] / len(y_val)

    print("=== ĐÃ CHUẨN BỊ DỮ LIỆU THÀNH CÔNG ===")
    print(f"Device      : {device}")
    print(f"Train set   : {X_tr_t.shape[0]} mẫu")
    print(f"Val set     : {X_val_t.shape[0]} mẫu")
    print(f"Eval set    : {X_eval_t.shape[0]} mẫu")
    print(f"Val Majority-class Acc (lớp {majority_class}): {majority_acc:.4f} (Mốc cơ bản tối thiểu)")
    print("======================================")

    return {
        "X_tr": X_tr_t,
        "y_tr": y_tr_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_eval": X_eval_t,
        "y_eval": y_eval_t,
        "eval_row_id": eval_row_id,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    """
    n_samples = X.shape[0]
    if shuffle:
        perm = torch.randperm(n_samples, generator=generator, device=X.device)
    else:
        perm = torch.arange(n_samples, device=X.device)

    for i in range(0, n_samples, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]