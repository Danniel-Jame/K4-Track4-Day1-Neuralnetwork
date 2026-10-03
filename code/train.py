"""train.py — Huấn luyện MLP cho bài toán 7 lớp, shape cố định (xem README mục 3 và GUIDE, "Quy định kiến trúc").

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base).
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # Chọn bằng val
    weight_decay=0.0, momentum=0.9,
    batch_size=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Giữ tính tái lập
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    f1_list = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        denom = 2 * tp + fp + fn
        if denom == 0:
            f1 = 0.0
        else:
            f1 = (2.0 * tp) / denom
        f1_list.append(f1)

    return float(np.mean(f1_list))


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    preds_list = []
    n_samples = X.shape[0]

    for i in range(0, n_samples, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        preds_list.append(preds)

    return torch.cat(preds_list, dim=0)


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        # Chuyển y sang one-hot shape (B, 7) float32
        num_classes = logits.shape[1]
        y_onehot = F.one_hot(y, num_classes=num_classes).float()
        # MSELoss lấy trung bình trên tất cả các phần tử (B * num_classes)
        return F.mse_loss(logits, y_onehot)
    else:
        raise ValueError(f"Loss '{loss_name}' không hợp lệ. Chọn 'ce' hoặc 'mse'.")


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad."""
    model.eval()
    n_samples = X.shape[0]
    total_loss = 0.0
    all_preds = []

    for i in range(0, n_samples, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]

        logits = model(xb)
        # Tính loss theo batch, quy đổi về tổng sum để lấy trung bình chính xác ở cuối
        if loss_name == "ce":
            loss_batch = F.cross_entropy(logits, yb, reduction="sum").item()
        elif loss_name == "mse":
            num_classes = logits.shape[1]
            y_onehot = F.one_hot(yb, num_classes=num_classes).float()
            # F.mse_loss(reduction='sum') cộng tổng tất cả (B * num_classes)
            # để tương đồng chuẩn ta chia cho num_classes cho mỗi mẫu
            loss_batch = F.mse_loss(logits, y_onehot, reduction="sum").item() / num_classes

        total_loss += loss_batch
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds)

    avg_loss = total_loss / n_samples
    y_pred = torch.cat(all_preds, dim=0)

    # Tính Accuracy
    correct = (y_pred == y).sum().item()
    acc = correct / n_samples

    # Dựng ma trận nhầm lẫn 7x7
    y_true_np = y.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    num_classes = 7

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true_np, y_pred_np):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    macro_f1 = macro_f1_from_confusion(cm)

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "macro_f1": float(macro_f1)
    }


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = data["X_tr"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = cfg.get("dropout", 0.0)
    init = cfg.get("init", "he")
    precision = cfg.get("precision", "fp32").lower()
    batch_size = cfg.get("batch_size", cfg.get("batch", 512))
    epochs = cfg.get("epochs", 20)
    loss_name = cfg.get("loss", "ce")
    clip_norm = cfg.get("clip_norm", None)

    # 0. Khởi tạo mô hình
    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)

    # Assert kiểm tra số tham số
    if hidden in EXPECTED_PARAMS:
        assert count_params(model) == EXPECTED_PARAMS[hidden], \
            f"Số tham số {count_params(model)} không khớp {EXPECTED_PARAMS[hidden]}"

    optimizer = build_optimizer(
        name=cfg.get("optimizer", "sgd_momentum"),
        params=model.parameters(),
        lr=cfg.get("lr", 0.01),
        weight_decay=cfg.get("weight_decay", 0.0),
        momentum=cfg.get("momentum", 0.9)
    )

    # Precision setup
    scaler = None
    if precision == "fp16" and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    autocast_dtype = torch.float32
    if precision == "fp16":
        autocast_dtype = torch.float16
    elif precision == "bf16":
        autocast_dtype = torch.bfloat16

    # 1. Đo loss bước 0 trên tập Validation (trước khi update)
    step0_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)
    step0_loss = step0_eval["loss"]

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": []
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    diverged = False

    generator = torch.Generator(device=device).manual_seed(seed)

    # 2. Vòng lặp huấn luyện qua các epoch
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        model.train()
        batch_grad_norms = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size, generator=generator, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if precision in ("fp16", "bf16") and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, loss_name=loss_name)

                if precision == "fp16" and scaler is not None:
                    scaler.scale(loss).backward()
                    if clip_norm is not None:
                        scaler.unscale_(optimizer)
                        gn = clip_gradients(model.parameters(), clip_norm)
                    else:
                        gn = clip_gradients(model.parameters(), None)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    gn = clip_gradients(model.parameters(), clip_norm)
                    optimizer.step()
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name=loss_name)
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            batch_grad_norms.append(gn)

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                print(f"[CẢNH BÁO] Loss bị NaN/Inf tại Epoch {epoch}! Dừng huấn luyện sớm.")
                break

        if diverged:
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - start_time

        # Đánh giá cuối epoch trên train và val (ở chế độ eval mode)
        # Để đảm bảo tốc độ, ta đánh giá trên tập cố định 50,000 mẫu train nếu train quá lớn
        n_tr = data["X_tr"].shape[0]
        if n_tr > 50000:
            eval_tr_res = evaluate(model, data["X_tr"][:50000], data["y_tr"][:50000], loss_name=loss_name)
        else:
            eval_tr_res = evaluate(model, data["X_tr"], data["y_tr"], loss_name=loss_name)

        eval_val_res = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)

        mean_gnorm = float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0

        history["epochs"].append(epoch)
        history["train_loss"].append(eval_tr_res["loss"])
        history["val_loss"].append(eval_val_res["loss"])
        history["val_acc"].append(eval_val_res["acc"])
        history["val_macro_f1"].append(eval_val_res["macro_f1"])
        history["grad_norm"].append(mean_gnorm)
        history["epoch_time_s"].append(epoch_time)

        # Lưu lại state tốt nhất theo val_loss
        if eval_val_res["loss"] < best_val_loss:
            best_val_loss = eval_val_res["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    # Peak memory GPU
    peak_mem_MB = 0.0
    if device.type == "cuda":
        peak_mem_MB = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # Tổng hợp kết quả summary tại best_epoch
    if best_epoch != -1 and best_epoch <= len(history["val_acc"]):
        best_idx = best_epoch - 1
        summary_acc = history["val_acc"][best_idx]
        summary_f1 = history["val_macro_f1"][best_idx]
    else:
        summary_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
        summary_f1 = history["val_macro_f1"][-1] if history["val_macro_f1"] else 0.0

    avg_time_per_epoch = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss) if not math.isinf(best_val_loss) else None,
        "best_epoch": int(best_epoch),
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else None,
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else None,
        "val_acc": float(summary_acc),
        "val_macro_f1": float(summary_f1),
        "time_per_epoch_s": float(avg_time_per_epoch),
        "peak_mem_MB": float(peak_mem_MB),
        "diverged": diverged,
    }

    return {
        "exp_id": cfg.get("exp_id", "exp"),
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("row_id,pred\n")
        for r_id, p in zip(row_id, preds):
            f.write(f"{r_id},{int(p)}\n")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    device = data["X_eval"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = cfg.get("dropout", 0.0)
    init = cfg.get("init", "he")

    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)

    if result.get("best_state") is not None:
        model.load_state_dict(result["best_state"])
    else:
        print("[CẢNH BÁO] Không tìm thấy best_state trong result, dùng trọng số mô hình hiện tại.")

    # Dự đoán trên toàn bộ tập eval (ở chế độ eval mode, float32)
    preds_t = predict(model, data["X_eval"])
    preds_np = preds_t.cpu().numpy()
    eval_row_id = data["eval_row_id"]

    write_predictions(eval_row_id, preds_np, pred_path)
    print(f"Đã lưu kết quả dự đoán Eval thành công tại: {pred_path}")