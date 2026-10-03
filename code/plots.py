"""plots.py — Vẽ biểu đồ cho từng thí nghiệm và so sánh nhiều thí nghiệm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    
    Tự động đánh dấu best_epoch bằng đường nét đứt màu đỏ.
    """
    exp_id = result.get("exp_id", "unknown")
    cfg = result.get("cfg", {})
    history = result.get("history", {})
    summary = result.get("summary", {})
    
    epochs = history.get("epochs", [])
    if not epochs:
        epochs = list(range(1, len(history.get("train_loss", [])) + 1))
        
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_acc", [])
    val_f1 = history.get("val_macro_f1", [])
    grad_norm = history.get("grad_norm", [])
    
    best_epoch = summary.get("best_epoch", None)

    # Đảm bảo thư mục lưu ảnh tồn tại
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Tiêu đề tổng quan chứa các siêu tham số chính
    opt_info = cfg.get("optimizer", "sgd")
    lr_info = cfg.get("lr", 0.01)
    bs_info = cfg.get("batch_size", 512)
    drop_info = cfg.get("dropout", 0.0)
    fig.suptitle(
        f"Experiment: {exp_id} | Opt: {opt_info} | LR: {lr_info} | Batch: {bs_info} | Dropout: {drop_info}",
        fontsize=14, fontweight="bold"
    )

    # --- Ô 1: Train Loss & Val Loss ---
    axes[0].plot(epochs, train_loss, label="Train Loss", color="blue", linestyle="--", marker="o", markersize=3)
    axes[0].plot(epochs, val_loss, label="Val Loss", color="red", linestyle="-", marker="s", markersize=3)
    axes[0].set_title("Loss Trajectory")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # --- Ô 2: Val Accuracy & Val Macro-F1 ---
    axes[1].plot(epochs, val_acc, label="Val Accuracy", color="green", linestyle="-", marker="^", markersize=3)
    axes[1].plot(epochs, val_f1, label="Val Macro-F1", color="purple", linestyle="-", marker="d", markersize=3)
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # --- Ô 3: Gradient Norm ---
    axes[2].plot(epochs, grad_norm, label="Grad Norm (Pre-clip)", color="orange", linestyle="-", marker="x", markersize=4)
    axes[2].set_title("Global Gradient Norm")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Norm L2")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    # Đánh dấu best epoch bằng đường dọc nét đứt trên cả 3 ô
    if best_epoch is not None and best_epoch in epochs:
        for ax in axes:
            ax.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Ep ({best_epoch})")
            # Cập nhật legend sau khi thêm line
            ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        print("Cảnh báo: Danh sách results rỗng, không thể vẽ biểu đồ so sánh.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for res in results:
        exp_id = res.get("exp_id", "unknown")
        history = res.get("history", {})
        epochs = history.get("epochs", [])
        metric_values = history.get(metric, [])

        if not epochs:
            epochs = list(range(1, len(metric_values) + 1))

        if metric_values:
            ax.plot(epochs, metric_values, label=exp_id, linewidth=2, marker="o", markersize=3)

    chart_title = title if title else f"Comparison of {metric}"
    ax.set_title(chart_title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)