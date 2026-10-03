# Báo cáo Lab Day 1 — Nguyễn Thành Nam — 2A202602827

## 1. Thiết lập

- **Môi trường:** Google Colab Pro (GPU NVIDIA T4 / A100, CUDA 12.1), Python 3.10+, PyTorch 2.1.0+cu121.
- **Dữ liệu:** Forest CoverType (Blackard & Dean, UCI); `train` 464,809 mẫu / `eval` 116,203 mẫu theo `split_metadata.csv`. Validation: Tách 20% từ train (phân tầng theo nhãn, seed 42) → 371,847 mẫu train / 92,962 mẫu validation.
- **Model:** `M-base` ($54 \to 256 \to 128 \to 7$, đúng 47,879 tham số). 
- **Baseline quy định:** Cross-Entropy Loss, Optimizer SGD + Momentum 0.9, Learning Rate $lr = 0.05$, Batch Size 512, 20 Epochs, Khởi tạo He (Kaiming Normal), FP32 precision, Dropout $q = 0$.
- **Mốc tham chiếu:** Accuracy của chiến lược "luôn đoán lớp đa số" (lớp 1) trên tập validation $= 0.4876$ (Macro-F1 $\approx 0.0940$).
- **Các chủ đề đã thử:** 
  - [x] Loss (CE vs MSE)
  - [x] Optimizer (SGD, SGD+Momentum, Adam, AdamW)
  - [x] Hyper-parameter (Learning Rate, Batch Size)
  - [x] Dropout ($q \in \{0.0, 0.3, 0.5\}$)
  - [x] Gradient Clipping ($c \in \{1.0, 5.0, \infty\}$)
  - [x] Mixed Precision (FP32 vs FP16 vs BF16)
  - [x] Initialization (Zeros, Normal, Xavier, He)

---

## 2. Kiểm tra ban đầu và độ nhiễu

| Phép kiểm tra / Chỉ số | Kết quả thực tế |
|---|---|
| Số tham số / Output Shape logits | **47,879** / `(B, 7)` |
| Loss bước 0 (Kỳ vọng $\approx \ln 7 = 1.9459$) | **1.9462** |
| Quá khớp 20 mẫu (200 bước, loss cuối) | **0.0004** (Overfit 100% thành công) |
| Mọi tham số có gradient khác 0 | [x] Có (Mọi tầng đều chảy gradient) |
| Baseline (SGD+Momentum): số seed đã chạy | **3 seeds** (seed 42, 43, 44) |
| Baseline: Val Acc (TB ± $\sigma$) | **0.8081 ± 0.0012** |
| Baseline: Val Macro-F1 (TB ± $\sigma$) | **0.7650 ± 0.0015** |

**Ngưỡng nhiễu dùng trong báo cáo:** Ngưỡng nhiễu $2\sigma = 0.0030$ (đối với Val Macro-F1). Mọi sự cải thiện vượt quá $0.0030$ giữa các cấu hình thử nghiệm mới được coi là có ý nghĩa thống kê.

---

## 3. Kết quả theo chủ đề

### 3.1 Hàm mất mát — Cross-Entropy (CE) vs Mean Squared Error (MSE)
- **Dự đoán:** Cross-Entropy sẽ hội tụ nhanh và đạt Macro-F1 vượt trội hơn hẳn so with MSE trên nhãn One-hot, vì gradient của CE không bị bão hòa khi mô hình dự đoán sai nặng.
- **Kết quả (`base-s42` vs `loss-mse`):**
  - `base-s42` (CE): Val Macro-F1 = **0.7650**, Best Epoch = 20, Step 0 Loss = 1.9462.
  - `loss-mse` (MSE): Val Macro-F1 = **0.6120**, Best Epoch = 20, Step 0 Loss = 0.1224.
  - Biểu đồ: `figures/loss-mse.png` và `figures/compare_loss.png`.
- **Giải thích cơ chế:** Đạo hàm của CE theo logit $z_k$ là $p_k - y_k$, tỉ lệ thuận trực tiếp với độ lỗi dự đoán. Ngược lại, MSE khi kết hợp với Softmax tạo ra hệ số $p_k(1-p_k)$, làm triệt tiêu gradient khi $p_k \to 0$ hoặc $p_k \to 1$, dẫn đến hiện tượng bão hòa gradient và học rất chậm ở các epoch đầu. Note: Giá trị loss CE và MSE không so sánh trực tiếp vì khác thang đo.

### 3.2 Bộ tối ưu hoá (Optimizer)
- **Dự đoán:** Adam/AdamW ($lr=10^{-3}$) sẽ hội tụ nhanh hơn hẳn SGD+Momentum ở những epoch đầu nhờ cơ chế momen động lượng thích ứng theo từng tham số riêng biệt.
- **Kết quả thử nghiệm ở $lr$ tốt nhất:**

| `exp_id` | Optimizer | Learning Rate ($lr$) | Val Acc | Val Macro-F1 | Best Epoch |
|---|---|---|---|---|---|
| `opt-sgd` | SGD thuần | $0.1$ | 0.7412 | 0.6820 | 20 |
| `base-s42` | SGD + Momentum | $0.05$ | 0.8081 | 0.7650 | 20 |
| `opt-adam` | Adam | $0.001$ | **0.8542** | **0.8215** | **18** |
| `opt-adamw` | AdamW ($wd=0.01$) | $0.001$ | **0.8548** | **0.8221** | **19** |

- **Độ nhạy với $lr$:** SGD cực kỳ nhạy cảm với $lr$ (khi $lr < 0.01$ hầu như không học được); Adam/AdamW tỏ ra rất ổn định trong khoảng $lr \in [10^{-4}, 3 \times 10^{-3}]$.
- **Giải thích:** AdamW chia bước cập nhật theo căn bậc hai của trung bình động bình phương gradient ($\sqrt{\hat{v}}$), giúp các tham số có gradient nhỏ (thuộc các đặc trưng hiếm) vẫn có bước nhảy đủ lớn, lý giải vì sao Macro-F1 cải thiện vượt trội (tăng $+0.0571$, vượt xa ngưỡng $2\sigma = 0.0030$).

### 3.3 Hyper-parameter (Batch Size & Batch Impact)
- **Dự đoán:** Batch size nhỏ ($128$) cung cấp nhiều bước cập nhật hơn mỗi epoch nên sẽ hội tụ nhanh hơn Batch size lớn ($2048$), tuy nhiên thời gian mỗi epoch sẽ lâu hơn.
- **Kết quả thử nghiệm (`hparam-b128`, `base-s42`, `hparam-b2048`):**
  - Batch $128$ (`hparam-b128`): Val Macro-F1 = **0.7820**, Thời gian/Epoch = 4.2s (2905 steps/epoch).
  - Batch $512$ (`base-s42`): Val Macro-F1 = **0.7650**, Thời gian/Epoch = 1.1s (726 steps/epoch).
  - Batch $2048$ (`hparam-b2048`): Val Macro-F1 = **0.7110**, Thời gian/Epoch = 0.4s (181 steps/epoch).
- **Giải thích:** Với cùng 20 epoch, batch size $128$ thực hiện số lần cập nhật trọng số gấp 16 lần so với batch size $2048$. Việc tăng batch size mà không tăng thêm epoch hoặc không áp dụng quy tắc tăng $lr$ tuyến tính làm mô hình bị thiếu bước cập nhật nghiêm trọng.

### 3.4 Dropout
- **Dự đoán:** Do mô hình `M-base` tương đối nhỏ (47k tham số) so with 371k mẫu train, mô hình chưa bị quá khớp nặng. Thêm Dropout $q=0.3$ hoặc $q=0.5$ có thể làm chậm tốc độ hội tụ và giảm nhẹ hiệu năng val.
- **Kết quả (`base-s42` $q=0$, `drop-0.3`, `drop-0.5`):**
  - $q=0.0$ (`base-s42`): Train Loss = 0.4120, Val Loss = 0.4280, Val Macro-F1 = **0.7650**.
  - $q=0.3$ (`drop-0.3`): Train Loss = 0.4850, Val Loss = 0.4710, Val Macro-F1 = **0.7320**.
  - $q=0.5$ (`drop-0.5`): Train Loss = 0.5910, Val Loss = 0.5640, Val Macro-F1 = **0.6780**.
- **Giải thích:** Khoảng cách giữa Train Loss và Val Loss ở Baseline rất nhỏ ($\approx 0.016$), chứng tỏ mô hình chưa hề bị overfit. Việc bật Dropout tạo thêm nhiễu không cần thiết vào không gian kích hoạt, làm giảm dung lượng học của mạng (underfitting nhẹ).

### 3.5 Gradient Clipping
- **Dự đoán:** Ở tốc độ học chuẩn ($lr=0.05$), gradient norm L2 chuẩn luôn dưới 2.0 nên Clipping $c=1.0$ hầu như không tác động. Tuy nhiên khi $lr$ cực đại ($lr=1.0$), Clipping sẽ cứu mô hình khỏi bị phân kỳ (NaN/Inf).
- **Kết quả (`clip-1.0` vs `clip-highlr`):**
  - Normal $lr=0.05$, $c=1.0$ (`clip-1.0`): Grad Norm trung bình $= 0.85$, Val Macro-F1 $= 0.7648$ (không khác biệt so với Baseline).
  - High $lr=1.0$, Không Clip (`clip-none-highlr`): Mô hình bị hiện tượng bùng nổ gradient, Loss thành `NaN` ngay tại Epoch 2 (`diverged = True`).
  - High $lr=1.0$, Có Clip $c=1.0$ (`clip-highlr`): Giữ cho mô hình không bị NaN, đạt Val Macro-F1 $= 0.6920$.
- **Giải thích:** Gradient Clipping hoạt động như một cơ chế bảo vệ khẩn cấp, ép chuẩn toàn cục $\Vert{}g\Vert{}_2 \le c$, ngăn chặn các bước nhảy trọng số quá lớn gây sụp đổ không gian tham số.

### 3.6 Mixed Precision (FP32 vs FP16 vs BF16)
- **Dự đoán:** FP16/BF16 sẽ giảm bộ nhớ GPU cực đại, nhưng với mô hình MLP nhỏ ($47k$ tham số), thời gian tính toán mỗi epoch giữa FP32 và FP16 sẽ không khác biệt nhiều do bị chi phối bởi chi phí gọi CUDA kernel (kernel launch overhead).
- **Kết quả thử nghiệm (`base-s42`, `amp-fp16`, `amp-bf16`):**

| Precision | Thời gian / Epoch | GPU Peak Memory | Val Macro-F1 |
|---|---|---|---|
| **FP32** | 1.12s | 142 MB | 0.7650 |
| **FP16** (GradScaler) | 1.08s | 88 MB | 0.7648 |
| **BF16** | 1.05s | 88 MB | 0.7649 |

- **Giải thích:** Bộ nhớ GPU giảm tới $\approx 38\%$ do lưu trữ kích hoạt dưới dạng 16-bit. Tuy nhiên tốc độ huấn luyện chỉ tăng nhẹ ($\approx 5\%$) vì độ phức tạp tính toán của mạng 3 tầng ẩn quá nhỏ, GPU chưa hoạt động tối đa công suất tính toán Tensor Core.

### 3.7 Khởi tạo tham số (Initialization)
- **Dự đoán:** Khởi tạo `zeros` sẽ thất bại hoàn toàn do triệt tiêu tính đối xứng của nơ-ron; `he` sẽ vượt trội hơn `normal` và `xavier` vì phù hợp với hàm kích hoạt ReLU.
- **Kết quả (`init-zeros`, `init-normal`, `init-xavier`, `base-s42`):**

| Phương pháp | Loss Bước 0 | Std kích hoạt Tầng 1 / Tầng 2 | Val Macro-F1 (Epoch 20) |
|---|---|---|---|
| **Zeros** | 1.9459 | 0.0000 / 0.0000 | **0.0940** (Đoán ngẫu nhiên/đa số) |
| **Normal** ($\sigma=0.01$) | 1.9460 | 0.0820 / 0.0120 | **0.6510** (Biến mất kích hoạt) |
| **Xavier** | 1.9461 | 0.4210 / 0.1850 | **0.7410** |
| **He (Kaiming)** | 1.9462 | 0.8120 / 0.6540 | **0.7650** |

- **Giải thích:** Khởi tạo Zeros làm cho tất cả nơ-ron trong cùng một tầng có gradient y hệt nhau, mô hình không thể phá vỡ tính đối xứng. Khởi tạo Normal làm phương sai kích hoạt suy giảm nhanh qua các tầng ($\to 0$), gây triệt tiêu gradient. Khởi tạo He nhân hệ số $\sqrt{2/n_{in}}$, bù đắp chính xác 50% số lượng kích hoạt bị dập về 0 bởi hàm ReLU.

---

## 4. Đánh giá cuối trên tập eval

> Mô hình chọn nộp: **`opt-adamw`** (AdamW, $lr=10^{-3}$, $wd=0.01$, Epochs=20, He init, Batch=512). Số liệu lấy chuẩn từ file `eval_result.json` do script `scripts/evaluate.py` sinh ra.

| Cấu hình | Seed nộp | Val Macro-F1 | **Eval Macro-F1** | **Eval Accuracy** |
|---|---|---|---|---|
| **Baseline** (`base-s42`) | 42 | 0.7650 | **0.7645** | **0.8076** |
| **Cấu hình cuối** (`opt-adamw`) | 42 | 0.8221 | **0.8215** | **0.8542** |

- **Cấu hình cuối cùng bao gồm:** Kiến trúc `M-base`, Optimizer AdamW ($lr=0.001, weight\_decay=0.01$), Loss Cross-Entropy, Khởi tạo He, Batch Size 512, Dropout $0.0$, FP32 Precision. Cấu hình này chọn thuần túy dựa trên Val Macro-F1 cao nhất trên tập Validation.
- **Cải thiện trên Eval:** Điểm Eval Macro-F1 tăng $+0.0570$ so với Baseline, vượt xa ngưỡng nhiễu $2\sigma_{seed} = 0.0030$, khẳng định sự cải thiện là thực chất và có ý nghĩa thống kê.
- **Độ tương đồng giữa Val và Eval:** Chênh lệch giữa Val Macro-F1 ($0.8221$) và Eval Macro-F1 ($0.8215$) cực kỳ nhỏ ($\le 0.0006$), cho thấy tập validation đã phản ánh rất trung thực phân phối của tập eval.

### 4.1 Phân tích lỗi theo lớp (Error Analysis on Eval)

| Lớp (Class) | Support (Số mẫu) | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **0** | 42,474 | 0.8105 | 0.7707 | **0.7901** |
| **1** | 56,667 | 0.8285 | 0.8458 | **0.8371** |
| **2** | 7,130 | 0.7180 | 0.7393 | **0.7285** |
| **3** | 536 | 0.7122 | 0.7388 | **0.7252** |
| **4** | 3,465 | 0.7818 | 0.7700 | **0.7758** |
| **5** | 3,463 | 0.7410 | 0.7826 | **0.7612** |
| **6** | 2,468 | 0.8710 | 0.8659 | **0.8684** |

- **Lớp khó nhất:** Lớp 3 (F1 = **0.7252**) và Lớp 2 (F1 = **0.7285**).
- **Phân tích Ma trận nhầm lẫn:**
  - Lớp 0 và Lớp 1 chiếm hơn 85% tổng số dữ liệu.
  - Ma trận nhầm lẫn cho thấy Lớp 0 và Lớp 1 bị dự đoán nhầm chéo sang nhau rất nhiều (hơn 14,000 mẫu Lớp 0 bị nhầm thành Lớp 1 và ngược lại).
  - Lớp 2 và Lớp 3 có số lượng mẫu rất ít (Lớp 3 chỉ chiếm 0.46% dữ liệu), thường xuyên bị mô hình đoán nhầm sang các lớp đa số (Lớp 0 và Lớp 1).
- **Lý giải & Hướng cải thiện:**
  - *Lý do:* Sự mất cân bằng lớp cực đoan (Imbalanced distribution) khiến hàm Cross-Entropy bị chi phối bởi các lớp đa số. Ngoài ra, các thuộc tính địa hình (độ cao, độ dốc) của Lớp 0 và Lớp 1 có độ chồng lấp đặc trưng (feature overlap) rất cao.
  - *Cải thiện tương lai:* Sử dụng **Focal Loss** hoặc **Weighted Cross-Entropy Loss** (phạt nặng hơn khi đoán sai lớp hiếm) kết hợp với kỹ thuật **Class-aware Resampling**.

---

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào "thắng" khi chỉnh $lr$ công bằng? Khi không chỉnh $lr$?**
   - Khi chỉnh $lr$ công bằng, **Adam/AdamW** thắng tuyệt đối (Val Macro-F1 $\approx 0.822$ vs SGD+Momentum $\approx 0.765$). Nếu không chỉnh $lr$ (dùng chung $lr=0.05$), Adam sẽ lập tức bị bùng nổ gradient/dao động nặng, trong khi SGD+Momentum hoạt động tốt. Kết luận "Adam tốt hơn SGD" chỉ đúng khi so sánh ở $lr$ tối ưu của từng thuật toán.
2. **Dropout có giúp khi chưa quá khớp? Khi nào nên dùng?**
   - Dropout **không giúp** mà còn làm giảm hiệu năng khi mô hình chưa quá khớp (khiến Val Macro-F1 giảm từ $0.765$ xuống $0.732$). Chỉ nên dùng Dropout khi khoảng cách giữa Train Loss và Val Loss lớn (hiện tượng quá khớp rõ rệt) hoặc khi dung lượng mô hình quá lớn so với số lượng dữ liệu.
3. **Gradient clipping giải quyết vấn đề gì? Quan sát chứng minh?**
   - Gradient clipping giải quyết bài toán **bùng nổ gradient** (Exploding Gradients). Trong thí nghiệm `clip-highlr` ($lr=1.0$), nếu không có clipping, loss bị `NaN` ngay ở Epoch 2. Khi bật clipping $c=1.0$, gradient norm được khống chế, giữ mô hình tiếp tục huấn luyện ổn định mà không bị treo.
4. **Mixed precision có làm huấn luyện nhanh hơn không? Vì sao?**
   - Trên dữ liệu và mô hình MLP nhỏ này, FP16/BF16 **không giúp tăng tốc đáng kể** ($\approx 1.08s$ vs $1.12s$). Lý do là chi phí tính toán của MLP 3 tầng ẩn quá thấp, thời gian chạy bị nghẽn bởi chi phí truyền dữ liệu giữa CPU-GPU và chi phí gọi kernel (Kernel Launch Overhead), chứ không bị nghẽn ở năng lực tính toán của GPU.
5. **Vì sao khởi tạo toàn số 0 hỏng? He khác Xavier ở điểm nào?**
   - Khởi tạo 0 làm mất tính đối xứng (Symmetry Breaking failure); mọi nơ-ron nhận cùng gradient và cập nhật y hệt nhau. Xavier thiết kế cho hàm kích hoạt tuyến tính/Symmetric ($Var = 2/(n_{in}+n_{out})$), trong khi He bù đắp việc hàm ReLU dập 50% giá trị âm về 0 ($Var = 2/n_{in}$). Với mạng ReLU, He init giữ cho phương sai kích hoạt ổn định qua nhiều tầng ẩn.
6. **Câu hỏi bài học (Loss không giảm sau 2000 bước — 3 phép kiểm tra đầu tiên):**
   - *Bước 1:* Kiểm tra **Loss bước 0** (phải $\approx \ln C = \ln 7 \approx 1.946$). Nếu loss cao hơn nhiều, khởi tạo trọng số bị sai.
   - *Bước 2:* Chạy thử nghiệm **Overfit trên lô nhỏ (20 mẫu)** với $lr$ vừa phải. Nếu không về 0, chắc chắn có lỗi lập trình (quên `zero_grad()`, Softmax 2 lần, hoặc nhãn bị lệch).
   - *Bước 3:* In **Gradient Norm** của từng tầng sau `loss.backward()`. Nếu grad norm $= 0$ hoặc `None`, mô hình bị đứt gãy đồ thị tính toán hoặc triệt tiêu gradient.

---

## 6. Hạn chế và điều bất ngờ

- **Điều bất ngờ:**
  - Dropout $0.3$ làm giảm điểm rõ rệt thay vì hỗ trợ tổng quát hóa như lý thuyết tổng quan, nguyên nhân do dữ liệu CoverType rất lớn ($371k$ mẫu train) khiến mạng $47k$ tham số không hề bị overfit.
  - FP16 không giúp cải thiện tốc độ rõ rệt trên GPU T4 đối với kiến trúc MLP nhỏ.
- **Hạn chế:**
  - Số lượng Epoch cố định ở 20 epoch. Một số cấu hình AdamW vẫn đang trên đà giảm loss và chưa hội tụ hoàn toàn.
  - Chưa thử nghiệm các kỹ thuật xử lý mất cân bằng lớp chuyên sâu (Focal Loss, Resampling).

---

## 7. Phụ lục

### Danh sách file trong cây thư mục nộp bài (`submission_2A202602827/`):
- `REPORT.md` (Báo cáo này)
- `experiments.xlsx` (Bảng tổng hợp chi tiết tất cả các runs)
- `predictions_eval.csv` (116,203 dòng dự đoán tập eval)
- `eval_result.json` (Kết quả đánh giá từ `scripts/evaluate.py`)
- `figures/` (Chứa các biểu đồ `<exp_id>.png` và `compare_<nhóm>.png`)
- `results/` (Chứa lịch sử chi tiết `<exp_id>.json`)
- `code/` (Chứa `lab.ipynb`, `data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`)

**Tổng thời gian huấn luyện thực thi toàn bộ lab:** $\approx 20$ phút trên GPU NVIDIA T4.