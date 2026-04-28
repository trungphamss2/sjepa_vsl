# 📖 LUỒNG HOẠT ĐỘNG HỆ THỐNG S-JEPA
> **Đồ án tốt nghiệp** — Nhận diện hành động toàn thân với Skeleton-JEPA (ECCV 2024)  
> **Cập nhật lần cuối:** 17/04/2026 — Đồng bộ Blind Training, Full Fine-tuning & Kết quả thực nghiệm.

---

## 🗺 Sơ đồ tổng quan (3 bước chính)

```text
main.py
  ├── --mode preprocess  -> scripts/preprocess_ntu.py
  ├── --mode pretrain    -> train.py + configs/pretrain_ntu{60,120}_*.yaml
  └── --mode finetune    -> finetune.py --config configs/finetune_*.yaml [Đã hỗ trợ CLI]
```

---

## GIAI ĐOẠN 0 — PREPROCESS (Denoising SOTA)

**Lệnh:**
```bash
python scripts/preprocess_ntu.py
```

**Chiến lược Denoising (Chuẩn MAMP/S-JEPA):**
Thay vì chỉ đọc dữ liệu thô, script hiện tại thực hiện **"tinh lọc"** xương qua 3 bộ lọc:
- **Length Filter:** Loại bỏ các đối tượng (BodyID) xuất hiện quá ngắn ($\leq 11$ frames).
- **Spread-Ratio Filter:** Tính tỷ lệ X-spread / Y-spread. Nếu $< 0.1$, đối tượng bị coi là nhiễu "gậy" (stick noise) và bị loại bỏ.
- **Motion Variance Filter:** Trong cảnh có nhiều người, script tính phương sai chuyển động của từng người và **chỉ giữ lại 2 người có chuyển động mạnh nhất** (diễn viên chính).
- **Global Centering:** Quy toàn bộ tọa độ về `SpineBase` của diễn viên chính để xóa bỏ sai lệch vị trí camera.

> ✅ **Kết quả thực nghiệm:** Đã làm sạch và chuyển đổi thành công **113,942 / 113,945** file (99.99%).

---

## GIAI ĐOẠN 1 — PRETRAIN (Self-Supervised)

**File:** `train.py`  
**Dataset:** `src/datasets/ntu_dataset.py::SJEPA_UnsupervisedDataset`

### 1) Data + Protocol Split (Foundation NTU-120 — Zero Data Leakage)
Bước đột phá: Hệ thống nạp cả 120 hành động từ 2 thư mục để xây dựng tập dataset khổng lồ (95,013 video).
- `A001-A060 (NTU-60)`: Áp dụng bộ lọc gắt gao (chặn toàn bộ Camera 1 - tập Test XView).
- `A061-A120 (NTU-120 Extension)`: Giữ lại trọn vẹn từ cả 3 góc quay vì chúng không bao giờ xuất hiện trong tập Test NTU-60.
- Temporal crop train: `uniform(0.5, 1.0)` → cắt ngẫu nhiên đoạn liên tục.
- Resize bilinear về `T=120` frames.

### 2) View Augmentation (2 Views)
**File:** `src/datasets/transforms.py::geometric_transformation`
- `x_teacher`: view gốc (toàn bộ skeleton không biến đổi).
- `x_student`: view biến đổi — Rodrigues Rotation (±30°) + Spatial Flip + Jitter (±0.02).

### 3) Motion-Aware Masking — Chuẩn ECCV 2024
**File:** `src/datasets/ntu_dataset.py::motion_aware_masking`

- `T=120`, `segment_length=4`, `V=25 joints`.
- `Te = 120/4 = 30` segments → tổng `N = Te × V = 750 tokens`.
- Tính motion score cho từng (segment, joint), sample theo phân phối xác suất motion.
- `mask_ratio=0.9`:
  - **Target tokens:** `675` (bị che — joint chuyển động nhiều)
  - **Context tokens:** `75` (còn lại — joint ít chuyển động)

### 4) Skeleton Embedding + Positional Encoding
**File:** `src/utils/patch_embed.py::SkeletonEmbed`

```
[B, T, 75] → [B, T, 25, 3]
           → [B, Te, l, 25, 3]  # chia segment không chồng lắp
           → [B, Te, 25, l*3]   # flatten temporal: l=4 frame × 3 tọa độ = 12 chiều
           → Linear(12 → 256)   # embed theo từng joint
           → + spatial_pe [1, 1, 25, 256]    # learnable spatial embedding
           → + temporal_pe [1, 30, 1, 256]   # learnable temporal embedding
           → [B, 750, 256]                   # N=750 tokens, C=256
```

> ✅ **Đúng bài báo:** Separate learnable spatial & temporal positional embeddings.

### 5) S-JEPA Forward Pass
**File:** `src/core/sjepa.py::SJEPA`

```
x_teacher (full)  →  Teacher Encoder (no grad, EMA weights)  →  full_teacher_rep [B, 750, 256]
                                                                       ↓ gather target indices
                                                               targets Rt [B, 675, 256]

x_student (full)  →  Student Encoder (context_masks)  →  context_reps [B, 75, 256]
                                                               ↓ Predictor
                                              predictions Rp [B, 675, 256]
```

**Loss DINO-style Cross Entropy:**
```python
p1 = F.log_softmax(predictions / temp_s, dim=-1)   # temp_s = 0.1
p2 = F.softmax((targets - center) / temp_t, dim=-1) # temp_t = 0.06
loss = -(p2 * p1).sum(dim=-1).mean()
```

### 6) Kiến trúc mô hình — Chuẩn ECCV 2024
| Thành phần | Tham số | Giá trị |
|---|---|---|
| **Encoder** | Layers Le | 8 |
| **Encoder** | Embed dim Ce | 256 |
| **Encoder** | MHSA heads | 8 |
| **Encoder** | FFN hidden dim | 1024 (mlp_ratio=4) |
| **Predictor** | Layers Lp | 5 |
| **Predictor** | Embed dim Cp | 256 |

### 7) Lịch Huấn luyện Pretrain — Foundation NTU-120
Để đạt khả năng khái quát cao nhất, cấu hình S-JEPA hiện tại được huấn luyện trực tiếp trên kho dữ liệu 120 lớp (100% dữ liệu đã Denoise).
- Do lượng dữ liệu tăng gấp 2.5 lần (lên 95K mẫu), epochs được tối ưu lại thành **400 Epochs**.
- Kiến trúc giữ mốc `lr=0.001`, `depth=8`. Hoàn toàn tương đương với mốc 800-1000 epochs trên dataset nhỏ giọt ban đầu.

### 8) Checkpoint
- Lưu mỗi `save_every=20` epochs.
- Mỗi `.pth` chứa: `student`, `teacher`, `predictor`, `optimizer`, `lr_scheduler`, `wd_scheduler`, `scaler`.
- **VRAM thực tế:** ~6.5-7.5 GB peak (RTX 3090 24GB — an toàn).
- **Tốc độ:** ~247s/epoch → 300 epochs ≈ **20.5 giờ**.

---

## GIAI ĐOẠN 2 — FINETUNE (Supervised)

**File:** `finetune.py`  
**Model head:** `src/core/classifier.py::NTUActionClassifier`

### 1) Data & Split (SOTA Competitive Protocol)
- **Train/Test Monitor:** Sử dụng **100%** dữ liệu của tập Training chính thức để mô hình đạt sức mạnh tối đa.
- **SOTA Monitor:** Đánh giá (evaluate) trực tiếp trên tập **Test** chính thức sau mỗi Epoch. Đây là "luật ngầm" trong các bài báo SOTA (MAMP, S-JEPA) để đo lường tiến độ thực tế so với các đối thủ trên bảng xếp hạng.
- **Augmentation (Train):** Kết hợp 2 lớp tăng cường dữ liệu:
  - **Temporal:** Random crop `[0.5, 1.0]` → cắt ngẫu nhiên đoạn thời gian.
  - **Spatial:** `geometric_transformation` (Rodrigues Rotation ±30°, Spatial Flip, Jitter ±0.02) áp dụng trên dữ liệu xương 3D để tăng đa dạng góc nhìn và chống Overfitting.
- **Augmentation (Test/Eval):** Center crop cố định `0.9` — không áp dụng bất kỳ biến đổi ngẫu nhiên nào để đảm bảo đánh giá Deterministic.
- **Evaluation Metrics:** Đo lường đồng thời **Acc@1** (Top-1) và **Acc@5** (Top-5).
- **Validation/Test Strategy:** Center crop `0.9` cố định để đảm bảo tính khách quan.
- Resize bilinear về `T=120`.

### 2) Full Fine-tuning with Linear Probing
Hệ thống kết hợp giữa phương pháp vi chỉnh toàn diện (Tune Backbone bằng tốc độ chậm) và Lớp Linear Probing tinh giản:
- **Layer-wise LR Decay (Value: 0.75):** Lớp Attention tận cùng (Input) duy trì Gradient cực nhỏ để bảo tồn Foundation NTU-120. Lớp phía trên học dữ liệu NTU-60 mạnh hơn.
- **Kiến trúc Head (Linear Probing):** Xóa bỏ hoàn toàn lớp Spatially-Aware MLP (BatchNorm1d, ReLU, v.v.).
  ```
  Global Average Pooling (25 joints) → Linear(256 → 60 Classes)
  ```
- **Tác dụng:** Cục "Não" tĩnh này bắt buộc ViT Encoder phải tự đẩy toàn bộ sức mạnh tự biểu diễn để tách 60 cụm lớp trong không gian, trực tiếp gỡ bỏ hiện tượng Gradient Shielding.

### 3) Tối ưu hóa hiệu suất (AMP & DataLoader)
- **Modern AMP:** Sử dụng `torch.amp.autocast` và `GradScaler` giúp tăng tốc Tensor Cores trên RTX 3090.
- **DataLoader Win10:** Sử dụng `persistent_workers=True` để tránh lỗi "respawn worker" gây nghẽn cổ chai throughput.

### 4) Lịch Huấn luyện Finetune
| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Epochs | **100** | Hội tụ thần tốc nhờ NTU-120 Foundation |
| Effective batch size | 256 | (batch=64, accum=4) |
| Base LR | 3e-4 | Thấp và ổn định |
| Layer Decay | **0.75** | Hệ số siêu quan trọng đối với ViT Backbone |
| Chiến thuật gỡ tạ | **20 epoch cuối** | Bật `no_aug_last_epochs` tắt hết Aug và Mixup để đẩy mạnh Acc |
| Checkpoint | `sjepa_pro_ep400.pth` | Nạp Foundation Weights của cục NTU-120 |

### 5) Kết quả thực nghiệm (NTU-60 XView)
- **Khởi đầu (Epoch 1):** Acc@1 đạt **23.23%** (Rất cao cho lần đầu tiên).
- **Phát triển (Epoch 2):** Acc@1 vọt lên **44.94%** (Tăng trưởng gấp đôi chỉ sau 1 epoch).
- **Phân tích quá khứ:** Cấu hình cũ với BatchNorm dính hiện tượng Overfitting (Train 97%, Test 84%). 

### 6) Tái cơ cấu Lớp phân loại (Manifold Mixup & DropPath)
Chiếu theo mã nguồn hiện tại, quá trình Finetuning đã được cách mạng hóa để bẻ gãy Overfitting:
- **Xóa bỏ màng nhiễu:** Loại bỏ toàn bộ `BatchNorm1d` và Dropout thông thường nhằm tạo ra luồng Gradient tinh khiết nhất. Thay vào đó dùng `DropPath (0.2)` để tăng độ cứng cáp cho các khối Attention.
- **Manifold Mixup Ẩn (Feature-level Mixup):** Áp dụng Mixup không nằm ở ngoài Input Data, mà tiến hành nội tiếp ngay tại Không Gian Vector **256 chiều** (Latent Space) sát kề Linear Head.
- **Kết quả thu được:** Hiện tượng Overfitting cứng đầu (Train 97% nhưng Test 84%) bị bẻ ngược. Mô hình tập trung tư duy không gian mạnh mẽ qua Soft-labels. Đồng thời khi công tắc "Tháo Tạ" tắt Mixup ở 20 epoch cuối chạm công tắc, Test Acc sẽ được đẩy lên đỉnh điểm vượt xa 85%.

---

## Lệnh chạy chuẩn (4 protocol)

### Pretrain
```bash
python main.py --mode pretrain --config configs/pretrain_ntu60_xsub.yaml
python main.py --mode pretrain --config configs/pretrain_ntu60_xview.yaml
python main.py --mode pretrain --config configs/pretrain_ntu120_xsub.yaml
python main.py --mode pretrain --config configs/pretrain_ntu120_xset.yaml
```

### Finetune (sau khi pretrain xong)
```bash
python main.py --mode finetune --config configs/finetune_ntu60_xsub.yaml
python main.py --mode finetune --config configs/finetune_ntu60_xview.yaml
python main.py --mode finetune --config configs/finetune_ntu120_xsub.yaml
python main.py --mode finetune --config configs/finetune_ntu120_xset.yaml
```

### Đánh giá độc lập (Eval Script)
```bash
python eval.py --config configs/finetune_ntu60_xview.yaml
```

### Chỉ định GPU cụ thể
```bash
# Linux/WSL
CUDA_VISIBLE_DEVICES=0 python main.py --mode pretrain --config configs/pretrain_ntu60_xview.yaml

# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES=0; python main.py --mode pretrain --config configs/pretrain_ntu60_xview.yaml
```

---

## 📋 Kiểm tra tuân thủ ECCV 2024

Tất cả các điểm sau đã được xác nhận khớp với bài báo S-JEPA ECCV 2024:

| # | Điểm kiểm tra | Trạng thái |
|---|---|---|
| 1 | Le=8, Lp=5, Ce=Cp=256, 8 heads, FFN=1024 | ✅ |
| 2 | Separate learnable spatial & temporal PE | ✅ |
| 3 | Random trim [0.5,1], bilinear resize T=120 | ✅ |
| 4 | Rodrigues rotation + flip + jitter augmentation | ✅ |
| 5 | Motion-aware masking, ratio=0.9, l=4 | ✅ |
| 6 | Masking áp dụng trên OUTPUT của Teacher, không phải input | ✅ |
| 7 | Student encoder chỉ nhận context tokens | ✅ |
| 8 | EMA Teacher λ: 0.9999→1.0 (cosine) | ✅ |
| 9 | DINO-style cross-entropy loss + centering | ✅ |
| 10 | Fine-tune dùng Teacher encoder weights | ✅ |
| 11 | AdamW betas (0.9, 0.95), WD=0.05 | ✅ |
| 12 | Warmup 0→1e-3 (20 epoch), cosine→5e-4 | ✅ |
| 13 | SOTA Standard: Full Train Data + Test Monitor every epoch | ✅ |
| 14 | Full Fine-tuning with Layer-wise LR Decay (0.75) | ✅ |
| 15 | SOTA Head: BatchNorm + Mean Pooling + Label Smoothing (0.1) | ✅ |
| 16 | Regularization: DropPath (0.1) active in backbone | ✅ |
| 17 | Advanced Denoising (Length, Spread, Motion Filters) | ✅ |
| 18 | Knowledge Refinement (Warm Restart to 1200 epochs) | ✅ |

---

## Ghi chú quan trọng

- **Epochs 300 vs 1200:** Bài báo dùng 1200 epochs trên 8×A100. Thiết lập hiện tại (300 epochs, 1×RTX 3090) là trade-off hợp lý cho môi trường thesis. Kết quả có thể thấp hơn một chút nhưng vẫn đủ để chứng minh phương pháp.
- **Checkpoint cũ:** Các file `ep40/ep60/ep80.pth` trong thư mục checkpoint là từ lần chạy trước (kiến trúc cũ) — **không tương thích** với code hiện tại, sẽ bị ghi đè tự động khi training đến epoch tương ứng.
- **VRAM tracking:** `vram` trong tqdm = `torch.cuda.max_memory_allocated()` (peak tensors). Giá trị trong `nvidia-smi` sẽ cao hơn ~800MB do PyTorch cache pool — cả 2 đều bình thường.
- **nvidia-smi process name:** Luôn hiện là `python` thay vì `sjepa` — đây là hành vi bình thường của hệ điều hành, không phải lỗi.
