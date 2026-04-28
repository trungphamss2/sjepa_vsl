"""
ensemble_eval.py — Đánh giá Ensemble Multi-Stream S-JEPA
=========================================================
Kết hợp Softmax từ 3 luồng: Joint + Bone + Velocity (Weighted Late Fusion)

CÁCH CHẠY:
    python ensemble_eval.py
    python ensemble_eval.py --weights 0.5 0.35 0.15   # Tuỳ chỉnh trọng số
    python ensemble_eval.py --no_velocity              # Chỉ Joint + Bone

CHECKPOINT MẶC ĐỊNH:
    Joint:    checkpoints_finetuned/finetune_NTU60_XView/best.pth
    Bone:     checkpoints_finetuned/finetune_NTU60_XView_bone/best.pth
    Velocity: checkpoints_finetuned/finetune_NTU60_XView_velocity_v2/best.pth
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

from src.core.classifier import NTUActionClassifier
from src.datasets.ntu_dataset import NTUActionDataset

# ─────────────────────────────────────────────
#  CẤU HÌNH MẶC ĐỊNH
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    "data_paths": ["./DATA/nturgb+d_skeletons"],
    "max_frames": 120,
    "protocol": "xview",
    "num_classes": 60,
    "embed_dim": 256,
    "depth": 8,
    "num_heads": 8,
    "segment_length": 4,
    "batch_size": 64,
    "num_workers": 8,
}

CHECKPOINT_PATHS = {
    "joint":    "checkpoints_finetuned/finetune_NTU60_XView/best.pth",
    "bone":     "checkpoints_finetuned/finetune_NTU60_XView_bone/best.pth",
    "velocity": "checkpoints_finetuned/finetune_NTU60_XView_velocity_v2/best.pth",
}


# ─────────────────────────────────────────────
#  HÀM PHỤ TRỢ
# ─────────────────────────────────────────────
def load_model(ckpt_path: str, device: torch.device, cfg: dict) -> NTUActionClassifier:
    """Tạo model và nạp trọng số từ checkpoint Finetune."""
    model = NTUActionClassifier(
        pretrained_path=None,           # Không cần pretrain — nạp full state_dict
        num_frames=cfg["max_frames"],
        num_classes=cfg["num_classes"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        segment_length=cfg["segment_length"],
        dropout=0.0,                    # Tắt Dropout khi eval
        drop_path=0.0,
    ).to(device)
    
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"  [OK] Loaded: {ckpt_path}")
    return model


@torch.no_grad()
def extract_softmax(model: NTUActionClassifier,
                    loader: DataLoader,
                    device: torch.device,
                    num_samples: int) -> torch.Tensor:
    """
    Trích xuất xác suất Softmax của từng mẫu trong Test Set.
    Trả về Tensor [N, num_classes] trên CPU.
    """
    all_probs = []
    for x, _ in loader:
        x = x.to(device)
        with autocast('cuda'):
            logits = model(x)
        probs = F.softmax(logits, dim=1)   # [B, C]
        all_probs.append(probs.cpu())
    return torch.cat(all_probs, dim=0)     # [N, C]


@torch.no_grad()
def extract_labels(loader: DataLoader) -> torch.Tensor:
    """Lấy toàn bộ nhãn thực từ DataLoader."""
    all_labels = []
    for _, y in loader:
        all_labels.append(y)
    return torch.cat(all_labels, dim=0)    # [N]


def topk_accuracy(probs: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Tính Top-K Accuracy từ ma trận xác suất."""
    _, topk_preds = probs.topk(k, dim=1)  # [N, k]
    correct = topk_preds.eq(labels.unsqueeze(1).expand_as(topk_preds))
    return 100.0 * correct.any(dim=1).float().mean().item()


def build_test_loader(modality: str, cfg: dict) -> DataLoader:
    """Tạo DataLoader cho tập Test với modality chỉ định."""
    ds = NTUActionDataset(
        data_path=cfg["data_paths"],
        max_frames=cfg["max_frames"],
        split='test',
        protocol=cfg["protocol"],
        modality=modality,
    )
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,                  # QUAN TRỌNG: giữ thứ tự để align nhãn
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=(cfg["num_workers"] > 0),
        prefetch_factor=2 if cfg["num_workers"] > 0 else None,
    )


# ─────────────────────────────────────────────
#  HÀM CHÍNH
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="S-JEPA Multi-Stream Ensemble Evaluation")
    parser.add_argument("--weights", nargs=3, type=float, default=None,
                        metavar=("W_JOINT", "W_BONE", "W_VEL"),
                        help="Trọng số cho [Joint, Bone, Velocity]. Mặc định: tự tính theo acc.")
    parser.add_argument("--no_velocity", action="store_true",
                        help="Chỉ dùng Joint + Bone (bỏ qua Velocity)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda hoặc cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = DEFAULT_CONFIG

    print("\n" + "="*60)
    print("  S-JEPA MULTI-STREAM ENSEMBLE EVALUATION")
    print("="*60)
    print(f"  Device: {device}")
    print(f"  Protocol: {cfg['protocol'].upper()}")
    print(f"  Num Classes: {cfg['num_classes']}")
    print("="*60 + "\n")

    # ── Kiểm tra checkpoint tồn tại ──
    streams_to_use = ["joint", "bone"]
    if not args.no_velocity:
        if os.path.exists(CHECKPOINT_PATHS["velocity"]):
            streams_to_use.append("velocity")
        else:
            print(f"  [WARN] Velocity checkpoint không tìm thấy -> bỏ qua Velocity")
            print(f"    ({CHECKPOINT_PATHS['velocity']})\n")

    for s in streams_to_use:
        if not os.path.exists(CHECKPOINT_PATHS[s]):
            raise FileNotFoundError(f"Checkpoint không tìm thấy: {CHECKPOINT_PATHS[s]}")

    # ── Bước 1: Load từng model và trích xuất Softmax ──
    print("[Buoc 1/3] Nap Models & Trich xuat Softmax...")
    softmax_dict = {}
    labels = None

    for stream in streams_to_use:
        print(f"\n  Luồng [{stream.upper()}]:")
        model = load_model(CHECKPOINT_PATHS[stream], device, cfg)
        loader = build_test_loader(stream, cfg)

        if labels is None:
            labels = extract_labels(loader)   # Chỉ cần lấy 1 lần (nhãn giống nhau)

        probs = extract_softmax(model, loader, device, len(loader.dataset))
        softmax_dict[stream] = probs          # [N, C] CPU

        acc1 = topk_accuracy(probs, labels, k=1)
        acc5 = topk_accuracy(probs, labels, k=5)
        print(f"  → Single-stream Acc@1: {acc1:.2f}%  |  Acc@5: {acc5:.2f}%")

        del model
        torch.cuda.empty_cache()

    # ── Bước 2: Thiết lập trọng số Ensemble ──
    print(f"\n[Buoc 2/3] Thiet lap trong so Ensemble...")
    
    if args.weights is not None:
        # Trọng số do người dùng chỉ định
        w_list = args.weights[:len(streams_to_use)]
        total = sum(w_list)
        weights = {s: w / total for s, w in zip(streams_to_use, w_list)}
        print("  Dùng trọng số tuỳ chỉnh (đã chuẩn hoá):")
    else:
        # Auto weight: tỉ lệ theo độ chính xác đơn luồng
        single_accs = {s: topk_accuracy(softmax_dict[s], labels, k=1)
                       for s in streams_to_use}
        total_acc = sum(single_accs.values())
        weights = {s: acc / total_acc for s, acc in single_accs.items()}
        print("  Tự động tính trọng số theo độ chính xác đơn luồng:")

    for s, w in weights.items():
        print(f"    {s.upper():10s}: {w:.4f}")

    # ── Bước 3: Weighted Ensemble ──
    print(f"\n[Buoc 3/3] Tong hop Ensemble...")
    
    # Thử tất cả các tổ hợp
    combos = []
    stream_list = streams_to_use

    # Thêm các combo con
    if len(stream_list) >= 2:
        for i in range(len(stream_list)):
            for j in range(i + 1, len(stream_list)):
                combos.append([stream_list[i], stream_list[j]])
    combos.append(stream_list)  # Full ensemble

    print("\n" + "─"*50)
    print(f"  {'Combo':<30} {'Acc@1':>8}  {'Acc@5':>8}")
    print("─"*50)

    best_acc1 = 0.0
    best_combo = None
    best_probs = None

    for combo in combos:
        # Weighted sum of softmax
        w_total = sum(weights[s] for s in combo)
        ensemble_probs = sum(weights[s] / w_total * softmax_dict[s] for s in combo)

        acc1 = topk_accuracy(ensemble_probs, labels, k=1)
        acc5 = topk_accuracy(ensemble_probs, labels, k=5)
        tag  = " + ".join(s.capitalize() for s in combo)
        
        is_best = ">>" if acc1 > best_acc1 else "  "
        print(f"  {is_best} {tag:<28} {acc1:>7.2f}%  {acc5:>7.2f}%")
        
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_combo = combo
            best_probs = ensemble_probs

    print("─"*50)
    print(f"\nKET QUA TOT NHAT:")
    print(f"   Combo:  {' + '.join(s.capitalize() for s in best_combo)}")
    print(f"   Acc@1:  {best_acc1:.2f}%")
    print(f"   Acc@5:  {topk_accuracy(best_probs, labels, k=5):.2f}%")
    print("="*60 + "\n")

    # Lưu softmax để dùng lại sau
    save_path = "checkpoints_finetuned/ensemble_probs.pt"
    torch.save({
        "softmax": softmax_dict,
        "labels": labels,
        "weights": weights,
        "best_combo": best_combo,
        "best_acc1": best_acc1,
    }, save_path)
    print(f"  [SAVED] Softmax cache -> {save_path}")


if __name__ == "__main__":
    main()
