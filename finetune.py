import os
import sys
import glob
import yaml
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast  # [AMP] Mixed Precision
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from src.utils.schedulers import WarmupCosineSchedule

# Import Pro structure components
from src.core.classifier import NTUActionClassifier
from src.datasets.ntu_dataset import NTUActionDataset

def calculate_topk_accuracy(output, target, topk=(1, 5)):
    """
    Tính toán Acc@1 và Acc@5 (Chuẩn ImageNet/NTU).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k) # Trả về số lượng đúng nguyên bản (sum)
        return res


def get_layerwise_param_groups(model, base_lr, layer_decay, weight_decay):
    """
    Layer-wise LR Decay — đúng chuẩn bài báo S-JEPA (mục 4.3).
    """
    num_layers = len(model.encoder.blocks)   # 8 lớp Transformer
    no_wd_keywords = {'bias', 'norm', 'pos_embed', 'spatial_pe', 'temporal_pe'}

    def get_layer_id(name):
        """Trả về layer index: 0 = gần input nhất, num_layers = gần output nhất"""
        if 'patch_embed' in name or 'pos_embed' in name:
            return 0
        if 'blocks.' in name:
            idx = int(name.split('blocks.')[1].split('.')[0])
            return idx + 1          # block 0 → layer 1, block 7 → layer 8
        return num_layers           # norm cuối

    # Nhóm params theo (layer_id, weight_decay)
    bucket = {}     # key: (layer_id, wd) → list of params
    for name, param in model.encoder.named_parameters():
        if not param.requires_grad:
            continue
        lid = get_layer_id(name)
        wd  = 0.0 if any(k in name for k in no_wd_keywords) else weight_decay
        key = (lid, wd)
        bucket.setdefault(key, []).append(param)

    param_groups = []
    for (lid, wd), params in bucket.items():
        # lr giảm dần khi lid giảm (lớp sâu hơn)
        lr = base_lr * (layer_decay ** (num_layers - lid))
        param_groups.append({'params': params, 'lr': lr,
                             'weight_decay': wd, 'layer_id': lid})

    # Gom tất cả các tham số KHÔNG thuộc encoder (ví dụ: fc1, bn, fc2) vào head
    head_params = [param for name, param in model.named_parameters() if not name.startswith('encoder.') and param.requires_grad]

    # Classifier head — full base_lr
    param_groups.append({'params': head_params,
                         'lr': base_lr, 'weight_decay': weight_decay,
                         'layer_id': 'head'})

    # Log LR phân bổ
    print("[Layer-wise LR] Phân bổ Learning Rate:")
    for g in param_groups:
        n = len(g['params'])
        print(f"  layer={g['layer_id']:>4}  lr={g['lr']:.2e}  wd={g['weight_decay']:.0e}  params={n}")

    return param_groups


def main_finetune(config_path=None):
    # 1. Nạp Configuration từ YAML (hỗ trợ truyền config từ dòng lệnh)
    if config_path is None:
        config_path = 'configs/finetune.yaml'
    print(f"Đang dùng config: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"--- KHỞI ĐỘNG HỆ THỐNG FINE-TUNING SOTA (FULL TRAIN + TEST MONITOR) ---")
    print(f"Tài nguyên sử dụng: {device}")
    
    # 2. Khởi tạo Dataset & DataLoader theo chuẩn X-Sub/X-View
    protocol = config['data'].get('protocol', 'xsub')
    modality = config['data'].get('modality', 'joint')  # 'joint' | 'bone' | 'velocity'
    print(f"[*] Modality: {modality.upper()}")

    # --- Nạp toàn bộ tập TRAIN (100% dữ liệu) ---
    train_ds = NTUActionDataset(
        data_path=config['data']['paths'],
        max_frames=config['data']['max_frames'],
        split='train',
        protocol=protocol,
        modality=modality,
    )

    test_ds = NTUActionDataset(
        data_path=config['data']['paths'],
        max_frames=config['data']['max_frames'],
        split='test',
        protocol=protocol,
        modality=modality,
    )
    
    num_workers = int(config['training'].get('num_workers', 8))
    pin_memory = bool(config['training'].get('pin_memory', torch.cuda.is_available()))
    
    # --- KHỞI TẠO 2 DATALOADER (TRAIN & TEST) ---
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),  
        prefetch_factor=2 if num_workers > 0 else None,  
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    # 3. Khởi tạo Model Classifier
    model = NTUActionClassifier(
        pretrained_path=config['model']['pretrained_path'],
        num_frames=config['data']['max_frames'],
        num_classes=config['model']['num_classes'],
        embed_dim=config['model'].get('embed_dim', 256),   # Đồng bộ với pretrain
        depth=config['model'].get('depth', 8),
        num_heads=config['model'].get('num_heads', 8),
        segment_length=int(config['model'].get('segment_length', 4)),
        dropout=float(config['model'].get('dropout', 0.1)),
        drop_path=float(config['training'].get('drop_path', 0.1)), # Thêm DropPath
        min_pretrained_match_ratio=float(config['model'].get('min_pretrained_match_ratio', 0.9)),
    ).to(device)
    
    # Optimizer — Layer-wise LR Decay (đúng bài báo S-JEPA mục 4.3)
    base_lr     = float(config['training']['lr'])            # LR đỉnh (classifier head)
    layer_decay = float(config['training'].get('layer_decay', 0.75))
    weight_decay = float(config['training']['weight_decay'])

    param_groups = get_layerwise_param_groups(model, base_lr, layer_decay, weight_decay)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    grad_accum_steps = int(config['training'].get('grad_accum_steps', 1))
    if grad_accum_steps < 1:
        raise ValueError("training.grad_accum_steps phải >= 1")
    grad_clip = float(config['training'].get('grad_clip', 1.0))

    # ĐÚNG BÀI BÁO S-JEPA: Sử dụng Label Smoothing 0.1
    smoothing = float(config['training'].get('label_smoothing', 0.1))
    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    
    # [KÍCH HOẠT MANIFOLD MIXUP]
    mixup_alpha = float(config['training'].get('mixup', 1.0))
    mixup_active = mixup_alpha > 0.0
    print(f"[*] Triển khai Manifold Mixup (alpha={mixup_alpha}) để dập Overfitting")
    
    scaler = GradScaler('cuda')  # [AMP] Scale gradient để tránh underflow với float16

    # Lịch học: Warmup 5 epochs → Cosine Decay về final_lr (configurable)
    steps_per_epoch = max(1, (len(train_loader) + grad_accum_steps - 1) // grad_accum_steps)
    total_steps = config['training']['epochs'] * steps_per_epoch
    warmup_epochs = int(config['training'].get('warmup_epochs', 5)) 
    warmup_steps = warmup_epochs * steps_per_epoch
    final_lr = float(config['training'].get('final_lr', 1e-5))

    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        start_lr=0.0,
        ref_lr=base_lr,           # Tỉ lệ scale dựa trên đỉnh (classifier)
        T_max=total_steps,
        final_lr=final_lr
    )
    
    # 4. Init Wandb
    wandb.init(
        project=config['wandb']['project'], 
        name=config['wandb']['name'], 
        config=config
    )
    
    # 5. Huấn luyện
    run_name = config['wandb']['name']  
    save_dir = os.path.join("checkpoints_finetuned", run_name)
    os.makedirs(save_dir, exist_ok=True)
    save_every = int(config['training'].get('save_every', 10))
    
    total_epochs = config['training']['epochs']
    no_aug_start  = config['training'].get('no_aug_last_epochs', 20)  # Tắt aug ở N epoch cuối

    for epoch in range(total_epochs):
        # --- TẮt mixup và augmentation ở chặng cuối ---
        is_final_phase = epoch >= (total_epochs - no_aug_start)
        if is_final_phase and mixup_active:
            mixup_active = False
            train_ds.disable_aug = True
            print(f"\n[Epoch {epoch+1}] ⭐ No-Aug Phase: Tắt Mixup + Aug để nén đặc trưng tinh khiết!")
        # --- TRAINING ---
        model.train()
        train_loss = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [TRAIN]")
        train_acc1_total = 0
        train_acc5_total = 0
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            with autocast('cuda'):  # [AMP] Forward pass dùng float16
                if model.training and mixup_active:
                    logits, y_a, y_b, lam = model(x, target=y, mixup_alpha=mixup_alpha)
                    loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                else:
                    logits = model(x)
                    loss = criterion(logits, y)

            scaler.scale(loss / grad_accum_steps).backward()  # [AMP] Scale gradient
            
            # Tính accuracy cho batch này (Top-1 và Top-5)
            acc1, acc5 = calculate_topk_accuracy(logits, y, topk=(1, 5))
            train_acc1_total += acc1.item()
            train_acc5_total += acc5.item()

            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)                                    # [AMP] Unscale trước khi clip
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # Gradient clipping
                scaler.step(optimizer)                                        # [AMP] Update weights
                scaler.update()                                               # [AMP] Cập nhật scale factor
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()       # Nhích tốc độ học mỗi optimizer step

            train_loss += loss.item()
            curr_acc1 = 100.0 * acc1.item() / x.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{curr_acc1:.2f}%"})
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc1 = 100.0 * train_acc1_total / len(train_ds)
        avg_train_acc5 = 100.0 * train_acc5_total / len(train_ds)
        current_lr = optimizer.param_groups[-1]['lr'] # Xem tốc độ của Head

        # --- ĐÁNH GIÁ TRỰC TIẾP TRÊN TEST SET SAU MỖI EPOCH (SOTA STYLE) ---
        model.eval()
        test_acc1_total = 0
        test_acc5_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                with autocast('cuda'):
                    logits = model(x)
                
                acc1, acc5 = calculate_topk_accuracy(logits, y, topk=(1, 5))
                test_acc1_total += acc1.item() # Cộng dồn số câu đúng
                test_acc5_total += acc5.item()
                
        # Tính trên tổng số mẫu thực tế của Test Set
        avg_test_acc1 = 100.0 * test_acc1_total / len(test_ds)
        avg_test_acc5 = 100.0 * test_acc5_total / len(test_ds)
        model.train()

        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | "
              f"Train Acc@1: {avg_train_acc1:.2f}% | Train Acc@5: {avg_train_acc5:.2f}% | "
              f"Test Acc@1: {avg_test_acc1:.2f}% | Test Acc@5: {avg_test_acc5:.2f}% | "
              f"LR: {current_lr:.2e}")
        wandb.log({
            "train_loss": avg_train_loss, 
            "train_acc1": avg_train_acc1,
            "train_acc5": avg_train_acc5,
            "test_acc1": avg_test_acc1, 
            "test_acc5": avg_test_acc5, 
            "lr_head": current_lr, 
            "epoch": epoch+1
        })

        # Lưu best checkpoint dựa trên Acc@1
        if not hasattr(main_finetune, 'best_acc'):
            main_finetune.best_acc = 0.0
        if avg_test_acc1 > main_finetune.best_acc:
            main_finetune.best_acc = avg_test_acc1
            best_path = os.path.join(save_dir, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  ★ Best model updated: {avg_test_acc1:.2f}% → {best_path}")

        # Lưu checkpoint định kỳ
        if (epoch + 1) % save_every == 0 or (epoch + 1) == config['training']['epochs']:
            ckpt_path = os.path.join(save_dir, f"last_ep{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"==> Đã lưu mô hình: {ckpt_path}")


    # 6. ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST
    # Đã báo cáo kết quả trong vòng lặp Epoch, kết thúc.
    print(f"\n=======================================================")
    print(f"HUẤN LUYỆN HOÀN TẤT. KẾT QUẢ TỐT NHẤT (Best Test Acc@1): {main_finetune.best_acc:.2f}%")
    print(f"Checkpoints lưu tại: {save_dir}")
    print(f"=======================================================")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='S-JEPA Fine-tuning')
    parser.add_argument('--config', type=str, default='configs/finetune.yaml', 
                        help='Path to the config file')
    args = parser.parse_args()
    
    main_finetune(config_path=args.config)