import os
import yaml
import time
import math
import glob
import torch
from torch.cuda.amp import GradScaler, autocast  # [AMP] Mixed Precision
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# Import Pro structure components
from src.core.sjepa import sjepa_base
from src.datasets.ntu_dataset import SJEPA_UnsupervisedDataset
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule

def main(config_path=None):
    # 1. Nạp Configuration từ YAML
    if config_path is None:
        config_path = 'configs/pretrain.yaml'
    print(f"Đang dùng config: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"--- KHỞI ĐỘNG HỆ THỐNG HUẤN LUYỆN S-JEPA PRO ---")
    print(f"Tài nguyên sử dụng: {device}")
    
    # 2. Khởi tạo Dataset & DataLoader
    dataset = SJEPA_UnsupervisedDataset(
        data_path=config['data']['paths'],
        max_frames=config['data']['max_frames'],
        mask_ratio=config['data'].get('mask_ratio', 0.9),
        segment_length=int(config['data'].get('segment_length', 4)),
        protocol=config['data'].get('protocol', None),      # Lọc theo protocol nếu có
        num_classes=config['data'].get('num_classes', None) # Giới hạn NTU-60 hoặc NTU-120
    )
    # Tối ưu hóa: num_workers=4 (Windows ổn định hơn với 4 thay vì 8)
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 3. Khởi tạo Model S-JEPA (đọc kiến trúc từ config)
    model = sjepa_base(
        num_frames=config['data']['max_frames'],
        skel_input_dim=config['model']['input_dim'],
        temp_s=config['model']['temp_s'],
        temp_t=config['model']['temp_t'],
        # Kiến trúc từ YAML (default = đúng bài báo nếu không có config)
        embed_dim=config['model'].get('embed_dim', 256),   # C_e = 256
        depth=config['model'].get('depth', 8),             # L_e = 8
        num_heads=config['model'].get('num_heads', 8),
        segment_length=int(config['model'].get('segment_length', 4)),
    ).to(device)
    
    # Optimizer (Sử dụng cấu hình từ YAML)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config['training']['lr']), 
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.95)
    )
    grad_accum_steps = int(config['training'].get('grad_accum_steps', 1))
    if grad_accum_steps < 1:
        raise ValueError("training.grad_accum_steps phải >= 1")
    
    steps_per_epoch = math.ceil(len(dataloader) / grad_accum_steps)
    total_steps = config['training']['epochs'] * steps_per_epoch
    warmup_steps = config['training']['warmup_epochs'] * steps_per_epoch

    # LR Scheduler: Linear Warmup + Cosine Decay
    # Bài báo: Peak 1e-3 -> Final 5e-4 (Tức là đáy LR bằng 50% đỉnh LR)
    peak_lr = float(config['training']['lr'])
    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        start_lr=0.0,
        ref_lr=peak_lr,
        T_max=total_steps,
        final_lr=peak_lr * 0.5   # [S-JEPA Paper]
    )

    # [FIX #2] WD Scheduler: Weight Decay tăng từ 0.04 → 0.4 theo Cosine (đúng theo bài báo)
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=config['training']['weight_decay'],
        T_max=total_steps,
        final_wd=config['training'].get('weight_decay_final', 0.4)
    )

    # [FIX #3] Gradient Clipping threshold
    grad_clip = config['training'].get('grad_clip', 0.3)
    
    # 4. Init Wandb 
    wandb.init(
        project=config['wandb']['project'], 
        name=config['wandb']['name'], 
        config=config
    )
    
    # [AMP] Khởi tạo GradScaler — giữ gradient ổn định khi dùng float16
    scaler = GradScaler()

    # 5. Vòng lặp Huấn luyện (Main Loop)
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    m_base, m_final = 0.9999, 1.0  # Teacher Momentum — đúng bài báo: λ từ 0.9999 → 1.0
    start_epoch = 0
    
    # [MỚI] LOGIC RESUME: Tự động tìm checkpoint gần nhất
    if config.get('training_options', {}).get('resume', False):
        checkpoints = glob.glob(os.path.join(config['training']['checkpoint_dir'], "sjepa_pro_ep*.pth"))
        if checkpoints:
            latest_ckpt = max(checkpoints, key=os.path.getctime)
            print(f"==> RESUME: Phát hiện checkpoint cũ tại {latest_ckpt}")
            try:
                checkpoint = torch.load(latest_ckpt, map_location=device)
                model.student_encoder.load_state_dict(checkpoint['student'])
                model.teacher_encoder.load_state_dict(checkpoint['teacher'])
                model.predictor.load_state_dict(checkpoint['predictor'])
                
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                wd_scheduler.load_state_dict(checkpoint['wd_scheduler'])
                scaler.load_state_dict(checkpoint['scaler'])

                # Tiêm kích T_max mới vào scheduler đề phòng trường hợp User trượt config Lên/Xuống epochs
                lr_scheduler.T_max = total_steps - warmup_steps
                wd_scheduler.T_max = total_steps

                
                start_epoch = checkpoint['epoch']
                print(f"==> Nạp thành công toàn bộ hệ thống! Tiếp tục từ Epoch {start_epoch + 1}")
            except Exception as e:
                print(f"BỎ QUA RESUME: Lỗi nạp checkpoint: {e}")

    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        dataset.set_epoch(epoch)
        epoch_loss = 0
        t0 = time.time()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (x_student, x_teacher, ctx_idx, tgt_idx) in enumerate(pbar):
            x_student, x_teacher = x_student.to(device), x_teacher.to(device)
            ctx_idx, tgt_idx = ctx_idx.to(device), tgt_idx.to(device)

            # Update Teacher Momentum (EMA) following Cosine Schedule
            progress = (epoch * len(dataloader) + batch_idx) / (config['training']['epochs'] * len(dataloader))
            # Cosine schedule from m_base (0.9999) to m_final (1.0)
            curr_m = m_final - (m_final - m_base) * 0.5 * (1.0 + math.cos(math.pi * progress))

            with autocast():  # [AMP] Forward pass dùng float16
                # Đưa thẳng ctx_idx (12 frames) và tgt_idx (108 frames) vào model 1 lần duy nhất!
                loss = model(x_student, x_teacher, [ctx_idx], [tgt_idx])
            loss_for_backward = loss / grad_accum_steps
            scaler.scale(loss_for_backward).backward()            # [AMP] Scale gradient

            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == len(dataloader))
            if should_step:
                scaler.unscale_(optimizer)               # [AMP] Unscale trước khi clip
                # [FIX #3] Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)                   # [AMP] Update weights
                scaler.update()                          # [AMP] Cập nhật scale factor
                optimizer.zero_grad(set_to_none=True)

                # Update LR và WD Schedulers trên mỗi optimizer step
                curr_lr = lr_scheduler.step()
                curr_wd = wd_scheduler.step()
                model.update_teacher(m=curr_m)
            else:
                curr_lr = optimizer.param_groups[0]['lr']
                curr_wd = optimizer.param_groups[0].get('weight_decay', 0.0)
            
            epoch_loss += loss.item()
            # [MỚI] Theo dõi VRAM
            vram_peak = torch.cuda.max_memory_allocated(device) / (1024**3) # Mức Đỉnh VRAM
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{curr_lr:.2e}", "vram": f"{vram_peak:.2f}GB"})
            
            if batch_idx % 10 == 0:
                wandb.log({
                    "iter_loss": loss.item(),
                    "lr": curr_lr,
                    "weight_decay": curr_wd,
                    "momentum": curr_m,
                    "vram_gb": vram_peak
                })

        avg_loss = epoch_loss / len(dataloader)
        print(f"Kết thúc Epoch {epoch+1} | Loss: {avg_loss:.4f} | Time: {time.time()-t0:.2f}s")
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch+1})
        
        if (epoch + 1) % config['training']['save_every'] == 0:
            save_path = os.path.join(config['training']['checkpoint_dir'], f"sjepa_pro_ep{epoch+1}.pth")
            checkpoint = {
                'epoch': epoch + 1,
                'student': model.student_encoder.state_dict(),
                'teacher': model.teacher_encoder.state_dict(),
                'predictor': model.predictor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'wd_scheduler': wd_scheduler.state_dict(),
                'scaler': scaler.state_dict() 
            }
            torch.save(checkpoint, save_path)
            print(f"==> Đã lưu checkpoint: {save_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="S-JEPA Pre-training Logic")
    parser.add_argument('--config', type=str, default='configs/pretrain.yaml', help="Path to YAML config")
    args = parser.parse_args()
    
    main(config_path=args.config)
