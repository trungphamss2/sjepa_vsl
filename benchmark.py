import time
import torch
import torch.nn as nn
from tqdm import tqdm

from src.core.sjepa import sjepa_base
from src.core.classifier import NTUActionClassifier

try:
    from thop import profile, clever_format
except ImportError:
    print("Vui lòng cài đặt thop để đo FLOPs: pip install thop")
    exit(1)

def benchmark_pretrain(device):
    print("\n" + "="*50)
    print("SECTION 1: PRE-TRAINING (Self-Supervised Step)")
    print("="*50)
    
    # 1. Khởi tạo mô hình S-JEPA (Gồm Student, Teacher, Predictor)
    model = sjepa_base(
        num_frames=120,
        skel_input_dim=75,
        embed_dim=256,
        depth=8,
        predictor_depth=5
    ).to(device)
    
    # Optimizer & Scaler (Giống thực tế training)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler()
    
    # 2. Tham số
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters:    {total_params / 1e6:.2f} M (Student + Teacher + Predictor)")
    print(f"Trainable Params:    {trainable_params / 1e6:.2f} M (Student + Predictor)")

    # 3. Dummy Data cho Training Step
    batch_size = 64
    x_student = torch.randn(batch_size, 120, 75).to(device)
    x_teacher = torch.randn(batch_size, 120, 75).to(device)
    
    # Tạo indices giả cho mask (context 75, target 675)
    N = 750
    ctx_idx = torch.arange(0, 75).repeat(batch_size, 1).to(device)
    tgt_idx = torch.arange(75, N).repeat(batch_size, 1).to(device)

    # 4. Đo FLOPs (Forward pass duy nhất)
    try:
        flops, _ = profile(model, inputs=(x_student, x_teacher, ctx_idx, tgt_idx), verbose=False)
        flops_fmt, _ = clever_format([flops, 0], "%.2f")
        print(f"Forward FLOPs:       {flops_fmt}")
    except Exception as e:
        print(f"Lỗi khi đo FLOPs: {e}")

    # 5. Đo Training Throughput (Steps/sec)
    print(f"Measuring Training Speed (Batch Size {batch_size})...")
    num_iterations = 100
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warm-up
    model.train()
    for _ in range(10):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            loss = model(x_student, x_teacher, ctx_idx, tgt_idx)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        model.update_teacher(m=0.999) # EMA Update

    start_event.record()
    for _ in range(num_iterations):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            loss = model(x_student, x_teacher, ctx_idx, tgt_idx)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        model.update_teacher(m=0.999)
    end_event.record()
    
    torch.cuda.synchronize()
    total_time_ms = start_event.elapsed_time(end_event)
    steps_per_sec = num_iterations / (total_time_ms / 1000)
    
    print(f"Training Throughput: {steps_per_sec:.2f} Steps/sec")
    print(f"Peak VRAM Usage:     {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

def benchmark_finetune(device):
    print("\n" + "="*50)
    print("SECTION 2: FINE-TUNING (Supervised Inference)")
    print("="*50)
    
    model = NTUActionClassifier(
        num_frames=120,
        num_classes=60,
        embed_dim=256,
        depth=8,
        num_heads=8
    ).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    dummy_input = torch.randn(1, 2, 120, 75).to(device)

    # 1. FLOPs & Params
    try:
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops_fmt, _ = clever_format([flops, 0], "%.2f")
        print(f"Total Parameters:    {total_params / 1e6:.2f} M")
        print(f"Inference FLOPs:     {flops_fmt}")
    except Exception as e:
         print(f"Lỗi khi đo FLOPs: {e}")

    # 2. Speed Test
    batch_size = 64
    batched_input = torch.randn(batch_size, 2, 120, 75).to(device)
    
    def run_fps_test(use_amp=False):
        # Warm-up
        with torch.no_grad():
            for _ in range(30):
                if use_amp:
                    with torch.amp.autocast('cuda'): _ = model(batched_input)
                else: _ = model(batched_input)

        num_iterations = 100
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(num_iterations):
                if use_amp:
                    with torch.amp.autocast('cuda'): _ = model(batched_input)
                else: _ = model(batched_input)
        end_event.record()
        
        torch.cuda.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)
        cps = (num_iterations * batch_size) / (total_time_ms / 1000)
        max_mem = torch.cuda.max_memory_allocated() / 1024**2
        return cps, max_mem

    torch.cuda.reset_peak_memory_stats()
    fps_32, _ = run_fps_test(use_amp=False)
    fps_16, mem_16 = run_fps_test(use_amp=True)
    
    print(f"Standard (FP32):     {fps_32:.1f} CPS")
    print(f"AMP (FP16):          {fps_16:.1f} CPS")
    print(f"Peak VRAM Usage:     {mem_16:.1f} MB")

if __name__ == '__main__':
    print("=== S-JEPA COMPREHENSIVE SYSTEM BENCHMARK ===")
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run Benchmarks
    benchmark_pretrain(dev)
    benchmark_finetune(dev)
    
    print("\n" + "="*50)
    print("BENCHMARK COMPLETED")
    print("="*50)
