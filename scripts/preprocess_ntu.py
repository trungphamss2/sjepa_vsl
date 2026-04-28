import os
import glob
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import sys
import re

# Thêm root vào path để import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets.ntu_dataset import read_ntu_skeleton

def denoise_skeleton(skel_25):
    """
    Denoising chuyên sâu theo phong cách MAMP/S-JEPA:
    1. Lọc theo độ dài (Length > 11)
    2. Lọc theo độ phủ (Spread Ratio: X-spread / Y-spread > 0.1)
    3. Lọc theo chuyển động (Motion Variance): Chọn 2 người "động" nhất
    """
    if skel_25 is None or len(skel_25) == 0:
        return None
    
    # NTU raw từ read_ntu_skeleton trả về [T, NumBodies, 25, 3]
    T, num_bodies, J, C = skel_25.shape
    valid_bodies = []
    
    for b in range(num_bodies):
        body = skel_25[:, b, :, :] # [T, 25, 3]
        
        # 1. Lọc theo độ dài: Kiểm tra số frame không trống
        non_zero_mask = (np.abs(body).sum(axis=(1, 2)) > 1e-6)
        actual_len = np.sum(non_zero_mask)
        if actual_len <= 11:
            continue
            
        # 2. Lọc theo Spread Ratio (Tránh stick noise)
        valid_frames = body[non_zero_mask]
        if len(valid_frames) == 0: continue
        
        x_min, x_max = valid_frames[:, :, 0].min(), valid_frames[:, :, 0].max()
        y_min, y_max = valid_frames[:, :, 1].min(), valid_frames[:, :, 1].max()
        x_spread = x_max - x_min
        y_spread = y_max - y_min
        
        # Tỷ lệ bề ngang / bề dọc (Người bình thường ~0.2 - 0.5)
        if y_spread == 0 or (x_spread / y_spread) < 0.1:
            continue
            
        # 3. Tính độ biến thiên chuyển động (Motion Variance)
        motion = np.diff(body, axis=0) # [T-1, 25, 3]
        variance = np.var(motion)
        
        valid_bodies.append({
            'data': body,
            'variance': variance
        })
        
    if not valid_bodies:
        return None
        
    # Sắp xếp theo chuyển động giảm dần và lấy tối đa 2 người
    valid_bodies.sort(key=lambda x: x['variance'], reverse=True)
    selected = valid_bodies[:2]
    
    # Hợp nhất lại thành [T, 2, 25, 3]
    final_skel = np.zeros((T, 2, 25, 3), dtype=np.float32)
    for i, b_info in enumerate(selected):
        final_skel[:, i, :, :] = b_info['data']
        
    # FIX LỖI BROADCASTING: Căn tâm (Centering) dựa trên SpineBase của diễn viên chính
    # main_root shape: [T, 1, 1, 3] để trừ cho [T, 2, 25, 3]
    main_root = final_skel[:, 0:1, 0:1, :] 
    final_skel = final_skel - main_root
    
    return final_skel.reshape(T, -1) # [T, 150]

def convert_one_file(file_path):
    """Đọc file .skeleton, tiền xử lý sạch và lưu thành .npy (Ghi đè)"""
    try:
        npy_path = file_path.replace('.skeleton', '.npy')
        raw_skel = read_ntu_skeleton(file_path)
        if raw_skel is None: return False
        
        processed_skel = denoise_skeleton(raw_skel)
        if processed_skel is None: return False
        
        np.save(npy_path, processed_skel)
        return True
    except Exception:
        return False

def main():
    print("--- S-JEPA SOTA PRE-PROCESSOR (DENOISING ENABLED) ---")
    DATA_PATHS = [
        "./DATA/nturgb+d_skeletons",
        "./DATA/nturgb+d120_skeletons"
    ]
    BLACKLIST_FILE = "./DATA/missing_skeletons.txt"
    
    blacklist = set()
    if os.path.exists(BLACKLIST_FILE):
        with open(BLACKLIST_FILE, 'r') as f:
            blacklist = {line.strip() for line in f}
        print(f"Bỏ qua {len(blacklist)} mẫu lỗi hệ thống.")

    all_skeletons = []
    for dp in DATA_PATHS:
        if os.path.exists(dp):
            files = glob.glob(os.path.join(dp, "*.skeleton"))
            all_skeletons.extend([f for f in files if os.path.basename(f).split('.')[0] not in blacklist])
    
    if not all_skeletons:
        print("Không tìm thấy file .skeleton nào!")
        return

    print(f"Bắt đầu làm sạch và xử lý {len(all_skeletons)} file...")
    num_workers = min(os.cpu_count() or 4, 16) 
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(convert_one_file, all_skeletons), total=len(all_skeletons)))
        
    print(f"HOÀN TẤT TUYỆT ĐỐI! Thành công: {sum(results)}/{len(all_skeletons)}")

if __name__ == "__main__":
    main()
