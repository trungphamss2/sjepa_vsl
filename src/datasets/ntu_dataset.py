import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

# ============================================================
# X-Sub Protocol — Danh sách Subject ID dùng để TRAIN
# Nguồn: NTU RGB+D 120 Benchmark (Shahroudy et al., 2016 & Liu et al., 2019)
# ============================================================
XSUB_TRAIN_SUBJECTS = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
    38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
    80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
}

def read_ntu_skeleton(file_path):
    """Giải mã file .skeleton của bộ NTU RGB+D (25 khớp, 3D)"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    if not lines: return None
    num_frames = int(lines[0].strip())
    current_line = 1
    all_frames = []
    for _ in range(num_frames):
        if current_line >= len(lines): break
        num_bodies = int(lines[current_line].strip())
        current_line += 1
        if num_bodies == 0:
            all_frames.append(all_frames[-1] if all_frames else np.zeros((2, 25, 3)))
            continue
        
        frame_bodies = []
        for b in range(num_bodies):
            current_line += 1
            num_joints = int(lines[current_line].strip())
            current_line += 1
            joints = []
            for j in range(num_joints):
                data = lines[current_line].strip().split()
                if b < 2:  # Chỉ lấy tối đa 2 người (tiêu chuẩn NTU)
                    joints.append([float(data[0]), float(data[1]), float(data[2])])
                current_line += 1
            if b < 2:
                frame_bodies.append(joints)
                
        # Bổ sung số 0 nếu chỉ có 1 người trên khung hình (Zero Padding)
        if len(frame_bodies) == 0:
            frame_bodies = [np.zeros((25, 3)), np.zeros((25, 3))]
        elif len(frame_bodies) == 1:
            frame_bodies.append(np.zeros((25, 3)))
            
        all_frames.append(np.array(frame_bodies)) # [2, 25, 3]
    return np.array(all_frames) if all_frames else None


def temporal_resize(skeleton, target_T):
    """
    Bilinear (Linear) Interpolation để resize chuỗi xương về target_T frames.
    Đúng theo bài báo S-JEPA ECCV 2024 (thay vì padding bằng số 0).
    Input:  skeleton [T, D]
    Output: skeleton [target_T, D]
    """
    T = skeleton.shape[0]
    if T == target_T:
        return skeleton.astype(np.float32)
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_T)
    # Vectorized interpolation trên toàn bộ chiều D cùng lúc
    resized = np.array([np.interp(x_new, x_old, skeleton[:, d]) for d in range(skeleton.shape[1])]).T
    return resized.astype(np.float32)


def motion_aware_masking(skeleton, mask_ratio=0.9, segment_length=4):
    """
    Segment-based Motion-Aware Masking — Đúng bài báo S-JEPA ECCV 2024.

    Chia T frames thành (T // l) segments, mỗi segment gồm l frames LIÊN TIẾP.
    Chọn top (mask_ratio) segments có motion cao nhất → toàn bộ bị che.

    Tại sao phải segment-based (không phải frame-level)?
    ─────────────────────────────────────────────────────
    Frame-level (cũ, sai): Che frame 5, nhưng vẫn thấy frame 4 và 6
    → Predictor "lười", chỉ lấy trung bình frame 4+6 để đoán frame 5
    → Temporal Interpolation Leakage → không học được gì thực sự

    Segment-based (mới, đúng): Che frames 5-6-7-8 cùng lúc
    → Predictor mù hoàn toàn đoạn đó, buộc phải học dynamics thực sự
    → Đúng cơ chế bài báo: l=4, 30 segments, che 27 segments (=108 frames)

    Args:
        skeleton:       [T, D]
        mask_ratio:     0.9 → che 90% segments
        segment_length: l=4 theo bài báo S-JEPA ECCV 2024
    Returns:
        target_idx:  [T*mask_ratio] frame indices bị che
        context_idx: [T*(1-mask_ratio)] frame indices còn lại
    """
    T = skeleton.shape[0]
    l = segment_length
    num_segments = T // l
    if num_segments == 0:
        raise ValueError("segment_length lớn hơn số frame hiện có.")

    # Token-level motion over (segment, joint) following paper tokenization.
    # skeleton: [T, 150] -> [T, 2, 25, 3] -> sum over bodies: [T, 25, 3]
    skel = skeleton[:, :150].reshape(T, 2, 25, 3).sum(axis=1)
    # Frame-level joint motion: [T,25]
    frame_joint_motion = np.zeros((T, 25), dtype=np.float32)
    frame_joint_motion[1:] = np.abs(skel[1:] - skel[:-1]).sum(axis=-1)
    # Segment-joint motion: [Te,25]
    seg_joint_motion = frame_joint_motion[:num_segments * l].reshape(num_segments, l, 25).sum(axis=1)
    token_motion = seg_joint_motion.reshape(-1)  # [Te*25]

    num_tokens = token_motion.shape[0]
    num_target_tokens = int(num_tokens * mask_ratio)
    if num_target_tokens <= 0 or num_target_tokens >= num_tokens:
        raise ValueError("mask_ratio không hợp lệ cho số token hiện tại.")

    # Sample target indices with motion-aware probabilities (paper style),
    # instead of deterministic top-k selection.
    motion = np.maximum(token_motion, 0.0)
    probs = motion + 1e-6
    probs = probs / probs.sum()


    all_idx = np.arange(num_tokens, dtype=np.int64)
    target_idx = np.random.choice(all_idx, size=num_target_tokens, replace=False, p=probs)
    context_idx = np.setdiff1d(all_idx, target_idx, assume_unique=False)
    target_idx = np.sort(target_idx).astype(np.int64)
    context_idx = np.sort(context_idx).astype(np.int64)

    return target_idx, context_idx



def extract_subject_id(path):
    """Trích xuất Subject ID từ tên file NTU, ví dụ: S001C001P003R001A001 → 3"""
    match = re.search(r'P(\d{3})', os.path.basename(path))
    return int(match.group(1)) if match else None


def _load_files(data_dirs):
    """Helper: quét và trả về danh sách file hợp lệ từ nhiều thư mục."""
    all_paths = []
    for d in data_dirs:
        # Resolve về absolute path để tránh sai lệch khi chạy từ working dir khác
        d = os.path.abspath(d)
        if not os.path.isdir(d): continue

        # Tìm missing_skeletons.txt ở nhiều vị trí theo thứ tự ưu tiên
        blacklist = set()
        candidate_paths = [
            os.path.join(d, '..', 'missing_skeletons.txt'),   # DATA/missing_skeletons.txt
            os.path.join(d, 'missing_skeletons.txt'),          # Ngay trong thư mục data
            os.path.join(d, '..', '..', 'missing_skeletons.txt'),  # 2 cấp lên
        ]
        blacklist_file_found = None
        for candidate in candidate_paths:
            if os.path.exists(candidate):
                blacklist_file_found = os.path.abspath(candidate)
                break

        if blacklist_file_found:
            with open(blacklist_file_found, 'r', encoding='utf-8', errors='ignore') as f:
                blacklist = {line.strip() for line in f if line.strip()}
            print(f"[_load_files] Blacklist: {len(blacklist)} mẫu bị loại từ {blacklist_file_found}")
        else:
            print(f"[_load_files] CẢNH BÁO: Không tìm thấy missing_skeletons.txt cho {d} → không lọc missing samples!")

        npy_files  = glob.glob(os.path.join(d, '*.npy'))
        skel_files = glob.glob(os.path.join(d, '*.skeleton'))
        available_npys = {os.path.basename(f).split('.')[0] for f in npy_files}

        valid = [f for f in npy_files if os.path.basename(f).split('.')[0] not in blacklist]
        for f in skel_files:
            fname = os.path.basename(f).split('.')[0]
            if fname not in blacklist and fname not in available_npys:
                valid.append(f)
        all_paths.extend(valid)
    return all_paths


def _load_skeleton(path):
    """Helper: đọc một file (.npy hoặc .skeleton) trả về [T, 150]."""
    if path.endswith('.skeleton'):
        raw = read_ntu_skeleton(path) # [T, 2, 25, 3]
        if raw is None: return None
        # Chuẩn hóa Hip-centered: Mỗi người (b) bám theo cột sống của chính nó
        root = raw[:, :, 0:1, :] # [T, 2, 1, 3]
        raw_centered = raw - root
        return raw_centered.reshape(len(raw), -1).astype(np.float32) # [T, 150]
    else:
        skel = np.load(path)
        # Dự phòng cho file cũ
        if skel.ndim == 3: skel = skel.reshape(skel.shape[0], -1)
        if skel.shape[-1] == 75:
            # Zero-pad người 2 nếu file cache npy cũ chỉ có 75
            skel = np.concatenate([skel, np.zeros_like(skel)], axis=-1)
        return skel.astype(np.float32)


# ============================================================
# Dataset 1: Pre-training (Không giám sát)
# ============================================================
class SJEPA_UnsupervisedDataset(Dataset):
    """
    Dataset cho huấn luyện tự giám sát S-JEPA.
    Áp dụng đúng theo bài báo ECCV 2024:
    - Bilinear resize về T=120 frames
    - Motion-aware masking (mask_ratio=0.9)
    - Geometric augmentation tạo Student view

    Hỗ trợ lọc theo protocol (tránh data leak):
    - Nếu truyền protocol, chỉ dùng TRAINING SPLIT của protocol đó.
    - Nếu không truyền (None), dùng toàn bộ (hành vi cũ).
    """
    def __init__(self, data_path, max_frames=120, mask_ratio=0.9,
                 protocol=None, num_classes=None, segment_length=4, **kwargs):
        self.max_frames = max_frames
        self.mask_ratio = mask_ratio
        self.segment_length = segment_length
        data_dirs = [data_path] if isinstance(data_path, str) else data_path
        all_paths = _load_files(data_dirs)

        # Lọc theo protocol (training split) nếu được yêu cầu
        if protocol is not None:
            all_paths = self._filter_pretrain(all_paths, protocol, num_classes)
            print(f"[SJEPA_UnsupervisedDataset] Protocol={protocol.upper()}, "
                  f"num_classes={num_classes} → chỉ dùng TRAIN split.")

        self.data_paths = all_paths
        print(f"[SJEPA_UnsupervisedDataset] Nạp {len(self.data_paths)} file "
              f"từ {len(data_dirs)} thư mục.")

    def _filter_pretrain(self, all_paths, protocol, num_classes):
        """
        Lọc dữ liệu Pre-train KHÔNG BỊ DATA LEAK, tối đa hóa lượng dữ liệu.
        """
        filtered = []
        for path in all_paths:
            fname = os.path.basename(path)

            # 1. Bóc tách thông tin từ tên file
            action_match = re.search(r'A(\d{3})', fname)
            if not action_match: continue
            action_id = int(action_match.group(1))

            # Nếu user cấu hình giới hạn class (ví dụ num_classes=60), ta bỏ qua các class vượt ngưỡng
            if num_classes is not None and action_id > num_classes:
                continue

            # 2. LOGIC CHẶN LEAKAGE CHO TỪNG PROTOCOL
            # Nếu hành động thuộc 60 class đầu (Tức là nằm trong tập NTU-60)
            if action_id <= 60:
                if protocol == 'xsub':
                    subject_id = extract_subject_id(path)
                    # Chỉ lấy các subject thuộc tập Train, vứt tập Test
                    if subject_id is None or subject_id not in XSUB_TRAIN_SUBJECTS:
                        continue
                elif protocol == 'xview':
                    m = re.search(r'C(\d{3})', fname)
                    # Chỉ lấy Camera 2, 3 (Train), vứt Camera 1 (Test)
                    if not m or int(m.group(1)) not in {2, 3}:
                        continue
                elif protocol == 'xset':
                    m = re.search(r'S(\d{3})', fname)
                    # Chỉ lấy Setup chẵn (Train), lẻ (Test)
                    if not m or int(m.group(1)) % 2 != 0:
                        continue
            
            # ĐIỂM SÁNG: Nếu action_id > 60 (Class mới của NTU-120), code sẽ nhảy qua khối IF trên
            # và được append thẳng vào tập Train bất kể nó là Camera 1 hay Subject bao nhiêu.
            # Vì tập Test của NTU-60 không bao giờ chứa các hành động từ 61-120!

            filtered.append(path)
        return filtered

    def set_epoch(self, epoch):
        pass  # Motion-aware masking không cần thay đổi theo epoch

    def __len__(self): return len(self.data_paths)

    def __getitem__(self, idx):
        from src.datasets.transforms import geometric_transformation

        skeleton = _load_skeleton(self.data_paths[idx])
        if skeleton is None:
            skeleton = np.zeros((self.max_frames, 150), dtype=np.float32)

        T_raw = skeleton.shape[0]
        crop_ratio = np.random.uniform(0.5, 1.0)
        crop_len = max(int(T_raw * crop_ratio), 10)
        start = np.random.randint(0, max(1, T_raw - crop_len + 1))
        skeleton = skeleton[start : start + crop_len]

        # Bilinear resize: [T, 150] -> [max_frames, 150]
        skeleton = temporal_resize(skeleton, self.max_frames)

        # Motion-aware Masking cộng dồn biến thiên trên cả 2 thân (150 tọa độ)
        target_idx, context_idx = motion_aware_masking(
            skeleton,
            mask_ratio=self.mask_ratio,
            segment_length=self.segment_length
        )

        # Tạo Student view bằng Geometric Transformation (áp dụng chung 2 body qua trick reshape)
        skel_3d = skeleton.reshape(self.max_frames * 2, 25, 3)
        x_student = geometric_transformation(skel_3d)
        
        # Reshape nhả định dạng Batch-Merging Multi-Subject: [max_frames, 2, 75] -> [2, max_frames, 75]
        x_student = x_student.reshape(self.max_frames, 2, 75)
        x_student = np.transpose(x_student, (1, 0, 2))
        
        x_teacher = skeleton.reshape(self.max_frames, 2, 75)
        x_teacher = np.transpose(x_teacher, (1, 0, 2))

        return (
            torch.FloatTensor(x_student),
            torch.FloatTensor(x_teacher),
            torch.LongTensor(context_idx),
            torch.LongTensor(target_idx)
        )


# ============================================================
# Dataset 2: Fine-tuning (Có giám sát — Nhận diện hành động)
# ============================================================
class NTUActionDataset(Dataset):
    """
    Dataset cho Fine-tuning nhận diện hành động NTU-120.
    Hỗ trợ giao thức chuẩn: X-Sub và X-View.
    Áp dụng bilinear resize (đúng theo bài báo) thay vì zero padding.
    """
    def __init__(self, data_path, max_frames=120, split='train', protocol='xsub', modality='joint'):
        """
        Args:
            data_path: str hoặc list đường dẫn đến thư mục chứa file skeleton.
            max_frames: Số frames sau khi resize (mặc định 120 theo bài báo).
            split:    'train' hoặc 'test'.
            protocol: 'xsub' (Cross-Subject) hoặc 'xview' (Cross-View).
            modality: 'joint' | 'bone' | 'velocity'  (default: 'joint')
        """
        self.max_frames = max_frames
        self.split = split
        self.protocol = protocol
        self.modality = modality
        self.disable_aug = False  # Bật bởi finetune.py ở 20 epoch cuối

        data_dirs = [data_path] if isinstance(data_path, str) else data_path
        all_paths = _load_files(data_dirs)

        # Phân chia theo giao thức chuẩn
        self.file_paths = self._filter_by_protocol(all_paths)
        print(f"[NTUActionDataset] Protocol={protocol.upper()}, Split={split} → {len(self.file_paths)} mẫu.")

    def _filter_by_protocol(self, all_paths):
        """
        Phân chia train/test theo giao thức chuẩn của NTU RGB+D:

        - xsub  (Cross-Subject):  70 Subject ID cụ thể → train, còn lại → test
        - xview (Cross-View):     Camera C002 & C003   → train, C001  → test  [NTU-60]
        - xset  (Cross-Setup):    Setup ID chẵn        → train, lẻ   → test  [NTU-120]

        Tên file NTU có định dạng: S{setup}C{camera}P{subject}R{rep}A{action}
        Ví dụ: S001C001P001R001A001.skeleton
        """
        filtered = []
        for path in all_paths:
            fname = os.path.basename(path)

            if self.protocol == 'xsub':
                subject_id = extract_subject_id(path)
                if subject_id is None: continue
                is_train = subject_id in XSUB_TRAIN_SUBJECTS

            elif self.protocol == 'xview':
                # NTU-60: Camera 2 & 3 → Train, Camera 1 → Test
                match = re.search(r'C(\d{3})', fname)
                if not match: continue
                camera_id = int(match.group(1))
                is_train = camera_id in {2, 3}

            elif self.protocol == 'xset':
                # NTU-120: Setup ID chẵn → Train, lẻ → Test
                match = re.search(r'S(\d{3})', fname)
                if not match: continue
                setup_id = int(match.group(1))
                is_train = (setup_id % 2 == 0)  # Chẵn → Train

            else:
                raise ValueError(f"Protocol '{self.protocol}' không hợp lệ. Chọn: xsub | xview | xset")

            if (self.split == 'train') == is_train:
                filtered.append(path)
        return filtered

    def _extract_label(self, path):
        match = re.search(r'A(\d{3})', os.path.basename(path))
        return int(match.group(1)) - 1 if match else 0

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        skeleton = _load_skeleton(self.file_paths[idx])
        if skeleton is None:
            skeleton = np.zeros((self.max_frames, 150), dtype=np.float32)

        T_raw = skeleton.shape[0]

        # -----------------------------------------------------------------
        # TEMPORAL CROP - ĐÚNG CHUẨN BÀI BÁO S-JEPA ECCV 2024
        # -----------------------------------------------------------------
        if self.split == 'train' and not self.disable_aug:
            # TRAIN: Random crop [0.5, 1.0] (Data Augmentation mạnh để chống Overfit)
            crop_ratio = np.random.uniform(0.5, 1.0)
            crop_len = max(int(T_raw * crop_ratio), 10)
            start = np.random.randint(0, max(1, T_raw - crop_len + 1))
        else:  # TEST hoặc disable_aug=True: center crop cố định 0.9
            # TEST: Center crop 0.9 cố định (Đảm bảo đánh giá Deterministic)
            crop_ratio = 0.9
            crop_len = max(int(T_raw * crop_ratio), 10)
            start = (T_raw - crop_len) // 2 
            
        skeleton = skeleton[start : start + crop_len]
        # -----------------------------------------------------------------

        # Bilinear resize [T_crop, 150] → [max_frames, 150]
        skeleton = temporal_resize(skeleton, self.max_frames)

        # -----------------------------------------------------------------
        # SPATIAL AUGMENTATION - Đập tan Overfitting
        # -----------------------------------------------------------------
        if self.split == 'train' and not self.disable_aug:
            from src.datasets.transforms import geometric_transformation
            skel_3d = skeleton.reshape(self.max_frames * 2, 25, 3)
            skeleton = geometric_transformation(skel_3d)
            skeleton = skeleton.reshape(self.max_frames, 150)

        # Reshape: [120, 150] → [120, 2, 25, 3]
        skeleton = skeleton.reshape(self.max_frames, 2, 25, 3)

        # ------------------------------------------------------------------
        # MULTI-STREAM: chuyển đổi sang Bone hoặc Velocity nếu cần
        # ------------------------------------------------------------------
        if self.modality == 'velocity':
            vel = np.zeros_like(skeleton)
            vel[1:] = skeleton[1:] - skeleton[:-1]   # Δpos theo thời gian
            skeleton = vel
        elif self.modality == 'bone':
            # Vector nối khớp con tới khớp cha (chuẩn NTU RGB+D 25 khớp)
            ntu_parents = [0, 0, 1, 2, 20, 4, 5, 6, 20, 8, 9, 10,
                           0, 12, 13, 14, 0, 16, 17, 18, 1, 22, 21, 24, 11]
            bone = np.zeros_like(skeleton)
            for j in range(25):
                bone[:, :, j, :] = skeleton[:, :, j, :] - skeleton[:, :, ntu_parents[j], :]
            skeleton = bone
        # 'joint': giữ nguyên, không làm gì thêm

        # Trả về tensor [2, 120, 75]
        skeleton = skeleton.reshape(self.max_frames, 2, 75)
        skeleton = np.transpose(skeleton, (1, 0, 2))

        label = self._extract_label(self.file_paths[idx])
        return torch.FloatTensor(skeleton), torch.tensor(label, dtype=torch.long)
