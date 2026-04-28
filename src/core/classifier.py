import torch
import torch.nn as nn
from src.core.vision_transformer import VisionTransformer

class NTUActionClassifier(nn.Module):
    def __init__(self, pretrained_path=None, num_frames=120, skel_input_dim=75,
                 num_classes=120, embed_dim=256, depth=8, num_heads=8,
                 segment_length=4, dropout=0.0, drop_path=0.1, 
                 min_pretrained_match_ratio=0.9):
        super().__init__()

        # 1. Base Encoder (S-JEPA ViT) — khởi tạo đúng kiến trúc bài báo
        self.encoder = VisionTransformer(
            num_frames=num_frames,
            skel_input_dim=skel_input_dim,
            embed_dim=embed_dim,   # 256 theo bài báo
            depth=depth,           # 8  theo bài báo
            num_heads=num_heads,   # 8  theo bài báo
            segment_length=segment_length,
            drop_path_rate=drop_path, # Kích hoạt DropPath để chống Overfit
        )

        # 2. Load Pre-trained weights if available
        if pretrained_path is not None:
            print(f"Loading Pre-trained weights from {pretrained_path}...")
            ckpt = torch.load(pretrained_path, map_location='cpu')
            if 'teacher' in ckpt:
                # Checkpoint Pretrain (S-JEPA) — trích xuất Teacher Encoder
                print("==> Phát hiện Checkpoint Pretrain, trích xuất 'teacher' (Target Encoder)...")
                state_dict = ckpt['teacher']
            elif any(k.startswith('encoder.') for k in ckpt.keys()):
                # Checkpoint Classifier (đã Finetune) — strip prefix 'encoder.'
                print("==> Phát hiện Checkpoint Classifier (Finetune), trích xuất Encoder weights...")
                state_dict = {k[len('encoder.'):]: v
                              for k, v in ckpt.items() if k.startswith('encoder.')}
            else:
                print("==> Nạp Checkpoint định dạng cũ (Legacy)...")
                state_dict = ckpt
            load_info = self.encoder.load_state_dict(state_dict, strict=False)
            
            # Filter matches for logging
            total_encoder_keys = len(self.encoder.state_dict())
            loaded_encoder_keys = total_encoder_keys - len(load_info.missing_keys)
            match_ratio = loaded_encoder_keys / max(1, total_encoder_keys)
            print(f"==> Pretrained match: {loaded_encoder_keys}/{total_encoder_keys} ({match_ratio * 100:.2f}%)")
            
            if match_ratio < min_pretrained_match_ratio:
                raise RuntimeError(f"Match ratio {match_ratio * 100:.2f}% < {min_pretrained_match_ratio * 100:.2f}%")

        # 3. Single Linear Head — "Linear Probing" style
        # Ly do dung Single Linear thay vi MLP:
        #   - Encoder da tao ra khong gian 256-dim tot. MLP se bop meo no.
        #   - Gradient di thang tu Loss xuong Encoder, khong bi "la chan" boi lop an.
        #   - Dung chuan Self-Supervised Learning: Linear(D, num_classes).
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, target=None, mixup_alpha=1.0):
        B = x.size(0)
        is_multi_subject = (x.dim() == 4 and x.size(1) == 2)

        # 1. Batch Merging: [B, 2, T, 75] -> [B*2, T, 75]
        if is_multi_subject:
            x = x.view(-1, x.size(2), x.size(3))

        # 2. Encoder: [B*2, T, 75] -> [B*2, L, 256]
        features = self.encoder(x)

        # 3. Global Average Pool over ALL tokens: [B*2, L, 256] -> [B*2, 256]
        feat = features.mean(dim=1)

        # 4. Average over 2 subjects: [B*2, 256] -> [B, 256]
        if is_multi_subject:
            feat = feat.view(B, 2, -1).mean(dim=1)

        # 5. Dropout on 256-dim representation
        feat = self.dropout(feat)

        # 6. Manifold Mixup tai tang 256-dim (truoc Linear head)
        #    Day la dung diem de Mixup: khong gian Pretrain chua bi bien dang.
        #    Gradient tu Loss se di thang: Loss -> fc -> feat -> encoder.
        if self.training and target is not None and mixup_alpha > 0.0:
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
            index = torch.randperm(B).to(x.device)
            target_a, target_b = target, target[index]
            feat = lam * feat + (1 - lam) * feat[index]
        else:
            lam = 1.0
            target_a, target_b = target, target

        # 7. Single Linear Classifier
        logits = self.fc(feat)

        if self.training and target is not None and mixup_alpha > 0.0:
            return logits, target_a, target_b, lam
        return logits
