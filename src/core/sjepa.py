# Copyright (c) Meta Platforms, Inc. and affiliates.
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.vision_transformer import VisionTransformer
from src.components.predictor import VisionTransformerPredictor

class SJEPA(nn.Module):
    """
    S-JEPA wrapper for Skeleton data.
    """
    def __init__(
        self,
        embed_dim=256,
        num_frames=120,   # Fix: 500 → 120 theo bài báo
        skel_input_dim=75,
        temp_s=0.1,
        temp_t=0.06,
        center_momentum=0.9,
        # Transformer configs
        depth=8,
        num_heads=8,
        predictor_depth=5,
        predictor_embed_dim=256,
        segment_length=4,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.temp_s = temp_s
        self.temp_t = temp_t
        self.center_momentum = center_momentum
        
        # 1. Student Encoder 
        self.student_encoder = VisionTransformer(
            num_frames=num_frames,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            skel_input_dim=skel_input_dim,
            segment_length=segment_length,
            **kwargs
        )
        
        # 2. Teacher Encoder (EMA)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
            
        # 3. Predictor 
        self.predictor = VisionTransformerPredictor(
            num_frames=num_frames,
            segment_length=segment_length,
            embed_dim=embed_dim,              # Input từ encoder: C_e = 256
            predictor_embed_dim=predictor_embed_dim,  # Bên trong predictor: C_p = 256
            out_embed_dim=embed_dim,          # Output = C_e = 256 để tính Loss
            depth=predictor_depth,
            num_heads=max(1, predictor_embed_dim // 32),  # 256//32 = 8 heads
            use_mask_tokens=True,
            **kwargs
        )
        
        # 4. Center buffer for Teacher
        self.register_buffer("teacher_center", torch.zeros(1, 1, embed_dim))
        
    @torch.no_grad()
    def update_teacher(self, m=0.999):
        for s_param, t_param in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            t_param.data.mul_(m).add_(s_param.data, alpha=1.0 - m)

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=(0, 1), keepdim=True)
        self.teacher_center = self.teacher_center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, x_student, x_teacher, context_masks, target_masks):
        # 1. Kỹ thuật Batch Merging cho Multi-Subject
        # Nếu đầu vào có dạng [Batch, 2, T, 75], trộn 2 Subject vào chiều Batch
        if x_student.dim() == 4 and x_student.size(1) == 2:
            x_student = x_student.view(-1, x_student.size(2), x_student.size(3))
            x_teacher = x_teacher.view(-1, x_teacher.size(2), x_teacher.size(3))
            
            # Nhân bản Masks cho cả 2 người (Người A và B có cùng khung hình bị che)
            if not isinstance(context_masks, list): context_masks = [context_masks]
            if not isinstance(target_masks, list): target_masks = [target_masks]
            context_masks = [m.repeat_interleave(2, dim=0) for m in context_masks]
            target_masks = [m.repeat_interleave(2, dim=0) for m in target_masks]
        else:
            if not isinstance(context_masks, list): context_masks = [context_masks]
            if not isinstance(target_masks, list): target_masks = [target_masks]
            
        with torch.no_grad():
            full_teacher_rep = self.teacher_encoder(x_teacher)
            if len(target_masks) == 1:
                tgt_idx = target_masks[0].unsqueeze(-1).expand(-1, -1, self.embed_dim)
            else:
                tgt_idx = torch.stack(target_masks).unsqueeze(-1).expand(-1, -1, self.embed_dim)
                
            targets = torch.gather(full_teacher_rep, 1, tgt_idx) 
            self.update_center(full_teacher_rep)
            targets_centered = targets - self.teacher_center

        context_reps = self.student_encoder(x_student, masks=context_masks)
        predictions = self.predictor(context_reps, context_masks, target_masks)
        
        p1 = F.log_softmax(predictions / self.temp_s, dim=-1)
        p2 = F.softmax(targets_centered / self.temp_t, dim=-1)
        return - (p2 * p1).sum(dim=-1).mean()

def sjepa_base(**kwargs):
    """
    S-JEPA Base — Đúng theo bài báo ECCV 2024:
    - Encoder:   L_e=8  lớp, C_e=256 chiều, 8 heads
    - Predictor: L_p=5  lớp, C_p=256 chiều (bằng encoder)
    Nhận tham số từ caller (train.py/config) nếu có, dùng default nếu không.
    """
    embed_dim        = kwargs.pop('embed_dim', 256)      # C_e = 256
    depth            = kwargs.pop('depth', 8)            # L_e = 8
    num_heads        = kwargs.pop('num_heads', 8)        # 8 heads
    predictor_depth  = kwargs.pop('predictor_depth', 5)  # L_p = 5
    # predictor_embed_dim = C_p = 256 (bằng encoder, đúng bài báo)
    predictor_embed_dim = kwargs.pop('predictor_embed_dim', embed_dim)

    return SJEPA(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        predictor_embed_dim=predictor_embed_dim,
        predictor_depth=predictor_depth,
        **kwargs   # num_frames, skel_input_dim, temp_s, temp_t, ...
    )

