# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn

class SkeletonEmbed(nn.Module):
    """Segment-joint patch embedding following S-JEPA paper."""
    def __init__(self, input_dim=75, embed_dim=768, num_frames=120, segment_length=4):
        super().__init__()
        assert input_dim % 3 == 0, "input_dim phải chia hết cho 3 (x,y,z)"
        assert num_frames % segment_length == 0, "num_frames phải chia hết cho segment_length"
        self.num_joints = input_dim // 3
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.segment_length = segment_length
        self.num_segments = num_frames // segment_length

        # Joint-wise linear embedding for each temporal segment ( flattened: l * 3 coords )
        self.joint_proj = nn.Linear(segment_length * 3, embed_dim, bias=True)

        # Paper uses separate learnable spatial and temporal positional embeddings.
        self.spatial_pe = nn.Parameter(torch.zeros(1, 1, self.num_joints, embed_dim))
        self.temporal_pe = nn.Parameter(torch.zeros(1, self.num_segments, 1, embed_dim))
        nn.init.trunc_normal_(self.spatial_pe, std=0.02)
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

    def forward(self, x):
        # x: [B, T, 75]
        B, T, _ = x.shape
        if T != self.num_frames:
            raise ValueError(f"Expected T={self.num_frames}, got T={T}")

        # [B, T, 75] -> [B, T, 25, 3]
        x_joints = x.view(B, T, self.num_joints, 3)
        # Group non-overlapping temporal segments and flatten temporal dimension into feature dimension
        # [B, T, 25, 3] -> [B, Te, l, 25, 3]
        x_segments = x_joints.view(B, self.num_segments, self.segment_length, self.num_joints, 3)
        # [B, Te, l, 25, 3] -> [B, Te, 25, l, 3] -> [B, Te, 25, l*3]
        x_flatten = x_segments.transpose(2, 3).reshape(B, self.num_segments, self.num_joints, self.segment_length * 3)
        
        # [B, Te, 25, l*3] -> [B, Te, 25, D]
        x_emb = self.joint_proj(x_flatten)
        x_emb = x_emb + self.spatial_pe + self.temporal_pe
        # [B, Te, 25, D] -> [B, Te*25, D]
        return x_emb.view(B, self.num_segments * self.num_joints, self.embed_dim)

