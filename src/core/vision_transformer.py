# Copyright (c) Meta Platforms, Inc. and affiliates.
import math
from functools import partial
import torch
import torch.nn as nn

from src.components.masks.utils import apply_masks
from src.utils.modules import Block
from src.utils.patch_embed import SkeletonEmbed
from src.utils.tensors import trunc_normal_

class VisionTransformer(nn.Module):
    """Skeleton-JEPA Vision Transformer (Pure Skeleton Version)"""
    def __init__(
        self,
        num_frames=120,       # Đúng theo bài báo S-JEPA
        skel_input_dim=75,
        embed_dim=256,        # C_e = 256 — đúng bài báo
        depth=8,              # L_e = 8  — đúng bài báo
        num_heads=8,          # 256 / 8 = 32 dim/head
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        use_sdpa=True,
        segment_length=4,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.num_frames = num_frames
        self.segment_length = segment_length
        self.num_segments = num_frames // segment_length

        # 1. Skeleton Embedding with learnable spatial+temporal positional embeddings.
        self.patch_embed = SkeletonEmbed(
            input_dim=skel_input_dim,
            embed_dim=embed_dim,
            num_frames=num_frames,
            segment_length=segment_length,
        )
        self.num_patches = self.num_segments * self.patch_embed.num_joints

        # 3. Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_sdpa=use_sdpa,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.init_std = init_std
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def forward(self, x, masks=None):
        if masks is not None and not isinstance(masks, list):
            masks = [masks]
        
        x = self.patch_embed(x)

        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        for blk in self.blocks:
            x = blk(x, mask=masks)
        return self.norm(x)


# ============================================================
# Các hàm factory dưới đây là tiện ích (KHÔNG dùng trong luồng chính).
# Luồng chính dùng sjepa_base() trong src/core/sjepa.py
# ============================================================
def vit_tiny(num_frames=120, **kwargs):
    return VisionTransformer(num_frames=num_frames, embed_dim=192, depth=12, num_heads=3, **kwargs)

def vit_small(num_frames=120, **kwargs):
    return VisionTransformer(num_frames=num_frames, embed_dim=384, depth=12, num_heads=6, **kwargs)

def vit_base(num_frames=120, **kwargs):
    return VisionTransformer(num_frames=num_frames, embed_dim=768, depth=12, num_heads=12, **kwargs)
