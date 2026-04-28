# Copyright (c) Meta Platforms, Inc. and affiliates.
import math
from functools import partial
import torch
import torch.nn as nn

from src.components.masks.utils import apply_masks
from src.utils.modules import Block
from src.utils.tensors import repeat_interleave_batch, trunc_normal_

class VisionTransformerPredictor(nn.Module):
    """Skeleton-JEPA Predictor Transformer (Pure Skeleton Version)"""
    def __init__(
        self,
        num_frames=120,
        segment_length=4,
        num_joints=25,
        embed_dim=768,
        predictor_embed_dim=384,
        out_embed_dim=None,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_mask_tokens=True,
        num_mask_tokens=10, # default 10 targets in V-JEPA
        **kwargs
    ):
        super().__init__()
        self.num_frames = num_frames
        self.segment_length = segment_length
        self.num_joints = num_joints
        self.num_patches = (num_frames // segment_length) * num_joints
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) for i in range(num_mask_tokens)]
            )

        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, predictor_embed_dim), requires_grad=True
        )
        trunc_normal_(self.predictor_pos_embed, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            ) for i in range(depth)
        ])

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, out_embed_dim or embed_dim, bias=True)
        self.init_std = init_std
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def forward(self, x, masks_x, masks_y, mask_index=0):
        if not isinstance(masks_x, list): masks_x = [masks_x]
        if not isinstance(masks_y, list): masks_y = [masks_y]
        B = x.size(0) // len(masks_x)

        x = self.predictor_embed(x)
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        mask_index = mask_index % self.num_mask_tokens
        pred_tokens = self.mask_tokens[mask_index].repeat(B, self.num_patches, 1)
        pred_tokens = apply_masks(pred_tokens, masks_y)
        
        pos_embs = apply_masks(x_pos_embed, masks_y)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        pred_tokens += pos_embs

        x = x.repeat(len(masks_x), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # Reorder tokens to temporal order — vectorized (thay Python for-loop)
        masks_x_cat = torch.cat(masks_x, dim=0)
        masks_y_cat = torch.cat(masks_y, dim=0)
        full_masks = torch.cat([masks_x_cat, masks_y_cat], dim=1)
        argsort = torch.argsort(full_masks, dim=1)
        # [B, N, D] — sắp xếp token theo thứ tự thời gian
        x = x.gather(1, argsort.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        full_masks = full_masks.gather(1, argsort)

        for blk in self.predictor_blocks:
            x = blk(x, mask=full_masks)
        
        x = self.predictor_norm(x)
        reverse_argsort = torch.argsort(argsort, dim=1)
        # Khôi phục thứ tự gốc — vectorized
        x = x.gather(1, reverse_argsort.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        x = x[:, x.shape[1] - pred_tokens.shape[1]:]
        return self.predictor_proj(x)
