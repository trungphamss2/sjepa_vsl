# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch

def apply_masks(x, masks, concat=True):
    """
    :param x: [B, N, D]
    :param masks: list of tensors of shape [B, K] containing indices of patches to KEEP
    """
    all_x = []
    for m in masks:
        # m: [B, K]
        # index: [B, K, D]
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)
