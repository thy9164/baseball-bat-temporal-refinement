"""Ground-truth parsing kept separate from observation construction."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SupervisionBatch:
    gt_head_norm: np.ndarray
    gt_tail_norm: np.ndarray
    head_visibility: np.ndarray
    tail_visibility: np.ndarray
    target_valid: np.ndarray


def build_supervision(*, gt_head_px, gt_tail_px, head_visibility, tail_visibility, image_width, image_height):
    gt_head = np.asarray(gt_head_px, dtype=np.float32)
    gt_tail = np.asarray(gt_tail_px, dtype=np.float32)
    if gt_head.shape != gt_tail.shape or gt_head.ndim != 2 or gt_head.shape[1] != 2:
        raise ValueError("GT coordinate arrays must have equal shape (N, 2)")
    head_vis = np.asarray(head_visibility, dtype=object)
    tail_vis = np.asarray(tail_visibility, dtype=object)
    if head_vis.shape != (len(gt_head),) or tail_vis.shape != (len(gt_head),):
        raise ValueError("visibility arrays must match GT length")
    scale = np.asarray([image_width, image_height], dtype=np.float32)
    target_valid = np.isfinite(gt_head).all(axis=1) & np.isfinite(gt_tail).all(axis=1) & (tail_vis != "ignore")
    return SupervisionBatch(
        gt_head_norm=gt_head / scale,
        gt_tail_norm=gt_tail / scale,
        head_visibility=head_vis,
        tail_visibility=tail_vis,
        target_valid=target_valid.astype(np.float32),
    )
