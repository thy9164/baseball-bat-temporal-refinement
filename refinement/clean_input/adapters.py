"""Explicit adapters from detector CSV records to clean observations."""

import numpy as np

from .observation import build_observation_features
from .schema import FLOW_FEATURES

FLOW_SOURCE_COLUMNS = (
    "has_prev_flow", "head_valid", "tail_valid", "bbox_valid",
    "head_flow_x_mean", "head_flow_y_mean", "head_flow_mag_mean", "head_flow_mag_p95",
    "tail_flow_x_mean", "tail_flow_y_mean", "tail_flow_mag_mean", "tail_flow_mag_p95",
    "bbox_flow_x_mean", "bbox_flow_y_mean", "bbox_flow_mag_mean", "bbox_flow_mag_p95",
    "global_flow_x_mean", "global_flow_y_mean", "global_flow_mag_mean", "global_flow_mag_p95",
)
assert len(FLOW_SOURCE_COLUMNS) == len(FLOW_FEATURES)


def parse_float(value, default=np.nan):
    if value in {"", None}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def prediction_exists_from_detector(x, y, confidence):
    values = np.asarray([parse_float(x), parse_float(y), parse_float(confidence)], dtype=np.float32)
    return bool(np.isfinite(values).all() and values[2] > 0.0)


def normalize_flow_rows(rows, flow_clip_px=50.0):
    clip = float(flow_clip_px)
    if not np.isfinite(clip) or clip <= 0.0:
        raise ValueError("flow_clip_px must be finite and positive")
    output = []
    for row in rows:
        values = []
        for column in FLOW_SOURCE_COLUMNS:
            value = parse_float(row.get(column), 0.0)
            if column.endswith("_valid") or column == "has_prev_flow":
                values.append(float(value > 0.0))
            elif "mag" in column:
                values.append(float(np.clip(value, 0.0, clip) / clip))
            else:
                values.append(float(np.clip(value, -clip, clip) / clip))
        output.append(values)
    return np.asarray(output, dtype=np.float32)


def build_from_wide_rows(rows, *, image_width, image_height, view_label, use_flow=False, flow_clip_px=50.0):
    """Build observations without reading GT, visibility, status, or target fields."""
    rows = list(rows)
    head_xy = np.asarray([[parse_float(r.get("head_pred_x")), parse_float(r.get("head_pred_y"))] for r in rows], dtype=np.float32)
    tail_xy = np.asarray([[parse_float(r.get("tail_pred_x")), parse_float(r.get("tail_pred_y"))] for r in rows], dtype=np.float32)
    head_conf = np.asarray([parse_float(r.get("head_pred_conf"), 0.0) for r in rows], dtype=np.float32)
    tail_conf = np.asarray([parse_float(r.get("tail_pred_conf"), 0.0) for r in rows], dtype=np.float32)
    head_exists = np.asarray([prediction_exists_from_detector(*xy, conf) for xy, conf in zip(head_xy, head_conf)], dtype=bool)
    tail_exists = np.asarray([prediction_exists_from_detector(*xy, conf) for xy, conf in zip(tail_xy, tail_conf)], dtype=bool)
    flow = normalize_flow_rows(rows, flow_clip_px) if use_flow else None
    return build_observation_features(
        head_xy_norm=head_xy / np.asarray([image_width, image_height], dtype=np.float32),
        tail_xy_norm=tail_xy / np.asarray([image_width, image_height], dtype=np.float32),
        head_confidence=head_conf,
        tail_confidence=tail_conf,
        head_prediction_exists=head_exists,
        tail_prediction_exists=tail_exists,
        image_width=image_width,
        image_height=image_height,
        view_label=view_label,
        flow_features=flow,
    )
