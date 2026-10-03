import numpy as np

from .data_utils import add_bat_vector_features


V8_EXTRA_FEATURE_DIM = 9
V8_INPUT_DIM = 20 + V8_EXTRA_FEATURE_DIM


def _prepend_zero_diff(values):
    out = np.zeros_like(values, dtype=np.float32)
    if len(values) > 1:
        out[1:] = values[1:] - values[:-1]
    return out


def _speed(points_px, diag):
    velocity = np.zeros_like(points_px, dtype=np.float32)
    if len(points_px) > 1:
        velocity[1:] = points_px[1:] - points_px[:-1]
    return np.linalg.norm(velocity, axis=1).astype(np.float32) / diag


def _acceleration(points_px, diag):
    velocity = np.zeros_like(points_px, dtype=np.float32)
    if len(points_px) > 1:
        velocity[1:] = points_px[1:] - points_px[:-1]
    acc = np.zeros_like(points_px, dtype=np.float32)
    if len(points_px) > 2:
        acc[2:] = velocity[2:] - velocity[1:-1]
    return np.linalg.norm(acc, axis=1).astype(np.float32) / diag


def add_v8_motion_features(seq8, image_width=1920.0, image_height=1080.0):
    """Append explicit bat geometry and temporal observation-quality features.

    Inputs stay normalized except for derived magnitudes, which are computed in
    pixel space and divided by the image diagonal to keep feature scale compact.
    """
    seq8 = np.asarray(seq8, dtype=np.float32)
    seq10 = add_bat_vector_features(seq8)
    scale = np.asarray([image_width, image_height], dtype=np.float32)
    diag = float(np.hypot(image_width, image_height))
    if not np.isfinite(diag) or diag <= 0.0:
        diag = 1.0

    head_px = seq8[:, 0:2] * scale
    tail_px = seq8[:, 2:4] * scale
    bat_vec_px = tail_px - head_px
    length_px = np.linalg.norm(bat_vec_px, axis=1).astype(np.float32)
    length = length_px / diag
    length_change = _prepend_zero_diff(length)

    unit = np.zeros_like(bat_vec_px, dtype=np.float32)
    valid_length = length_px > 1e-6
    unit[valid_length] = bat_vec_px[valid_length] / length_px[valid_length, None]

    direction_change = np.zeros(len(seq8), dtype=np.float32)
    if len(seq8) > 1:
        dot = np.sum(unit[1:] * unit[:-1], axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        valid_pairs = valid_length[1:] & valid_length[:-1]
        direction_change[1:] = np.where(valid_pairs, 1.0 - dot, 0.0).astype(np.float32)

    extra = np.column_stack(
        [
            length.astype(np.float32),
            length_change.astype(np.float32),
            unit[:, 0].astype(np.float32),
            unit[:, 1].astype(np.float32),
            direction_change.astype(np.float32),
            _speed(head_px, diag),
            _speed(tail_px, diag),
            _acceleration(head_px, diag),
            _acceleration(tail_px, diag),
        ]
    )
    return np.concatenate([seq10, extra], axis=1).astype(np.float32)
