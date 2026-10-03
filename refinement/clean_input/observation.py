"""Build model observations from inference-time information only."""

from dataclasses import dataclass

import numpy as np

from .schema import CLEAN_NO_FLOW_SCHEMA, CLEAN_WITH_FLOW_SCHEMA, FLOW_FEATURES

VIEW_LABELS = ("1b_side", "front", "3b_side", "home_side", "random")


@dataclass(frozen=True)
class ObservationBatch:
    features: np.ndarray
    feature_schema: tuple[str, ...]
    detector_observations: np.ndarray
    head_valid: np.ndarray
    tail_valid: np.ndarray
    head_missing: np.ndarray
    tail_missing: np.ndarray


def _as_xy(value, length: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (length, 2):
        raise ValueError(f"{name} must have shape ({length}, 2), got {arr.shape}")
    return arr


def _as_vector(value, length: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {arr.shape}")
    return arr


def _prediction_valid(xy, confidence, prediction_exists=None):
    confidence = np.asarray(confidence, dtype=np.float32)
    inferred_exists = np.isfinite(xy).all(axis=1) & np.isfinite(confidence) & (confidence > 0.0)
    if prediction_exists is None:
        return inferred_exists
    exists = np.asarray(prediction_exists, dtype=bool)
    if exists.shape != confidence.shape:
        raise ValueError("prediction_exists must match confidence shape")
    return exists & inferred_exists


def _fill_missing_xy(xy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Nearest-neighbour fill only for genuinely missing detector observations."""
    out = np.asarray(xy, dtype=np.float32).copy()
    missing = ~valid
    if not missing.any():
        return out
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        out[missing] = 0.0
        return out
    for index in np.flatnonzero(missing):
        nearest = valid_indices[np.argmin(np.abs(valid_indices - index))]
        out[index] = out[nearest]
    return out


def _prepend_zero_diff(values):
    out = np.zeros_like(values, dtype=np.float32)
    if len(values) > 1:
        out[1:] = values[1:] - values[:-1]
    return out


def _speed(points_px, diagonal):
    velocity = np.zeros_like(points_px, dtype=np.float32)
    if len(points_px) > 1:
        velocity[1:] = points_px[1:] - points_px[:-1]
    return np.linalg.norm(velocity, axis=1).astype(np.float32) / diagonal


def _acceleration(points_px, diagonal):
    velocity = np.zeros_like(points_px, dtype=np.float32)
    if len(points_px) > 1:
        velocity[1:] = points_px[1:] - points_px[:-1]
    acceleration = np.zeros_like(points_px, dtype=np.float32)
    if len(points_px) > 2:
        acceleration[2:] = velocity[2:] - velocity[1:-1]
    return np.linalg.norm(acceleration, axis=1).astype(np.float32) / diagonal


def _view_one_hot(view_label: str, length: int) -> np.ndarray:
    label = view_label if view_label in VIEW_LABELS else "random"
    vector = np.zeros(len(VIEW_LABELS), dtype=np.float32)
    vector[VIEW_LABELS.index(label)] = 1.0
    return np.tile(vector, (length, 1))


def build_observation_features(
    *,
    head_xy_norm,
    tail_xy_norm,
    head_confidence,
    tail_confidence,
    image_width: float,
    image_height: float,
    view_label: str,
    head_prediction_exists=None,
    tail_prediction_exists=None,
    flow_features=None,
) -> ObservationBatch:
    """Return the canonical 25- or 45-dimensional clean observation.

    Coordinates are normalized detector outputs. No ground truth, visibility,
    evaluation status, ignore flag, or target-validity argument is accepted.
    """
    head_conf = np.asarray(head_confidence, dtype=np.float32)
    tail_conf = np.asarray(tail_confidence, dtype=np.float32)
    if head_conf.ndim != 1 or tail_conf.shape != head_conf.shape:
        raise ValueError("head_confidence and tail_confidence must be equal-length vectors")
    length = len(head_conf)
    head_xy = _as_xy(head_xy_norm, length, "head_xy_norm")
    tail_xy = _as_xy(tail_xy_norm, length, "tail_xy_norm")
    head_valid = _prediction_valid(head_xy, head_conf, head_prediction_exists)
    tail_valid = _prediction_valid(tail_xy, tail_conf, tail_prediction_exists)

    filled_head = _fill_missing_xy(head_xy, head_valid)
    filled_tail = _fill_missing_xy(tail_xy, tail_valid)
    clean_head_conf = np.where(head_valid, head_conf, 0.0).astype(np.float32)
    clean_tail_conf = np.where(tail_valid, tail_conf, 0.0).astype(np.float32)
    seq8 = np.column_stack(
        [
            filled_head,
            filled_tail,
            clean_head_conf,
            clean_tail_conf,
            head_valid.astype(np.float32),
            tail_valid.astype(np.float32),
        ]
    ).astype(np.float32)

    scale = np.asarray([image_width, image_height], dtype=np.float32)
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("image_width and image_height must be finite and positive")
    diagonal = float(np.hypot(image_width, image_height))
    head_px = filled_head * scale
    tail_px = filled_tail * scale
    bat_vector_norm = filled_tail - filled_head
    bat_vector_px = tail_px - head_px
    length_px = np.linalg.norm(bat_vector_px, axis=1).astype(np.float32)
    length_norm = length_px / diagonal
    unit = np.zeros_like(bat_vector_px, dtype=np.float32)
    valid_length = length_px > 1e-6
    unit[valid_length] = bat_vector_px[valid_length] / length_px[valid_length, None]
    direction_change = np.zeros(length, dtype=np.float32)
    if length > 1:
        dot = np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1.0, 1.0)
        valid_pairs = valid_length[1:] & valid_length[:-1]
        direction_change[1:] = np.where(valid_pairs, 1.0 - dot, 0.0)
    motion = np.column_stack(
        [
            length_norm,
            _prepend_zero_diff(length_norm),
            unit[:, 0],
            unit[:, 1],
            direction_change,
            _speed(head_px, diagonal),
            _speed(tail_px, diagonal),
            _acceleration(head_px, diagonal),
            _acceleration(tail_px, diagonal),
        ]
    ).astype(np.float32)
    window_time = np.linspace(-1.0, 1.0, length, dtype=np.float32)[:, None]
    features = np.concatenate(
        [seq8, bat_vector_norm, motion, window_time, _view_one_hot(view_label, length)], axis=1
    ).astype(np.float32)
    schema = CLEAN_NO_FLOW_SCHEMA

    if flow_features is not None:
        flow = np.asarray(flow_features, dtype=np.float32)
        if flow.shape != (length, len(FLOW_FEATURES)):
            raise ValueError(f"flow_features must have shape ({length}, {len(FLOW_FEATURES)}), got {flow.shape}")
        if not np.isfinite(flow).all():
            raise ValueError("flow_features must be finite and pre-normalized")
        features = np.concatenate([features, flow], axis=1).astype(np.float32)
        schema = CLEAN_WITH_FLOW_SCHEMA

    if features.shape[1] != len(schema):
        raise AssertionError(f"feature width {features.shape[1]} does not match schema {len(schema)}")
    return ObservationBatch(
        features=features,
        feature_schema=schema,
        detector_observations=seq8,
        head_valid=head_valid,
        tail_valid=tail_valid,
        head_missing=~head_valid,
        tail_missing=~tail_valid,
    )
