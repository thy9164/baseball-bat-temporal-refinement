"""Deterministic YOLO candidate selection using inference-time inputs only.

This module intentionally has no ground-truth, visibility, status, error, or
evaluation-row dependencies. It preserves the scoring and local-outlier logic
from ``export_detector_evaluation_rows.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionConfig:
    width: float
    height: float
    bbox_conf_weight: float
    anchor_radius: float
    anchor_weight: float


def detection_point_pixels(
    detection: dict[str, object],
    keypoint: str,
    width: float,
    height: float,
) -> tuple[float, float]:
    x_norm, y_norm, _ = detection[keypoint]  # type: ignore[index]
    return float(x_norm) * width, float(y_norm) * height


def detection_bbox_center_pixels(
    detection: dict[str, object], width: float, height: float
) -> tuple[float, float]:
    bbox = detection["bbox"]  # type: ignore[index]
    xc, yc, _, _ = bbox
    return float(xc) * width, float(yc) * height


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def detection_length_pixels(
    detection: dict[str, object], width: float, height: float
) -> float:
    head = detection_point_pixels(detection, "head", width, height)
    tail = detection_point_pixels(detection, "tail", width, height)
    return distance(head, tail)


def detection_direction_degrees(
    detection: dict[str, object], width: float, height: float
) -> float | None:
    head = detection_point_pixels(detection, "head", width, height)
    tail = detection_point_pixels(detection, "tail", width, height)
    dx = head[0] - tail[0]
    dy = head[1] - tail[1]
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dy, dx))


def angle_delta_degrees(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


def score_detection(
    detection: dict[str, object],
    previous: dict[str, object] | None,
    frame_gap: int,
    width: float,
    height: float,
    bbox_conf_weight: float,
    anchor: tuple[float, float] | None,
    anchor_radius: float,
    anchor_weight: float,
) -> None:
    bbox_conf = float(detection["bbox_conf"])
    bbox_conf_score = -bbox_conf_weight * bbox_conf
    temporal_tail_jump = 0.0
    temporal_head_jump = 0.0
    length_jump = 0.0
    direction_change = 0.0
    bbox_center_jump = 0.0
    anchor_cost = 0.0

    if previous is not None:
        dt = max(1, frame_gap)
        temporal_tail_jump = distance(
            detection_point_pixels(detection, "tail", width, height),
            detection_point_pixels(previous, "tail", width, height),
        ) / dt
        temporal_head_jump = distance(
            detection_point_pixels(detection, "head", width, height),
            detection_point_pixels(previous, "head", width, height),
        ) / dt
        length_jump = abs(
            detection_length_pixels(detection, width, height)
            - detection_length_pixels(previous, width, height)
        )
        direction_change = angle_delta_degrees(
            detection_direction_degrees(detection, width, height),
            detection_direction_degrees(previous, width, height),
        )
        bbox_center_jump = distance(
            detection_bbox_center_pixels(detection, width, height),
            detection_bbox_center_pixels(previous, width, height),
        ) / dt

    if anchor is not None:
        anchor_dist = distance(
            detection_bbox_center_pixels(detection, width, height), anchor
        )
        anchor_cost = anchor_weight * max(0.0, anchor_dist - anchor_radius)

    score = (
        bbox_conf_score
        + temporal_tail_jump
        + temporal_head_jump
        + length_jump
        + direction_change
        + bbox_center_jump
        + anchor_cost
    )
    detection["score"] = score
    detection["bbox_conf_score"] = bbox_conf_score
    detection["temporal_tail_jump"] = temporal_tail_jump
    detection["temporal_head_jump"] = temporal_head_jump
    detection["length_jump"] = length_jump
    detection["direction_change"] = direction_change
    detection["bbox_center_jump"] = bbox_center_jump
    detection["anchor_cost"] = anchor_cost


def select_best_detection(
    detections: list[dict[str, object]],
    previous: dict[str, object] | None,
    frame_gap: int,
    width: float,
    height: float,
    bbox_conf_weight: float,
    anchor: tuple[float, float] | None,
    anchor_radius: float,
    anchor_weight: float,
) -> dict[str, object] | None:
    if not detections:
        return None

    valid_detections = []
    for detection in detections:
        head = detection.get("head")
        tail = detection.get("tail")
        head_conf = float(head[2]) if head else 0.0  # type: ignore[index]
        tail_conf = float(tail[2]) if tail else 0.0  # type: ignore[index]
        if head_conf >= 0.1 and tail_conf >= 0.1:
            valid_detections.append(detection)

    if not valid_detections:
        return None

    for detection in valid_detections:
        score_detection(
            detection=detection,
            previous=previous,
            frame_gap=frame_gap,
            width=width,
            height=height,
            bbox_conf_weight=bbox_conf_weight,
            anchor=anchor,
            anchor_radius=anchor_radius,
            anchor_weight=anchor_weight,
        )

    if len(valid_detections) == 1:
        return valid_detections[0]

    return min(
        valid_detections,
        key=lambda detection: (
            float(detection["score"]),
            int(detection["detection_rank"]),
        ),
    )


def compute_swing_anchors(
    records: list[dict[str, object]], width: float, height: float
) -> dict[str, tuple[float, float]]:
    centers_by_swing: dict[str, list[tuple[float, float]]] = {}
    for record in records:
        detections = record["detections"]  # type: ignore[assignment]
        if not detections:
            continue
        swing_id = str(record["swing_id"])
        center = detection_bbox_center_pixels(detections[0], width, height)  # type: ignore[index]
        centers_by_swing.setdefault(swing_id, []).append(center)

    anchors: dict[str, tuple[float, float]] = {}
    for swing_id, centers in centers_by_swing.items():
        median_x = median([center[0] for center in centers])
        median_y = median([center[1] for center in centers])
        if median_x is not None and median_y is not None:
            anchors[swing_id] = (median_x, median_y)
    return anchors


def median_point(
    points: list[tuple[float, float]],
) -> tuple[float, float] | None:
    if not points:
        return None
    mx = median([point[0] for point in points])
    my = median([point[1] for point in points])
    if mx is None or my is None:
        return None
    return mx, my


def check_local_outlier(
    selected_items: list[dict[str, object]],
    idx: int,
    width: float,
    height: float,
    window: int = 5,
    min_neighbors: int = 3,
    min_length_ratio: float = 0.35,
    max_length_ratio: float = 2.2,
    center_base_thres: float = 300.0,
    center_spread_weight: float = 2.5,
) -> tuple[bool, str]:
    item = selected_items[idx]
    curr_pred = item.get("selected_pred")
    if curr_pred is None:
        return False, ""

    swing_id = item["swing_id"]
    neighbors: list[dict[str, object]] = []
    for j in range(max(0, idx - window), min(len(selected_items), idx + window + 1)):
        if j == idx:
            continue
        other = selected_items[j]
        if other["swing_id"] != swing_id:
            continue
        other_pred = other.get("selected_pred")
        if other_pred is not None:
            neighbors.append(other_pred)  # type: ignore[arg-type]

    if len(neighbors) < min_neighbors:
        return False, "not_enough_neighbors"

    curr_length = detection_length_pixels(curr_pred, width, height)  # type: ignore[arg-type]
    neighbor_lengths = [
        detection_length_pixels(pred, width, height) for pred in neighbors
    ]
    med_length = median(neighbor_lengths)
    length_outlier = False
    length_ratio = 1.0
    if med_length is not None and med_length > 1:
        length_ratio = curr_length / med_length
        length_outlier = (
            length_ratio < min_length_ratio or length_ratio > max_length_ratio
        )

    curr_center = detection_bbox_center_pixels(curr_pred, width, height)  # type: ignore[arg-type]
    neighbor_centers = [
        detection_bbox_center_pixels(pred, width, height) for pred in neighbors
    ]
    med_center = median_point(neighbor_centers)
    center_outlier = False
    center_dist = 0.0
    center_threshold = center_base_thres
    if med_center is not None:
        center_dist = distance(curr_center, med_center)
        spreads = [distance(center, med_center) for center in neighbor_centers]
        center_spread = median(spreads) or 0.0
        center_threshold = max(center_base_thres, center_spread_weight * center_spread)
        center_outlier = center_dist > center_threshold

    if length_outlier and center_outlier:
        return True, (
            "length_and_center_outlier:"
            f"length_ratio={length_ratio:.3f},"
            f"center_dist={center_dist:.1f},"
            f"center_threshold={center_threshold:.1f}"
        )
    if length_outlier:
        return True, f"length_outlier:length_ratio={length_ratio:.3f}"
    if center_outlier:
        return True, (
            "center_outlier:"
            f"center_dist={center_dist:.1f},"
            f"center_threshold={center_threshold:.1f}"
        )
    return False, ""


def candidate_record_sort_key(
    record: dict[str, object],
) -> tuple[str, int, str, str]:
    frame_no = record.get("frame_no")
    return (
        str(record["swing_id"]),
        int(frame_no) if frame_no is not None else 10**9,
        str(record["frame_name"]),
        str(record["prediction_stem"]),
    )


def select_sequence_candidates(
    records: list[dict[str, object]], config: SelectionConfig
) -> list[dict[str, object]]:
    """Select and screen candidates without accepting annotation fields."""
    required_record_fields = {
        "prediction_stem",
        "image_stem",
        "swing_id",
        "frame_no",
        "frame_name",
        "detections",
    }
    allowed_detection_fields = {
        "bbox_conf",
        "bbox",
        "head",
        "tail",
        "detection_rank",
        "score",
        "bbox_conf_score",
        "temporal_tail_jump",
        "temporal_head_jump",
        "length_jump",
        "direction_change",
        "bbox_center_jump",
        "anchor_cost",
        "is_outlier",
        "outlier_reason",
    }
    for record in records:
        missing = required_record_fields - set(record)
        unexpected = set(record) - required_record_fields
        if missing or unexpected:
            raise ValueError(
                "Candidate selection record violates the inference-only schema: "
                f"missing={sorted(missing)} unexpected/evaluation-only={sorted(unexpected)}"
            )
        for detection in record["detections"]:  # type: ignore[union-attr]
            unexpected_detection = set(detection) - allowed_detection_fields
            if unexpected_detection:
                raise ValueError(
                    "Detection contains unsupported/evaluation-only fields: "
                    f"{sorted(unexpected_detection)}"
                )

    anchors = compute_swing_anchors(records, config.width, config.height)
    previous_by_swing: dict[str, tuple[int | None, dict[str, object]]] = {}
    selected_items: list[dict[str, object]] = []

    for record in sorted(records, key=candidate_record_sort_key):
        swing_id = str(record["swing_id"])
        current_frame_no = record.get("frame_no")
        previous_state = previous_by_swing.get(swing_id)
        previous_frame_no = previous_state[0] if previous_state else None
        previous_detection = previous_state[1] if previous_state else None
        frame_gap = (
            current_frame_no - previous_frame_no
            if current_frame_no is not None and previous_frame_no is not None
            else 1
        )
        detections = record["detections"]  # type: ignore[assignment]
        selected_pred = select_best_detection(
            detections=detections,  # type: ignore[arg-type]
            previous=previous_detection,
            frame_gap=int(frame_gap),
            width=config.width,
            height=config.height,
            bbox_conf_weight=config.bbox_conf_weight,
            anchor=anchors.get(swing_id),
            anchor_radius=config.anchor_radius,
            anchor_weight=config.anchor_weight,
        )
        if selected_pred is not None:
            previous_by_swing[swing_id] = (current_frame_no, selected_pred)  # type: ignore[arg-type]
        selected_items.append({**record, "selected_pred": selected_pred})

    for idx, item in enumerate(selected_items):
        selected_pred = item["selected_pred"]
        if selected_pred is None:
            continue
        is_outlier, reason = check_local_outlier(
            selected_items, idx, config.width, config.height, window=5
        )
        selected_pred["is_outlier"] = is_outlier
        selected_pred["outlier_reason"] = reason
        if not is_outlier:
            continue
        if reason.startswith("center_outlier") or reason.startswith(
            "length_and_center_outlier"
        ):
            item["drop_bbox_reason"] = "center_outlier"
            item["outlier_detail"] = reason
        elif reason.startswith("length_outlier"):
            item["drop_keypoints_reason"] = "length_outlier"
            item["outlier_detail"] = reason

    return selected_items
