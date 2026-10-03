"""Fail-fast alignment checks for selected detector rows and RAFT feature CSVs."""

import csv
import hashlib
from pathlib import Path

import numpy as np

from .adapters import FLOW_SOURCE_COLUMNS, parse_float


PREDICTION_FIELDS = ("pred_x", "pred_y", "pred_conf")
BBOX_FIELDS = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(row):
    swing = str(row.get("swing_id") or "").strip()
    frame = str(row.get("image_stem") or row.get("frame") or "").strip()
    if not swing or not frame:
        raise ValueError("alignment row is missing swing_id or frame/image_stem")
    return swing, frame


def _same_number(left, right):
    a = parse_float(left)
    b = parse_float(right)
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isfinite(a) and np.isfinite(b) and a == b)


def _read_selected(path):
    frames = {}
    with open(path, newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            key = _key(row)
            keypoint = str(row.get("keypoint") or "").strip().lower()
            if keypoint not in {"head", "tail"}:
                continue
            frame = frames.setdefault(key, {"swing_id": key[0], "image_stem": key[1], "keypoints": {}})
            if keypoint in frame["keypoints"]:
                raise ValueError(f"duplicate selected detector key: {key + (keypoint,)}")
            frame["keypoints"][keypoint] = row
    return frames


def _read_flow(path):
    frames = {}
    with open(path, newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        missing_columns = [name for name in FLOW_SOURCE_COLUMNS if name not in (reader.fieldnames or [])]
        if missing_columns:
            raise ValueError(f"flow CSV missing required columns: {missing_columns}")
        for row in reader:
            key = _key(row)
            if key in frames:
                raise ValueError(f"duplicate flow frame key: {key}")
            frames[key] = row
    return frames


def verify_selected_flow_alignment(selected_csv, flow_csv, *, expected_frames=None, expected_swings=None):
    """Verify exact frame population and detector values before flow dataset creation."""
    selected_csv = Path(selected_csv)
    flow_csv = Path(flow_csv)
    selected = _read_selected(selected_csv)
    flow = _read_flow(flow_csv)
    selected_keys = set(selected)
    flow_keys = set(flow)
    missing = sorted(selected_keys - flow_keys)
    extra = sorted(flow_keys - selected_keys)
    if missing or extra:
        raise ValueError(f"selected/flow key mismatch: missing={missing[:5]} extra={extra[:5]}")

    for key in sorted(selected_keys):
        selected_frame = selected[key]
        flow_row = flow[key]
        for keypoint in ("head", "tail"):
            row = selected_frame["keypoints"].get(keypoint)
            expected = row or {name: "" for name in PREDICTION_FIELDS}
            for field in PREDICTION_FIELDS:
                flow_field = f"{keypoint}_{field}"
                if not _same_number(expected.get(field), flow_row.get(flow_field)):
                    raise ValueError(f"detector mismatch at {key} {keypoint}: {field}")

        bbox_rows = list(selected_frame["keypoints"].values())
        for field in BBOX_FIELDS:
            values = [row.get(field) for row in bbox_rows]
            nonmissing = [value for value in values if np.isfinite(parse_float(value))]
            if nonmissing and any(not _same_number(nonmissing[0], value) for value in nonmissing[1:]):
                raise ValueError(f"selected bbox is inconsistent within frame {key}: {field}")
            expected = nonmissing[0] if nonmissing else ""
            if not _same_number(expected, flow_row.get(field)):
                raise ValueError(f"detector bbox mismatch at {key}: {field}")

    swings = sorted({key[0] for key in selected_keys})
    if expected_frames is not None and len(selected_keys) != int(expected_frames):
        raise ValueError(f"frame count mismatch: {len(selected_keys)} != {expected_frames}")
    if expected_swings is not None and len(swings) != int(expected_swings):
        raise ValueError(f"swing count mismatch: {len(swings)} != {expected_swings}")
    return {
        "selected_csv": str(selected_csv),
        "selected_csv_sha256": file_sha256(selected_csv),
        "flow_csv": str(flow_csv),
        "flow_csv_sha256": file_sha256(flow_csv),
        "frame_count": len(selected_keys),
        "swing_count": len(swings),
        "swing_ids": swings,
        "flow_columns": list(FLOW_SOURCE_COLUMNS),
        "selected_detector_observations": True,
        "alignment": "exact_keys_predictions_confidence_and_bbox",
    }


def verify_disjoint_splits(reports):
    named = list(reports.items())
    for index, (left_name, left) in enumerate(named):
        left_ids = set(left["swing_ids"])
        for right_name, right in named[index + 1 :]:
            overlap = sorted(left_ids & set(right["swing_ids"]))
            if overlap:
                raise ValueError(f"split overlap {left_name}/{right_name}: {overlap}")
    return True
