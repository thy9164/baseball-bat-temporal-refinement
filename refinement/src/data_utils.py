"""Canonical behavior-preserving data helpers extracted from src_v2.

The source definitions are retained verbatim so strict-clean code no longer needs
the full historical src_v2 tree. Algorithm changes belong in a separate review.
"""

import csv
import json
import random
from pathlib import Path

import numpy as np

def parse_float(value, default=np.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def parse_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default

def clean_keypoint_columns(fieldnames):
    fields = set(fieldnames or [])
    if {"head_u", "head_v", "tail_u", "tail_v"}.issubset(fields):
        return "head_u", "head_v", "tail_u", "tail_v"
    if {"head_x", "head_y", "tail_x", "tail_y"}.issubset(fields):
        return "head_x", "head_y", "tail_x", "tail_y"
    raise ValueError("CSV must contain head_u/head_v/tail_u/tail_v or head_x/head_y/tail_x/tail_y.")

def row_swing_id(row):
    return row.get("swing_id") or row.get("sequence_id") or row.get("source_c3d")

def row_view_id(row):
    view = row.get("view_id") or row.get("camera_id") or row.get("camera_name") or "view_0"
    mirror = row.get("mirror")
    if mirror and mirror != "none":
        return f"{view}__{mirror}"
    return view

def row_frame_idx(row):
    return parse_int(row.get("frame_idx") or row.get("frame") or row.get("image_stem"))

def list_csv_files(input_path):
    path = Path(input_path)
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    return [path]

def collect_swing_ids(csv_files):
    swing_ids = set()
    for csv_path in csv_files:
        print(f"Collecting swing IDs from {csv_path}...")
        with open(csv_path, newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                swing_id = row_swing_id(row)
                if swing_id:
                    swing_ids.add(swing_id)
    return sorted(swing_ids)

def make_splits(swing_ids, train_ratio, val_ratio, seed):
    swing_ids = list(swing_ids)
    rng = random.Random(seed)
    rng.shuffle(swing_ids)
    n = len(swing_ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    return {
        "train": swing_ids[:n_train],
        "val": swing_ids[n_train:n_train + n_val],
        "test": swing_ids[n_train + n_val:],
    }

def load_or_create_splits(args, csv_files):
    split_path = Path(args.splits_json)
    if args.reuse_splits and split_path.exists():
        with open(split_path, encoding="utf-8") as fp:
            return json.load(fp)
    splits = make_splits(collect_swing_ids(csv_files), args.train_ratio, args.val_ratio, args.seed)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as fp:
        json.dump(splits, fp, indent=2)
    print(f"Created new splits and saved to {split_path}")
    return splits

def split_lookup(splits):
    lookup = {}
    for split, swing_ids in splits.items():
        print(f"Split '{split}' has {len(swing_ids)} swing IDs.")
        for swing_id in swing_ids:
            lookup[swing_id] = split
    return lookup

def infer_downsample_step(times, target_fps, source_fps):
    if target_fps <= 0:
        return 1
    finite_times = np.asarray([t for t in times if np.isfinite(t)], dtype=np.float64)
    if len(finite_times) >= 3:
        diffs = np.diff(finite_times[: min(len(finite_times), 300)])
        diffs = diffs[diffs > 0]
        if len(diffs):
            inferred_fps = 1.0 / float(np.median(diffs))
            return max(1, int(round(inferred_fps / target_fps)))
    if source_fps > 0:
        return max(1, int(round(source_fps / target_fps)))
    return 1

def normalize_xy(x, y, image_width, image_height):
    return x / image_width, y / image_height

def clip01(values):
    return np.clip(values, 0.0, 1.0)

def fill_missing_xy(seq):
    out = seq.copy()
    for x_col, y_col, conf_col in ((0, 1, 4), (2, 3, 5)):
        valid = np.isfinite(out[:, x_col]) & np.isfinite(out[:, y_col]) & (out[:, conf_col] > 0.0)
        missing = ~valid
        if not np.any(missing):
            continue
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) == 0:
            out[missing, x_col] = 0.0
            out[missing, y_col] = 0.0
            continue
        for idx in np.flatnonzero(missing):
            nearest = valid_indices[np.argmin(np.abs(valid_indices - idx))]
            out[idx, x_col] = out[nearest, x_col]
            out[idx, y_col] = out[nearest, y_col]
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32)

def add_bat_vector_features(seq8):
    bat_dx = seq8[:, 2] - seq8[:, 0]
    bat_dy = seq8[:, 3] - seq8[:, 1]
    return np.column_stack([seq8, bat_dx, bat_dy]).astype(np.float32)

def tail_aligned_starts(length, window_size, stride, include_first_window=False):
    if length < window_size:
        return []
    max_start = length - window_size
    starts = list(range(max_start, -1, -stride))
    if include_first_window and starts[-1] != 0:
        starts.append(0)
    return starts

def usable_row(row, cols):
    values = [parse_float(row.get(col)) for col in cols]
    return all(np.isfinite(v) for v in values)

def corrupt_clean_window(clean_window, sampler, rng):
    sampled = sampler.sample(rng)
    errors = sampled["errors"]
    out = np.zeros((len(clean_window), 8), dtype=np.float32)
    out[:, :4] = np.nan

    for point_idx, (xy_cols, error_cols, conf_col, valid_col) in enumerate(
        [((0, 1), (0, 1), 2, 3), ((2, 3), (4, 5), 6, 7)]
    ):
        valid = errors[:, valid_col] > 0.0
        out[:, 4 + point_idx] = np.where(valid, errors[:, conf_col], 0.0)
        out[:, 6 + point_idx] = valid.astype(np.float32)
        noisy_xy = clean_window[:, xy_cols] + errors[:, error_cols]
        out[valid, xy_cols[0]] = clip01(noisy_xy[valid, 0])
        out[valid, xy_cols[1]] = clip01(noisy_xy[valid, 1])

    filled = fill_missing_xy(out[:, :6])
    full = np.zeros((len(clean_window), 8), dtype=np.float32)
    full[:, :6] = filled
    full[:, 6:8] = out[:, 6:8]
    return add_bat_vector_features(full), sampled["head_status"], sampled["tail_status"]
