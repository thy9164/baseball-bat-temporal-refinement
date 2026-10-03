import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import (  # noqa: E402
    clean_keypoint_columns,
    clip01,
    collect_swing_ids,
    corrupt_clean_window,
    infer_downsample_step,
    list_csv_files,
    load_or_create_splits,
    normalize_xy,
    parse_float,
    row_frame_idx,
    row_swing_id,
    row_view_id,
    split_lookup,
    tail_aligned_starts,
    usable_row,
)
from clean_input.adapters import prediction_exists_from_detector  # noqa: E402
from clean_input.observation import build_observation_features  # noqa: E402
from clean_input.schema import CLEAN_NO_FLOW_SCHEMA  # noqa: E402
from src.geometry_features import add_v8_motion_features  # noqa: E402
from src.view_features import (  # noqa: E402
    ID_TO_VIEW,
    VIEW_LABELS,
    normalize_batter_hand,
    normalize_view_label,
)


VIS_TO_ID = {
    "fully_visible": 0,
    "partially_occluded": 1,
    "fully_occluded": 2,
    "ignore": 3,
}
ID_TO_VIS = {value: key for key, value in VIS_TO_ID.items()}
STATUS_TO_ID = {
    "matched": 0,
    "missing_prediction": 1,
    "ignore_with_prediction": 2,
    "ignore_no_prediction": 3,
}
ID_TO_STATUS = {value: key for key, value in STATUS_TO_ID.items()}
PREDICTION_STATUSES = {"matched", "ignore_with_prediction"}
MISSING_STATUSES = {"missing_prediction", "ignore_no_prediction"}


def load_visibility_csv(path):
    if not path:
        return {}
    lookup = {}
    with open(path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            swing_id = row.get("swing_id")
            frame = row.get("frame") or row.get("image_stem") or ""
            digits = "".join(ch for ch in frame if ch.isdigit())
            if not swing_id or not digits:
                continue
            lookup[(swing_id, int(digits[-6:]))] = row.get("tail_visibility") or "fully_visible"
    return lookup


def load_visibility_from_keypoint_eval(path):
    if not path:
        return {}
    lookup = {}
    with open(path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if row.get("keypoint") != "tail":
                continue
            swing_id = row.get("swing_id")
            visibility = row.get("visibility") or "fully_visible"
            if not swing_id or visibility not in VIS_TO_ID:
                continue
            frame_idx = eval_frame_idx(row)
            lookup[(swing_id, frame_idx)] = visibility
    return lookup


def eval_frame_idx(row):
    frame = row.get("frame") or row.get("image_stem") or ""
    digits = "".join(ch for ch in frame if ch.isdigit())
    if digits:
        return int(digits[-6:])
    return 0


def tail_status_from_visibility(label, obs_valid):
    if label == "ignore":
        return "ignore_with_prediction" if obs_valid > 0.0 else "ignore_no_prediction"
    return "matched" if obs_valid > 0.0 else "missing_prediction"


class RealYoloWindowSamplerV52:
    def __init__(self, csv_path, window_size, stride, image_width, image_height):
        self.window_size = window_size
        self.stride = stride
        self.image_width = image_width
        self.image_height = image_height
        self.windows = []
        self.load(csv_path)

    def load(self, csv_path):
        frames = defaultdict(lambda: defaultdict(dict))
        print(f"Loading real YOLO error windows from {csv_path}...")
        with open(csv_path, newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                keypoint = row.get("keypoint")
                if keypoint not in {"head", "tail"}:
                    continue
                swing_id = row.get("swing_id") or "unknown"
                image_stem = row.get("image_stem") or row.get("frame")
                if not image_stem:
                    continue
                frames[swing_id][image_stem][keypoint] = row

        for swing_id, by_frame in frames.items():
            print(f"Processing swing_id={swing_id} with {len(by_frame)} frames...")
            seq = []
            for _image_stem, keypoints in sorted(by_frame.items(), key=lambda item: eval_frame_idx(next(iter(item[1].values())))):
                if "head" not in keypoints or "tail" not in keypoints:
                    continue
                frame = []
                statuses = []
                visibilities = []
                for keypoint in ("head", "tail"):
                    row = keypoints[keypoint]
                    status = row.get("status") or "unknown"
                    dx = parse_float(row.get("dx"))
                    dy = parse_float(row.get("dy"))
                    conf = parse_float(row.get("pred_conf"), 0.0)
                    valid = 1.0 if prediction_exists_from_detector(row.get("pred_x"), row.get("pred_y"), conf) and np.isfinite(dx) and np.isfinite(dy) else 0.0
                    if valid:
                        frame.extend([dx / self.image_width, dy / self.image_height, conf, valid])
                    else:
                        frame.extend([np.nan, np.nan, 0.0, 0.0])
                    statuses.append(status)
                    visibilities.append(row.get("visibility") or "fully_visible")
                seq.append({"frame": np.asarray(frame, dtype=np.float32), "statuses": statuses, "visibilities": visibilities})

            starts = tail_aligned_starts(len(seq), self.window_size, self.stride, include_first_window=True)
            for start in starts:
                window = seq[start:start + self.window_size]
                self.windows.append(
                    {
                        "errors": np.stack([item["frame"] for item in window]).astype(np.float32),
                        "head_status": [item["statuses"][0] for item in window],
                        "tail_status": [item["statuses"][1] for item in window],
                        "head_visibility": [item["visibilities"][0] for item in window],
                        "tail_visibility": [item["visibilities"][1] for item in window],
                    }
                )

        if not self.windows:
            raise ValueError(f"No real YOLO error windows built from {csv_path}")

    def sample(self, rng):
        return self.windows[int(rng.integers(0, len(self.windows)))]

    def summary(self):
        head_status = defaultdict(int)
        tail_status = defaultdict(int)
        tail_visibility = defaultdict(int)
        valid = {"head": 0, "tail": 0}
        total = 0
        for window in self.windows:
            errors = window["errors"]
            total += len(errors)
            valid["head"] += int(np.sum(errors[:, 3] > 0.0))
            valid["tail"] += int(np.sum(errors[:, 7] > 0.0))
            for status in window["head_status"]:
                head_status[status] += 1
            for status in window["tail_status"]:
                tail_status[status] += 1
            for visibility in window["tail_visibility"]:
                tail_visibility[visibility] += 1
        return {
            "windows": len(self.windows),
            "frames": total,
            "valid_rate": {
                "head": valid["head"] / max(total, 1),
                "tail": valid["tail"] / max(total, 1),
            },
            "head_status": dict(head_status),
            "tail_status": dict(tail_status),
            "tail_visibility": dict(tail_visibility),
            "prediction_statuses": sorted(PREDICTION_STATUSES),
            "missing_statuses": sorted(MISSING_STATUSES),
            "status_ids": ID_TO_STATUS,
        }


class TailInpaintDatasetWriter:
    def __init__(self, args):
        self.args = args
        self.output_npz = Path(args.output_npz)
        self.shard_dir = Path(args.shard_dir) if args.shard_dir else self.output_npz.with_suffix("")
        self.use_shards = args.shard_size > 0
        self.buffers = defaultdict(self.new_buffer)
        self.counts = defaultdict(int)
        self.shard_ids = defaultdict(int)

    @staticmethod
    def new_buffer():
        return {
            "X": [],
            "y_tail": [],
            "y_head": [],
            "tail_weight": [],
            "tail_visibility": [],
            "tail_status": [],
            "tail_obs_valid": [],
            "swing_id": [],
            "view_id": [],
            "start_frame_idx": [],
            "end_frame_idx": [],
        }

    def add(self, split, x, y_tail, y_head, tail_weight, tail_visibility, tail_status, tail_obs_valid, meta):
        buf = self.buffers[split]
        buf["X"].append(x)
        buf["y_tail"].append(y_tail)
        buf["y_head"].append(y_head)
        buf["tail_weight"].append(tail_weight)
        buf["tail_visibility"].append(tail_visibility)
        buf["tail_status"].append(tail_status)
        buf["tail_obs_valid"].append(tail_obs_valid)
        for key in ["swing_id", "view_id", "start_frame_idx", "end_frame_idx"]:
            buf[key].append(meta[key])
        self.counts[split] += 1
        if self.use_shards and len(buf["X"]) >= self.args.shard_size:
            self.flush_split(split)

    def arrays_from_buffer(self, buf):
        return {
            "X": np.stack(buf["X"]).astype(np.float32),
            "y_tail": np.stack(buf["y_tail"]).astype(np.float32),
            "y_head": np.stack(buf["y_head"]).astype(np.float32),
            "tail_weight": np.stack(buf["tail_weight"]).astype(np.float32),
            "tail_visibility": np.stack(buf["tail_visibility"]).astype(np.int8),
            "tail_status": np.stack(buf["tail_status"]).astype(np.int8),
            "tail_obs_valid": np.stack(buf["tail_obs_valid"]).astype(np.float32),
            "swing_id": np.asarray(buf["swing_id"]),
            "view_id": np.asarray(buf["view_id"]),
            "start_frame_idx": np.asarray(buf["start_frame_idx"], dtype=np.int32),
            "end_frame_idx": np.asarray(buf["end_frame_idx"], dtype=np.int32),
        }

    def save_npz(self, path, arrays):
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.args.compress:
            np.savez_compressed(path, **arrays)
        else:
            np.savez(path, **arrays)

    def flush_split(self, split):
        buf = self.buffers[split]
        if not buf["X"]:
            return
        path = self.shard_dir / split / f"shard_{self.shard_ids[split]:05d}.npz"
        self.save_npz(path, self.arrays_from_buffer(buf))
        print(f"[shard] split={split} path={path} samples={len(buf['X'])}")
        self.shard_ids[split] += 1
        self.buffers[split] = self.new_buffer()

    def close(self):
        if self.use_shards:
            for split in list(self.buffers.keys()):
                self.flush_split(split)
            return
        arrays = {}
        for split, buf in self.buffers.items():
            if not buf["X"]:
                continue
            for key, value in self.arrays_from_buffer(buf).items():
                arrays[f"{key}_{split}"] = value
        self.save_npz(self.output_npz, arrays)
        print(f"[saved] path={self.output_npz}")


def gaussian_corrupt_clean_window(clean_window, rng, args):
    scale = np.asarray([args.image_width, args.image_height, args.image_width, args.image_height], dtype=np.float32)
    noise_px = rng.normal(0.0, args.noise_std_px, size=clean_window.shape).astype(np.float32)
    noisy = clip01(clean_window + noise_px / scale)
    seq8 = np.zeros((len(clean_window), 8), dtype=np.float32)
    seq8[:, :4] = noisy
    seq8[:, 4] = args.default_head_conf
    seq8[:, 5] = args.default_tail_conf
    seq8[:, 6] = 1.0
    seq8[:, 7] = 1.0
    return add_v8_motion_features(seq8, args.image_width, args.image_height)


def sample_block_length(rng, args):
    bucket = rng.choice(3, p=[args.short_block_prob, args.medium_block_prob, args.long_block_prob])
    if bucket == 0:
        lo, hi = args.short_block_min, args.short_block_max
    elif bucket == 1:
        lo, hi = args.medium_block_min, args.medium_block_max
    else:
        lo, hi = args.long_block_min, args.long_block_max
    return int(rng.integers(lo, hi + 1))


def sample_block_visibility(rng, args):
    return rng.choice(
        ["partially_occluded", "fully_occluded", "ignore"],
        p=[args.partial_block_prob, args.full_block_prob, args.ignore_block_prob],
    )


def synthetic_visibility(window_size, rng, args):
    labels = np.asarray(["fully_visible"] * window_size, dtype=object)
    if rng.random() >= args.occlusion_block_prob:
        return labels
    block_len = min(sample_block_length(rng, args), window_size)
    if args.occlusion_position_bias == "middle_late":
        weights = np.linspace(0.5, 1.5, window_size - block_len + 1)
        weights = weights / weights.sum()
        start = int(rng.choice(window_size - block_len + 1, p=weights))
    else:
        start = int(rng.integers(0, window_size - block_len + 1))
    labels[start:start + block_len] = sample_block_visibility(rng, args)
    if args.second_block_prob > 0.0 and rng.random() < args.second_block_prob:
        block_len = min(sample_block_length(rng, args), window_size)
        start = int(rng.integers(0, window_size - block_len + 1))
        labels[start:start + block_len] = sample_block_visibility(rng, args)
    return labels


def visibility_for_window(swing_id, frame_indices, visibility_lookup, rng, args):
    labels = []
    found = 0
    for frame_idx in frame_indices:
        label = visibility_lookup.get((swing_id, frame_idx))
        if label in VIS_TO_ID:
            found += 1
        labels.append(label if label in VIS_TO_ID else None)
    if found >= len(frame_indices) * args.real_visibility_min_coverage:
        return np.asarray([label or "fully_visible" for label in labels], dtype=object)
    return synthetic_visibility(len(frame_indices), rng, args)


def fill_tail_after_mask(seq8):
    out = seq8.copy()
    valid = out[:, 7] > 0.0
    missing = ~valid
    if not np.any(missing):
        return out
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        out[:, 2] = 0.0
        out[:, 3] = 0.0
        return out
    for idx in np.flatnonzero(missing):
        nearest = valid_indices[np.argmin(np.abs(valid_indices - idx))]
        out[idx, 2] = out[nearest, 2]
        out[idx, 3] = out[nearest, 3]
    return out


def add_tail_noise_px(seq8, idx, std_px, rng, args):
    if std_px <= 0.0:
        return
    scale = np.asarray([args.image_width, args.image_height], dtype=np.float32)
    noise = rng.normal(0.0, std_px, size=2).astype(np.float32) / scale
    seq8[idx, 2:4] = clip01(seq8[idx, 2:4] + noise)


def mask_tail_observation(seq8, tail_obs_valid, idx):
    seq8[idx, 5] = 0.0
    seq8[idx, 7] = 0.0
    tail_obs_valid[idx] = 0.0


def apply_tail_visibility(seq10, visibility_labels, rng, args):
    seq8 = seq10[:, :8].copy()
    tail_weight = np.ones(len(seq8), dtype=np.float32)
    tail_obs_valid = seq8[:, 7].copy().astype(np.float32)
    visibility_ids = np.asarray([VIS_TO_ID[label] for label in visibility_labels], dtype=np.int8)

    for idx, label in enumerate(visibility_labels):
        if label == "fully_visible":
            tail_weight[idx] = args.weight_fully_visible
            continue
        if label == "partially_occluded":
            tail_weight[idx] = args.weight_partially_occluded
            if seq8[idx, 7] <= 0.0 or rng.random() < args.partial_mask_prob:
                mask_tail_observation(seq8, tail_obs_valid, idx)
            else:
                seq8[idx, 5] = min(seq8[idx, 5], args.partial_tail_conf)
                add_tail_noise_px(seq8, idx, args.partial_tail_extra_noise_px, rng, args)
            continue
        if label == "fully_occluded":
            tail_weight[idx] = args.weight_fully_occluded
            if seq8[idx, 7] <= 0.0 or rng.random() < args.fully_occluded_mask_prob:
                mask_tail_observation(seq8, tail_obs_valid, idx)
            else:
                seq8[idx, 5] = min(seq8[idx, 5], args.fully_occluded_tail_conf)
                add_tail_noise_px(seq8, idx, args.fully_occluded_tail_extra_noise_px, rng, args)
            continue
        if label == "ignore":
            tail_weight[idx] = args.weight_ignore
            if seq8[idx, 7] <= 0.0 or rng.random() >= args.ignore_keep_observation_prob:
                mask_tail_observation(seq8, tail_obs_valid, idx)
            else:
                seq8[idx, 5] = min(seq8[idx, 5], args.ignore_tail_conf)
                add_tail_noise_px(seq8, idx, args.ignore_tail_extra_noise_px, rng, args)

    status_labels = [tail_status_from_visibility(label, tail_obs_valid[idx]) for idx, label in enumerate(visibility_labels)]
    status_ids = np.asarray([STATUS_TO_ID[label] for label in status_labels], dtype=np.int8)
    return seq8, tail_weight, visibility_ids, status_ids, tail_obs_valid


def build_clean_features_from_pseudo_detector(seq8, *, image_width, image_height, view_label):
    """Build model X from the already-observed pseudo-detector sequence only.

    Latent synthetic visibility is intentionally absent from this interface. It
    may create corruption before this boundary and remain available as
    supervision, but cannot directly affect feature construction afterwards.
    """
    seq8 = np.asarray(seq8, dtype=np.float32)
    if seq8.ndim != 2 or seq8.shape[1] != 8:
        raise ValueError(f"expected pseudo-detector sequence shaped [T, 8], got {seq8.shape}")
    return build_observation_features(
        head_xy_norm=seq8[:, 0:2],
        tail_xy_norm=seq8[:, 2:4],
        head_confidence=seq8[:, 4],
        tail_confidence=seq8[:, 5],
        head_prediction_exists=seq8[:, 6] > 0.0,
        tail_prediction_exists=seq8[:, 7] > 0.0,
        image_width=image_width,
        image_height=image_height,
        view_label=view_label,
    )


def process_group(rows, split, writer, args, rng, cols, sampler, visibility_lookup):
    rows = sorted(rows, key=row_frame_idx)
    times = [parse_float(row.get("time_sec")) for row in rows]
    step = infer_downsample_step(times, args.target_fps, args.source_fps)
    rows = rows[::step]
    if len(rows) < args.window_size:
        return 0

    clean = []
    frame_indices = []
    camera_names = []
    for row in rows:
        # Keep only the already-canonical right-handed copy:
        # original R -> mirror=none, original L -> mirror=horizontal.
        if normalize_batter_hand(row.get("batter_hand")) != "R":
            continue
        if not usable_row(row, cols):
            continue
        width = parse_float(row.get("image_width"), args.image_width)
        height = parse_float(row.get("image_height"), args.image_height)
        head_x = parse_float(row.get(cols[0]))
        head_y = parse_float(row.get(cols[1]))
        tail_x = parse_float(row.get(cols[2]))
        tail_y = parse_float(row.get(cols[3]))
        clean.append([
            *normalize_xy(head_x, head_y, width, height),
            *normalize_xy(tail_x, tail_y, width, height),
        ])
        frame_indices.append(row_frame_idx(row))
        camera_names.append(row.get("camera_name") or row.get("view_id") or row.get("camera_id") or "random")

    if len(clean) < args.window_size:
        return 0

    clean = np.asarray(clean, dtype=np.float32)
    made = 0
    for start in tail_aligned_starts(len(clean), args.window_size, args.stride, args.include_first_window):
        end = start + args.window_size
        clean_window = clean[start:end]
        window_cameras = camera_names[start:end]
        camera_name = max(set(window_cameras), key=window_cameras.count) if window_cameras else "random"
        view_label = normalize_view_label(camera_name)

        if sampler is not None:
            seq10, _head_status, _tail_status = corrupt_clean_window(clean_window, sampler, rng)
        else:
            seq10 = gaussian_corrupt_clean_window(clean_window, rng, args)

        window_frame_indices = frame_indices[start:end]
        visibility_labels = visibility_for_window(rows[0]["_swing_id"], window_frame_indices, visibility_lookup, rng, args)
        seq8, tail_weight, visibility_ids, status_ids, tail_obs_valid = apply_tail_visibility(seq10, visibility_labels, rng, args)
        observation = build_clean_features_from_pseudo_detector(
            seq8,
            image_width=args.image_width,
            image_height=args.image_height,
            view_label=view_label,
        )
        x = observation.features
        tail_obs_valid = observation.tail_valid.astype(np.float32)
        meta = {
            "swing_id": rows[0]["_swing_id"],
            "view_id": rows[0]["_view_id"],
            "start_frame_idx": window_frame_indices[0],
            "end_frame_idx": window_frame_indices[-1],
        }
        writer.add(
            split,
            x,
            clean_window[:, 2:4].astype(np.float32),
            clean_window[:, 0:2].astype(np.float32),
            tail_weight,
            visibility_ids,
            status_ids,
            tail_obs_valid,
            meta,
        )
        made += 1
    return made


def process_csv(csv_path, split_map, writer, args, rng, sampler, visibility_lookup):
    groups = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        cols = clean_keypoint_columns(reader.fieldnames)
        for row in reader:
            swing_id = row_swing_id(row)
            if not swing_id:
                continue
            split = split_map.get(swing_id)
            if split is None:
                continue
            row["_swing_id"] = swing_id
            row["_view_id"] = row_view_id(row)
            groups[(split, swing_id, row["_view_id"])].append(row)

    made = 0
    for (split, _swing_id, _view_id), rows in groups.items():
        made += process_group(rows, split, writer, args, rng, cols, sampler, visibility_lookup)
    return made, len(groups)


def normalize_probabilities(args):
    total = args.short_block_prob + args.medium_block_prob + args.long_block_prob
    args.short_block_prob /= total
    args.medium_block_prob /= total
    args.long_block_prob /= total
    total = args.partial_block_prob + args.full_block_prob + args.ignore_block_prob
    args.partial_block_prob /= total
    args.full_block_prob /= total
    args.ignore_block_prob /= total


def parse_args():
    parser = argparse.ArgumentParser(description="Build v8 prior-guided correction + trust tail trajectory dataset.")
    parser.add_argument("--input_csv", required=True, help="CSV file or directory containing clean C3D 2D coordinates.")
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--shard_dir", default=None)
    parser.add_argument("--shard_size", type=int, default=0)
    parser.add_argument("--splits_json", required=True)
    parser.add_argument("--reuse_splits", action="store_true")
    parser.add_argument("--yolo_error_csv", default=None, help="Optional keypoint_eval_details.csv for visible-observation YOLO-like noise.")
    parser.add_argument("--visibility_csv", default=None, help="Optional all_keypoints.csv with tail_visibility labels.")
    parser.add_argument("--image_width", type=float, default=1920.0)
    parser.add_argument("--image_height", type=float, default=1080.0)
    parser.add_argument("--source_fps", type=float, default=360.0)
    parser.add_argument("--target_fps", type=float, default=60.0)
    parser.add_argument("--window_size", type=int, default=31)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--include_first_window", action="store_true")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--noise_std_px", type=float, default=6.0)
    parser.add_argument("--default_head_conf", type=float, default=0.95)
    parser.add_argument("--default_tail_conf", type=float, default=0.90)
    parser.add_argument("--occlusion_block_prob", type=float, default=0.8)
    parser.add_argument("--second_block_prob", type=float, default=0.1)
    parser.add_argument("--short_block_prob", type=float, default=0.75)
    parser.add_argument("--medium_block_prob", type=float, default=0.20)
    parser.add_argument("--long_block_prob", type=float, default=0.05)
    parser.add_argument("--short_block_min", type=int, default=3)
    parser.add_argument("--short_block_max", type=int, default=10)
    parser.add_argument("--medium_block_min", type=int, default=10)
    parser.add_argument("--medium_block_max", type=int, default=20)
    parser.add_argument("--long_block_min", type=int, default=20)
    parser.add_argument("--long_block_max", type=int, default=30)
    parser.add_argument("--partial_block_prob", type=float, default=0.35)
    parser.add_argument("--full_block_prob", type=float, default=0.45)
    parser.add_argument("--ignore_block_prob", type=float, default=0.20)
    parser.add_argument("--partial_mask_prob", type=float, default=0.5)
    parser.add_argument("--partial_tail_conf", type=float, default=0.4)
    parser.add_argument("--partial_tail_extra_noise_px", type=float, default=4.0)
    parser.add_argument("--fully_occluded_mask_prob", type=float, default=0.35)
    parser.add_argument("--fully_occluded_tail_conf", type=float, default=0.25)
    parser.add_argument("--fully_occluded_tail_extra_noise_px", type=float, default=12.0)
    parser.add_argument("--ignore_keep_observation_prob", type=float, default=0.05)
    parser.add_argument("--ignore_tail_conf", type=float, default=0.10)
    parser.add_argument("--ignore_tail_extra_noise_px", type=float, default=24.0)
    parser.add_argument("--occlusion_position_bias", choices=["uniform", "middle_late"], default="middle_late")
    parser.add_argument("--real_visibility_min_coverage", type=float, default=0.8)
    parser.add_argument("--weight_fully_visible", type=float, default=0.75)
    parser.add_argument("--weight_partially_occluded", type=float, default=2.0)
    parser.add_argument("--weight_fully_occluded", type=float, default=3.0)
    parser.add_argument("--weight_ignore", type=float, default=4.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.window_size < args.long_block_max:
        print(f"[warn] window_size={args.window_size} is smaller than long_block_max={args.long_block_max}.")
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    normalize_probabilities(args)

    csv_files = list_csv_files(args.input_csv)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found: {args.input_csv}")
    print(f"Found {len(csv_files)} CSV files. Collecting swing IDs and creating splits...")
    splits = load_or_create_splits(args, csv_files)
    split_map = split_lookup(splits)
    rng = np.random.default_rng(args.seed)
    print(f"Using splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(
        "dataset_config "
        f"window_size={args.window_size} stride={args.stride} target_fps={args.target_fps:g} "
        f"block_probs=short:{args.short_block_prob:g},medium:{args.medium_block_prob:g},long:{args.long_block_prob:g}"
    )
    print(
        "visibility_observation_config "
        f"partial_mask_prob={args.partial_mask_prob:g} "
        f"fully_occluded_mask_prob={args.fully_occluded_mask_prob:g} "
        f"ignore_keep_observation_prob={args.ignore_keep_observation_prob:g}"
    )
    print(
        "view_hand_config "
        f"views={','.join(VIEW_LABELS)} use_batter_hand=R skip_batter_hand=L "
        "canonical_source='R none or L horizontal'"
    )

    sampler = None
    if args.yolo_error_csv:
        sampler = RealYoloWindowSamplerV52(
            args.yolo_error_csv,
            window_size=args.window_size,
            stride=max(1, args.stride),
            image_width=args.image_width,
            image_height=args.image_height,
        )
        print("[real_yolo_window_sampler]")
        print(json.dumps(sampler.summary(), indent=2))
    visibility_lookup = load_visibility_csv(args.visibility_csv)
    visibility_source = args.visibility_csv
    if not visibility_lookup and args.yolo_error_csv:
        visibility_lookup = load_visibility_from_keypoint_eval(args.yolo_error_csv)
        visibility_source = args.yolo_error_csv
    if visibility_lookup:
        print(f"Loaded visibility labels: {len(visibility_lookup)} frame labels from {visibility_source}")

    writer = TailInpaintDatasetWriter(args)
    total_samples = 0
    total_groups = 0
    for idx, csv_path in enumerate(csv_files, start=1):
        samples, groups = process_csv(csv_path, split_map, writer, args, rng, sampler, visibility_lookup)
        total_samples += samples
        total_groups += groups
        print(f"[{idx}/{len(csv_files)}] {csv_path} groups={groups} samples={samples}")
    writer.close()

    summary = {
        "csv_files": len(csv_files),
        "groups": total_groups,
        "samples": total_samples,
        "counts": dict(writer.counts),
        "window_size": args.window_size,
        "stride": args.stride,
        "target_fps": args.target_fps,
        "input_csv": args.input_csv,
        "seed": args.seed,
        "splits_json": args.splits_json,
        "split_counts": {key: len(value) for key, value in splits.items()},
        "yolo_error_csv": args.yolo_error_csv,
        "visibility_csv": args.visibility_csv,
        "input_dim": len(CLEAN_NO_FLOW_SCHEMA),
        "feature_schema": list(CLEAN_NO_FLOW_SCHEMA),
        "feature_policy": {
            "latent_visibility_in_model_input": False,
            "validity_source": "pseudo-detector prediction-exists flags plus finite x/y",
            "fill_policy": "nearest fill only where pseudo-detector observation is missing",
            "geometry_motion_source": "clean pseudo-detector observations after missing-value fill",
        },
        "output_dim": 10,
        "target": "prior_tail_delta_trust_yolo_error_bucket_quality",
        "version": "v8_geometry_prior_guided_yolo_residual_trust_pretrain",
        "canonical_batter_hand": "R",
        "row_filter": "keep only rows where batter_hand is R after mirror generation",
        "canonical_source": "original R mirror=none, original L mirror=horizontal",
        "view_labels": VIEW_LABELS,
        "view_ids": ID_TO_VIEW,
        "left_hand_horizontal_flip_to_right": False,
        "flipped_view_swap": None,
        "visibility_ids": ID_TO_VIS,
        "status_ids": ID_TO_STATUS,
        "ignore_statuses": {
            "ignore_with_prediction": "visibility=ignore with a YOLO prediction but no GT in real eval CSV",
            "ignore_no_prediction": "visibility=ignore without YOLO prediction and no GT in real eval CSV",
        },
        "block_distribution": {
            "short_3_10": args.short_block_prob,
            "medium_10_20": args.medium_block_prob,
            "long_20_30": args.long_block_prob,
        },
        "loss_weights": {
            "fully_visible": args.weight_fully_visible,
            "partially_occluded": args.weight_partially_occluded,
            "fully_occluded": args.weight_fully_occluded,
            "ignore": args.weight_ignore,
        },
        "visibility_observation_config": {
            "partial_mask_prob": args.partial_mask_prob,
            "partial_tail_conf": args.partial_tail_conf,
            "partial_tail_extra_noise_px": args.partial_tail_extra_noise_px,
            "fully_occluded_mask_prob": args.fully_occluded_mask_prob,
            "fully_occluded_tail_conf": args.fully_occluded_tail_conf,
            "fully_occluded_tail_extra_noise_px": args.fully_occluded_tail_extra_noise_px,
            "ignore_keep_observation_prob": args.ignore_keep_observation_prob,
            "ignore_tail_conf": args.ignore_tail_conf,
            "ignore_tail_extra_noise_px": args.ignore_tail_extra_noise_px,
        },
    }
    summary_path = Path(args.output_npz).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
