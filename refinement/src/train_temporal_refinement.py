import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from clean_input.adapters import FLOW_SOURCE_COLUMNS, prediction_exists_from_detector  # noqa: E402
from clean_input.observation import build_observation_features  # noqa: E402
from clean_input.flow_alignment import verify_disjoint_splits, verify_selected_flow_alignment  # noqa: E402
from clean_input.schema import CLEAN_NO_FLOW_SCHEMA, CLEAN_WITH_FLOW_SCHEMA, FLOW_FEATURES  # noqa: E402
from clean_input.checkpoint import (  # noqa: E402
    checkpoint_contract,
    checkpoint_sha256,
    strict_load_model_state,
    validate_checkpoint_contract,
    verified_strict_clean_parent,
)
from src.models import build_model  # noqa: E402
from src.real_data_pipeline import tail_aligned_starts  # noqa: E402
from src.train_synthetic_pretraining import (  # noqa: E402
    compute_loss,
    coordinate_scale,
    decode_prediction,
    finish_running,
    metric_text,
    update_running,
)
from src.view_features import (  # noqa: E402
    canonical_view_label,
    flip_seq8,
    flip_xy2,
    normalize_batter_hand,
    normalize_view_label,
    should_flip_to_right,
    swap_side_view,
)


FLOW_FEATURE_COLS = [
    "has_prev_flow",
    "head_valid",
    "tail_valid",
    "bbox_valid",
    "head_flow_x_mean",
    "head_flow_y_mean",
    "head_flow_mag_mean",
    "head_flow_mag_p95",
    "tail_flow_x_mean",
    "tail_flow_y_mean",
    "tail_flow_mag_mean",
    "tail_flow_mag_p95",
    "bbox_flow_x_mean",
    "bbox_flow_y_mean",
    "bbox_flow_mag_mean",
    "bbox_flow_mag_p95",
    "global_flow_x_mean",
    "global_flow_y_mean",
    "global_flow_mag_mean",
    "global_flow_mag_p95",
]

VIS_TO_ID = {"fully_visible": 0, "partially_occluded": 1, "fully_occluded": 2, "ignore": 3}

def get_flow_feature(row, col, args):
    value = parse_float(row.get(col), 0.0)
    clip = float(getattr(args, "flow_clip_px", 50.0))
    if col.endswith("_valid") or col == "has_prev_flow":
        return float(value > 0.0)
    if "mag" in col:
        return float(np.clip(value, 0.0, clip) / clip)
    return float(np.clip(value, -clip, clip) / clip)


def flow_vector_from_row(row, args):
    if not getattr(args, "use_flow_features", False):
        return np.zeros((0,), dtype=np.float32)
    return np.asarray([get_flow_feature(row, col, args) for col in FLOW_FEATURE_COLS], dtype=np.float32)


def flip_flow_features(flow_vec):
    if flow_vec.size == 0:
        return flow_vec
    out = flow_vec.copy()
    for idx, col in enumerate(FLOW_FEATURE_COLS):
        if "flow_x" in col:
            out[idx] *= -1.0
    return out


def parse_float(value, default=np.nan):
    if value in {"", None}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def frame_idx(row):
    frame = row.get("frame") or row.get("image_stem") or ""
    digits = "".join(ch for ch in frame if ch.isdigit())
    return int(digits[-6:]) if digits else 0


def normalize_batter_side(value):
    text = str(value or "").strip().lower()
    if text.startswith("l"):
        return "left"
    if text.startswith("r"):
        return "right"
    return "right"


def clip_view_to_canonical_label(view, batter_side):
    view_text = str(view or "").strip().lower()
    side = normalize_batter_side(batter_side)
    if view_text == "side":
        label = "1b_side" if side == "right" else "3b_side"
    elif view_text == "back":
        label = "3b_side" if side == "right" else "1b_side"
    else:
        label = normalize_view_label(view_text)
    return swap_side_view(label) if side == "left" else label


def load_clip_metadata(path):
    if not path:
        return {}
    import csv

    meta = {}
    with open(path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            swing_id = row.get("name") or row.get("swing_id") or row.get("clip") or row.get("clip_name")
            if not swing_id:
                continue
            side = normalize_batter_side(row.get("batter_side"))
            meta[swing_id] = {
                "batter_side": side,
                "batter_hand": "L" if side == "left" else "R",
                "flip_to_right": side == "left",
                "source_view": row.get("view") or "random",
                "view_label": clip_view_to_canonical_label(row.get("view"), side),
            }
    return meta


def combine_clip_metadata(*metas):
    combined = {}
    for meta in metas:
        combined.update(meta or {})
    return combined


def metadata_for_swing(args, swing_id):
    clip_meta = getattr(args, "clip_metadata", {}) or {}
    if swing_id in clip_meta:
        return clip_meta[swing_id]
    batter_hand = normalize_batter_hand(args.batter_hand)
    return {
        "batter_side": "left" if batter_hand == "L" else "right",
        "batter_hand": batter_hand,
        "flip_to_right": should_flip_to_right(batter_hand),
        "source_view": args.view_label,
        "view_label": canonical_view_label(args.view_label, batter_hand),
    }


def load_real_yolo_details(csv_path, image_width, image_height):
    frames = defaultdict(lambda: defaultdict(dict))
    with open(csv_path, newline="", encoding="utf-8") as fp:
        import csv

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

    sequences = {}
    for swing_id, by_frame in frames.items():
        items = []
        for image_stem, keypoints in by_frame.items():
            if "head" not in keypoints or "tail" not in keypoints:
                continue
            head = keypoints["head"]
            tail = keypoints["tail"]
            gt_head = np.asarray(
                [parse_float(head.get("gt_x")) / image_width, parse_float(head.get("gt_y")) / image_height],
                dtype=np.float32,
            )
            gt_tail = np.asarray(
                [parse_float(tail.get("gt_x")) / image_width, parse_float(tail.get("gt_y")) / image_height],
                dtype=np.float32,
            )
            head_valid = prediction_exists_from_detector(head.get("pred_x"), head.get("pred_y"), head.get("pred_conf"))
            tail_valid = prediction_exists_from_detector(tail.get("pred_x"), tail.get("pred_y"), tail.get("pred_conf"))
            raw = np.asarray(
                [
                    parse_float(head.get("pred_x")) / image_width,
                    parse_float(head.get("pred_y")) / image_height,
                    parse_float(tail.get("pred_x")) / image_width,
                    parse_float(tail.get("pred_y")) / image_height,
                    parse_float(head.get("pred_conf"), 0.0),
                    parse_float(tail.get("pred_conf"), 0.0),
                    1.0 if head_valid else 0.0,
                    1.0 if tail_valid else 0.0,
                ],
                dtype=np.float32,
            )
            head_visibility = head.get("visibility") or "fully_visible"
            if head_visibility not in VIS_TO_ID:
                head_visibility = "fully_visible"
            visibility = tail.get("visibility") or "fully_visible"
            if visibility not in VIS_TO_ID:
                visibility = "fully_visible"
            items.append(
                {
                    "swing_id": swing_id,
                    "image_stem": image_stem,
                    "frame_idx": frame_idx(head),
                    "gt_head": gt_head,
                    "gt_tail": gt_tail,
                    "raw": raw,
                    "head_status": head.get("status") or "unknown",
                    "tail_status": tail.get("status") or "unknown",
                    "head_visibility": head_visibility,
                    "tail_visibility": visibility,
                }
            )
        sequences[swing_id] = sorted(items, key=lambda item: item["frame_idx"])
    return sequences




def load_real_yolo_details_with_flow(csv_path, image_width, image_height, args):
    sequences = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            swing_id = row.get("swing_id") or "unknown"
            image_stem = row.get("image_stem") or row.get("frame")
            if not image_stem:
                continue
            head_valid = prediction_exists_from_detector(row.get("head_pred_x"), row.get("head_pred_y"), row.get("head_pred_conf"))
            tail_valid = prediction_exists_from_detector(row.get("tail_pred_x"), row.get("tail_pred_y"), row.get("tail_pred_conf"))
            raw = np.asarray(
                [
                    parse_float(row.get("head_pred_x")) / image_width,
                    parse_float(row.get("head_pred_y")) / image_height,
                    parse_float(row.get("tail_pred_x")) / image_width,
                    parse_float(row.get("tail_pred_y")) / image_height,
                    parse_float(row.get("head_pred_conf"), 0.0) if head_valid else 0.0,
                    parse_float(row.get("tail_pred_conf"), 0.0) if tail_valid else 0.0,
                    1.0 if head_valid else 0.0,
                    1.0 if tail_valid else 0.0,
                ],
                dtype=np.float32,
            )
            gt_head = np.asarray(
                [parse_float(row.get("head_gt_x")) / image_width, parse_float(row.get("head_gt_y")) / image_height],
                dtype=np.float32,
            )
            gt_tail = np.asarray(
                [parse_float(row.get("tail_gt_x")) / image_width, parse_float(row.get("tail_gt_y")) / image_height],
                dtype=np.float32,
            )
            head_visibility = row.get("head_visibility") or "fully_visible"
            if head_visibility not in VIS_TO_ID:
                head_visibility = "fully_visible"
            tail_visibility = row.get("tail_visibility") or "fully_visible"
            if tail_visibility not in VIS_TO_ID:
                tail_visibility = "fully_visible"
            status = row.get("status") or "unknown"
            sequences[swing_id].append(
                {
                    "swing_id": swing_id,
                    "image_stem": image_stem,
                    "frame_idx": frame_idx(row),
                    "gt_head": gt_head,
                    "gt_tail": gt_tail,
                    "raw": raw,
                    "flow": flow_vector_from_row(row, args),
                    "head_status": status,
                    "tail_status": status,
                    "head_visibility": head_visibility,
                    "tail_visibility": tail_visibility,
                }
            )
    return {key: sorted(value, key=lambda item: item["frame_idx"]) for key, value in sequences.items()}

def fill_target_xy(xy):
    out = xy.copy()
    frames = np.arange(len(out), dtype=np.float32)
    for dim in range(out.shape[1]):
        values = out[:, dim]
        valid = np.isfinite(values)
        if valid.any():
            out[:, dim] = np.interp(frames, frames[valid], values[valid]).astype(np.float32)
        else:
            out[:, dim] = 0.0
    return out.astype(np.float32)


def make_windows_v52(sequences, args):
    windows = []
    for swing_id, items in sequences.items():
        meta = metadata_for_swing(args, swing_id)
        flip_to_right = bool(meta["flip_to_right"])
        view_label = meta["view_label"]
        starts = tail_aligned_starts(len(items), args.window_size, args.stride, args.include_first_window)
        raw_seq = np.stack([item["raw"] for item in items]).astype(np.float32)
        flow_seq = np.stack([item.get("flow", np.zeros((0,), dtype=np.float32)) for item in items]).astype(np.float32)
        gt_tail = np.stack([item["gt_tail"] for item in items]).astype(np.float32)
        gt_head = np.stack([item["gt_head"] for item in items]).astype(np.float32)
        if flip_to_right:
            raw_seq = flip_seq8(raw_seq)
            flow_seq = np.stack([flip_flow_features(f) for f in flow_seq]).astype(np.float32)
            gt_tail = flip_xy2(gt_tail)
            gt_head = flip_xy2(gt_head)
        target_valid = (
            np.isfinite(gt_tail).all(axis=1)
            & np.isfinite(gt_head).all(axis=1)
            & np.asarray([item["tail_visibility"] != "ignore" for item in items], dtype=bool)
        ).astype(np.float32)
        gt_tail = fill_target_xy(gt_tail)
        gt_head = fill_target_xy(gt_head)
        visibility_ids_all = np.asarray([VIS_TO_ID[item["tail_visibility"]] for item in items], dtype=np.int64)
        for start in starts:
            end = start + args.window_size
            seq = raw_seq[start:end]
            observation = build_observation_features(
                head_xy_norm=seq[:, 0:2],
                tail_xy_norm=seq[:, 2:4],
                head_confidence=seq[:, 4],
                tail_confidence=seq[:, 5],
                head_prediction_exists=seq[:, 6] > 0.0,
                tail_prediction_exists=seq[:, 7] > 0.0,
                image_width=args.image_width,
                image_height=args.image_height,
                view_label=view_label,
                flow_features=flow_seq[start:end] if getattr(args, "use_flow_features", False) else None,
            )
            windows.append(
                {
                    "swing_id": swing_id,
                    "start": start,
                    "end": end,
                    "items": items[start:end],
                    "metadata": meta,
                    "raw_seq": raw_seq[start:end],
                    "X": observation.features,
                    "feature_schema": observation.feature_schema,
                    "gt_tail": gt_tail[start:end],
                    "gt_head": gt_head[start:end],
                    "target_valid": target_valid[start:end],
                    "tail_obs_valid": observation.tail_valid.astype(np.float32),
                    "tail_missing": observation.tail_missing,
                    "tail_visibility": visibility_ids_all[start:end],
                }
            )
    if not windows:
        raise ValueError("No windows were created. Check window_size/stride and input CSV.")
    return windows


make_windows_v51 = make_windows_v52


class RealYoloWindowDataset(Dataset):
    def __init__(self, windows, visibility_weights):
        self.samples = []
        for window in windows:
            tail_visibility = window["tail_visibility"].astype(np.int64)
            tail_weight = np.asarray([visibility_weights[int(vis_id)] for vis_id in tail_visibility], dtype=np.float32)
            self.samples.append(
                (
                    window["X"].astype(np.float32),
                    window["gt_tail"].astype(np.float32),
                    window["gt_head"].astype(np.float32),
                    tail_weight,
                    tail_visibility,
                    window["tail_obs_valid"].astype(np.float32),
                    window["target_valid"].astype(np.float32),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return tuple(torch.from_numpy(array) for array in self.samples[idx])


def split_windows_by_swing(windows, args):
    swing_ids = sorted({window["swing_id"] for window in windows})
    if args.val_swing_ids:
        val_ids = set(args.val_swing_ids)
    else:
        rng = random.Random(args.seed)
        shuffled = swing_ids[:]
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * args.val_fraction))) if len(shuffled) > 1 else 0
        val_ids = set(shuffled[:val_count])
    train_windows = [window for window in windows if window["swing_id"] not in val_ids]
    val_windows = [window for window in windows if window["swing_id"] in val_ids]
    if not train_windows:
        raise ValueError("No training windows after split. Use fewer --val_swing_ids or lower --val_fraction.")
    if not val_windows:
        print("[warn] no validation windows; using train windows for validation metrics")
        val_windows = train_windows
    train_ids = sorted({window["swing_id"] for window in train_windows})
    val_ids_sorted = sorted({window["swing_id"] for window in val_windows})
    return train_windows, val_windows, train_ids, val_ids_sorted


def describe_sequences(sequences, args, label):
    print(f"loaded_real_yolo_{label} swings={len(sequences)} details_csv={args.details_csv if label == 'train' else args.val_details_csv}")
    for swing_id, items in sorted(sequences.items()):
        meta = metadata_for_swing(args, swing_id)
        tail_status = defaultdict(int)
        tail_prediction = 0
        tail_ignore = 0
        target_valid = 0
        for item in items:
            tail_status[item["tail_status"]] += 1
            tail_prediction += int(item["raw"][7] > 0.0)
            tail_ignore += int(item["tail_visibility"] == "ignore")
            target_valid += int(np.isfinite(item["gt_tail"]).all() and np.isfinite(item["gt_head"]).all() and item["tail_visibility"] != "ignore")
        print(
            f"  {swing_id}: frames={len(items)} target_frames={target_valid} tail_prediction={tail_prediction} "
            f"tail_ignore={tail_ignore} tail_status={dict(tail_status)} "
            f"side={meta['batter_side']} view={meta['view_label']} flip={meta['flip_to_right']}"
        )


def make_datasets(args):
    args.clip_metadata = combine_clip_metadata(load_clip_metadata(args.clips_csv), load_clip_metadata(args.val_clips_csv))
    if args.clip_metadata:
        side_counts = defaultdict(int)
        view_counts = defaultdict(int)
        for meta in args.clip_metadata.values():
            side_counts[meta["batter_side"]] += 1
            view_counts[meta["view_label"]] += 1
        print(
            f"loaded_clip_metadata swings={len(args.clip_metadata)} clips_csv={args.clips_csv} "
            f"batter_side={dict(side_counts)} canonical_views={dict(view_counts)}"
        )
    if getattr(args, "use_flow_features", False):
        sequences = load_real_yolo_details_with_flow(args.details_csv, args.image_width, args.image_height, args)
    else:
        sequences = load_real_yolo_details(args.details_csv, args.image_width, args.image_height)
    describe_sequences(sequences, args, "train")
    train_source_windows = make_windows_v52(sequences, args)
    if args.val_details_csv:
        if getattr(args, "use_flow_features", False):
            val_sequences = load_real_yolo_details_with_flow(args.val_details_csv, args.image_width, args.image_height, args)
        else:
            val_sequences = load_real_yolo_details(args.val_details_csv, args.image_width, args.image_height)
        describe_sequences(val_sequences, args, "val")
        train_windows = train_source_windows
        val_windows = make_windows_v52(val_sequences, args)
        train_ids = sorted({window["swing_id"] for window in train_windows})
        val_ids = sorted({window["swing_id"] for window in val_windows})
        total_windows = len(train_windows) + len(val_windows)
    else:
        train_windows, val_windows, train_ids, val_ids = split_windows_by_swing(train_source_windows, args)
        total_windows = len(train_source_windows)
    visibility_weights = {
        0: args.weight_fully_visible,
        1: args.weight_partially_occluded,
        2: args.weight_fully_occluded,
        3: args.weight_ignore,
    }
    train_ds = RealYoloWindowDataset(train_windows, visibility_weights)
    val_ds = RealYoloWindowDataset(val_windows, visibility_weights)
    return train_ds, val_ds, train_ids, val_ids, total_windows


EXPANDED_INPUT_WEIGHT_KEYS = ("gru.weight_ih_l0", "gru.weight_ih_l0_reverse")
EXPANSION_TYPE = "strict_clean25_to_clean45_zero_flow_columns"


def expand_clean25_state_to_clean45(parent_model, target_model):
    """Copy a clean25 model into clean45, zeroing only the 20 new flow columns."""
    parent_state = parent_model.state_dict()
    target_state = target_model.state_dict()
    if set(parent_state) != set(target_state):
        raise ValueError("parent/target architecture state keys differ")
    expanded = {}
    for key, parent_value in parent_state.items():
        target_value = target_state[key]
        if key in EXPANDED_INPUT_WEIGHT_KEYS:
            expected_parent = (target_value.shape[0], len(CLEAN_NO_FLOW_SCHEMA))
            expected_target = (target_value.shape[0], len(CLEAN_WITH_FLOW_SCHEMA))
            if tuple(parent_value.shape) != expected_parent or tuple(target_value.shape) != expected_target:
                raise ValueError(
                    f"unexpected input weight shape for {key}: "
                    f"parent={tuple(parent_value.shape)} target={tuple(target_value.shape)}"
                )
            value = torch.zeros_like(target_value)
            value[:, : len(CLEAN_NO_FLOW_SCHEMA)].copy_(parent_value)
            expanded[key] = value
        else:
            if tuple(parent_value.shape) != tuple(target_value.shape):
                raise ValueError(
                    f"architecture mismatch at {key}: "
                    f"parent={tuple(parent_value.shape)} target={tuple(target_value.shape)}"
                )
            expanded[key] = parent_value.detach().clone()
    target_model.load_state_dict(expanded, strict=True)
    return target_model


def _require_matching_architecture(ckpt, args, output_dim):
    expected = {
        "model_type": args.model,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "output_dim": output_dim,
    }
    for key, value in expected.items():
        actual = ckpt.get(key)
        if key == "dropout":
            matches = actual is not None and float(actual) == float(value)
        else:
            matches = actual == value
        if not matches:
            raise ValueError(f"parent architecture mismatch for {key}: {actual!r} != {value!r}")


def verify_flow_dataset_alignment(args):
    if not getattr(args, "use_flow_features", False):
        args.flow_alignment_reports = {}
        return {}
    if not args.selected_details_csv or not args.val_selected_details_csv:
        raise ValueError(
            "--use_flow_features requires --selected_details_csv and "
            "--val_selected_details_csv for deterministic provenance verification"
        )
    reports = {
        "train": verify_selected_flow_alignment(args.selected_details_csv, args.details_csv),
        "validation": verify_selected_flow_alignment(args.val_selected_details_csv, args.val_details_csv),
    }
    verify_disjoint_splits(reports)
    args.flow_alignment_reports = reports
    return reports


def load_pretrained(args, input_dim, output_dim, device):
    if args.init_from_scratch:
        expected_input_dim = 45 if args.use_flow_features else 25
        if input_dim != expected_input_dim:
            raise ValueError(
                f"Clean-input scratch initialization requires input_dim={expected_input_dim}, got {input_dim}."
            )
        model = build_model(
            args.model,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=True,
            output_dim=output_dim,
        ).to(device)
        args.prediction_mode = "prior_guided_yolo_residual_trust"
        args.prior_mode = args.prediction_mode
        print(
            "initialized_from_scratch "
            f"model={args.model} input_dim={input_dim} output_dim={output_dim} "
            f"hidden_dim={args.hidden_dim} num_layers={args.num_layers} dropout={args.dropout:g}"
        )
        return model, {
            "model_type": args.model,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "init_from_scratch": True,
        }
    ckpt = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
    ckpt_input_dim = ckpt.get("input_dim", input_dim)
    if args.expand_clean25_to_clean45:
        if not args.use_flow_features or input_dim != len(CLEAN_WITH_FLOW_SCHEMA):
            raise ValueError("clean25->clean45 expansion requires --use_flow_features and target input_dim=45")
        validate_checkpoint_contract(
            ckpt,
            use_flow_features=False,
            important_settings={
                "window_size": args.window_size,
                "image_width": args.image_width,
                "image_height": args.image_height,
            },
        )
        if ckpt_input_dim != len(CLEAN_NO_FLOW_SCHEMA):
            raise ValueError(f"clean25->clean45 expansion requires parent input_dim=25, got {ckpt_input_dim}")
        if ckpt.get("feature_schema") != list(CLEAN_NO_FLOW_SCHEMA):
            raise ValueError("clean25->clean45 expansion requires the canonical clean25 schema/order")
        _require_matching_architecture(ckpt, args, output_dim)
        parent_model = build_model(
            ckpt["model_type"],
            input_dim=len(CLEAN_NO_FLOW_SCHEMA),
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
            bidirectional=True,
            output_dim=ckpt["output_dim"],
        ).to(device)
        strict_load_model_state(parent_model, ckpt)
        model = build_model(
            args.model,
            input_dim=len(CLEAN_WITH_FLOW_SCHEMA),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=True,
            output_dim=output_dim,
        ).to(device)
        expand_clean25_state_to_clean45(parent_model, model)
        lineage = verified_strict_clean_parent(
            ckpt,
            args.pretrained_checkpoint,
            use_flow_features=False,
            current_stage="strict_clean_real_finetune_45d",
        )
        lineage.update({
            "expansion_type": EXPANSION_TYPE,
            "preserved_columns_count": len(CLEAN_NO_FLOW_SCHEMA),
            "added_flow_columns_count": len(FLOW_FEATURES),
            "target_feature_schema": list(CLEAN_WITH_FLOW_SCHEMA),
            "target_input_dim": len(CLEAN_WITH_FLOW_SCHEMA),
            "raft20_feature_order": list(FLOW_FEATURES),
        })
        args.verified_parent_lineage = lineage
        args.prediction_mode = "prior_guided_yolo_residual_trust"
        args.prior_mode = args.prediction_mode
        print(
            f"expanded_pretrained checkpoint={args.pretrained_checkpoint} epoch={ckpt.get('epoch')} "
            f"expansion={EXPANSION_TYPE} input_dim=25->45"
        )
        return model, ckpt
    ckpt_output_dim = ckpt.get("output_dim", output_dim)
    if ckpt_input_dim != input_dim:
        raise ValueError(
            f"Checkpoint input_dim={ckpt_input_dim} but clean-input windows have input_dim={input_dim}. "
            "Train a new clean-input checkpoint with the matching feature schema."
        )
    if ckpt_output_dim != output_dim:
        raise ValueError(
            f"Checkpoint output_dim={ckpt_output_dim} but prior-guided residual+trust requires output_dim={output_dim}."
        )
    model = build_model(
        ckpt.get("model_type", args.model),
        input_dim=ckpt_input_dim,
        hidden_dim=ckpt.get("hidden_dim", args.hidden_dim),
        num_layers=ckpt.get("num_layers", args.num_layers),
        dropout=ckpt.get("dropout", args.dropout),
        bidirectional=True,
        output_dim=ckpt_output_dim,
    ).to(device)
    validate_checkpoint_contract(
        ckpt,
        use_flow_features=args.use_flow_features,
        important_settings={"window_size": args.window_size, "image_width": args.image_width, "image_height": args.image_height},
    )
    args.verified_parent_lineage = verified_strict_clean_parent(
        ckpt, args.pretrained_checkpoint, use_flow_features=args.use_flow_features
    )
    strict_load_model_state(model, ckpt)
    args.prediction_mode = "prior_guided_yolo_residual_trust"
    args.prior_mode = args.prediction_mode
    print(
        f"loaded_pretrained checkpoint={args.pretrained_checkpoint} "
        f"epoch={ckpt.get('epoch')} prediction_mode={args.prediction_mode}"
    )
    return model, ckpt


def run_epoch(model, loader, optimizer, device, args, scale):
    model.train()
    running = defaultdict(float)
    running["count"] = 0
    for x, y_tail, y_head, tail_weight, tail_visibility, tail_obs_valid, target_valid in loader:
        x = x.to(device)
        y_tail = y_tail.to(device)
        y_head = y_head.to(device)
        tail_weight = tail_weight.to(device)
        tail_visibility = tail_visibility.to(device)
        tail_obs_valid = tail_obs_valid.to(device)
        target_valid = target_valid.to(device)
        loss, components = compute_loss(
            model, x, y_tail, y_head, tail_weight, tail_visibility, tail_obs_valid, args, scale, target_valid=target_valid
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        update_running(running, components, len(x))
    return finish_running(running)


@torch.no_grad()
def evaluate_loss(model, loader, device, args, scale):
    model.eval()
    running = defaultdict(float)
    running["count"] = 0
    final_squared_error_sum = 0.0
    final_valid_count = 0
    for x, y_tail, y_head, tail_weight, tail_visibility, tail_obs_valid, target_valid in loader:
        x = x.to(device)
        y_tail = y_tail.to(device)
        y_head = y_head.to(device)
        tail_weight = tail_weight.to(device)
        tail_visibility = tail_visibility.to(device)
        tail_obs_valid = tail_obs_valid.to(device)
        target_valid = target_valid.to(device)
        _loss, components = compute_loss(
            model, x, y_tail, y_head, tail_weight, tail_visibility, tail_obs_valid, args, scale, target_valid=target_valid
        )
        update_running(running, components, len(x))
        final_tail = decode_prediction(model, x, args, scale)[0]
        valid = target_valid > 0.0
        squared_error = (((final_tail - y_tail) * scale) ** 2).sum(dim=-1)
        final_squared_error_sum += float(squared_error[valid].sum().item())
        final_valid_count += int(valid.sum().item())
    metrics = finish_running(running)
    metrics["final_position_rmse_px"] = (
        float(np.sqrt(final_squared_error_sum / final_valid_count)) if final_valid_count else None
    )
    return metrics


def make_lr_scheduler(args, optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
    raise ValueError(f"Unsupported lr_scheduler={args.lr_scheduler}")


def step_lr_scheduler(scheduler, metrics, args):
    if scheduler is None:
        return None
    metric_value = metrics.get(args.lr_scheduler_metric)
    if metric_value is None:
        metric_value = metrics.get("total", 0.0)
    scheduler.step(float(metric_value))
    return float(metric_value)


def current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune v8 prior-guided correction + trust model on real YOLO prediction + GT details CSV.")
    parser.add_argument("--details_csv", required=True)
    parser.add_argument("--val_details_csv", default=None, help="Optional separate validation details CSV. When set, no swing-level random split is used.")
    parser.add_argument("--pretrained_checkpoint", default=None)
    parser.add_argument("--init_from_scratch", action="store_true", help="Train the real-data finetune model from random initialization instead of loading a checkpoint.")
    parser.add_argument("--expand_clean25_to_clean45", action="store_true", help="Verified strict-clean 25D parent expansion with zero-initialized RAFT20 input columns.")
    parser.add_argument("--selected_details_csv", default=None, help="Selected long-form detector CSV aligned to --details_csv.")
    parser.add_argument("--val_selected_details_csv", default=None, help="Selected long-form detector CSV aligned to --val_details_csv.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", choices=["bigru", "gru"], default="bigru")
    parser.add_argument("--prediction_mode", default="prior_guided_yolo_residual_trust", help="Fixed to prior_guided_yolo_residual_trust for v8.")
    parser.add_argument("--prior_mode", default=None, help="Ignored backward-compatible alias.")
    parser.add_argument("--window_size", type=int, default=31)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--include_first_window", action="store_true")
    parser.add_argument("--clips_csv", default=None, help="Optional clips_list.csv with name/view/batter_side metadata.")
    parser.add_argument("--val_clips_csv", default=None, help="Optional extra clips metadata for --val_details_csv; defaults to --clips_csv when omitted.")
    parser.add_argument("--view_label", default="random", help="Coarse camera/view label for real YOLO details when metadata is unavailable.")
    parser.add_argument("--batter_hand", choices=["R", "L", "r", "l"], default="R", help="Observed batter hand; L is flipped to canonical right-handed space.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", choices=["none", "reduce_on_plateau"], default="none")
    parser.add_argument("--lr_scheduler_metric", default="total")
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=2e-7)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_width", type=float, default=1920.0)
    parser.add_argument("--image_height", type=float, default=1080.0)
    parser.add_argument("--use_flow_features", action="store_true")
    parser.add_argument("--flow_clip_px", type=float, default=50.0)
    parser.add_argument("--val_fraction", type=float, default=0.25)
    parser.add_argument("--val_swing_ids", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--weight_fully_visible", type=float, default=0.75)
    parser.add_argument("--weight_partially_occluded", type=float, default=2.0)
    parser.add_argument("--weight_fully_occluded", type=float, default=3.0)
    parser.add_argument("--weight_ignore", type=float, default=4.0)
    parser.add_argument("--lambda_dir", type=float, default=1.0)
    parser.add_argument("--lambda_velocity", type=float, default=0.02)
    parser.add_argument("--lambda_acceleration", type=float, default=0.0)
    parser.add_argument("--lambda_observation", type=float, default=0.02)
    parser.add_argument("--lambda_line", type=float, default=0.05)
    parser.add_argument("--lambda_end", type=float, default=0.02)
    parser.add_argument("--line_delta_px", type=float, default=3.0)
    parser.add_argument("--end_delta_px", type=float, default=10.0)
    parser.add_argument("--lambda_prior", type=float, default=0.5)
    parser.add_argument("--lambda_correction", type=float, default=1.0)
    parser.add_argument("--lambda_prior_guidance", type=float, default=0.05)
    parser.add_argument("--lambda_trust", type=float, default=0.2)
    parser.add_argument("--lambda_delta", type=float, default=0.01)
    parser.add_argument("--lambda_good_yolo_damage", type=float, default=0.0)
    parser.add_argument("--lambda_prior_velocity", type=float, default=0.02)
    parser.add_argument("--lambda_prior_acceleration", type=float, default=0.005)
    parser.add_argument("--lambda_length", type=float, default=0.0)
    parser.add_argument("--lambda_yolo_error_bucket", type=float, default=0.0)
    parser.add_argument("--lambda_yolo_quality", type=float, default=0.0)
    parser.add_argument("--yolo_quality_good_error_px", type=float, default=5.0)
    parser.add_argument("--yolo_quality_bad_error_px", type=float, default=10.0)
    parser.add_argument("--trust_margin_px", type=float, default=2.0, help="Deprecated; kept for old command compatibility.")
    parser.add_argument("--trust_target_temp_px", type=float, default=3.0, help="Deprecated; kept for old command compatibility.")
    parser.add_argument("--trust_min_delta_px", type=float, default=0.5)
    parser.add_argument("--trust_min_gain_px", type=float, default=0.5, help="Minimum best possible gain before trusting a predicted correction.")
    parser.add_argument("--trust_good_gain_px", type=float, default=3.0, help="Best possible gain that maps the gain quality term to 1.")
    parser.add_argument("--trust_target_eps", type=float, default=1e-6)
    parser.add_argument("--delta_regularize_error_px", type=float, default=5.0, help="Deprecated; damage loss now applies to all observed frames.")
    parser.add_argument("--delta_damage_margin_px", type=float, default=1.0)
    parser.add_argument("--good_yolo_damage_error_px", type=float, default=5.0)
    parser.add_argument("--good_yolo_damage_margin_px", type=float, default=0.25)
    parser.add_argument("--error_weight_5_px", type=float, default=5.0)
    parser.add_argument("--error_weight_10_px", type=float, default=10.0)
    parser.add_argument("--error_weight_20_px", type=float, default=20.0)
    parser.add_argument("--weight_error_lt5", type=float, default=0.3)
    parser.add_argument("--weight_error_5_10", type=float, default=1.0)
    parser.add_argument("--weight_error_10_20", type=float, default=2.0)
    parser.add_argument("--weight_error_gt20", type=float, default=4.0)
    parser.add_argument("--weight_missing_occluded", type=float, default=4.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    args = parser.parse_args()
    args.prediction_mode = "prior_guided_yolo_residual_trust"
    args.prior_mode = args.prediction_mode
    return args


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if not args.init_from_scratch and not args.pretrained_checkpoint:
        raise ValueError("Provide --pretrained_checkpoint or pass --init_from_scratch.")
    if args.init_from_scratch and args.expand_clean25_to_clean45:
        raise ValueError("--init_from_scratch and --expand_clean25_to_clean45 are mutually exclusive")
    verify_flow_dataset_alignment(args)

    train_ds, val_ds, train_ids, val_ids, total_windows = make_datasets(args)
    sample_x, _sample_y_tail, _sample_y_head, _sample_weight, _sample_vis, _sample_valid, _sample_target_valid = train_ds[0]
    window_size, input_dim = sample_x.shape
    output_dim = 10
    print(
        f"real_finetune windows={total_windows} train_windows={len(train_ds)} val_windows={len(val_ds)} "
        f"window_size={window_size} input_dim={input_dim}"
    )
    print(f"train_swing_ids={train_ids}")
    print(f"val_swing_ids={val_ids}")

    device = torch.device(args.device)
    scale = coordinate_scale(args, device)
    model, pretrained = load_pretrained(args, input_dim, output_dim, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = make_lr_scheduler(args, optimizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    print(
        "finetune_config "
        f"lr={args.lr:g} batch_size={args.batch_size} epochs={args.epochs} patience={args.patience} "
        f"lr_scheduler={args.lr_scheduler} lr_scheduler_metric={args.lr_scheduler_metric} "
        f"lambda_prior={args.lambda_prior:g} "
        f"lambda_correction={args.lambda_correction:g} lambda_trust={args.lambda_trust:g} "
        f"lambda_prior_guidance={args.lambda_prior_guidance:g} "
        f"lambda_delta={args.lambda_delta:g} lambda_good_yolo_damage={args.lambda_good_yolo_damage:g} "
        f"lambda_line={args.lambda_line:g} lambda_end={args.lambda_end:g} "
        f"lambda_length={args.lambda_length:g} "
        f"lambda_yolo_error_bucket={args.lambda_yolo_error_bucket:g} lambda_yolo_quality={args.lambda_yolo_quality:g} "
        f"prediction_mode={args.prediction_mode}"
    )
    print(
        "v8_visibility "
    )
    print(
        "view_hand_config "
        f"clips_csv={args.clips_csv} fallback_view={canonical_view_label(args.view_label, args.batter_hand)} "
        f"fallback_batter_hand={normalize_batter_hand(args.batter_hand)} canonical_batter_hand=R"
    )

    best_val = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, scale)
        val_metrics = evaluate_loss(model, val_loader, device, args, scale)
        lr_scheduler_metric = step_lr_scheduler(lr_scheduler, val_metrics, args)
        lr_now = current_lr(optimizer)
        history.append({
            "epoch": epoch,
            "lr": lr_now,
            "lr_scheduler_metric": lr_scheduler_metric,
            "train": train_metrics,
            "val": val_metrics,
        })
        val_rmse_text = f" val_final_rmse_px={val_metrics['final_position_rmse_px']:.6f}"
        print(
            f"epoch={epoch:03d} lr={lr_now:.8g} "
            f"{metric_text(train_metrics, 'train')} {metric_text(val_metrics, 'val')}{val_rmse_text}"
        )

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_epoch = epoch
            ckpt = {
                "model_state": model.state_dict(),
                **checkpoint_contract(
                    use_flow_features=args.use_flow_features,
                    split_identifier=f"train:{','.join(train_ids)}|val:{','.join(val_ids)}",
                    important_settings={
                        "window_size": window_size,
                        "stride": args.stride,
                        "image_width": args.image_width,
                        "image_height": args.image_height,
                        "flow_clip_px": getattr(args, "flow_clip_px", None),
                        "stability_gate": False,
                    },
                ),
                "model_type": pretrained.get("model_type", args.model),
                "task": "prior_guided_tail_residual_trust_refiner_real_finetune_v8_flow" if args.use_flow_features else "prior_guided_tail_residual_trust_refiner_real_finetune_v8",
                "pretrained_checkpoint": args.pretrained_checkpoint,
                "pretrained_checkpoint_sha256": checkpoint_sha256(args.pretrained_checkpoint),
                "lineage": dict(args.verified_parent_lineage),
                "initialization_contract": dict(args.verified_parent_lineage),
                "seed": args.seed,
                "dataset_provenance": {
                    "train_csv": args.details_csv,
                    "train_csv_sha256": checkpoint_sha256(args.details_csv),
                    "validation_csv": args.val_details_csv,
                    "validation_csv_sha256": checkpoint_sha256(args.val_details_csv),
                    "selected_detector_rows": bool(args.flow_alignment_reports) or bool(
                        str(args.details_csv).endswith("_selected.csv")
                        and str(args.val_details_csv).endswith("_selected.csv")
                    ),
                    "flow_features_generated_and_aligned_from_selected_detector_observations": bool(
                        args.flow_alignment_reports
                    ),
                    "flow_alignment": dict(args.flow_alignment_reports),
                },
                "prediction_mode": args.prediction_mode,
                "prior_mode": args.prediction_mode,
                "window_size": window_size,
                "input_dim": input_dim,
                "output_dim": pretrained.get("output_dim", output_dim),
                "hidden_dim": pretrained.get("hidden_dim", args.hidden_dim),
                "num_layers": pretrained.get("num_layers", args.num_layers),
                "dropout": pretrained.get("dropout", args.dropout),
                "image_width": args.image_width,
                "image_height": args.image_height,
                "train_swing_ids": train_ids,
                "val_swing_ids": val_ids,
                "clips_csv": args.clips_csv,
                "clip_metadata": {key: value for key, value in getattr(args, "clip_metadata", {}).items() if key in set(train_ids + val_ids)},
                "fallback_view_label": canonical_view_label(args.view_label, args.batter_hand),
                "fallback_batter_hand": normalize_batter_hand(args.batter_hand),
                "canonical_batter_hand": "R",
                "left_hand_horizontal_flip_to_right": True,
                "loss_config": vars(args),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"[best] saved {out_dir / 'best.pt'}")

        if epoch - best_epoch >= args.patience:
            print(f"[early_stop] best_epoch={best_epoch} best_val={best_val:.6f}")
            break

    with open(out_dir / "history.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)
    print(f"done best_epoch={best_epoch} best_val={best_val:.6f}")


if __name__ == "__main__":
    main()
