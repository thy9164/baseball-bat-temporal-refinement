import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation_metrics import (  # noqa: E402
    VIS_NAMES,
    add_correction_effect,
    add_prior_effect,
    add_trust_effect,
    point_errors_px,
    stats,
)
from src.real_data_pipeline import VIS_TO_ID, load_clip_metadata, load_real_yolo_details, make_windows_v52  # noqa: E402
from src.train_temporal_refinement import (  # noqa: E402
    load_real_yolo_details_with_flow,
    make_windows_v52 as make_windows_v52_with_flow,
)
from clean_input.checkpoint import strict_load_model_state, validate_checkpoint_contract  # noqa: E402
from src.models import build_model  # noqa: E402
from src.view_features import canonical_view_label, normalize_batter_hand  # noqa: E402


def load_model(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    use_flow_features = bool(args.use_flow_features)
    validate_checkpoint_contract(
        ckpt,
        use_flow_features=use_flow_features,
        important_settings={"window_size": args.window_size, "image_width": args.image_width, "image_height": args.image_height},
    )
    output_dim = ckpt.get("output_dim", 10)
    if output_dim != 10:
        raise ValueError(
            f"Checkpoint output_dim={output_dim} but prior-guided residual+trust evaluation requires output_dim=10."
        )
    model = build_model(
        ckpt.get("model_type", args.model),
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt.get("hidden_dim", args.hidden_dim),
        num_layers=ckpt.get("num_layers", args.num_layers),
        dropout=ckpt.get("dropout", args.dropout),
        bidirectional=True,
        output_dim=output_dim,
    ).to(device)
    strict_load_model_state(model, ckpt)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_windows(model, ckpt, windows, args, device):
    scale = torch.tensor([args.image_width, args.image_height], dtype=torch.float32, device=device)
    outputs = []
    for start in range(0, len(windows), args.batch_size):
        batch = windows[start:start + args.batch_size]
        x = np.stack([window["X"] for window in batch]).astype(np.float32)
        xb = torch.from_numpy(x).to(device)
        raw = model(xb)
        prior = raw[:, :, :2]
        delta_px = raw[:, :, 2:4]
        trust = torch.sigmoid(raw[:, :, 4:5])
        bucket_probs = torch.softmax(raw[:, :, 5:9], dim=-1)
        yolo_quality = torch.sigmoid(raw[:, :, 9])
        final = xb[:, :, 2:4] + trust * delta_px / scale
        corrected = xb[:, :, 2:4] + delta_px / scale
        for local_idx in range(len(batch)):
            outputs.append(
                {
                    "prior": np.clip(prior[local_idx].cpu().numpy(), 0.0, 1.0),
                    "final": np.clip(final[local_idx].cpu().numpy(), 0.0, 1.0),
                    "corrected": np.clip(corrected[local_idx].cpu().numpy(), 0.0, 1.0),
                    "delta_px": delta_px[local_idx].cpu().numpy(),
                    "trust": trust[local_idx, :, 0].cpu().numpy(),
                    "yolo_error_bucket_probs": bucket_probs[local_idx].cpu().numpy(),
                    "yolo_quality": yolo_quality[local_idx].cpu().numpy(),
                }
            )
    return outputs


def aggregate(windows, outputs):
    records = {}
    prior_lists = defaultdict(list)
    final_lists = defaultdict(list)
    corrected_lists = defaultdict(list)
    delta_lists = defaultdict(list)
    trust_lists = defaultdict(list)
    bucket_prob_lists = defaultdict(list)
    quality_lists = defaultdict(list)
    filled_lists = defaultdict(list)
    for window, output in zip(windows, outputs):
        raw_seq = window.get("raw_seq")
        for idx, item in enumerate(window["items"]):
            key = (item["swing_id"], item["frame_idx"])
            records[key] = {
                **item,
                "raw": raw_seq[idx] if raw_seq is not None else item["raw"],
                "gt_head": window["gt_head"][idx] if window.get("target_valid", np.ones(len(window["items"])))[idx] > 0.0 else np.asarray([np.nan, np.nan], dtype=np.float32),
                "gt_tail": window["gt_tail"][idx] if window.get("target_valid", np.ones(len(window["items"])))[idx] > 0.0 else np.asarray([np.nan, np.nan], dtype=np.float32),
                "target_valid": bool(window.get("target_valid", np.ones(len(window["items"])))[idx] > 0.0),
                "tail_observation_valid": window["tail_obs_valid"][idx] > 0.0,
                "head_visibility_id": int(VIS_TO_ID.get(item.get("head_visibility", "fully_visible"), VIS_TO_ID["fully_visible"])),
                "tail_visibility_id": int(window["tail_visibility"][idx]),
            }
            prior_lists[key].append(output["prior"][idx])
            final_lists[key].append(output["final"][idx])
            corrected_lists[key].append(output["corrected"][idx])
            delta_lists[key].append(output["delta_px"][idx])
            trust_lists[key].append(output["trust"][idx])
            bucket_prob_lists[key].append(output["yolo_error_bucket_probs"][idx])
            quality_lists[key].append(output["yolo_quality"][idx])
            filled_lists[key].append(window["X"][idx, 2:4])
    frames = []
    for key in sorted(records, key=lambda item: (item[0], item[1])):
        record = records[key]
        record["prior_tail"] = np.mean(prior_lists[key], axis=0).astype(np.float32)
        record["final_tail"] = np.mean(final_lists[key], axis=0).astype(np.float32)
        record["corrected_tail"] = np.mean(corrected_lists[key], axis=0).astype(np.float32)
        record["delta_px"] = np.mean(delta_lists[key], axis=0).astype(np.float32)
        record["trust"] = float(np.mean(trust_lists[key]))
        record["yolo_error_bucket_probs"] = np.mean(bucket_prob_lists[key], axis=0).astype(np.float32)
        record["yolo_error_bucket_pred"] = int(np.argmax(record["yolo_error_bucket_probs"]))
        record["yolo_quality"] = float(np.mean(quality_lists[key]))
        record["filled_head"] = record["raw"][0:2].astype(np.float32)
        record["filled_tail"] = np.mean(filled_lists[key], axis=0).astype(np.float32)
        frames.append(record)
    return frames


def denorm_point(point, args):
    point = np.asarray(point, dtype=np.float32)
    return np.asarray([point[0] * args.image_width, point[1] * args.image_height], dtype=np.float32)


def write_prediction_csv(frames, path, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "swing_id",
        "image_stem",
        "frame_idx",
        "keypoint",
        "visibility",
        "target_valid",
        "obs_valid",
        "raw_x",
        "raw_y",
        "filled_x",
        "filled_y",
        "prior_x",
        "prior_y",
        "final_x",
        "final_y",
        "corrected_x",
        "corrected_y",
        "gt_x",
        "gt_y",
        "pred_conf",
        "trust",
        "yolo_quality",
        "raw_error_px",
        "filled_error_px",
        "prior_error_px",
        "final_error_px",
        "corrected_error_px",
        "final_shift_px",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frames:
            raw = np.asarray(frame["raw"], dtype=np.float32)
            raw_head = denorm_point(raw[:2], args)
            raw_tail = denorm_point(raw[2:4], args)
            filled_head = denorm_point(frame["filled_head"], args)
            filled_tail = denorm_point(frame["filled_tail"], args)
            final_tail = denorm_point(frame["final_tail"], args)
            prior_tail = denorm_point(frame["prior_tail"], args)
            corrected_tail = denorm_point(frame["corrected_tail"], args)
            gt_head = denorm_point(frame["gt_head"], args) if frame["target_valid"] else np.asarray([np.nan, np.nan], dtype=np.float32)
            gt_tail = denorm_point(frame["gt_tail"], args) if frame["target_valid"] else np.asarray([np.nan, np.nan], dtype=np.float32)
            target_valid = bool(frame["target_valid"])
            tail_obs_valid = bool(frame["tail_observation_valid"])
            head_error = float(np.linalg.norm(raw_head - gt_head)) if target_valid and np.all(np.isfinite(gt_head)) else None
            tail_raw_error = float(np.linalg.norm(raw_tail - gt_tail)) if target_valid and tail_obs_valid else None
            tail_filled_error = float(np.linalg.norm(filled_tail - gt_tail)) if target_valid else None
            tail_prior_error = float(np.linalg.norm(prior_tail - gt_tail)) if target_valid else None
            tail_final_error = float(np.linalg.norm(final_tail - gt_tail)) if target_valid else None
            tail_corrected_error = float(np.linalg.norm(corrected_tail - gt_tail)) if target_valid else None
            final_shift = float(np.linalg.norm(final_tail - filled_tail))

            common = {
                "swing_id": frame.get("swing_id"),
                "image_stem": frame.get("image_stem"),
                "frame_idx": frame.get("frame_idx"),
                "target_valid": int(target_valid),
                "trust": frame.get("trust"),
                "yolo_quality": frame.get("yolo_quality"),
            }
            writer.writerow(
                {
                    **common,
                    "keypoint": "head",
                    "visibility": frame.get("head_visibility"),
                    "obs_valid": 1,
                    "raw_x": float(raw_head[0]),
                    "raw_y": float(raw_head[1]),
                    "filled_x": float(filled_head[0]),
                    "filled_y": float(filled_head[1]),
                    "prior_x": "",
                    "prior_y": "",
                    "final_x": float(raw_head[0]),
                    "final_y": float(raw_head[1]),
                    "corrected_x": float(raw_head[0]),
                    "corrected_y": float(raw_head[1]),
                    "gt_x": float(gt_head[0]) if np.isfinite(gt_head[0]) else "",
                    "gt_y": float(gt_head[1]) if np.isfinite(gt_head[1]) else "",
                    "pred_conf": frame.get("head_conf"),
                    "raw_error_px": head_error,
                    "filled_error_px": head_error,
                    "prior_error_px": "",
                    "final_error_px": head_error,
                    "corrected_error_px": head_error,
                    "final_shift_px": 0.0,
                }
            )
            writer.writerow(
                {
                    **common,
                    "keypoint": "tail",
                    "visibility": frame.get("tail_visibility"),
                    "obs_valid": int(tail_obs_valid),
                    "raw_x": float(raw_tail[0]) if tail_obs_valid else "",
                    "raw_y": float(raw_tail[1]) if tail_obs_valid else "",
                    "filled_x": float(filled_tail[0]),
                    "filled_y": float(filled_tail[1]),
                    "prior_x": float(prior_tail[0]),
                    "prior_y": float(prior_tail[1]),
                    "final_x": float(final_tail[0]),
                    "final_y": float(final_tail[1]),
                    "corrected_x": float(corrected_tail[0]),
                    "corrected_y": float(corrected_tail[1]),
                    "gt_x": float(gt_tail[0]) if np.isfinite(gt_tail[0]) else "",
                    "gt_y": float(gt_tail[1]) if np.isfinite(gt_tail[1]) else "",
                    "pred_conf": frame.get("tail_conf"),
                    "raw_error_px": tail_raw_error,
                    "filled_error_px": tail_filled_error,
                    "prior_error_px": tail_prior_error,
                    "final_error_px": tail_final_error,
                    "corrected_error_px": tail_corrected_error,
                    "final_shift_px": final_shift,
                }
            )


def scalar_stats(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.percentile(values, 50)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def component_stats(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "count": 0,
            "mean_signed": None,
            "mae": None,
            "rmse": None,
            "median_abs": None,
            "p90_abs": None,
            "gt5": None,
            "gt10": None,
            "gt20": None,
        }
    abs_values = np.abs(values)
    return {
        "count": int(len(values)),
        "mean_signed": float(np.mean(values)),
        "mae": float(np.mean(abs_values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "median_abs": float(np.percentile(abs_values, 50)),
        "p90_abs": float(np.percentile(abs_values, 90)),
        "gt5": float(np.mean(abs_values > 5.0)),
        "gt10": float(np.mean(abs_values > 10.0)),
        "gt20": float(np.mean(abs_values > 20.0)),
    }


def head_axis_quality_counts(u_values, v_values):
    u_values = np.asarray(u_values, dtype=np.float32)
    v_values = np.asarray(v_values, dtype=np.float32)
    valid = np.isfinite(u_values) & np.isfinite(v_values)
    labels = ["ok", "slightly_bad", "bad", "very_bad"]
    v_labels = ["ok", "slightly_bad", "bad"]
    if not np.any(valid):
        return {
            "count": 0,
            "u_counts": {label: 0 for label in labels},
            "u_rates": {label: None for label in labels},
            "v_counts": {label: 0 for label in v_labels},
            "v_rates": {label: None for label in v_labels},
            "axis_counts": {label: 0 for label in labels},
            "axis_rates": {label: None for label in labels},
        }

    abs_u = np.abs(u_values[valid])
    v = v_values[valid]
    u_quality = np.full(len(abs_u), "very_bad", dtype=object)
    u_quality[abs_u < 10.0] = "bad"
    u_quality[abs_u < 5.0] = "slightly_bad"
    u_quality[abs_u < 2.0] = "ok"

    v_quality = np.full(len(v), "bad", dtype=object)
    v_quality[((v >= -7.0) & (v < -3.0)) | ((v > 5.0) & (v <= 10.0))] = "slightly_bad"
    v_quality[(v >= -3.0) & (v <= 5.0)] = "ok"

    axis_quality = np.full(len(v), "ok", dtype=object)
    axis_quality[(u_quality == "slightly_bad") | (v_quality == "slightly_bad")] = "slightly_bad"
    axis_quality[(u_quality == "bad") | (v_quality == "bad")] = "bad"
    axis_quality[u_quality == "very_bad"] = "very_bad"

    def counts_and_rates(values, label_list):
        total = len(values)
        counts = {label: int(np.sum(values == label)) for label in label_list}
        rates = {label: float(counts[label] / total) for label in label_list}
        return counts, rates

    u_counts, u_rates = counts_and_rates(u_quality, labels)
    v_counts, v_rates = counts_and_rates(v_quality, v_labels)
    axis_counts, axis_rates = counts_and_rates(axis_quality, labels)
    return {
        "count": int(len(v)),
        "u_counts": u_counts,
        "u_rates": u_rates,
        "v_counts": v_counts,
        "v_rates": v_rates,
        "axis_counts": axis_counts,
        "axis_rates": axis_rates,
    }


def head_axis_components_px(pred_head, gt_head, gt_tail, image_width, image_height):
    scale = np.asarray([image_width, image_height], dtype=np.float32)
    pred_head_px = pred_head.astype(np.float32) * scale
    gt_head_px = gt_head.astype(np.float32) * scale
    gt_tail_px = gt_tail.astype(np.float32) * scale
    axis_v = gt_tail_px - gt_head_px
    length = np.linalg.norm(axis_v, axis=1)
    valid = np.isfinite(pred_head_px).all(axis=1) & np.isfinite(gt_head_px).all(axis=1) & np.isfinite(gt_tail_px).all(axis=1) & (length > 1e-6)
    unit_v = np.zeros_like(axis_v, dtype=np.float32)
    unit_v[valid] = axis_v[valid] / length[valid, None]
    unit_u = np.stack([-unit_v[:, 1], unit_v[:, 0]], axis=1)
    diff = pred_head_px - gt_head_px
    u = np.full(len(pred_head), np.nan, dtype=np.float32)
    v = np.full(len(pred_head), np.nan, dtype=np.float32)
    u[valid] = np.sum(diff[valid] * unit_u[valid], axis=1)
    v[valid] = np.sum(diff[valid] * unit_v[valid], axis=1)
    euclidean = np.full(len(pred_head), np.nan, dtype=np.float32)
    euclidean[valid] = np.linalg.norm(diff[valid], axis=1)
    return u, v, euclidean, valid


def summarize_head_axis_error(pred_head, gt_head, gt_tail, masks, target_valid, head_valid, args):
    u, v, euclidean, axis_valid = head_axis_components_px(pred_head, gt_head, gt_tail, args.image_width, args.image_height)
    out = {}
    for group, mask in masks.items():
        valid = mask & target_valid & head_valid & axis_valid
        out[group] = {
            "count": int(np.sum(valid)),
            "u_px": component_stats(u[valid]),
            "v_px": component_stats(v[valid]),
            "euclidean_px": stats(euclidean[valid]),
            "quality": head_axis_quality_counts(u[valid], v[valid]),
        }
    return out


def yolo_error_bucket_from_error(raw_error):
    bucket = np.full(raw_error.shape, -1, dtype=np.int64)
    finite = np.isfinite(raw_error)
    bucket[finite & (raw_error < 5.0)] = 0
    bucket[finite & (raw_error >= 5.0) & (raw_error < 10.0)] = 1
    bucket[finite & (raw_error >= 10.0) & (raw_error < 20.0)] = 2
    bucket[finite & (raw_error >= 20.0)] = 3
    return bucket


def bucket_summary(pred_bucket, true_bucket, mask):
    mask = mask & (true_bucket >= 0)
    total = int(np.sum(mask))
    if total == 0:
        return {"count": 0, "accuracy": None, "pred_counts": {}, "true_counts": {}}
    labels = ["lt5", "5_10", "10_20", "gt20"]
    pred_counts = {labels[idx]: int(np.sum(pred_bucket[mask] == idx)) for idx in range(4)}
    true_counts = {labels[idx]: int(np.sum(true_bucket[mask] == idx)) for idx in range(4)}
    return {
        "count": total,
        "accuracy": float(np.mean(pred_bucket[mask] == true_bucket[mask])),
        "pred_counts": pred_counts,
        "true_counts": true_counts,
    }


def summarize_tail(frames, args):
    gt_head = np.stack([frame["gt_head"] for frame in frames]).astype(np.float32)
    gt = np.stack([frame["gt_tail"] for frame in frames]).astype(np.float32)
    raw_head = np.stack([frame["raw"][0:2] for frame in frames]).astype(np.float32)
    filled_head = np.stack([frame["filled_head"] for frame in frames]).astype(np.float32)
    raw = np.stack([frame["raw"][2:4] for frame in frames]).astype(np.float32)
    filled = np.stack([frame["filled_tail"] for frame in frames]).astype(np.float32)
    prior = np.stack([frame["prior_tail"] for frame in frames]).astype(np.float32)
    final = np.stack([frame["final_tail"] for frame in frames]).astype(np.float32)
    trust = np.asarray([frame["trust"] for frame in frames], dtype=np.float32)
    yolo_quality = np.asarray([frame["yolo_quality"] for frame in frames], dtype=np.float32)
    pred_bucket = np.asarray([frame["yolo_error_bucket_pred"] for frame in frames], dtype=np.int64)
    delta_px = np.stack([frame["delta_px"] for frame in frames]).astype(np.float32)
    head_raw_valid = np.asarray([frame["raw"][6] > 0.0 for frame in frames], dtype=bool)
    raw_valid = np.asarray([frame["raw"][7] > 0.0 for frame in frames], dtype=bool)
    target_valid = np.asarray([frame.get("target_valid", True) for frame in frames], dtype=bool)
    obs_valid = np.asarray([frame["tail_observation_valid"] for frame in frames], dtype=bool)
    head_visibility_ids = np.asarray([frame["head_visibility_id"] for frame in frames], dtype=np.int64)
    visibility_ids = np.asarray([frame["tail_visibility_id"] for frame in frames], dtype=np.int64)

    errors = {
        "raw_yolo": point_errors_px(raw, gt, args.image_width, args.image_height),
        "filled_input": point_errors_px(filled, gt, args.image_width, args.image_height),
        "prior": point_errors_px(prior, gt, args.image_width, args.image_height),
        "final": point_errors_px(final, gt, args.image_width, args.image_height),
    }
    raw_error = errors["raw_yolo"]
    true_bucket = yolo_error_bucket_from_error(raw_error)
    filled_error = errors["filled_input"]
    prior_error = errors["prior"]
    final_error = errors["final"]
    delta_mag = np.linalg.norm(delta_px, axis=-1)
    final_shift = point_errors_px(final, filled, args.image_width, args.image_height)
    masks = {
        "all": np.ones(len(frames), dtype=bool),
        "raw_matched": raw_valid,
        "raw_missing": ~raw_valid,
        "obs_valid": obs_valid,
        "obs_missing": ~obs_valid,
    }
    raw_error_valid = raw_valid & target_valid & np.isfinite(raw_error)
    masks.update(
        {
            "raw_error_lt5": raw_error_valid & (raw_error < 5.0),
            "raw_error_5_10": raw_error_valid & (raw_error >= 5.0) & (raw_error < 10.0),
            "raw_error_10_20": raw_error_valid & (raw_error >= 10.0) & (raw_error < 20.0),
            "raw_error_gt20": raw_error_valid & (raw_error >= 20.0),
        }
    )
    for vis_id, vis_name in VIS_NAMES.items():
        masks[vis_name] = visibility_ids == vis_id

    result = {}
    for name, values in errors.items():
        result[name] = {
            group: stats(values[mask & target_valid & raw_valid] if name == "raw_yolo" else values[mask & target_valid])
            for group, mask in masks.items()
        }
    result["trust"] = {
        group: {
            "count": int(np.sum(mask)),
            "mean": float(np.mean(trust[mask])) if np.any(mask) else None,
            "p50": float(np.percentile(trust[mask], 50)) if np.any(mask) else None,
            "p90": float(np.percentile(trust[mask], 90)) if np.any(mask) else None,
        }
        for group, mask in masks.items()
    }
    result["yolo_quality"] = {
        group: {
            "count": int(np.sum(mask & target_valid)),
            "mean": float(np.mean(yolo_quality[mask & target_valid])) if np.any(mask & target_valid) else None,
            "p50": float(np.percentile(yolo_quality[mask & target_valid], 50)) if np.any(mask & target_valid) else None,
            "p90": float(np.percentile(yolo_quality[mask & target_valid], 90)) if np.any(mask & target_valid) else None,
        }
        for group, mask in masks.items()
    }
    result["yolo_error_bucket"] = {
        group: bucket_summary(pred_bucket, true_bucket, mask & target_valid & raw_valid)
        for group, mask in masks.items()
    }
    head_axis_base_masks = {
        group: mask
        for group, mask in masks.items()
        if group not in set(VIS_NAMES.values())
    }
    head_visibility_masks = {
        f"head_{vis_name}": head_visibility_ids == vis_id
        for vis_id, vis_name in VIS_NAMES.items()
    }
    tail_visibility_masks = {
        f"tail_{vis_name}": visibility_ids == vis_id
        for vis_id, vis_name in VIS_NAMES.items()
    }
    result["head_axis_error"] = {
        "description": (
            "GT head is origin. v axis is the GT head->GT tail unit vector; "
            "u axis is perpendicular to v. u_px is lateral/head-to-bat-centerline error, "
            "v_px is axial/end-direction error. Visibility groups here are split explicitly "
            "into by_head_visibility and by_tail_visibility."
        ),
        "base": {
            "raw_yolo": summarize_head_axis_error(raw_head, gt_head, gt, head_axis_base_masks, target_valid, head_raw_valid, args),
            "filled_input": summarize_head_axis_error(filled_head, gt_head, gt, head_axis_base_masks, target_valid, head_raw_valid, args),
        },
        "by_head_visibility": {
            "raw_yolo": summarize_head_axis_error(raw_head, gt_head, gt, head_visibility_masks, target_valid, head_raw_valid, args),
            "filled_input": summarize_head_axis_error(filled_head, gt_head, gt, head_visibility_masks, target_valid, head_raw_valid, args),
        },
        "by_tail_visibility": {
            "raw_yolo": summarize_head_axis_error(raw_head, gt_head, gt, tail_visibility_masks, target_valid, head_raw_valid, args),
            "filled_input": summarize_head_axis_error(filled_head, gt_head, gt, tail_visibility_masks, target_valid, head_raw_valid, args),
        },
    }
    add_correction_effect(
        result,
        masks,
        target_valid,
        raw_valid,
        raw_error,
        filled_error,
        final_error,
        final_shift,
        args,
        delta_mag=delta_mag,
    )
    add_prior_effect(
        result,
        masks,
        target_valid,
        raw_valid,
        raw_error,
        filled_error,
        prior_error,
        args,
    )
    add_trust_effect(
        result,
        masks,
        target_valid,
        raw_valid,
        raw_error,
        filled_error,
        final_error,
        trust,
        args,
    )
    result["improvement_final_vs_raw_rmse_px"] = {}
    result["improvement_final_vs_filled_rmse_px"] = {}
    result["improvement_prior_vs_raw_rmse_px"] = {}
    result["improvement_prior_vs_filled_rmse_px"] = {}
    for group in masks:
        raw_rmse = result["raw_yolo"][group]["rmse"]
        filled_rmse = result["filled_input"][group]["rmse"]
        prior_rmse = result["prior"][group]["rmse"]
        final_rmse = result["final"][group]["rmse"]
        result["improvement_final_vs_raw_rmse_px"][group] = None if raw_rmse is None or final_rmse is None else raw_rmse - final_rmse
        result["improvement_final_vs_filled_rmse_px"][group] = None if filled_rmse is None or final_rmse is None else filled_rmse - final_rmse
        result["improvement_prior_vs_raw_rmse_px"][group] = None if raw_rmse is None or prior_rmse is None else raw_rmse - prior_rmse
        result["improvement_prior_vs_filled_rmse_px"][group] = None if filled_rmse is None or prior_rmse is None else filled_rmse - prior_rmse
    return result


def evaluate(args):
    args.clip_metadata = load_clip_metadata(args.clips_csv)
    if getattr(args, "use_flow_features", False):
        sequences = load_real_yolo_details_with_flow(args.details_csv, args.image_width, args.image_height, args)
        windows = make_windows_v52_with_flow(sequences, args)
    else:
        sequences = load_real_yolo_details(args.details_csv, args.image_width, args.image_height)
        windows = make_windows_v52(sequences, args)
    device = torch.device(args.device)
    model, ckpt = load_model(args, device)
    sample_input_dim = int(windows[0]["X"].shape[1]) if windows else None
    ckpt_input_dim = int(ckpt.get("input_dim", sample_input_dim))
    if sample_input_dim != ckpt_input_dim:
        raise ValueError(
            f"Eval input_dim={sample_input_dim} but checkpoint input_dim={ckpt_input_dim}. "
            "Use --use_flow_features with flow checkpoints and refinement_*_with_flow.csv, or omit it for coord-only checkpoints."
        )
    outputs = predict_windows(model, ckpt, windows, args, device)
    frames = aggregate(windows, outputs)
    if args.output_predictions_csv:
        write_prediction_csv(frames, args.output_predictions_csv, args)
    result = summarize_tail(frames, args)
    result.update(
        {
            "details_csv": args.details_csv,
            "checkpoint": args.checkpoint,
            "prediction_mode": ckpt.get("prediction_mode") or ckpt.get("prior_mode") or "prior_guided_yolo_residual_trust",
            "prior_mode": ckpt.get("prediction_mode") or ckpt.get("prior_mode") or "prior_guided_yolo_residual_trust",
            "swings": len(sequences),
            "windows": len(windows),
            "frames_evaluated": len(frames),
            "window_size": args.window_size,
            "stride": args.stride,
            "include_first_window": args.include_first_window,
            "use_flow_features": args.use_flow_features,
            "flow_clip_px": args.flow_clip_px,
            "clips_csv": args.clips_csv,
            "fallback_view_label": canonical_view_label(args.view_label, args.batter_hand),
            "fallback_batter_hand": normalize_batter_hand(args.batter_hand),
            "canonical_batter_hand": "R",
            "left_hand_horizontal_flip_to_right": True,
        }
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate v8 prior-guided correction + trust model on real YOLO details.")
    parser.add_argument("--details_csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_predictions_csv", default=None)
    parser.add_argument("--window_size", type=int, default=31)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--include_first_window", action="store_true")
    parser.add_argument("--clips_csv", default=None, help="Optional clips_list.csv with name/view/batter_side metadata.")
    parser.add_argument("--view_label", default="random", help="Coarse camera/view label for real YOLO details when metadata is unavailable.")
    parser.add_argument("--batter_hand", choices=["R", "L", "r", "l"], default="R", help="Observed batter hand; L is flipped to canonical right-handed space.")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_width", type=float, default=1920.0)
    parser.add_argument("--image_height", type=float, default=1080.0)
    parser.add_argument("--use_flow_features", action="store_true")
    parser.add_argument("--flow_clip_px", type=float, default=50.0)
    parser.add_argument("--model", default="bigru")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--prediction_mode", default="prior_guided_yolo_residual_trust", help="Fixed to prior_guided_yolo_residual_trust for v8.")
    parser.add_argument("--prior_mode", default=None, help="Ignored backward-compatible alias.")
    parser.add_argument("--correction_shift_threshold_px", type=float, default=1.0, help="Minimum final-vs-input shift counted as a corrected frame.")
    parser.add_argument("--correction_gain_threshold_px", type=float, default=1.0, help="Minimum error reduction counted as an effective correction.")
    parser.add_argument("--prior_gain_threshold_px", type=float, default=1.0, help="Minimum prior-vs-YOLO error reduction counted as an effective prior.")
    args = parser.parse_args()
    args.prediction_mode = "prior_guided_yolo_residual_trust"
    args.prior_mode = args.prediction_mode
    return args


def main():
    args = parse_args()
    result = evaluate(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(text)


if __name__ == "__main__":
    main()
