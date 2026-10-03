import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean_input.checkpoint import checkpoint_contract, strict_load_model_state, validate_checkpoint_contract
from clean_input.schema import CLEAN_NO_FLOW_SCHEMA

try:
    from src.models import build_model
except ImportError:
    from models import build_model


VIS_FULLY_VISIBLE = 0
VIS_PARTIAL = 1
VIS_FULLY_OCCLUDED = 2
VIS_IGNORE = 3


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset_provenance(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


class PreloadedTailDataset(Dataset):
    def __init__(self, dataset_path, split):
        path = Path(dataset_path)
        keys = ("X", "y_tail", "y_head", "tail_weight", "tail_visibility", "tail_obs_valid")
        if path.is_dir():
            files = sorted((path / split).glob("*.npz"))
            if not files:
                raise FileNotFoundError(f"No NPZ shards found under {path / split}")
            arrays = defaultdict(list)
            for file_path in files:
                with np.load(file_path) as data:
                    for key in keys:
                        arrays[key].append(data[key])
            self.arrays = {
                key: np.concatenate(arrays[key], axis=0).astype(np.float32 if key != "tail_visibility" else np.int64)
                for key in keys
            }
        else:
            with np.load(path) as data:
                self.arrays = {
                    key: data[f"{key}_{split}"].astype(np.float32 if key != "tail_visibility" else np.int64)
                    for key in keys
                }

    def __len__(self):
        return len(self.arrays["X"])

    def __getitem__(self, idx):
        return tuple(
            torch.from_numpy(self.arrays[key][idx])
            for key in ("X", "y_tail", "y_head", "tail_weight", "tail_visibility", "tail_obs_valid")
        )


class ShardedTailDataset(Dataset):
    def __init__(self, root_dir, split):
        self.files = sorted((Path(root_dir) / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No NPZ shards found under {Path(root_dir) / split}")
        self.lengths = []
        for path in self.files:
            with np.load(path) as data:
                self.lengths.append(len(data["X"]))
        self.cumulative = np.cumsum(self.lengths)
        self.cache_path = None
        self.cache = None

    def __len__(self):
        return int(self.cumulative[-1])

    def load_file(self, file_idx):
        path = self.files[file_idx]
        if self.cache_path != path:
            self.cache = np.load(path)
            self.cache_path = path
        return self.cache

    def __getitem__(self, idx):
        file_idx = int(np.searchsorted(self.cumulative, idx, side="right"))
        prev = 0 if file_idx == 0 else int(self.cumulative[file_idx - 1])
        local_idx = idx - prev
        data = self.load_file(file_idx)
        return (
            torch.from_numpy(data["X"][local_idx].astype(np.float32)),
            torch.from_numpy(data["y_tail"][local_idx].astype(np.float32)),
            torch.from_numpy(data["y_head"][local_idx].astype(np.float32)),
            torch.from_numpy(data["tail_weight"][local_idx].astype(np.float32)),
            torch.from_numpy(data["tail_visibility"][local_idx].astype(np.int64)),
            torch.from_numpy(data["tail_obs_valid"][local_idx].astype(np.float32)),
        )


def make_dataset(dataset_path, split, preload=True):
    if preload:
        return PreloadedTailDataset(dataset_path, split)
    return ShardedTailDataset(dataset_path, split)


def coordinate_scale(args, device):
    return torch.tensor([args.image_width, args.image_height], dtype=torch.float32, device=device)


def weighted_smooth_l1(pred, target, weight):
    loss = F.smooth_l1_loss(pred, target, reduction="none").sum(dim=-1)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def decode_prediction(model, x, args, scale):
    raw = model(x)
    prior_tail = raw[:, :, :2]
    delta_px = raw[:, :, 2:4]
    trust_logit = raw[:, :, 4]
    yolo_error_bucket_logits = raw[:, :, 5:9]
    yolo_quality_logit = raw[:, :, 9]
    trust = torch.sigmoid(trust_logit).unsqueeze(-1)
    final_tail = x[:, :, 2:4] + trust * delta_px / scale
    corrected_tail = x[:, :, 2:4] + delta_px / scale
    return (
        final_tail,
        corrected_tail,
        prior_tail,
        delta_px,
        trust_logit,
        trust,
        yolo_error_bucket_logits,
        yolo_quality_logit,
        raw,
    )


def prediction_mode(args):
    return "prior_guided_yolo_residual_trust"


def refinement_frame_weight(obs_px, target_px, tail_visibility, tail_obs_valid, args, target_valid=None):
    obs_error = torch.linalg.norm(obs_px - target_px, dim=-1)
    weight = torch.where(
        obs_error < args.error_weight_5_px,
        torch.full_like(obs_error, args.weight_error_lt5),
        torch.where(
            obs_error < args.error_weight_10_px,
            torch.full_like(obs_error, args.weight_error_5_10),
            torch.where(
                obs_error < args.error_weight_20_px,
                torch.full_like(obs_error, args.weight_error_10_20),
                torch.full_like(obs_error, args.weight_error_gt20),
            ),
        ),
    )
    occluded = tail_visibility != VIS_FULLY_VISIBLE
    missing = tail_obs_valid <= 0.0
    weight = torch.where(occluded | missing, torch.full_like(weight, args.weight_missing_occluded), weight)
    if target_valid is not None:
        weight = weight * target_valid.to(dtype=weight.dtype)
    return weight, obs_error


def weighted_bce_with_logits(logits, target, weight):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def weighted_scalar_smooth_l1(pred, target, weight):
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def yolo_error_bucket_target(obs_error, args):
    return torch.where(
        obs_error < args.error_weight_5_px,
        torch.zeros_like(obs_error, dtype=torch.long),
        torch.where(
            obs_error < args.error_weight_10_px,
            torch.ones_like(obs_error, dtype=torch.long),
            torch.where(
                obs_error < args.error_weight_20_px,
                torch.full_like(obs_error, 2, dtype=torch.long),
                torch.full_like(obs_error, 3, dtype=torch.long),
            ),
        ),
    )


def yolo_quality_target(obs_error, args):
    span = max(args.yolo_quality_bad_error_px - args.yolo_quality_good_error_px, 1e-6)
    return ((args.yolo_quality_bad_error_px - obs_error) / span).clamp(0.0, 1.0)


def weighted_bucket_ce(logits, target, weight):
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
    weight = weight.reshape(-1)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def compute_loss(model, x, y_tail, y_head, tail_weight, tail_visibility, tail_obs_valid, args, scale, target_valid=None):
    valid_mask = None
    if target_valid is not None:
        target_valid = target_valid.to(dtype=tail_weight.dtype)
        valid_mask = target_valid.unsqueeze(-1) > 0.0
        y_tail = torch.where(valid_mask, y_tail, x[:, :, 2:4])
        y_head = torch.where(valid_mask, y_head, x[:, :, 0:2])
    (
        final_tail,
        corrected_tail,
        prior_tail,
        delta_px,
        trust_logit,
        trust,
        yolo_error_bucket_logits,
        yolo_quality_logit,
        raw,
    ) = decode_prediction(model, x, args, scale)
    final_px = final_tail * scale
    corrected_px = corrected_tail * scale
    prior_px = prior_tail * scale
    obs_px = x[:, :, 2:4] * scale
    target_px = y_tail * scale
    head_px = y_head * scale
    frame_weight, obs_error = refinement_frame_weight(
        obs_px, target_px, tail_visibility, tail_obs_valid, args, target_valid=target_valid
    )

    position_loss = weighted_smooth_l1(final_px, target_px, frame_weight)
    prior_loss = weighted_smooth_l1(prior_px, target_px, frame_weight)
    pred_length = torch.linalg.norm(final_px - head_px, dim=-1)
    target_length = torch.linalg.norm(target_px - head_px, dim=-1)
    length_loss = weighted_scalar_smooth_l1(pred_length, target_length, frame_weight)
    correction_target_px = target_px - obs_px
    correction_loss = weighted_smooth_l1(delta_px, correction_target_px, frame_weight)
    prior_guidance_loss = weighted_smooth_l1(delta_px, (prior_px.detach() - obs_px), frame_weight)
    bucket_target = yolo_error_bucket_target(obs_error.detach(), args)
    bucket_loss = weighted_bucket_ce(yolo_error_bucket_logits, bucket_target, frame_weight)
    quality_loss = weighted_bce_with_logits(
        yolo_quality_logit,
        yolo_quality_target(obs_error.detach(), args),
        frame_weight,
    )

    pred_delta = delta_px.detach()
    target_delta = correction_target_px.detach()
    pred_delta_norm_sq = (pred_delta ** 2).sum(dim=-1)
    best_trust = (target_delta * pred_delta).sum(dim=-1) / pred_delta_norm_sq.clamp_min(args.trust_target_eps)
    best_trust = best_trust.clamp(0.0, 1.0)
    best_final_px = obs_px + best_trust.unsqueeze(-1) * pred_delta
    best_final_error = torch.linalg.norm(best_final_px - target_px, dim=-1)
    best_gain = obs_error.detach() - best_final_error
    gain_range = max(args.trust_good_gain_px - args.trust_min_gain_px, args.trust_target_eps)
    trust_quality = ((best_gain - args.trust_min_gain_px) / gain_range).clamp(0.0, 1.0)
    trust_target = best_trust * trust_quality
    trust_target = torch.where(
        pred_delta_norm_sq >= args.trust_min_delta_px ** 2,
        trust_target,
        torch.zeros_like(trust_target),
    )
    if target_valid is not None:
        trust_weight = frame_weight * target_valid
    else:
        trust_weight = frame_weight
    trust_loss = weighted_bce_with_logits(trust_logit, trust_target, trust_weight)

    pred_vec = final_px - head_px
    target_vec = target_px - head_px
    pred_unit = F.normalize(pred_vec, p=2, dim=-1, eps=1e-6)
    target_unit = F.normalize(target_vec, p=2, dim=-1, eps=1e-6)
    direction_per_frame = 1.0 - (pred_unit * target_unit).sum(dim=-1)
    direction_loss = (direction_per_frame * frame_weight).sum() / frame_weight.sum().clamp_min(1.0)

    rel_to_gt_tail = final_px - target_px
    axis_offset = (rel_to_gt_tail * target_unit).sum(dim=-1)
    perpendicular = rel_to_gt_tail - axis_offset.unsqueeze(-1) * target_unit
    perpendicular_distance = torch.linalg.norm(perpendicular, dim=-1)
    line_excess = F.relu(perpendicular_distance - args.line_delta_px)
    end_excess = F.relu(axis_offset.abs() - args.end_delta_px)
    line_loss = ((line_excess ** 2) * frame_weight).sum() / frame_weight.sum().clamp_min(1.0)
    end_loss = ((end_excess ** 2) * frame_weight).sum() / frame_weight.sum().clamp_min(1.0)

    pred_velocity = final_px[:, 1:, :] - final_px[:, :-1, :]
    prior_velocity = prior_px[:, 1:, :] - prior_px[:, :-1, :]
    target_velocity = target_px[:, 1:, :] - target_px[:, :-1, :]
    velocity_weight = 0.5 * (frame_weight[:, 1:] + frame_weight[:, :-1])
    if target_valid is not None:
        velocity_weight = velocity_weight * target_valid[:, 1:] * target_valid[:, :-1]
    velocity_loss = weighted_smooth_l1(pred_velocity, target_velocity, velocity_weight)
    prior_velocity_loss = weighted_smooth_l1(prior_velocity, target_velocity, velocity_weight)

    pred_acc = final_px[:, 2:, :] - 2.0 * final_px[:, 1:-1, :] + final_px[:, :-2, :]
    prior_acc = prior_px[:, 2:, :] - 2.0 * prior_px[:, 1:-1, :] + prior_px[:, :-2, :]
    target_acc = target_px[:, 2:, :] - 2.0 * target_px[:, 1:-1, :] + target_px[:, :-2, :]
    acc_weight = (frame_weight[:, 2:] + frame_weight[:, 1:-1] + frame_weight[:, :-2]) / 3.0
    if target_valid is not None:
        acc_weight = acc_weight * target_valid[:, 2:] * target_valid[:, 1:-1] * target_valid[:, :-2]
    acceleration_loss = weighted_smooth_l1(pred_acc, target_acc, acc_weight)
    prior_acceleration_loss = weighted_smooth_l1(prior_acc, target_acc, acc_weight)

    obs_weight = tail_obs_valid * x[:, :, 5].clamp(0.0, 1.0)
    if target_valid is not None:
        obs_weight = obs_weight * target_valid
    observation_loss = weighted_smooth_l1(final_px, x[:, :, 2:4] * scale, obs_weight)

    damage_weight = (tail_obs_valid > 0.0).to(delta_px.dtype)
    if target_valid is not None:
        damage_weight = damage_weight * target_valid
    final_error = torch.linalg.norm(final_px - target_px, dim=-1)
    delta_damage = F.relu(final_error - obs_error - args.delta_damage_margin_px)
    delta_loss = ((delta_damage ** 2) * damage_weight).sum() / damage_weight.sum().clamp_min(1.0)
    good_yolo_damage_mask = (
        (tail_obs_valid > 0.0)
        & (obs_error.detach() < args.good_yolo_damage_error_px)
    ).to(delta_px.dtype)
    if target_valid is not None:
        good_yolo_damage_mask = good_yolo_damage_mask * target_valid
    good_yolo_damage = F.relu(final_error - obs_error - args.good_yolo_damage_margin_px)
    good_yolo_damage_loss = (
        (good_yolo_damage ** 2) * good_yolo_damage_mask
    ).sum() / good_yolo_damage_mask.sum().clamp_min(1.0)

    total = (
        position_loss
        + args.lambda_prior * prior_loss
        + args.lambda_correction * correction_loss
        + args.lambda_prior_guidance * prior_guidance_loss
        + args.lambda_length * length_loss
        + args.lambda_yolo_error_bucket * bucket_loss
        + args.lambda_yolo_quality * quality_loss
        + args.lambda_trust * trust_loss
        + args.lambda_dir * direction_loss
        + args.lambda_velocity * velocity_loss
        + args.lambda_prior_velocity * prior_velocity_loss
        + args.lambda_acceleration * acceleration_loss
        + args.lambda_prior_acceleration * prior_acceleration_loss
        + args.lambda_observation * observation_loss
        + args.lambda_line * line_loss
        + args.lambda_end * end_loss
        + args.lambda_delta * delta_loss
        + args.lambda_good_yolo_damage * good_yolo_damage_loss
    )
    return total, {
        "total": total.detach(),
        "position": position_loss.detach(),
        "prior": prior_loss.detach(),
        "correction": correction_loss.detach(),
        "prior_guidance": prior_guidance_loss.detach(),
        "length": length_loss.detach(),
        "yolo_error_bucket": bucket_loss.detach(),
        "yolo_quality": quality_loss.detach(),
        "trust": trust_loss.detach(),
        "direction": direction_loss.detach(),
        "line": line_loss.detach(),
        "end": end_loss.detach(),
        "velocity": velocity_loss.detach(),
        "prior_velocity": prior_velocity_loss.detach(),
        "acceleration": acceleration_loss.detach(),
        "prior_acceleration": prior_acceleration_loss.detach(),
        "observation": observation_loss.detach(),
        "delta": delta_loss.detach(),
        "good_yolo_damage": good_yolo_damage_loss.detach(),
        "trust_mean": trust.detach().mean(),
        "frame_weight": frame_weight.detach().mean(),
    }


def update_running(running, components, batch_size):
    for key, value in components.items():
        running[key] += float(value.item()) * batch_size
    running["count"] += batch_size


def finish_running(running):
    count = max(running.pop("count"), 1)
    return {key: value / count for key, value in running.items()}


def run_epoch(model, loader, optimizer, device, args, scale):
    model.train()
    running = defaultdict(float)
    running["count"] = 0
    for x, y_tail, y_head, tail_weight, tail_visibility, tail_obs_valid in loader:
        x = x.to(device)
        y_tail = y_tail.to(device)
        y_head = y_head.to(device)
        tail_weight = tail_weight.to(device)
        tail_visibility = tail_visibility.to(device)
        tail_obs_valid = tail_obs_valid.to(device)
        loss, components = compute_loss(model, x, y_tail, y_head, tail_weight, tail_visibility, tail_obs_valid, args, scale)
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
    for x, y_tail, y_head, tail_weight, tail_visibility, tail_obs_valid in loader:
        x = x.to(device)
        y_tail = y_tail.to(device)
        y_head = y_head.to(device)
        tail_weight = tail_weight.to(device)
        tail_visibility = tail_visibility.to(device)
        tail_obs_valid = tail_obs_valid.to(device)
        _loss, components = compute_loss(model, x, y_tail, y_head, tail_weight, tail_visibility, tail_obs_valid, args, scale)
        update_running(running, components, len(x))
    return finish_running(running)


def parse_args():
    parser = argparse.ArgumentParser(description="Train v8 prior-guided YOLO residual correction + trust tail refiner.")
    parser.add_argument("--dataset", required=True, help="v5.3/v7/v8 NPZ file or sharded NPZ directory.")
    parser.add_argument("--model", choices=["bigru", "gru"], default="bigru")
    parser.add_argument("--prediction_mode", default="prior_guided_yolo_residual_trust", help="Fixed to prior_guided_yolo_residual_trust for v8.")
    parser.add_argument("--prior_mode", default=None, help="Ignored backward-compatible alias.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_preload", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume_checkpoint", default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--disable_cudnn", action="store_true", help="Disable cuDNN RNN kernels for CUDA stability.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset_summary", default=None, help="Builder summary JSON recorded in checkpoints.")
    parser.add_argument("--image_width", type=float, default=1920.0)
    parser.add_argument("--image_height", type=float, default=1080.0)
    parser.add_argument("--lambda_dir", type=float, default=1.0)
    parser.add_argument("--lambda_velocity", type=float, default=0.05)
    parser.add_argument("--lambda_acceleration", type=float, default=0.01)
    parser.add_argument("--lambda_observation", type=float, default=0.05)
    parser.add_argument("--lambda_line", type=float, default=0.05)
    parser.add_argument("--lambda_end", type=float, default=0.02)
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
    parser.add_argument("--line_delta_px", type=float, default=3.0)
    parser.add_argument("--end_delta_px", type=float, default=10.0)
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


def metric_text(metrics, prefix):
    return (
        f"{prefix}_total={metrics['total']:.6f} {prefix}_pos={metrics['position']:.6f} "
        f"{prefix}_prior={metrics['prior']:.6f} "
        f"{prefix}_corr={metrics['correction']:.6f} {prefix}_trust={metrics['trust']:.6f} "
        f"{prefix}_prior_guidance={metrics['prior_guidance']:.6f} "
        f"{prefix}_len={metrics['length']:.6f} "
        f"{prefix}_bucket={metrics['yolo_error_bucket']:.6f} {prefix}_quality={metrics['yolo_quality']:.6f} "
        f"{prefix}_trust_mean={metrics['trust_mean']:.4f} {prefix}_w={metrics['frame_weight']:.4f} "
        f"{prefix}_dir={metrics['direction']:.6f} "
        f"{prefix}_line={metrics['line']:.6f} {prefix}_end={metrics['end']:.6f} "
        f"{prefix}_vel={metrics['velocity']:.6f} {prefix}_prior_vel={metrics['prior_velocity']:.6f} "
        f"{prefix}_acc={metrics['acceleration']:.6f} {prefix}_prior_acc={metrics['prior_acceleration']:.6f} "
        f"{prefix}_obs={metrics['observation']:.6f} {prefix}_delta={metrics['delta']:.6f} "
        f"{prefix}_good_yolo_damage={metrics['good_yolo_damage']:.6f}"
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("[cudnn] disabled")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = make_dataset(args.dataset, "train", preload=not args.no_preload)
    val_ds = make_dataset(args.dataset, "val", preload=not args.no_preload)
    sample_x, sample_y_tail, _sample_y_head, _sample_weight, _sample_vis, _sample_valid = train_ds[0]
    window_size, input_dim = sample_x.shape
    output_dim = 10
    split_identifier = f"synthetic:{Path(args.dataset).name}"
    dataset_provenance = load_dataset_provenance(args.dataset_summary)
    if input_dim != len(CLEAN_NO_FLOW_SCHEMA):
        raise ValueError(f"clean synthetic dataset must have {len(CLEAN_NO_FLOW_SCHEMA)} features, got {input_dim}")

    print(f"train_ds={len(train_ds)} val_ds={len(val_ds)} window_size={window_size} input_dim={input_dim} output_dim={output_dim}")
    print(f"prediction_mode={args.prediction_mode} image_width={args.image_width:g} image_height={args.image_height:g}")
    print(
        "loss_weights "
        f"lambda_prior={args.lambda_prior:g} "
        f"lambda_correction={args.lambda_correction:g} lambda_trust={args.lambda_trust:g} "
        f"lambda_prior_guidance={args.lambda_prior_guidance:g} "
        f"lambda_dir={args.lambda_dir:g} lambda_velocity={args.lambda_velocity:g} "
        f"lambda_prior_velocity={args.lambda_prior_velocity:g} "
        f"lambda_acceleration={args.lambda_acceleration:g} lambda_prior_acceleration={args.lambda_prior_acceleration:g} "
        f"lambda_observation={args.lambda_observation:g} "
        f"lambda_line={args.lambda_line:g} lambda_end={args.lambda_end:g} lambda_delta={args.lambda_delta:g} "
        f"lambda_good_yolo_damage={args.lambda_good_yolo_damage:g} "
        f"lambda_length={args.lambda_length:g} "
        f"lambda_yolo_error_bucket={args.lambda_yolo_error_bucket:g} lambda_yolo_quality={args.lambda_yolo_quality:g}"
    )
    print(
        "dynamic_frame_weights "
        f"<5={args.weight_error_lt5:g} 5-10={args.weight_error_5_10:g} "
        f"10-20={args.weight_error_10_20:g} >20={args.weight_error_gt20:g} "
        f"missing_or_occluded={args.weight_missing_occluded:g}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device)
    scale = coordinate_scale(args, device)
    model = build_model(
        args.model,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=True,
        output_dim=output_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = 0
    history = []
    start_epoch = 1
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        validate_checkpoint_contract(
            ckpt,
            use_flow_features=False,
            expected_split_identifier=split_identifier,
            important_settings={"window_size": window_size, "image_width": args.image_width, "image_height": args.image_height},
        )
        strict_load_model_state(model, ckpt)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        val_metrics = ckpt.get("val_metrics") or {}
        best_val = float(val_metrics.get("total", best_val))
        best_epoch = int(ckpt.get("epoch", 0))
        history_path = out_dir / "history.json"
        if history_path.exists():
            with open(history_path, encoding="utf-8") as fp:
                history = json.load(fp)
        print(
            f"[resume] checkpoint={args.resume_checkpoint} "
            f"start_epoch={start_epoch} best_epoch={best_epoch} best_val={best_val:.6f}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, scale)
        val_metrics = evaluate_loss(model, val_loader, device, args, scale)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(f"epoch={epoch:03d} {metric_text(train_metrics, 'train')} {metric_text(val_metrics, 'val')}")

        latest_ckpt = {
            "model_state": model.state_dict(),
            **checkpoint_contract(
                use_flow_features=False,
                split_identifier=split_identifier,
                important_settings={
                    "window_size": window_size,
                    "image_width": args.image_width,
                    "image_height": args.image_height,
                    "stability_gate": False,
                },
            ),
            "optimizer_state": optimizer.state_dict(),
            "model_type": args.model,
            "task": "prior_guided_tail_residual_trust_refiner_v8",
            "prediction_mode": args.prediction_mode,
            "prior_mode": args.prediction_mode,
            "window_size": window_size,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "loss_config": vars(args),
            "seed": args.seed,
            "dataset_provenance": dataset_provenance,
            "provenance_improvement": "Python, NumPy, Torch, CUDA, and DataLoader seeds explicitly fixed to 7; legacy trainer did not fully record/fix them.",
            "epoch": epoch,
            "val_metrics": val_metrics,
            "best_epoch": best_epoch,
            "best_val": best_val,
        }
        torch.save(latest_ckpt, out_dir / "latest.pt")
        with open(out_dir / "history.json", "w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_epoch = epoch
            ckpt = {
                "model_state": model.state_dict(),
            **checkpoint_contract(
                use_flow_features=False,
                split_identifier=split_identifier,
                important_settings={
                    "window_size": window_size,
                    "image_width": args.image_width,
                    "image_height": args.image_height,
                    "stability_gate": False,
                },
            ),
                "optimizer_state": optimizer.state_dict(),
                "model_type": args.model,
                "task": "prior_guided_tail_residual_trust_refiner_v8",
                "prediction_mode": args.prediction_mode,
                "prior_mode": args.prediction_mode,
                "window_size": window_size,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "image_width": args.image_width,
                "image_height": args.image_height,
                "loss_config": vars(args),
                "seed": args.seed,
                "dataset_provenance": dataset_provenance,
                "provenance_improvement": "Python, NumPy, Torch, CUDA, and DataLoader seeds explicitly fixed to 7; legacy trainer did not fully record/fix them.",
                "epoch": epoch,
                "val_metrics": val_metrics,
                "best_epoch": best_epoch,
                "best_val": best_val,
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
