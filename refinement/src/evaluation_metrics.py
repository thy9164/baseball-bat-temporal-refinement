"""Canonical behavior-preserving evaluation helpers extracted from src_v4.

Metric definitions, visibility names, missing-value handling, and numeric behavior
are intentionally unchanged from the historical source.
"""

import numpy as np


VIS_NAMES = {
    0: "fully_visible",
    1: "partially_occluded",
    2: "fully_occluded",
    3: "ignore",
}

def point_errors_px(pred_norm, target_norm, image_width, image_height):
    scale = np.asarray([image_width, image_height], dtype=np.float32)
    return np.linalg.norm((pred_norm - target_norm) * scale, axis=-1)


def stats(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return {"count": 0, "rmse": None, "mae": None, "p90": None, "gt20": None, "gt40": None}
    return {
        "count": int(len(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "mae": float(np.mean(np.abs(values))),
        "p90": float(np.percentile(values, 90)),
        "gt20": float(np.mean(values > 20.0)),
        "gt40": float(np.mean(values > 40.0)),
    }


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

def add_correction_effect(
    result,
    masks,
    target_valid,
    raw_valid,
    raw_error,
    filled_error,
    final_error,
    final_shift,
    args,
    delta_mag=None,
):
    gain_threshold = getattr(args, "correction_gain_threshold_px", 1.0)
    shift_threshold = getattr(args, "correction_shift_threshold_px", 1.0)
    target_valid = np.asarray(target_valid, dtype=bool)
    raw_valid = np.asarray(raw_valid, dtype=bool)
    raw_error = np.asarray(raw_error, dtype=np.float32)
    filled_error = np.asarray(filled_error, dtype=np.float32)
    final_error = np.asarray(final_error, dtype=np.float32)
    final_shift = np.asarray(final_shift, dtype=np.float32)
    delta_mag = None if delta_mag is None else np.asarray(delta_mag, dtype=np.float32)

    correction_gain_vs_raw = raw_error - final_error
    correction_gain_vs_filled = filled_error - final_error
    result["correction_effect"] = {}

    for group, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        raw_mask = mask & target_valid & raw_valid & np.isfinite(correction_gain_vs_raw)
        filled_mask = mask & target_valid & np.isfinite(correction_gain_vs_filled)

        raw_helped_mask = raw_mask & (correction_gain_vs_raw >= gain_threshold)
        raw_hurt_mask = raw_mask & (correction_gain_vs_raw <= -gain_threshold)
        raw_neutral_mask = raw_mask & (np.abs(correction_gain_vs_raw) < gain_threshold)
        raw_improved_mask = raw_mask & (correction_gain_vs_raw > 0.0)
        raw_worsened_mask = raw_mask & (correction_gain_vs_raw < 0.0)
        filled_helped_mask = filled_mask & (correction_gain_vs_filled >= gain_threshold)
        filled_hurt_mask = filled_mask & (correction_gain_vs_filled <= -gain_threshold)
        filled_neutral_mask = filled_mask & (np.abs(correction_gain_vs_filled) < gain_threshold)
        filled_improved_mask = filled_mask & (correction_gain_vs_filled > 0.0)
        filled_worsened_mask = filled_mask & (correction_gain_vs_filled < 0.0)
        moved_mask = filled_mask & np.isfinite(final_shift) & (final_shift >= shift_threshold)

        group_result = {
            "vs_raw_error_delta_px": scalar_stats(correction_gain_vs_raw[raw_mask]),
            "vs_filled_error_delta_px": scalar_stats(correction_gain_vs_filled[filled_mask]),
            "improved_rate_vs_raw": float(np.mean(raw_improved_mask[raw_mask])) if np.any(raw_mask) else None,
            "improved_count_vs_raw": int(np.sum(raw_improved_mask)),
            "worsened_rate_vs_raw": float(np.mean(raw_worsened_mask[raw_mask])) if np.any(raw_mask) else None,
            "worsened_count_vs_raw": int(np.sum(raw_worsened_mask)),
            "effective_correction_rate_vs_raw": float(np.mean(raw_helped_mask[raw_mask])) if np.any(raw_mask) else None,
            "effective_correction_count_vs_raw": int(np.sum(raw_helped_mask)),
            "hurt_rate_vs_raw": float(np.mean(raw_hurt_mask[raw_mask])) if np.any(raw_mask) else None,
            "hurt_count_vs_raw": int(np.sum(raw_hurt_mask)),
            "neutral_rate_vs_raw": float(np.mean(raw_neutral_mask[raw_mask])) if np.any(raw_mask) else None,
            "neutral_count_vs_raw": int(np.sum(raw_neutral_mask)),
            "eligible_count_vs_raw": int(np.sum(raw_mask)),
            "improved_rate_vs_filled": float(np.mean(filled_improved_mask[filled_mask])) if np.any(filled_mask) else None,
            "improved_count_vs_filled": int(np.sum(filled_improved_mask)),
            "worsened_rate_vs_filled": float(np.mean(filled_worsened_mask[filled_mask])) if np.any(filled_mask) else None,
            "worsened_count_vs_filled": int(np.sum(filled_worsened_mask)),
            "effective_correction_rate_vs_filled": float(np.mean(filled_helped_mask[filled_mask])) if np.any(filled_mask) else None,
            "effective_correction_count_vs_filled": int(np.sum(filled_helped_mask)),
            "hurt_rate_vs_filled": float(np.mean(filled_hurt_mask[filled_mask])) if np.any(filled_mask) else None,
            "hurt_count_vs_filled": int(np.sum(filled_hurt_mask)),
            "neutral_rate_vs_filled": float(np.mean(filled_neutral_mask[filled_mask])) if np.any(filled_mask) else None,
            "neutral_count_vs_filled": int(np.sum(filled_neutral_mask)),
            "eligible_count_vs_filled": int(np.sum(filled_mask)),
            "moved_frame_rate": float(np.mean(moved_mask[filled_mask])) if np.any(filled_mask) else None,
            "moved_count": int(np.sum(moved_mask)),
            "mean_shift_px": float(np.mean(final_shift[filled_mask])) if np.any(filled_mask) else None,
            "median_shift_px": float(np.percentile(final_shift[filled_mask], 50)) if np.any(filled_mask) else None,
        }
        if delta_mag is not None:
            group_result["mean_delta_mag_px"] = float(np.mean(delta_mag[filled_mask])) if np.any(filled_mask) else None
        result["correction_effect"][group] = group_result

    return result

def add_prior_effect(
    result,
    masks,
    target_valid,
    raw_valid,
    raw_error,
    filled_error,
    prior_error,
    args,
):
    gain_threshold = getattr(args, "prior_gain_threshold_px", getattr(args, "correction_gain_threshold_px", 1.0))
    target_valid = np.asarray(target_valid, dtype=bool)
    raw_valid = np.asarray(raw_valid, dtype=bool)
    raw_error = np.asarray(raw_error, dtype=np.float32)
    filled_error = np.asarray(filled_error, dtype=np.float32)
    prior_error = np.asarray(prior_error, dtype=np.float32)

    prior_gain_vs_raw = raw_error - prior_error
    prior_gain_vs_filled = filled_error - prior_error
    result["prior_effect"] = {}

    for group, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        raw_mask = mask & target_valid & raw_valid & np.isfinite(prior_gain_vs_raw)
        filled_mask = mask & target_valid & np.isfinite(prior_gain_vs_filled)

        raw_helped_mask = raw_mask & (prior_gain_vs_raw >= gain_threshold)
        raw_hurt_mask = raw_mask & (prior_gain_vs_raw <= -gain_threshold)
        raw_neutral_mask = raw_mask & (np.abs(prior_gain_vs_raw) < gain_threshold)
        raw_improved_mask = raw_mask & (prior_gain_vs_raw > 0.0)
        raw_worsened_mask = raw_mask & (prior_gain_vs_raw < 0.0)
        filled_helped_mask = filled_mask & (prior_gain_vs_filled >= gain_threshold)
        filled_hurt_mask = filled_mask & (prior_gain_vs_filled <= -gain_threshold)
        filled_neutral_mask = filled_mask & (np.abs(prior_gain_vs_filled) < gain_threshold)
        filled_improved_mask = filled_mask & (prior_gain_vs_filled > 0.0)
        filled_worsened_mask = filled_mask & (prior_gain_vs_filled < 0.0)

        result["prior_effect"][group] = {
            "vs_raw_error_delta_px": scalar_stats(prior_gain_vs_raw[raw_mask]),
            "vs_filled_error_delta_px": scalar_stats(prior_gain_vs_filled[filled_mask]),
            "prior_minus_raw_error_px": scalar_stats((-prior_gain_vs_raw)[raw_mask]),
            "prior_minus_filled_error_px": scalar_stats((-prior_gain_vs_filled)[filled_mask]),
            "improved_rate_vs_raw": float(np.mean(raw_improved_mask[raw_mask])) if np.any(raw_mask) else None,
            "improved_count_vs_raw": int(np.sum(raw_improved_mask)),
            "worsened_rate_vs_raw": float(np.mean(raw_worsened_mask[raw_mask])) if np.any(raw_mask) else None,
            "worsened_count_vs_raw": int(np.sum(raw_worsened_mask)),
            "effective_prior_rate_vs_raw": float(np.mean(raw_helped_mask[raw_mask])) if np.any(raw_mask) else None,
            "effective_prior_count_vs_raw": int(np.sum(raw_helped_mask)),
            "hurt_rate_vs_raw": float(np.mean(raw_hurt_mask[raw_mask])) if np.any(raw_mask) else None,
            "hurt_count_vs_raw": int(np.sum(raw_hurt_mask)),
            "neutral_rate_vs_raw": float(np.mean(raw_neutral_mask[raw_mask])) if np.any(raw_mask) else None,
            "neutral_count_vs_raw": int(np.sum(raw_neutral_mask)),
            "eligible_count_vs_raw": int(np.sum(raw_mask)),
            "improved_rate_vs_filled": float(np.mean(filled_improved_mask[filled_mask])) if np.any(filled_mask) else None,
            "improved_count_vs_filled": int(np.sum(filled_improved_mask)),
            "worsened_rate_vs_filled": float(np.mean(filled_worsened_mask[filled_mask])) if np.any(filled_mask) else None,
            "worsened_count_vs_filled": int(np.sum(filled_worsened_mask)),
            "effective_prior_rate_vs_filled": float(np.mean(filled_helped_mask[filled_mask])) if np.any(filled_mask) else None,
            "effective_prior_count_vs_filled": int(np.sum(filled_helped_mask)),
            "hurt_rate_vs_filled": float(np.mean(filled_hurt_mask[filled_mask])) if np.any(filled_mask) else None,
            "hurt_count_vs_filled": int(np.sum(filled_hurt_mask)),
            "neutral_rate_vs_filled": float(np.mean(filled_neutral_mask[filled_mask])) if np.any(filled_mask) else None,
            "neutral_count_vs_filled": int(np.sum(filled_neutral_mask)),
            "eligible_count_vs_filled": int(np.sum(filled_mask)),
        }

    return result

def _trust_bucket_summary(gain, trust, eligible_mask, threshold):
    gain = np.asarray(gain, dtype=np.float32)
    trust = np.asarray(trust, dtype=np.float32)
    eligible_mask = np.asarray(eligible_mask, dtype=bool) & np.isfinite(gain) & np.isfinite(trust)
    edges = (0.0, 0.2, 0.4, 0.6, 0.8, 1.000001)
    buckets = {}
    total = int(np.sum(eligible_mask))
    for start, end in zip(edges[:-1], edges[1:]):
        label = f"{start:.1f}_{min(end, 1.0):.1f}"
        bucket_mask = eligible_mask & (trust >= start) & (trust < end)
        bucket_gain = gain[bucket_mask]
        helped = bucket_gain >= threshold
        hurt = bucket_gain <= -threshold
        neutral = np.abs(bucket_gain) < threshold
        improved = bucket_gain > 0.0
        worsened = bucket_gain < 0.0
        buckets[label] = {
            "count": int(np.sum(bucket_mask)),
            "frame_rate": float(np.sum(bucket_mask) / total) if total else None,
            "trust_mean": float(np.mean(trust[bucket_mask])) if np.any(bucket_mask) else None,
            "error_delta_px": scalar_stats(bucket_gain),
            "improved_rate": float(np.mean(improved)) if len(bucket_gain) else None,
            "improved_count": int(np.sum(improved)),
            "worsened_rate": float(np.mean(worsened)) if len(bucket_gain) else None,
            "worsened_count": int(np.sum(worsened)),
            "effective_rate": float(np.mean(helped)) if len(bucket_gain) else None,
            "effective_count": int(np.sum(helped)),
            "hurt_rate": float(np.mean(hurt)) if len(bucket_gain) else None,
            "hurt_count": int(np.sum(hurt)),
            "neutral_rate": float(np.mean(neutral)) if len(bucket_gain) else None,
            "neutral_count": int(np.sum(neutral)),
        }
    return {"eligible_count": total, "buckets": buckets}

def add_trust_effect(
    result,
    masks,
    target_valid,
    raw_valid,
    raw_error,
    filled_error,
    final_error,
    trust,
    args,
):
    threshold = getattr(args, "correction_gain_threshold_px", 1.0)
    target_valid = np.asarray(target_valid, dtype=bool)
    raw_valid = np.asarray(raw_valid, dtype=bool)
    raw_error = np.asarray(raw_error, dtype=np.float32)
    filled_error = np.asarray(filled_error, dtype=np.float32)
    final_error = np.asarray(final_error, dtype=np.float32)
    trust = np.asarray(trust, dtype=np.float32)

    gain_vs_raw = raw_error - final_error
    gain_vs_filled = filled_error - final_error
    result["trust_effect"] = {}

    for group, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        raw_mask = mask & target_valid & raw_valid
        filled_mask = mask & target_valid
        result["trust_effect"][group] = {
            "vs_raw": _trust_bucket_summary(gain_vs_raw, trust, raw_mask, threshold),
            "vs_filled": _trust_bucket_summary(gain_vs_filled, trust, filled_mask, threshold),
        }

    return result
