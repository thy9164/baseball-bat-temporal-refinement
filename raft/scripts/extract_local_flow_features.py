import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def frame_to_idx(frame_name: str) -> int:
    """
    Supports:
    000001.jpg
    swing_042_000001.jpg
    """
    stem = Path(str(frame_name)).stem
    nums = re.findall(r"\d+", stem)
    if not nums:
        raise ValueError(f"Cannot parse frame index from: {frame_name}")
    return int(nums[-1])


def safe_patch(flow: np.ndarray, x: float, y: float, radius: int):
    """
    flow shape: [2, H, W]
    x, y are pixel coordinates.
    Return patch flow [2, h, w], or None if invalid.
    """
    if pd.isna(x) or pd.isna(y):
        return None

    _, h, w = flow.shape
    cx = int(round(float(x)))
    cy = int(round(float(y)))

    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return None

    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius + 1)
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius + 1)

    if x2 <= x1 or y2 <= y1:
        return None

    return flow[:, y1:y2, x1:x2]


def safe_bbox_patch(flow: np.ndarray, x1, y1, x2, y2, pad: int = 0):
    if pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2):
        return None

    _, h, w = flow.shape

    x1 = int(round(float(x1))) - pad
    y1 = int(round(float(y1))) - pad
    x2 = int(round(float(x2))) + pad
    y2 = int(round(float(y2))) + pad

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return flow[:, y1:y2, x1:x2]


def stats_from_patch(patch, prefix: str):
    if patch is None:
        return {
            f"{prefix}_valid": 0,
            f"{prefix}_flow_x_mean": 0.0,
            f"{prefix}_flow_y_mean": 0.0,
            f"{prefix}_flow_x_std": 0.0,
            f"{prefix}_flow_y_std": 0.0,
            f"{prefix}_flow_mag_mean": 0.0,
            f"{prefix}_flow_mag_std": 0.0,
            f"{prefix}_flow_mag_max": 0.0,
            f"{prefix}_flow_mag_p90": 0.0,
            f"{prefix}_flow_mag_p95": 0.0,
            f"{prefix}_flow_mag_p99": 0.0,
        }

    fx = patch[0]
    fy = patch[1]
    mag = np.sqrt(fx * fx + fy * fy)

    return {
        f"{prefix}_valid": 1,
        f"{prefix}_flow_x_mean": float(np.mean(fx)),
        f"{prefix}_flow_y_mean": float(np.mean(fy)),
        f"{prefix}_flow_x_std": float(np.std(fx)),
        f"{prefix}_flow_y_std": float(np.std(fy)),
        f"{prefix}_flow_mag_mean": float(np.mean(mag)),
        f"{prefix}_flow_mag_std": float(np.std(mag)),
        f"{prefix}_flow_mag_max": float(np.max(mag)),
        f"{prefix}_flow_mag_p90": float(np.percentile(mag, 90)),
        f"{prefix}_flow_mag_p95": float(np.percentile(mag, 95)),
        f"{prefix}_flow_mag_p99": float(np.percentile(mag, 99)),
    }


def zero_all_stats():
    out = {}
    for prefix in ["head", "tail", "bbox"]:
        out.update(stats_from_patch(None, prefix))
    return out


def build_frame_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long format:
    one row per keypoint
    into one row per frame.
    """
    required = [
        "image_stem", "swing_id", "frame", "keypoint",
        "pred_x", "pred_y", "pred_conf",
        "bbox_conf", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rows = []

    group_cols = ["image_stem", "swing_id", "frame"]

    for (image_stem, swing_id, frame), g in df.groupby(group_cols, sort=False):
        row = {
            "image_stem": image_stem,
            "swing_id": swing_id,
            "frame": frame,
            "frame_idx": frame_to_idx(frame),
        }

        # bbox should be same for head/tail rows, so take first valid row
        first = g.iloc[0]
        row.update({
            "bbox_conf": first["bbox_conf"],
            "bbox_x1": first["bbox_x1"],
            "bbox_y1": first["bbox_y1"],
            "bbox_x2": first["bbox_x2"],
            "bbox_y2": first["bbox_y2"],
        })

        for kp in ["head", "tail"]:
            kg = g[g["keypoint"] == kp]
            if len(kg) == 0:
                row[f"{kp}_pred_x"] = np.nan
                row[f"{kp}_pred_y"] = np.nan
                row[f"{kp}_pred_conf"] = np.nan
            else:
                krow = kg.iloc[0]
                row[f"{kp}_pred_x"] = krow["pred_x"]
                row[f"{kp}_pred_y"] = krow["pred_y"]
                row[f"{kp}_pred_conf"] = krow["pred_conf"]

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-csv", type=str, required=True)
    parser.add_argument("--flows-root", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--bbox-pad", type=int, default=5)
    args = parser.parse_args()

    yolo_csv = Path(args.yolo_csv)
    flows_root = Path(args.flows_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(yolo_csv)
    frame_df = build_frame_table(df)

    rows = []

    for _, r in tqdm(frame_df.iterrows(), total=len(frame_df), desc="local flow features"):
        swing_id = r["swing_id"]
        frame_idx = int(r["frame_idx"])
        prev_idx = frame_idx - 1

        base = {
            "image_stem": r["image_stem"],
            "swing_id": swing_id,
            "frame": r["frame"],
            "frame_idx": frame_idx,
            "prev_frame_idx": "" if frame_idx <= 1 else prev_idx,
            "flow_file": "",
            "has_prev_flow": 0,
            "head_pred_x": r["head_pred_x"],
            "head_pred_y": r["head_pred_y"],
            "head_pred_conf": r["head_pred_conf"],
            "tail_pred_x": r["tail_pred_x"],
            "tail_pred_y": r["tail_pred_y"],
            "tail_pred_conf": r["tail_pred_conf"],
            "bbox_conf": r["bbox_conf"],
            "bbox_x1": r["bbox_x1"],
            "bbox_y1": r["bbox_y1"],
            "bbox_x2": r["bbox_x2"],
            "bbox_y2": r["bbox_y2"],
        }

        if frame_idx <= 1:
            rows.append({**base, **zero_all_stats()})
            continue

        flow_path = flows_root / swing_id / "flow_npy" / f"{prev_idx:06d}_to_{frame_idx:06d}.npy"

        if not flow_path.exists():
            print(f"[WARN] missing flow: {flow_path}")
            rows.append({**base, **zero_all_stats()})
            continue

        flow = np.load(flow_path)  # [2, H, W]

        head_patch = safe_patch(
            flow,
            r["head_pred_x"],
            r["head_pred_y"],
            radius=args.radius,
        )

        tail_patch = safe_patch(
            flow,
            r["tail_pred_x"],
            r["tail_pred_y"],
            radius=args.radius,
        )

        bbox_patch = safe_bbox_patch(
            flow,
            r["bbox_x1"],
            r["bbox_y1"],
            r["bbox_x2"],
            r["bbox_y2"],
            pad=args.bbox_pad,
        )

        stats = {}
        stats.update(stats_from_patch(head_patch, "head"))
        stats.update(stats_from_patch(tail_patch, "tail"))
        stats.update(stats_from_patch(bbox_patch, "bbox"))

        rows.append({
            **base,
            "prev_frame_idx": prev_idx,
            "flow_file": str(flow_path),
            "has_prev_flow": 1,
            **stats,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] saved: {out_csv}")
    print(f"[INFO] rows: {len(out_df)}")


if __name__ == "__main__":
    main()