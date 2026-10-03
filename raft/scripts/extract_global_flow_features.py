import argparse
import csv
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm


def get_frame_idx_from_name(path: Path):
    """
    Supports names like:
    000001.jpg
    swing_001_000001.jpg
    """
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        return None
    return int(nums[-1])


def parse_flow_name(path: Path):
    """
    Supports names like:
    000001_to_000002.npy
    swing_001_000001_to_swing_001_000002.npy
    """
    nums = re.findall(r"\d+", path.stem)
    if len(nums) < 2:
        return None, None
    return int(nums[-2]), int(nums[-1])


def list_images(frames_dir: Path):
    imgs = (
        list(frames_dir.glob("*.jpg"))
        + list(frames_dir.glob("*.png"))
        + list(frames_dir.glob("*.jpeg"))
    )
    return sorted(imgs)


def compute_flow_stats(flow_path: Path):
    flow = np.load(flow_path)  # [2, H, W]
    fx = flow[0]
    fy = flow[1]
    mag = np.sqrt(fx * fx + fy * fy)

    return {
        "flow_x_mean": float(np.mean(fx)),
        "flow_y_mean": float(np.mean(fy)),
        "flow_x_std": float(np.std(fx)),
        "flow_y_std": float(np.std(fy)),
        "flow_mag_mean": float(np.mean(mag)),
        "flow_mag_std": float(np.std(mag)),
        "flow_mag_max": float(np.max(mag)),
        "flow_mag_p50": float(np.percentile(mag, 50)),
        "flow_mag_p90": float(np.percentile(mag, 90)),
        "flow_mag_p95": float(np.percentile(mag, 95)),
        "flow_mag_p99": float(np.percentile(mag, 99)),
    }


def zero_stats():
    return {
        "flow_x_mean": 0.0,
        "flow_y_mean": 0.0,
        "flow_x_std": 0.0,
        "flow_y_std": 0.0,
        "flow_mag_mean": 0.0,
        "flow_mag_std": 0.0,
        "flow_mag_max": 0.0,
        "flow_mag_p50": 0.0,
        "flow_mag_p90": 0.0,
        "flow_mag_p95": 0.0,
        "flow_mag_p99": 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-root", type=str, required=True)
    parser.add_argument("--flows-root", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    inputs_root = Path(args.inputs_root)
    flows_root = Path(args.flows_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    swing_dirs = sorted([p for p in inputs_root.iterdir() if p.is_dir()])

    fieldnames = [
        "swing_id",
        "frame_idx",
        "frame_name",
        "prev_frame_idx",
        "flow_file",
        "has_prev_flow",
        "flow_x_mean",
        "flow_y_mean",
        "flow_x_std",
        "flow_y_std",
        "flow_mag_mean",
        "flow_mag_std",
        "flow_mag_max",
        "flow_mag_p50",
        "flow_mag_p90",
        "flow_mag_p95",
        "flow_mag_p99",
    ]

    rows = []

    for swing_dir in tqdm(swing_dirs, desc="swings"):
        swing_id = swing_dir.name
        frames = list_images(swing_dir)

        if not frames:
            print(f"[SKIP] {swing_id}: no frames")
            continue

        flow_dir = flows_root / swing_id / "flow_npy"
        flow_files = sorted(flow_dir.glob("*.npy"))

        # Map target frame index -> previous flow file
        # flow 000001_to_000002 is aligned to frame 000002
        target_to_flow = {}
        target_to_prev = {}

        for fp in flow_files:
            prev_idx, target_idx = parse_flow_name(fp)
            if prev_idx is None or target_idx is None:
                print(f"[WARN] cannot parse flow filename: {fp}")
                continue
            target_to_flow[target_idx] = fp
            target_to_prev[target_idx] = prev_idx

        for frame_path in frames:
            frame_idx = get_frame_idx_from_name(frame_path)
            if frame_idx is None:
                print(f"[WARN] cannot parse frame filename: {frame_path}")
                continue

            base = {
                "swing_id": swing_id,
                "frame_idx": frame_idx,
                "frame_name": frame_path.name,
            }

            if frame_idx in target_to_flow:
                fp = target_to_flow[frame_idx]
                stats = compute_flow_stats(fp)
                row = {
                    **base,
                    "prev_frame_idx": target_to_prev[frame_idx],
                    "flow_file": str(fp),
                    "has_prev_flow": 1,
                    **stats,
                }
            else:
                # Usually the first frame has no previous flow.
                row = {
                    **base,
                    "prev_frame_idx": "",
                    "flow_file": "",
                    "has_prev_flow": 0,
                    **zero_stats(),
                }

            rows.append(row)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] saved: {out_csv}")
    print(f"[INFO] rows: {len(rows)}")


if __name__ == "__main__":
    main()