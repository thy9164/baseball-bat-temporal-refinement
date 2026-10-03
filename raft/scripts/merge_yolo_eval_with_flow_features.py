import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_wide_yolo_table(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "image_stem", "swing_id", "view", "batter_side", "frame",
        "keypoint", "visibility",
        "gt_x", "gt_y", "pred_x", "pred_y", "pred_conf",
        "bbox_conf", "detection_rank",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "status",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in yolo csv: {missing}")

    rows = []

    group_cols = ["image_stem", "swing_id", "view", "batter_side", "frame"]

    for keys, g in df.groupby(group_cols, sort=False):
        image_stem, swing_id, view, batter_side, frame = keys

        first = g.iloc[0]

        row = {
            "image_stem": image_stem,
            "swing_id": swing_id,
            "view": view,
            "batter_side": batter_side,
            "frame": frame,
            "bbox_conf": first["bbox_conf"],
            "detection_rank": first["detection_rank"],
            "bbox_x1": first["bbox_x1"],
            "bbox_y1": first["bbox_y1"],
            "bbox_x2": first["bbox_x2"],
            "bbox_y2": first["bbox_y2"],
            "status": first["status"],
        }

        for kp in ["head", "tail"]:
            kg = g[g["keypoint"] == kp]

            if len(kg) == 0:
                row[f"{kp}_visibility"] = ""
                row[f"{kp}_gt_x"] = np.nan
                row[f"{kp}_gt_y"] = np.nan
                row[f"{kp}_pred_x"] = np.nan
                row[f"{kp}_pred_y"] = np.nan
                row[f"{kp}_pred_conf"] = np.nan
                row[f"{kp}_dx"] = np.nan
                row[f"{kp}_dy"] = np.nan
                row[f"{kp}_error_px"] = np.nan
            else:
                kr = kg.iloc[0]
                row[f"{kp}_visibility"] = kr["visibility"]
                row[f"{kp}_gt_x"] = kr["gt_x"]
                row[f"{kp}_gt_y"] = kr["gt_y"]
                row[f"{kp}_pred_x"] = kr["pred_x"]
                row[f"{kp}_pred_y"] = kr["pred_y"]
                row[f"{kp}_pred_conf"] = kr["pred_conf"]
                row[f"{kp}_dx"] = kr["dx"] if "dx" in kr else np.nan
                row[f"{kp}_dy"] = kr["dy"] if "dy" in kr else np.nan
                row[f"{kp}_error_px"] = kr["error_px"] if "error_px" in kr else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-csv", type=str, required=True)
    parser.add_argument("--local-flow-csv", type=str, required=True)
    parser.add_argument("--global-flow-csv", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    yolo_csv = Path(args.yolo_csv)
    local_flow_csv = Path(args.local_flow_csv)
    global_flow_csv = Path(args.global_flow_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    yolo_df = pd.read_csv(yolo_csv)
    local_df = pd.read_csv(local_flow_csv)
    global_df = pd.read_csv(global_flow_csv)

    wide_df = build_wide_yolo_table(yolo_df)

    # Keep only useful local flow columns.
    local_keep = [
        c for c in local_df.columns
        if c not in [
            "head_pred_x", "head_pred_y", "head_pred_conf",
            "tail_pred_x", "tail_pred_y", "tail_pred_conf",
            "bbox_conf", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "flow_file",
        ]
    ]

    local_df = local_df[local_keep]

    # Rename global flow columns to avoid collision with local has_prev_flow / flow_file.
    global_df = global_df.rename(columns={
        "has_prev_flow": "global_has_prev_flow",
        "prev_frame_idx": "global_prev_frame_idx",
        "flow_file": "global_flow_file",
        "flow_x_mean": "global_flow_x_mean",
        "flow_y_mean": "global_flow_y_mean",
        "flow_x_std": "global_flow_x_std",
        "flow_y_std": "global_flow_y_std",
        "flow_mag_mean": "global_flow_mag_mean",
        "flow_mag_std": "global_flow_mag_std",
        "flow_mag_max": "global_flow_mag_max",
        "flow_mag_p50": "global_flow_mag_p50",
        "flow_mag_p90": "global_flow_mag_p90",
        "flow_mag_p95": "global_flow_mag_p95",
        "flow_mag_p99": "global_flow_mag_p99",
    })

    global_keep = [
        "swing_id", "frame_idx", "frame_name",
        "global_prev_frame_idx", "global_flow_file", "global_has_prev_flow",
        "global_flow_x_mean", "global_flow_y_mean",
        "global_flow_x_std", "global_flow_y_std",
        "global_flow_mag_mean", "global_flow_mag_std", "global_flow_mag_max",
        "global_flow_mag_p50", "global_flow_mag_p90", "global_flow_mag_p95", "global_flow_mag_p99",
    ]
    global_df = global_df[global_keep]

    # Merge local flow by swing_id + frame
    merged = wide_df.merge(
        local_df,
        on=["image_stem", "swing_id", "frame"],
        how="left",
        validate="one_to_one",
    )

    # Merge global flow by swing_id + frame_idx
    merged = merged.merge(
        global_df,
        on=["swing_id", "frame_idx"],
        how="left",
        validate="one_to_one",
    )

    # Basic sanity columns
    merged["bbox_w"] = merged["bbox_x2"] - merged["bbox_x1"]
    merged["bbox_h"] = merged["bbox_y2"] - merged["bbox_y1"]
    merged["pred_bat_length"] = np.sqrt(
        (merged["head_pred_x"] - merged["tail_pred_x"]) ** 2
        + (merged["head_pred_y"] - merged["tail_pred_y"]) ** 2
    )
    merged["gt_bat_length"] = np.sqrt(
        (merged["head_gt_x"] - merged["tail_gt_x"]) ** 2
        + (merged["head_gt_y"] - merged["tail_gt_y"]) ** 2
    )

    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] saved: {out_csv}")
    print(f"[INFO] rows: {len(merged)}")
    print(f"[INFO] columns: {len(merged.columns)}")

    missing_local = merged["has_prev_flow"].isna().sum() if "has_prev_flow" in merged.columns else -1
    missing_global = merged["global_has_prev_flow"].isna().sum() if "global_has_prev_flow" in merged.columns else -1
    print(f"[CHECK] missing local flow rows : {missing_local}")
    print(f"[CHECK] missing global flow rows: {missing_global}")


if __name__ == "__main__":
    main()