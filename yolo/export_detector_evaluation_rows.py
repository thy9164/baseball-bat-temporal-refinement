from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from pathlib import Path

try:
    from .candidate_selection import SelectionConfig, select_sequence_candidates
except ImportError:  # Direct script execution from the yolo directory.
    from candidate_selection import SelectionConfig, select_sequence_candidates


KEYPOINTS = ("head", "tail")
VISIBLE_AUX_KEYPOINT = "tail_visible_aux"
IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SWING_FRAME_RE = re.compile(r"^(swing_\d+)_(.+)$")
FIELDNAMES = [
    "image_stem",
    "swing_id",
    "view",
    "batter_side",
    "frame",
    "keypoint",
    "visibility",
    "gt_x",
    "gt_y",
    "pred_x",
    "pred_y",
    "pred_conf",
    "bbox_conf",
    "detection_rank",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "dx",
    "dy",
    "error_px",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export keypoint-level YOLO pose prediction details for one or more "
            "Ultralytics predict folders."
        )
    )
    parser.add_argument(
        "prediction_dirs",
        nargs="*",
        type=Path,
        help="Optional YOLO predict run directories or labels directories.",
    )
    parser.add_argument(
        "--pred-dirs",
        "--prediction-dirs",
        nargs="+",
        action="append",
        type=Path,
        default=[],
        help=(
            "One or more YOLO predict run directories or labels directories. "
            "Can be used multiple times and supports wildcards."
        ),
    )
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument(
        "--clips-list",
        type=Path,
        default=None,
        help="Optional CSV containing swing metadata columns: name, view, batter_side.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--swing-id",
        default=None,
        help="Optional swing_id for predict labels/images named like 000001.txt.",
    )
    parser.add_argument("--width", type=float, default=1920.0)
    parser.add_argument("--height", type=float, default=1080.0)
    parser.add_argument("--bbox-conf-thres", type=float, default=0.0)
    parser.add_argument("--kpt-conf-thres", type=float, default=0.0)
    parser.add_argument(
        "--bbox-conf-weight",
        type=float,
        default=100.0,
        help="Weight used in score = -weight*bbox_conf + temporal/shape jump terms.",
    )
    parser.add_argument(
        "--anchor-radius",
        type=float,
        default=300.0,
        help="No anchor cost is applied within this pixel radius. Defaults to 300.",
    )
    parser.add_argument(
        "--anchor-weight",
        type=float,
        default=1,
        help="Weight for max(0, bbox_center_to_anchor - anchor_radius). Defaults to 1.",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Only evaluate frames with prediction label files; skips images with no label file.",
    )
    return parser.parse_args()


def has_glob_pattern(path: Path) -> bool:
    return any(char in str(path) for char in "*?[")


def expand_prediction_dirs(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        matches = [Path(match) for match in sorted(glob.glob(str(path)))] if has_glob_pattern(path) else [path]
        if not matches:
            print(f"[WARN] Prediction path pattern matched nothing: {path}")
            continue
        expanded.extend(matches)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in expanded:
        normalized = path.resolve() if path.exists() else path
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(path)
    return unique


def selected_prediction_dirs(args: argparse.Namespace) -> list[Path]:
    pred_dirs_from_flag = [path for group in args.pred_dirs for path in group]
    chosen = args.prediction_dirs + pred_dirs_from_flag
    if chosen:
        return expand_prediction_dirs(chosen)
    return []


def resolve_prediction_dir(path: Path) -> tuple[Path, Path]:
    if path.name == "labels":
        return path.parent, path
    labels_dir = path / "labels"
    if labels_dir.exists():
        return path, labels_dir
    return path, path


def parse_swing_and_frame(stem: str, swing_id: str | None = None) -> tuple[str, str]:
    match = SWING_FRAME_RE.match(stem)
    if match:
        return match.group(1), match.group(2)
    if swing_id:
        return swing_id, stem
    raise ValueError(f"Cannot parse swing_id/frame_id from filename stem: {stem}")


def image_stem_from_gt(row: dict[str, str]) -> str:
    return f"{row['swing_id']}_{Path(row['frame']).stem}"


def prediction_stem_to_image_stem(stem: str, swing_id: str | None) -> str:
    if SWING_FRAME_RE.match(stem):
        return stem
    if swing_id:
        return f"{swing_id}_{stem}"
    return stem


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return float(value)


def format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def frame_number(frame_name: str) -> int | None:
    stem = Path(frame_name).stem
    if stem.isdigit():
        return int(stem)
    match = re.search(r"(\d+)$", stem)
    return int(match.group(1)) if match else None


def load_gt_rows(gt_csv: Path) -> dict[str, dict[str, str]]:
    if not gt_csv.exists():
        raise FileNotFoundError(f"Cannot find GT CSV: {gt_csv}")

    rows: dict[str, dict[str, str]] = {}
    with gt_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[image_stem_from_gt(row)] = row
    return rows


def load_clip_metadata(clips_list: Path | None) -> dict[str, dict[str, str]]:
    if clips_list is None:
        return {}
    if not clips_list.exists():
        print(f"[WARN] clips_list.csv not found; view/batter_side will be blank: {clips_list}")
        return {}

    metadata: dict[str, dict[str, str]] = {}
    with clips_list.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"name", "view", "batter_side"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            print(f"[WARN] Missing columns in {clips_list}: {sorted(missing_columns)}")
            return metadata

        for row in reader:
            swing_id = (row.get("name") or "").strip()
            if not swing_id:
                continue
            metadata[swing_id] = {
                "view": (row.get("view") or "").strip(),
                "batter_side": (row.get("batter_side") or "").strip(),
            }
    return metadata


def bbox_xyxy_pixels(bbox: tuple[float, float, float, float], width: float, height: float) -> tuple[float, float, float, float]:
    xc, yc, box_w, box_h = bbox
    x1 = (xc - box_w / 2.0) * width
    y1 = (yc - box_h / 2.0) * height
    x2 = (xc + box_w / 2.0) * width
    y2 = (yc + box_h / 2.0) * height
    return x1, y1, x2, y2


def read_yolo_predictions(label_path: Path) -> list[dict[str, object]]:
    """
    Expected Ultralytics pose save_txt save_conf format for 2 keypoints:

    class xc yc w h head_x head_y head_conf tail_x tail_y tail_conf bbox_conf
    """
    if not label_path.exists():
        return []

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    detections: list[dict[str, object]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue

        try:
            vals = [float(part) for part in parts]
        except ValueError:
            print(f"[WARN] Non-numeric label values in {label_path}:{line_no}: {line}")
            continue

        if len(vals) < 11:
            print(f"[WARN] Unexpected label format in {label_path}:{line_no}: {line}")
            continue

        bbox_conf = vals[-1] if len(vals) >= 12 else 1.0
        detections.append(
            {
                "bbox_conf": bbox_conf,
                "bbox": tuple(vals[1:5]),
                "head": tuple(vals[5:8]),
                "tail": tuple(vals[8:11]),
            }
        )

    detections.sort(key=lambda item: float(item["bbox_conf"]), reverse=True)
    for rank, detection in enumerate(detections, start=1):
        detection["detection_rank"] = rank
    return detections


def collect_eval_stems(run_dir: Path, labels_dir: Path, labels_only: bool) -> list[str]:
    stems = set()
    if labels_dir.exists():
        stems.update(label_path.stem for label_path in labels_dir.glob("*.txt"))

    if not labels_only and run_dir.exists():
        for image_path in run_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                stems.add(image_path.stem)

    return sorted(stems)


def make_detail_row(
    gt_row: dict[str, str],
    clip_meta: dict[str, str],
    prediction_stem: str,
    image_stem: str,
    keypoint: str,
    pred: dict[str, object] | None,
    width: float,
    height: float,
    bbox_conf_thres: float,
    kpt_conf_thres: float,
    drop_bbox_reason: str | None = None,
    drop_keypoints_reason: str | None = None,
) -> dict[str, str]:
    visibility = (gt_row.get(f"{keypoint}_visibility") or "").strip()
    gt_x = safe_float(gt_row.get(f"{keypoint}_x"))
    gt_y = safe_float(gt_row.get(f"{keypoint}_y"))

    detail = {
        "image_stem": image_stem,
        "swing_id": gt_row["swing_id"],
        "view": clip_meta.get("view", ""),
        "batter_side": clip_meta.get("batter_side", ""),
        "frame": gt_row["frame"],
        "keypoint": keypoint,
        "visibility": visibility,
        "gt_x": format_optional_float(gt_x),
        "gt_y": format_optional_float(gt_y),
        "pred_x": "",
        "pred_y": "",
        "pred_conf": "",
        "bbox_conf": "",
        "detection_rank": "",
        "bbox_x1": "",
        "bbox_y1": "",
        "bbox_x2": "",
        "bbox_y2": "",
        "dx": "",
        "dy": "",
        "error_px": "",
        "status": "missing_prediction",
    }

    if pred is None:
        detail["status"] = "ignore_no_prediction" if visibility == "ignore" else "missing_prediction"
        return detail

    if drop_bbox_reason is not None:
        detail["status"] = drop_bbox_reason
        return detail

    bbox_conf = float(pred["bbox_conf"])
    detail["bbox_conf"] = format_optional_float(bbox_conf)
    detail["detection_rank"] = str(pred.get("detection_rank", ""))
    bbox = pred.get("bbox")
    if bbox is not None:
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox_xyxy_pixels(bbox, width, height)  # type: ignore[arg-type]
        detail["bbox_x1"] = format_optional_float(bbox_x1)
        detail["bbox_y1"] = format_optional_float(bbox_y1)
        detail["bbox_x2"] = format_optional_float(bbox_x2)
        detail["bbox_y2"] = format_optional_float(bbox_y2)

    if drop_keypoints_reason is not None:
        detail["status"] = drop_keypoints_reason
        return detail

    pred_xn, pred_yn, pred_conf = pred[keypoint]  # type: ignore[index]
    pred_x = float(pred_xn) * width
    pred_y = float(pred_yn) * height
    pred_conf = float(pred_conf)
    detail["pred_x"] = format_optional_float(pred_x)
    detail["pred_y"] = format_optional_float(pred_y)
    detail["pred_conf"] = format_optional_float(pred_conf)

    if visibility == "ignore":
        detail["status"] = "ignore_with_prediction"
        return detail

    if bbox_conf < bbox_conf_thres:
        detail["status"] = "bbox_conf_below_threshold"
        return detail

    if pred_conf < kpt_conf_thres:
        detail["status"] = "kpt_conf_below_threshold"
        return detail

    if gt_x is None or gt_y is None:
        detail["status"] = "missing_gt"
        return detail

    dx = pred_x - gt_x
    dy = pred_y - gt_y
    detail["dx"] = format_optional_float(dx)
    detail["dy"] = format_optional_float(dy)
    detail["error_px"] = format_optional_float(math.hypot(dx, dy))
    detail["status"] = "matched"
    return detail


def has_visible_aux_eval(gt_row: dict[str, str]) -> bool:
    tail_visibility = (gt_row.get("tail_visibility") or "").strip()
    aux_x = safe_float(gt_row.get("bat_visible_aux_x"))
    aux_y = safe_float(gt_row.get("bat_visible_aux_y"))
    return tail_visibility == "ignore" and aux_x is not None and aux_y is not None


def make_visible_aux_detail_row(
    gt_row: dict[str, str],
    clip_meta: dict[str, str],
    image_stem: str,
) -> dict[str, str]:
    gt_x = safe_float(gt_row.get("bat_visible_aux_x"))
    gt_y = safe_float(gt_row.get("bat_visible_aux_y"))

    detail = {
        "image_stem": image_stem,
        "swing_id": gt_row["swing_id"],
        "view": clip_meta.get("view", ""),
        "batter_side": clip_meta.get("batter_side", ""),
        "frame": gt_row["frame"],
        "keypoint": VISIBLE_AUX_KEYPOINT,
        "visibility": "",
        "gt_x": format_optional_float(gt_x),
        "gt_y": format_optional_float(gt_y),
        "pred_x": "",
        "pred_y": "",
        "pred_conf": "",
        "bbox_conf": "",
        "detection_rank": "",
        "bbox_x1": "",
        "bbox_y1": "",
        "bbox_x2": "",
        "bbox_y2": "",
        "dx": "",
        "dy": "",
        "error_px": "",
        "status": "gt_only",
    }
    return detail


def write_details(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def detail_sort_key(row: dict[str, str]) -> tuple[str, str, int, str]:
    rank_text = row.get("detection_rank", "")
    detection_rank = int(rank_text) if rank_text.isdigit() else 10**9
    return row["swing_id"], row["frame"], detection_rank, row["keypoint"]



def select_for_evaluation_records(
    records: list[dict[str, object]], config: SelectionConfig
) -> list[dict[str, object]]:
    """Run inference-only selection, then reattach evaluation context."""
    selection_records = [
        {
            "prediction_stem": record["prediction_stem"],
            "image_stem": record["image_stem"],
            "swing_id": record["swing_id"],
            "frame_no": record["frame_no"],
            "frame_name": record["frame_name"],
            "detections": record["detections"],
        }
        for record in records
    ]
    selected = select_sequence_candidates(selection_records, config)
    evaluation_by_key = {
        (str(record["prediction_stem"]), str(record["image_stem"])): record
        for record in records
    }
    for item in selected:
        source = evaluation_by_key[(str(item["prediction_stem"]), str(item["image_stem"]))]
        item["gt_row"] = source["gt_row"]
        item["clip_meta"] = source["clip_meta"]
    return selected


def main() -> None:
    args = parse_args()
    prediction_dirs = selected_prediction_dirs(args)
    if not prediction_dirs:
        raise FileNotFoundError(
            "No prediction directories provided. Pass positional paths or --pred-dirs explicitly."
        )

    print("[INFO] Prediction inputs:")
    for prediction_dir in prediction_dirs:
        print(f"  - {prediction_dir}")

    gt_rows = load_gt_rows(args.gt_csv)
    clip_metadata = load_clip_metadata(args.clips_list)
    records: list[dict[str, object]] = []
    missing_gt_count = 0

    for input_dir in prediction_dirs:
        run_dir, labels_dir = resolve_prediction_dir(input_dir)
        if not labels_dir.exists():
            print(f"[WARN] Prediction labels dir does not exist: {labels_dir}")

        for prediction_stem in collect_eval_stems(run_dir, labels_dir, args.labels_only):
            image_stem = prediction_stem_to_image_stem(prediction_stem, args.swing_id)
            gt_row = gt_rows.get(image_stem)
            if gt_row is None:
                missing_gt_count += 1
                continue

            records.append(
                {
                    "prediction_stem": prediction_stem,
                    "image_stem": image_stem,
                    "gt_row": gt_row,
                    "clip_meta": clip_metadata.get(gt_row["swing_id"], {}),
                    "swing_id": gt_row["swing_id"],
                    "frame_no": frame_number(gt_row["frame"]),
                    "frame_name": gt_row["frame"],
                    "detections": read_yolo_predictions(labels_dir / f"{prediction_stem}.txt"),
                }
            )

    config = SelectionConfig(
        width=args.width,
        height=args.height,
        bbox_conf_weight=args.bbox_conf_weight,
        anchor_radius=args.anchor_radius,
        anchor_weight=args.anchor_weight,
    )
    selected_items = select_for_evaluation_records(records, config)

    details: list[dict[str, str]] = []
    missing_prediction_count = sum(not item["detections"] for item in selected_items)
    ignore_with_prediction_count = 0
    ignore_no_prediction_count = 0
    visible_aux_eval_count = 0
    rank_not_one_count = 0
    center_outlier_drop_count = 0
    length_keypoints_drop_count = 0

    for item in selected_items:
        selected_pred = item["selected_pred"]
        if selected_pred is not None and int(selected_pred["detection_rank"]) != 1:
            rank_not_one_count += 1
        if item.get("drop_bbox_reason") == "center_outlier":
            center_outlier_drop_count += 1
        if item.get("drop_keypoints_reason") == "length_outlier":
            length_keypoints_drop_count += 1

    for item in selected_items:
        prediction_stem = str(item["prediction_stem"])
        image_stem = str(item["image_stem"])
        gt_row = item["gt_row"]  # type: ignore[assignment]
        clip_meta = item["clip_meta"]  # type: ignore[assignment]
        selected_pred = item["selected_pred"]  # type: ignore[assignment]
        drop_bbox_reason = item.get("drop_bbox_reason")
        drop_keypoints_reason = item.get("drop_keypoints_reason")

        for keypoint in KEYPOINTS:
            detail = make_detail_row(
                gt_row=gt_row,  # type: ignore[arg-type]
                clip_meta=clip_meta,  # type: ignore[arg-type]
                prediction_stem=prediction_stem,
                image_stem=image_stem,
                keypoint=keypoint,
                pred=selected_pred,
                width=args.width,
                height=args.height,
                bbox_conf_thres=args.bbox_conf_thres,
                kpt_conf_thres=args.kpt_conf_thres,
                drop_bbox_reason=drop_bbox_reason if isinstance(drop_bbox_reason, str) else None,
                drop_keypoints_reason=drop_keypoints_reason if isinstance(drop_keypoints_reason, str) else None,
            )
            ignore_with_prediction_count += int(detail["status"] == "ignore_with_prediction")
            ignore_no_prediction_count += int(detail["status"] == "ignore_no_prediction")
            details.append(detail)

        if has_visible_aux_eval(gt_row):  # type: ignore[arg-type]
            detail = make_visible_aux_detail_row(
                gt_row=gt_row,  # type: ignore[arg-type]
                clip_meta=clip_meta,  # type: ignore[arg-type]
                image_stem=image_stem,
            )
            visible_aux_eval_count += 1
            details.append(detail)

    details.sort(key=detail_sort_key)
    write_details(args.output, details)

    print(f"[DONE] Wrote {len(details)} detail rows: {args.output}")
    print(f"[INFO] Missing GT frames skipped: {missing_gt_count}")
    print(f"[INFO] Frames with no prediction label: {missing_prediction_count}")
    print(f"[INFO] ignore_with_prediction rows: {ignore_with_prediction_count}")
    print(f"[INFO] ignore_no_prediction rows: {ignore_no_prediction_count}")
    print(f"[INFO] visible aux eval rows: {visible_aux_eval_count}")
    print(f"[INFO] selected rank != 1 frames: {rank_not_one_count}")
    print(f"[INFO] center outlier frames with prediction fields cleared: {center_outlier_drop_count}")
    print(f"[INFO] length outlier frames with keypoint fields cleared: {length_keypoints_drop_count}")


if __name__ == "__main__":
    main()
