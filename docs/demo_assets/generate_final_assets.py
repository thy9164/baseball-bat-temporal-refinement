from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "bg": "#0A0E16",
    "text": "#F8FAFC",
    "muted": "#CBD5E1",
    "gt": "#FFFFFF",
    "detector": "#FF784F",
    "refined": "#16D9E3",
}
CANVAS_SIZE = (1800, 1000)
PANEL_SIZE = (820, 615)
PANEL_Y = 140
SHORT_FRAMES = range(18, 31)
FULL_FRAMES = range(1, 64)


@functools.lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a portable sans-serif font without bundling proprietary files."""
    requested = os.environ.get("DEMO_FONT_BOLD" if bold else "DEMO_FONT_REGULAR")
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        requested,
        Path("C:/Windows/Fonts") / ("seguisb.ttf" if bold else "segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") / filename,
        Path("/System/Library/Fonts/Supplemental") / ("Arial Bold.ttf" if bold else "Arial.ttf"),
        filename,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow versions before the scalable default-font option.
        return ImageFont.load_default()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point(row: pd.Series, x_column: str, y_column: str) -> tuple[float, float]:
    x, y = float(row[x_column]), float(row[y_column])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"Non-finite point in {x_column}/{y_column}")
    return x, y


def rows_for(df: pd.DataFrame, swing: str, frame: int) -> dict[str, pd.Series]:
    subset = df[(df["swing_id"] == swing) & (df["frame_idx"] == frame)]
    rows = {str(row["keypoint"]): row for _, row in subset.iterrows()}
    if set(rows) != {"head", "tail"}:
        raise ValueError(f"Missing head/tail rows for {swing} frame {frame}")
    return rows


def fixed_tail_crop(
    df: pd.DataFrame, swing: str, image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Create one fixed 4:3 crop from the valid full-swing GT tail envelope."""
    tail = df[
        (df["swing_id"] == swing)
        & (df["keypoint"] == "tail")
        & (df["target_valid"].astype(bool))
    ]
    xs = pd.to_numeric(tail["gt_x"], errors="coerce").dropna()
    ys = pd.to_numeric(tail["gt_y"], errors="coerce").dropna()
    if xs.empty or ys.empty:
        raise ValueError(f"No valid GT tail envelope for {swing}")
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    width, height = (max_x - min_x) * 1.60, (max_y - min_y) * 1.60
    if width / height < 4 / 3:
        width = height * 4 / 3
    else:
        height = width * 3 / 4
    source_w, source_h = image_size
    width, height = min(width, source_w), min(height, source_h)
    left = max(0.0, min(cx - width / 2, source_w - width))
    top = max(0.0, min(cy - height / 2, source_h - height))
    return round(left), round(top), round(left + width), round(top + height)


def crop_transform(
    location: tuple[float, float],
    crop: tuple[int, int, int, int],
    size: tuple[int, int],
) -> tuple[int, int]:
    left, top, right, bottom = crop
    return (
        round((location[0] - left) * size[0] / (right - left)),
        round((location[1] - top) * size[1] / (bottom - top)),
    )


def draw_marker(draw: ImageDraw.ImageDraw, location: tuple[int, int], color: str) -> None:
    radius = 15
    draw.ellipse(
        (location[0] - radius, location[1] - radius, location[0] + radius, location[1] + radius),
        fill=color,
        outline=COLORS["bg"],
        width=5,
    )


def draw_gt_ring(draw: ImageDraw.ImageDraw, location: tuple[int, int]) -> None:
    radius = 20
    bounds = (location[0] - radius, location[1] - radius, location[0] + radius, location[1] + radius)
    draw.ellipse(bounds, fill=None, outline=COLORS["bg"], width=10)
    draw.ellipse(bounds, fill=None, outline=COLORS["gt"], width=5)


def panel(
    source: Image.Image,
    rows: dict[str, pd.Series],
    crop: tuple[int, int, int, int],
    method: str,
    label: str,
    error: float,
) -> Image.Image:
    image = source.crop(crop).resize(PANEL_SIZE, Image.Resampling.LANCZOS).convert("RGBA")
    pred_x, pred_y = ("raw_x", "raw_y") if method == "detector" else ("final_x", "final_y")
    color = COLORS[method]
    gt_head = crop_transform(point(rows["head"], "gt_x", "gt_y"), crop, PANEL_SIZE)
    gt_tail = crop_transform(point(rows["tail"], "gt_x", "gt_y"), crop, PANEL_SIZE)
    pred_head = crop_transform(point(rows["head"], pred_x, pred_y), crop, PANEL_SIZE)
    pred_tail = crop_transform(point(rows["tail"], pred_x, pred_y), crop, PANEL_SIZE)

    axis_layer = Image.new("RGBA", PANEL_SIZE, (0, 0, 0, 0))
    axis_draw = ImageDraw.Draw(axis_layer)
    axis_draw.line([gt_head, gt_tail], fill=(255, 255, 255, 70), width=2)
    axis_rgb = (255, 120, 79, 82) if method == "detector" else (22, 217, 227, 82)
    axis_draw.line([pred_head, pred_tail], fill=axis_rgb, width=2)
    axis_draw.line([pred_tail, gt_tail], fill=axis_rgb[:3] + (165,), width=3)
    image.alpha_composite(axis_layer)

    draw = ImageDraw.Draw(image)
    draw_gt_ring(draw, gt_tail)
    draw_marker(draw, pred_tail, color)
    label_layer = Image.new("RGBA", (PANEL_SIZE[0], 92), (10, 14, 22, 218))
    label_draw = ImageDraw.Draw(label_layer)
    label_draw.text((24, 12), label, fill=COLORS["text"], font=font(29, True))
    label_draw.text((24, 53), f"Tail error  {error:5.1f} px", fill=color, font=font(23, True))
    image.alpha_composite(label_layer, (0, 0))
    return image


def comparison_canvas(
    source: Image.Image,
    rows: dict[str, pd.Series],
    crop: tuple[int, int, int, int],
    title: str,
    subtitle: str,
) -> Image.Image:
    detector_error = float(rows["tail"]["raw_error_px"])
    refined_error = float(rows["tail"]["final_error_px"])
    canvas = Image.new("RGB", CANVAS_SIZE, COLORS["bg"])
    draw = ImageDraw.Draw(canvas)
    draw.text((60, 30), title, fill=COLORS["text"], font=font(38, True))
    draw.text((60, 82), subtitle, fill=COLORS["muted"], font=font(21))
    detector = panel(source, rows, crop, "detector", "FRAME-WISE DETECTOR", detector_error)
    refined = panel(source, rows, crop, "refined", "TEMPORAL REFINEMENT", refined_error)
    canvas.paste(detector.convert("RGB"), (60, PANEL_Y))
    canvas.paste(refined.convert("RGB"), (920, PANEL_Y))
    draw.text((530, 815), "Tail error", fill=COLORS["muted"], font=font(24))
    draw.text((675, 799), f"{detector_error:5.1f} px", fill=COLORS["detector"], font=font(42, True))
    draw.text((905, 801), "→", fill=COLORS["text"], font=font(42, True))
    draw.text((978, 799), f"{refined_error:5.1f} px", fill=COLORS["refined"], font=font(42, True))
    legend_center = (706, 905)
    draw.ellipse(
        (legend_center[0] - 11, legend_center[1] - 11, legend_center[0] + 11, legend_center[1] + 11),
        fill=None,
        outline=COLORS["gt"],
        width=3,
    )
    draw.text((728, 887), "Ground-truth tail", fill=COLORS["text"], font=font(24))
    return canvas


def render_video(
    ffmpeg: Path,
    output_path: Path,
    predictions: pd.DataFrame,
    frames_root: Path,
    crop: tuple[int, int, int, int],
    frame_range: range,
    fps: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"{output_path.stem}_") as temp_name:
        temp = Path(temp_name)
        for output_index, frame in enumerate(frame_range, start=1):
            source = Image.open(frames_root / f"swing_074/{frame:06d}.jpg").convert("RGB")
            image = comparison_canvas(
                source,
                rows_for(predictions, "swing_074", frame),
                crop,
                "Frame-wise detector vs temporal refinement",
                f"Frozen qualitative example  •  swing_074  •  frame {frame:03d}",
            )
            image.save(temp / f"frame_{output_index:06d}.png", compress_level=4)
        subprocess.run(
            [
                str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(fps), "-start_number", "1",
                "-i", str(temp / "frame_%06d.png"), "-frames:v", str(len(frame_range)),
                "-c:v", "libx264", "-crf", "20", "-preset", "medium",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
            ],
            check=True,
        )


def asset_record(path: Path) -> dict[str, object]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render all final public demo assets from frozen predictions.")
    parser.add_argument("--predictions", type=Path, required=True, help="Frozen formal prediction CSV.")
    parser.add_argument("--selected-csv", type=Path, required=True, help="Selected-detector CSV used for provenance hashing.")
    parser.add_argument("--frames-root", type=Path, required=True, help="Rights-cleared source-frame root containing swing directories.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the four assets and manifests.")
    parser.add_argument("--ffmpeg", type=Path, required=True, help="FFmpeg executable with the CPU libx264 encoder.")
    parser.add_argument(
        "--write-private-manifest",
        action="store_true",
        help="Also write ASSET_MANIFEST.json with absolute local provenance paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.predictions
    selected_path = args.selected_csv
    predictions = pd.read_csv(predictions_path)
    frames_root = args.frames_root
    crop_074 = fixed_tail_crop(predictions, "swing_074", Image.open(frames_root / "swing_074/000001.jpg").size)
    crop_076 = fixed_tail_crop(predictions, "swing_076", Image.open(frames_root / "swing_076/000001.jpg").size)

    hero = comparison_canvas(
        Image.open(frames_root / "swing_074/000023.jpg").convert("RGB"),
        rows_for(predictions, "swing_074", 23), crop_074,
        "Frame-wise detector vs temporal refinement",
        "Frozen qualitative example  •  swing_074  •  frame 023",
    )
    hero.save(output / "swing074_hero.png", optimize=True)
    limitation = comparison_canvas(
        Image.open(frames_root / "swing_076/000043.jpg").convert("RGB"),
        rows_for(predictions, "swing_076", 43), crop_076,
        "Limitation: temporal refinement can also worsen a detector estimate",
        "Frozen qualitative counterexample  •  swing_076  •  frame 043",
    )
    limitation.save(output / "swing076_limitation.png", optimize=True)

    render_video(args.ffmpeg, output / "swing074_detector_vs_temporal.mp4", predictions, frames_root, crop_074, SHORT_FRAMES, 6)
    render_video(args.ffmpeg, output / "swing074_full_swing_comparison.mp4", predictions, frames_root, crop_074, FULL_FRAMES, 8)

    source_frames = [frames_root / f"swing_074/{frame:06d}.jpg" for frame in FULL_FRAMES]
    source_frames.append(frames_root / "swing_076/000043.jpg")
    assets = [
        output / "swing074_hero.png",
        output / "swing074_detector_vs_temporal.mp4",
        output / "swing074_full_swing_comparison.mp4",
        output / "swing076_limitation.png",
    ]
    common = {
        "purpose": "Final public demo assets rendered from frozen strict-clean no-flow predictions.",
        "frozen_prediction_lineage": "strict_clean_real_no_flow_v1_mysplit_v2_seed7 / formal_test_predictions",
        "crop_policy": "Valid full-swing GT tail trajectory envelope, 30% margin on every side, expanded to 4:3, clamped to the source image; fixed across every frame in a swing.",
        "crop_rectangles_ltrb": {"swing_074": list(crop_074), "swing_076": list(crop_076)},
        "rendering_policy": {
            "comparison": "frame-wise selected detector vs strict-clean no-flow temporal refinement",
            "ground_truth": "white hollow ring",
            "detector": "orange marker",
            "temporal_refinement": "cyan marker",
            "flow_included": False,
            "bat_axes": "faint two-pixel reference lines",
            "focused_short_video": {"swing_id": "swing_074", "first_frame": 18, "last_frame": 30, "total_frames": 13, "playback_fps": 6, "duration_seconds": 13 / 6},
            "full_swing_video": {"swing_id": "swing_074", "first_frame": 1, "last_frame": 63, "total_frames": 63, "playback_fps": 8, "duration_seconds": 7.875},
            "prediction_interpolation": False,
            "display_smoothing": False,
            "hand_edited_points": False,
            "encoding": "CPU libx264; no hardware encoder or GPU workload",
            "font_policy": "Optional DEMO_FONT_* override, then common system sans-serif fonts, then DejaVu Sans, then Pillow default; no font file is bundled.",
        },
        "public_boundary": "Frozen qualitative examples and counterexample; none replaces complete 15-swing test-set performance.",
        "model_operations": "No training, inference, checkpoint selection, prediction regeneration, or metric recomputation.",
        "assets": [asset_record(path) for path in assets],
    }
    internal_manifest = {
        **common,
        "inputs": {
            "frozen_predictions": {"path": str(predictions_path.resolve()), "sha256": sha256(predictions_path)},
            "selected_test_csv_provenance": {"path": str(selected_path.resolve()), "sha256": sha256(selected_path)},
            "source_frames": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in source_frames],
        },
    }
    public_manifest = {
        **common,
        "inputs": {
            "frozen_predictions": {"logical_id": "formal_test_predictions/no_flow_v1/mysplit_v2/seed7", "sha256": sha256(predictions_path)},
            "selected_test_csv_provenance": {"logical_id": "selected_detector/test/mysplit_v2", "sha256": sha256(selected_path)},
            "source_frames": [
                {"logical_id": f"self_recorded/{path.parent.name}/{path.stem}", "sha256": sha256(path)}
                for path in source_frames
            ],
        },
    }
    (output / "ASSET_MANIFEST_PUBLIC.json").write_text(json.dumps(public_manifest, indent=2) + "\n", encoding="utf-8")
    if args.write_private_manifest:
        (output / "ASSET_MANIFEST.json").write_text(json.dumps(internal_manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
