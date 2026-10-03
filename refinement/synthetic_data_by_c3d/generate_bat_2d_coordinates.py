import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2
import ezc3d
import numpy as np


BAT_MARKERS = ("Marker1", "Marker2", "Marker3")
BASE_DISTANCE_SCALE = 5.0
BASE_FOCAL_MM = 35.0
SENSOR_WIDTH_MM = 36.0

BODY_MARKERS = [
    "LFHD", "RFHD", "LBHD", "RBHD",
    "C7", "CLAV", "STRN", "T10",
    "LSHO", "RSHO",
    "LELB", "RELB",
    "LWRA", "RWRA",
    "LFIN", "RFIN",
    "LASI", "RASI",
    "LPSI", "RPSI",
    "LKNE", "RKNE",
    "LANK", "RANK",
    "LTOE", "RTOE",
]

SKELETON_SEGMENTS = [
    ("LFHD", "RFHD"), ("RFHD", "RBHD"), ("RBHD", "LBHD"), ("LBHD", "LFHD"),
    ("LSHO", "RSHO"), ("LSHO", "LASI"), ("RSHO", "RASI"), ("LASI", "RASI"),
    ("C7", "T10"), ("CLAV", "STRN"),
    ("LSHO", "LELB"), ("LELB", "LWRA"), ("LWRA", "LFIN"),
    ("RSHO", "RELB"), ("RELB", "RWRA"), ("RWRA", "RFIN"),
    ("LASI", "LKNE"), ("LKNE", "LANK"), ("LANK", "LTOE"),
    ("RASI", "RKNE"), ("RKNE", "RANK"), ("RANK", "RTOE"),
]


def valid_xyz(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    return (
        xyz.shape == (3,)
        and np.isfinite(xyz).all()
        and np.linalg.norm(xyz) > 1e-9
    )


def parse_batter_hand(c3d_path):
    match = re.search(r"_(R|L)_", c3d_path.stem, re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).upper()


def flip_batter_hand(hand):
    if hand == "R":
        return "L"
    if hand == "L":
        return "R"
    return hand


def horizontal_mirror_camera_name(camera_name):
    if "1B_side" in camera_name:
        return camera_name.replace("1B_side", "3B_side")
    if "3B_side" in camera_name:
        return camera_name.replace("3B_side", "1B_side")
    return camera_name


def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def estimate_unit_scale(tail_3d, head_3d):
    lengths = []
    for tail, head in zip(tail_3d, head_3d):
        if valid_xyz(tail) and valid_xyz(head):
            lengths.append(np.linalg.norm(head - tail))

    if not lengths:
        return 1.0, "unknown"

    median_len = float(np.median(lengths))
    if median_len > 10.0:
        return 0.001, "mm_to_m"
    return 1.0, "already_m"


def read_motion_points(c3d_path):
    c3d = ezc3d.c3d(str(c3d_path))
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    label_to_idx = {name: i for i, name in enumerate(labels)}
    label_upper_to_idx = {name.upper(): i for i, name in enumerate(labels)}

    missing = [name for name in BAT_MARKERS if name.upper() not in label_upper_to_idx]
    if missing:
        raise ValueError(f"missing bat markers: {missing}")

    points = c3d["data"]["points"][:3].astype(np.float64)
    marker1 = points[:, label_upper_to_idx["MARKER1"], :].T
    marker2 = points[:, label_upper_to_idx["MARKER2"], :].T
    marker3 = points[:, label_upper_to_idx["MARKER3"], :].T

    tail_3d = marker1
    head_3d = (marker2 + marker3) / 2.0
    scale, unit_mode = estimate_unit_scale(tail_3d, head_3d)

    body_points = {}
    for marker in BODY_MARKERS:
        if marker in label_to_idx:
            body_points[marker] = points[:, label_to_idx[marker], :].T * scale

    fps = float(c3d["header"]["points"].get("frame_rate", 120.0))
    return tail_3d * scale, head_3d * scale, body_points, fps, unit_mode


def has_uppercase_only_bat_markers(c3d_path):
    c3d = ezc3d.c3d(str(c3d_path))
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    label_set = set(labels)
    label_upper_set = {name.upper() for name in labels}

    has_bat_markers = all(name.upper() in label_upper_set for name in BAT_MARKERS)
    has_canonical_case = all(name in label_set for name in BAT_MARKERS)
    return has_bat_markers and not has_canonical_case


def make_camera(center, radius, yaw_deg, pitch_deg, roll_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    radius = max(float(radius), 1.0)
    distance = radius * BASE_DISTANCE_SCALE

    direction = np.array(
        [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ],
        dtype=np.float64,
    )
    eye = center + direction * distance
    forward = normalize(center - eye)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(right, forward))

    if abs(roll) > 1e-9:
        c, s = math.cos(roll), math.sin(roll)
        right, up = c * right + s * up, -s * right + c * up

    return {
        "eye": eye,
        "forward": forward,
        "right": right,
        "up": up,
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "distance": distance,
    }


def project(point, camera, width, height, focal_px, pan_x=0.0, pan_y=0.0):
    rel = np.asarray(point, dtype=np.float64) - camera["eye"]
    x = float(np.dot(rel, camera["right"]))
    y = float(np.dot(rel, camera["up"]))
    z = float(np.dot(rel, camera["forward"]))

    if z <= 1e-6:
        return np.nan, np.nan, z, False

    u = focal_px * x / z + width * (0.5 + pan_x)
    v = height * (0.5 + pan_y) - focal_px * y / z
    in_frame = 0.0 <= u < width and 0.0 <= v < height
    return u, v, z, in_frame


def valid_scene_points(tail_3d, head_3d, body_points=None):
    pts = []
    for tail, head in zip(tail_3d, head_3d):
        if valid_xyz(tail):
            pts.append(tail)
        if valid_xyz(head):
            pts.append(head)
    if body_points:
        for marker_frames in body_points.values():
            for point in marker_frames:
                if valid_xyz(point):
                    pts.append(point)
    if not pts:
        return None
    return np.stack(pts, axis=0)


def parse_float_range(value, name):
    try:
        lo_text, hi_text = str(value).split(",", maxsplit=1)
        lo = float(lo_text)
        hi = float(hi_text)
    except ValueError as exc:
        raise ValueError(f"{name} must be formatted like min,max") from exc

    if lo > hi:
        raise ValueError(f"{name} min must be <= max")

    return lo, hi


def parse_float_list(value, name):
    try:
        values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of numbers") from exc

    if not values:
        raise ValueError(f"{name} cannot be empty")

    return values


def sample_range(rng, value_range):
    lo, hi = value_range
    if lo == hi:
        return float(lo)
    return float(rng.uniform(lo, hi))


def sample_config_value(rng, view, key, default):
    value = view.get(key, view.get(f"{key}_range", default))

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        text = value.strip()
        if "," in text:
            return sample_range(rng, parse_float_range(text, key))
        return float(text)

    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return float(value[0])
        if len(value) == 2:
            return sample_range(rng, (float(value[0]), float(value[1])))

    raise ValueError(f"Invalid view config value for {key}: {value!r}")


def load_configured_camera_specs(config_path, rng):
    path = Path(config_path)
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)

    views = data["views"] if isinstance(data, dict) else data
    if not isinstance(views, list) or not views:
        raise ValueError("view config must contain a non-empty views list")

    specs = []
    for idx, view in enumerate(views):
        if not isinstance(view, dict):
            raise ValueError(f"view {idx} must be an object")

        specs.append(
            (
                str(view.get("type", "configured")),
                str(view.get("name", f"view_{idx + 1:02d}")),
                sample_config_value(rng, view, "yaw", 0.0),
                sample_config_value(rng, view, "pitch", 8.0),
                sample_config_value(rng, view, "roll", 0.0),
                sample_config_value(rng, view, "zoom", 1.0),
                sample_config_value(rng, view, "pan_x", 0.0),
                sample_config_value(rng, view, "pan_y", 0.0),
            )
        )

    return specs


def camera_specs(args, rng):
    if args.view_config:
        return load_configured_camera_specs(args.view_config, rng)

    specs = []

    fixed_yaws = parse_float_list(args.fixed_yaws, "--fixed-yaws")
    fixed_pitch_range = parse_float_range(args.fixed_pitch_range, "--fixed-pitch-range")
    fixed_roll_range = parse_float_range(args.fixed_roll_range, "--fixed-roll-range")
    fixed_zoom_range = parse_float_range(args.fixed_zoom_range, "--fixed-zoom-range")
    fixed_pan_x_range = parse_float_range(args.fixed_pan_x_range, "--fixed-pan-x-range")
    fixed_pan_y_range = parse_float_range(args.fixed_pan_y_range, "--fixed-pan-y-range")

    random_yaw_range = parse_float_range(args.random_yaw_range, "--random-yaw-range")
    random_pitch_range = parse_float_range(args.random_pitch_range, "--random-pitch-range")
    random_roll_range = parse_float_range(args.random_roll_range, "--random-roll-range")
    random_zoom_range = parse_float_range(args.random_zoom_range, "--random-zoom-range")
    random_pan_x_range = parse_float_range(args.random_pan_x_range, "--random-pan-x-range")
    random_pan_y_range = parse_float_range(args.random_pan_y_range, "--random-pan-y-range")

    for yaw in fixed_yaws:
        specs.append(
            (
                "fixed",
                f"fixed_yaw_{yaw:g}",
                float(yaw),
                sample_range(rng, fixed_pitch_range),
                sample_range(rng, fixed_roll_range),
                sample_range(rng, fixed_zoom_range),
                sample_range(rng, fixed_pan_x_range),
                sample_range(rng, fixed_pan_y_range),
            )
        )

    for _ in range(args.random_views):
        specs.append(
            (
                "random",
                "random",
                sample_range(rng, random_yaw_range),
                sample_range(rng, random_pitch_range),
                sample_range(rng, random_roll_range),
                sample_range(rng, random_zoom_range),
                sample_range(rng, random_pan_x_range),
                sample_range(rng, random_pan_y_range),
            )
        )

    return specs


def skeleton_projection_row(body_points, frame_idx, camera, width, height, focal_px, pan_x, pan_y):
    row = {}
    for marker in BODY_MARKERS:
        prefix = f"skel_{marker}"
        if marker not in body_points or frame_idx >= len(body_points[marker]):
            row[f"{prefix}_u"] = np.nan
            row[f"{prefix}_v"] = np.nan
            row[f"{prefix}_depth"] = np.nan
            row[f"{prefix}_in_frame"] = False
            continue

        point = body_points[marker][frame_idx]
        if not valid_xyz(point):
            row[f"{prefix}_u"] = np.nan
            row[f"{prefix}_v"] = np.nan
            row[f"{prefix}_depth"] = np.nan
            row[f"{prefix}_in_frame"] = False
            continue

        u, v, depth, in_frame = project(point, camera, width, height, focal_px, pan_x, pan_y)
        row[f"{prefix}_u"] = u
        row[f"{prefix}_v"] = v
        row[f"{prefix}_depth"] = depth
        row[f"{prefix}_in_frame"] = in_frame

    return row


def mirror_u(value, width):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value

    if not np.isfinite(value):
        return value

    return (width - 1) - value


def horizontal_mirror_row(row, width):
    mirrored = dict(row)
    mirrored["mirror"] = "horizontal"
    mirrored["batter_hand"] = flip_batter_hand(row.get("batter_hand", ""))
    mirrored["camera_name"] = horizontal_mirror_camera_name(row.get("camera_name", ""))

    for key in list(mirrored.keys()):
        if key.endswith("_u"):
            mirrored[key] = mirror_u(mirrored[key], width)

    mirrored["tail_from_head_dx"] = -float(row["tail_from_head_dx"])
    mirrored["tail_from_head_dy"] = row["tail_from_head_dy"]

    return mirrored


def to_int_point(u, v):
    return int(round(float(u))), int(round(float(v)))


def draw_text(img, text, x, y, color=(230, 230, 230)):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, lineType=cv2.LINE_AA)


def project_body_points(body_points, frame_idx, camera, width, height, focal_px, pan_x, pan_y, mirror):
    points_2d = {}
    for name, marker_frames in body_points.items():
        if frame_idx >= len(marker_frames):
            continue
        xyz = marker_frames[frame_idx]
        if not valid_xyz(xyz):
            continue

        u, v, _, in_frame = project(xyz, camera, width, height, focal_px, pan_x, pan_y)
        if not in_frame:
            continue
        if mirror:
            u = mirror_u(u, width)
        points_2d[name] = to_int_point(u, v)
    return points_2d


def draw_skeleton(img, points_2d, line_thickness=4, point_radius=5):
    for a, b in SKELETON_SEGMENTS:
        if a in points_2d and b in points_2d:
            cv2.line(img, points_2d[a], points_2d[b], (135, 135, 135), line_thickness, lineType=cv2.LINE_AA)

    for pt in points_2d.values():
        cv2.circle(img, pt, point_radius, (205, 205, 205), -1, lineType=cv2.LINE_AA)


def safe_name(value):
    keep = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "view"


def render_camera_preview_video(
    args,
    c3d_path,
    sequence_id,
    camera_id,
    camera_name,
    mirror,
    tail_3d,
    head_3d,
    body_points,
    source_fps,
    camera,
    focal_px,
    pan_x,
    pan_y,
    yaw,
    pitch,
    roll,
    zoom,
):
    preview_dir = Path(args.preview_dir) / safe_name(sequence_id)
    preview_dir.mkdir(parents=True, exist_ok=True)

    mirror_name = "horizontal" if mirror else "none"
    video_name = f"{safe_name(sequence_id)}_{safe_name(camera_id)}_{safe_name(camera_name)}_{mirror_name}.mp4"
    video_path = preview_dir / video_name

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.preview_fps,
        (args.width, args.height),
    )

    frame_step = max(1, int(round(source_fps / args.preview_fps)))
    trail = []

    for frame_idx in range(0, len(tail_3d), frame_step):
        img = np.zeros((args.height, args.width, 3), dtype=np.uint8)

        body_2d = project_body_points(
            body_points,
            frame_idx,
            camera,
            args.width,
            args.height,
            focal_px,
            pan_x,
            pan_y,
            mirror,
        )
        draw_skeleton(img, body_2d)

        tail = tail_3d[frame_idx]
        head = head_3d[frame_idx]
        if valid_xyz(tail) and valid_xyz(head):
            tail_u, tail_v, _, tail_in_frame = project(tail, camera, args.width, args.height, focal_px, pan_x, pan_y)
            head_u, head_v, _, head_in_frame = project(head, camera, args.width, args.height, focal_px, pan_x, pan_y)
        else:
            tail_in_frame = False
            head_in_frame = False

        if tail_in_frame and head_in_frame:
            if mirror:
                tail_u = mirror_u(tail_u, args.width)
                head_u = mirror_u(head_u, args.width)

            tail_pt = to_int_point(tail_u, tail_v)
            head_pt = to_int_point(head_u, head_v)
            trail.append((tail_pt, head_pt))

            for i, (old_tail, old_head) in enumerate(trail[-args.preview_trail_frames:]):
                alpha = (i + 1) / max(args.preview_trail_frames, 1)
                color = (int(20 * alpha), int(130 * alpha), int(160 * alpha))
                cv2.line(img, old_tail, old_head, color, 2, lineType=cv2.LINE_AA)

            cv2.line(img, tail_pt, head_pt, (0, 220, 255), 8, lineType=cv2.LINE_AA)
            cv2.circle(img, tail_pt, 13, (0, 70, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(img, head_pt, 13, (255, 80, 40), -1, lineType=cv2.LINE_AA)
            draw_text(img, "tail", tail_pt[0] + 18, tail_pt[1] + 6, (0, 120, 255))
            draw_text(img, "head", head_pt[0] + 18, head_pt[1] + 6, (255, 120, 80))

        draw_text(img, f"frame {frame_idx}", 30, 50)
        draw_text(img, f"{sequence_id} / {camera_id} / {camera_name} / mirror={mirror_name}", 30, 86, (180, 180, 180))
        draw_text(
            img,
            f"yaw={yaw:.2f} pitch={pitch:.2f} roll={roll:.2f} zoom={zoom:.3f} pan=({pan_x:.3f},{pan_y:.3f})",
            30,
            122,
            (180, 180, 180),
        )
        draw_text(img, f"source: {c3d_path}", 30, 158, (150, 150, 150))
        writer.write(img)

    writer.release()
    return video_path


def process_file(c3d_path, args, rng):
    tail_3d, head_3d, body_points, fps, unit_mode = read_motion_points(c3d_path)
    scene_body_points = body_points if args.include_skeleton_in_bounds else None
    scene_pts = valid_scene_points(tail_3d, head_3d, scene_body_points)
    if scene_pts is None:
        return []

    center = scene_pts.mean(axis=0)
    radius = np.linalg.norm(scene_pts.max(axis=0) - scene_pts.min(axis=0)) * 0.5
    base_focal_px = (BASE_FOCAL_MM / SENSOR_WIDTH_MM) * args.width

    rows = []
    rel_path = str(c3d_path).replace("\\", "/")
    sequence_id = c3d_path.stem
    batter_hand = parse_batter_hand(c3d_path)

    for cam_idx, (camera_type, camera_name, yaw, pitch, roll, zoom, pan_x, pan_y) in enumerate(
        camera_specs(args, rng)
    ):
        camera = make_camera(center, radius, yaw, pitch, roll)
        camera_id = f"cam_{cam_idx:03d}"
        focal_px = base_focal_px * zoom
        print(
            "[view] "
            f"sequence={sequence_id} "
            f"camera={camera_id} "
            f"name={camera_name} "
            f"type={camera_type} "
            f"yaw={yaw:.2f} "
            f"pitch={pitch:.2f} "
            f"roll={roll:.2f} "
            f"zoom={zoom:.3f} "
            f"pan=({pan_x:.3f},{pan_y:.3f})"
        )

        for frame_idx, (tail, head) in enumerate(zip(tail_3d, head_3d)):
            if not (valid_xyz(tail) and valid_xyz(head)):
                continue

            tail_u, tail_v, tail_depth, tail_in_frame = project(
                tail, camera, args.width, args.height, focal_px, pan_x, pan_y
            )
            head_u, head_v, head_depth, head_in_frame = project(
                head, camera, args.width, args.height, focal_px, pan_x, pan_y
            )

            if args.keep_only_visible and not (tail_in_frame and head_in_frame):
                continue

            dx = tail_u - head_u
            dy = tail_v - head_v
            pixel_len = math.sqrt(dx * dx + dy * dy) if np.isfinite(dx + dy) else np.nan

            row = {
                "source_c3d": rel_path,
                "sequence_id": sequence_id,
                "camera_id": camera_id,
                "camera_type": camera_type,
                "camera_name": camera_name,
                "mirror": "none",
                "batter_hand": batter_hand,
                "frame_idx": frame_idx,
                "time_sec": frame_idx / fps if fps > 0 else "",
                "unit_mode": unit_mode,
                "image_width": args.width,
                "image_height": args.height,
                "camera_yaw_deg": camera["yaw_deg"],
                "camera_pitch_deg": camera["pitch_deg"],
                "camera_roll_deg": camera["roll_deg"],
                "camera_zoom": zoom,
                "camera_pan_x": pan_x,
                "camera_pan_y": pan_y,
                "camera_distance_m": camera["distance"],
                "head_u": head_u,
                "head_v": head_v,
                "head_depth": head_depth,
                "head_in_frame": head_in_frame,
                "tail_u": tail_u,
                "tail_v": tail_v,
                "tail_depth": tail_depth,
                "tail_in_frame": tail_in_frame,
                "tail_from_head_dx": dx,
                "tail_from_head_dy": dy,
                "bat_pixel_length": pixel_len,
                "head_3d_x": head[0],
                "head_3d_y": head[1],
                "head_3d_z": head[2],
                "tail_3d_x": tail[0],
                "tail_3d_y": tail[1],
                "tail_3d_z": tail[2],
            }

            if args.include_skeleton:
                row.update(
                    skeleton_projection_row(
                        body_points,
                        frame_idx,
                        camera,
                        args.width,
                        args.height,
                        focal_px,
                        pan_x,
                        pan_y,
                    )
                )

            rows.append(row)
            if args.horizontal_flip:
                rows.append(horizontal_mirror_row(row, args.width))

        if not args.no_preview_videos:
            preview_path = render_camera_preview_video(
                args,
                c3d_path,
                sequence_id,
                camera_id,
                camera_name,
                False,
                tail_3d,
                head_3d,
                body_points,
                fps,
                camera,
                focal_px,
                pan_x,
                pan_y,
                yaw,
                pitch,
                roll,
                zoom,
            )
            print(
                "[preview] "
                f"sequence={sequence_id} "
                f"camera={camera_id} "
                f"mirror=none "
                f"path={preview_path}"
            )
            if args.horizontal_flip:
                preview_path = render_camera_preview_video(
                    args,
                    c3d_path,
                    sequence_id,
                    camera_id,
                    camera_name,
                    True,
                    tail_3d,
                    head_3d,
                    body_points,
                    fps,
                    camera,
                    focal_px,
                    pan_x,
                    pan_y,
                    yaw,
                    pitch,
                    roll,
                    zoom,
                )
                print(
                    "[preview] "
                    f"sequence={sequence_id} "
                    f"camera={camera_id} "
                    f"mirror=horizontal "
                    f"path={preview_path}"
                )

    return rows


def write_single_csv(c3d_files, args, rng):
    if args.split_by_folder:
        write_split_by_folder(c3d_files, args, rng)
    else:
        write_single_csv(c3d_files, args, rng)


def write_split_by_folder(c3d_files, args, rng):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writers = {}
    handles = {}
    fieldnames_by_folder = {}
    folder_rows = {}
    folder_files = {}
    skipped = 0

    try:
        for c3d_path in c3d_files:
            folder_name = safe_name(c3d_path.parent.name)
            out_path = out_dir / f"{folder_name}.csv"

            try:
                rows = process_file(c3d_path, args, rng)
            except Exception as exc:
                skipped += 1
                print(f"[skip] {c3d_path}: {exc}")
                continue

            if not rows:
                skipped += 1
                continue

            if folder_name not in writers:
                mode = "a" if args.append else "w"
                fp = open(out_path, mode, newline="", encoding="utf-8")
                handles[folder_name] = fp
                fieldnames = list(rows[0].keys())
                fieldnames_by_folder[folder_name] = fieldnames
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writers[folder_name] = writer
                if not args.append:
                    writer.writeheader()

            writers[folder_name].writerows(rows)
            folder_rows[folder_name] = folder_rows.get(folder_name, 0) + len(rows)
            folder_files[folder_name] = folder_files.get(folder_name, 0) + 1

            print(
                "[folder-progress] "
                f"folder={folder_name} "
                f"files={folder_files[folder_name]} "
                f"rows={folder_rows[folder_name]} "
                f"csv={out_path}"
            )
    finally:
        for fp in handles.values():
            fp.close()

    total_files = sum(folder_files.values())
    total_rows = sum(folder_rows.values())
    print(f"saved_dir: {out_dir}")
    print(f"folders: {len(folder_files)}")
    print(f"processed_files: {total_files}")
    print(f"skipped_files: {skipped}")
    print(f"rows: {total_rows}")


def main():
    parser = argparse.ArgumentParser(
        description="Project C3D bat head/tail 3D coordinates into synthetic 2D coordinate sequences."
    )
    parser.add_argument("--c3d-root", default="c3d")
    parser.add_argument("--out", default="bat_2d_coordinates.csv")
    parser.add_argument("--out-dir", default="bat_2d_coordinates_by_folder")
    parser.add_argument("--split-by-folder", action="store_true")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--view-config", default=None)
    parser.add_argument("--fixed-yaws", default="0,45,90,135,180")
    parser.add_argument("--fixed-pitch-range", default="6,10")
    parser.add_argument("--fixed-roll-range", default="-1,1")
    parser.add_argument("--fixed-zoom-range", default="0.95,1.05")
    parser.add_argument("--fixed-pan-x-range", default="-0.04,0.04")
    parser.add_argument("--fixed-pan-y-range", default="-0.03,0.03")
    parser.add_argument("--random-views", type=int, default=10)
    parser.add_argument("--random-yaw-range", default="0,360")
    parser.add_argument("--random-pitch-range", default="-12,28")
    parser.add_argument("--random-roll-range", default="-4,4")
    parser.add_argument("--random-zoom-range", default="0.7,1.4")
    parser.add_argument("--random-pan-x-range", default="-0.18,0.18")
    parser.add_argument("--random-pan-y-range", default="-0.14,0.14")
    parser.add_argument("--include-skeleton", action="store_true")
    parser.add_argument("--include-skeleton-in-bounds", action="store_true")
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--keep-only-visible", action="store_true")
    parser.add_argument("--preview-dir", default="bat_2d_preview_videos")
    parser.add_argument("--preview-fps", type=float, default=60.0)
    parser.add_argument("--preview-trail-frames", type=int, default=25)
    parser.add_argument("--no-preview-videos", action="store_true")
    parser.add_argument("--include-model-c3d", action="store_true")
    parser.add_argument("--only-uppercase-bat-markers", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    c3d_files = sorted(Path(args.c3d_root).glob("*/*.c3d"))
    if not args.include_model_c3d:
        c3d_files = [path for path in c3d_files if "model" not in path.name.lower()]
    if args.only_uppercase_bat_markers:
        filtered = []
        for path in c3d_files:
            try:
                if has_uppercase_only_bat_markers(path):
                    filtered.append(path)
            except Exception as exc:
                print(f"[skip-filter] {path}: {exc}")
        c3d_files = filtered
    if args.max_files is not None:
        c3d_files = c3d_files[: args.max_files]
    if not c3d_files:
        raise SystemExit(f"No .c3d files found under {args.c3d_root}")

    if args.split_by_folder:
        write_split_by_folder(c3d_files, args, rng)
    else:
        write_single_csv(c3d_files, args, rng)


if __name__ == "__main__":
    main()
