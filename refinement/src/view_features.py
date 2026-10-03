import numpy as np


VIEW_LABELS = ["1b_side", "front", "3b_side", "home_side", "random"]
VIEW_TO_ID = {name: idx for idx, name in enumerate(VIEW_LABELS)}
ID_TO_VIEW = {idx: name for name, idx in VIEW_TO_ID.items()}


def normalize_batter_hand(value):
    text = str(value or "").strip().upper()
    if text.startswith("L"):
        return "L"
    if text.startswith("R"):
        return "R"
    return "R"


def should_flip_to_right(batter_hand):
    return normalize_batter_hand(batter_hand) == "L"


def normalize_view_label(camera_name):
    text = str(camera_name or "").strip().lower()
    if "1b" in text or "first" in text:
        return "1b_side"
    if "3b" in text or "third" in text:
        return "3b_side"
    if "front" in text:
        return "front"
    if "home" in text:
        return "home_side"
    return "random"


def swap_side_view(view_label):
    if view_label == "1b_side":
        return "3b_side"
    if view_label == "3b_side":
        return "1b_side"
    return view_label


def canonical_view_label(camera_name, batter_hand):
    view_label = normalize_view_label(camera_name)
    if should_flip_to_right(batter_hand):
        return swap_side_view(view_label)
    return view_label


def view_one_hot(view_label):
    out = np.zeros(len(VIEW_LABELS), dtype=np.float32)
    out[VIEW_TO_ID.get(view_label, VIEW_TO_ID["random"])] = 1.0
    return out


def append_view_features(x, view_label):
    onehot = np.tile(view_one_hot(view_label), (len(x), 1))
    return np.concatenate([x, onehot], axis=1).astype(np.float32)


def flip_x_columns(arr, columns):
    out = arr.copy()
    for col in columns:
        values = out[:, col]
        finite = np.isfinite(values)
        out[finite, col] = 1.0 - values[finite]
    return out


def flip_clean_xy4(clean_xy4):
    return flip_x_columns(clean_xy4, [0, 2])


def flip_seq8(seq8):
    return flip_x_columns(seq8, [0, 2])


def flip_xy2(xy):
    out = xy.copy()
    out[..., 0] = np.where(np.isfinite(out[..., 0]), 1.0 - out[..., 0], out[..., 0])
    return out
