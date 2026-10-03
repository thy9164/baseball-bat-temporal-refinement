import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms.functional import pil_to_tensor


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = pil_to_tensor(img).float() / 255.0
    return tensor


def pad_to_multiple_of_8(img: torch.Tensor):
    """
    RAFT usually works better when H and W are divisible by 8.
    img shape: [C, H, W]
    """
    _, h, w = img.shape
    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8

    pad_h = new_h - h
    pad_w = new_w - w

    img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
    return img, h, w


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """
    flow shape: [2, H, W]
    output: BGR image for cv2.imwrite
    """
    fx = flow[0]
    fy = flow[1]

    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)

    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = ang / 2
    hsv[..., 1] = 255

    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = mag_norm.astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--small-test", type=int, default=0, help="Only process first N pairs. 0 means all.")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)

    flow_dir = out_dir / "flow_npy"
    vis_dir = out_dir / "flow_vis"

    flow_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(frames_dir.glob("*.jpg"))
        + list(frames_dir.glob("*.png"))
        + list(frames_dir.glob("*.jpeg"))
    )

    if len(image_paths) < 2:
        raise ValueError(f"Need at least 2 images in {frames_dir}, but got {len(image_paths)}")

    if args.small_test > 0:
        image_paths = image_paths[: args.small_test + 1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")
    print(f"[INFO] frames: {len(image_paths)}")
    print(f"[INFO] pairs : {len(image_paths) - 1}")

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True).to(device)
    model.eval()

    transforms = weights.transforms()

    for i in tqdm(range(len(image_paths) - 1), desc="RAFT flow"):
        img1_path = image_paths[i]
        img2_path = image_paths[i + 1]

        img1 = load_image(img1_path)
        img2 = load_image(img2_path)

        img1, orig_h, orig_w = pad_to_multiple_of_8(img1)
        img2, _, _ = pad_to_multiple_of_8(img2)

        batch1, batch2 = transforms(img1.unsqueeze(0), img2.unsqueeze(0))
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        flow_predictions = model(batch1, batch2)
        flow = flow_predictions[-1][0]  # [2, H, W]
        flow = flow[:, :orig_h, :orig_w]
        flow_np = flow.detach().cpu().numpy().astype(np.float32)

        stem1 = img1_path.stem
        stem2 = img2_path.stem
        out_name = f"{stem1}_to_{stem2}.npy"

        np.save(flow_dir / out_name, flow_np)

        if args.save_vis:
            vis = flow_to_rgb(flow_np)
            cv2.imwrite(str(vis_dir / f"{stem1}_to_{stem2}.jpg"), vis)

    print(f"[DONE] flow npy saved to: {flow_dir}")
    if args.save_vis:
        print(f"[DONE] flow vis saved to: {vis_dir}")


if __name__ == "__main__":
    main()