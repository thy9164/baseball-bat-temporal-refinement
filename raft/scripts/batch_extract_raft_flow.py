import argparse
import subprocess
import sys
from pathlib import Path


def count_images(folder: Path) -> int:
    return len(
        list(folder.glob("*.jpg"))
        + list(folder.glob("*.png"))
        + list(folder.glob("*.jpeg"))
    )


def count_flows(folder: Path) -> int:
    flow_dir = folder / "flow_npy"
    if not flow_dir.exists():
        return 0
    return len(list(flow_dir.glob("*.npy")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-root", type=str, required=True)
    parser.add_argument("--outputs-root", type=str, required=True)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--small-test", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs_root = Path(args.inputs_root)
    outputs_root = Path(args.outputs_root)

    swing_dirs = sorted([p for p in inputs_root.iterdir() if p.is_dir()])

    if not swing_dirs:
        raise ValueError(f"No swing folders found in {inputs_root}")

    print(f"[INFO] found {len(swing_dirs)} swing folders")

    for swing_dir in swing_dirs:
        swing_name = swing_dir.name
        out_dir = outputs_root / swing_name

        n_frames = count_images(swing_dir)
        expected_flows = max(0, n_frames - 1)
        existing_flows = count_flows(out_dir)

        if n_frames < 2:
            print(f"[SKIP] {swing_name}: less than 2 frames")
            continue

        # For full extraction, skip already completed swing.
        if args.small_test == 0 and not args.overwrite:
            if existing_flows == expected_flows:
                print(f"[SKIP] {swing_name}: already complete ({existing_flows}/{expected_flows})")
                continue

        cmd = [
            sys.executable,
            str(Path(__file__).with_name("extract_raft_flow.py")),
            "--frames-dir",
            str(swing_dir),
            "--out-dir",
            str(out_dir),
        ]

        if args.save_vis:
            cmd.append("--save-vis")

        if args.small_test > 0:
            cmd.extend(["--small-test", str(args.small_test)])

        print()
        print(f"[RUN] {swing_name}: frames={n_frames}, expected_flows={expected_flows}, existing_flows={existing_flows}")
        print(" ".join(cmd))

        subprocess.run(cmd, check=True)

    print()
    print("[DONE] all swings processed")


if __name__ == "__main__":
    main()
