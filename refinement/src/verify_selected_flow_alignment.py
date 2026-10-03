import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean_input.flow_alignment import verify_disjoint_splits, verify_selected_flow_alignment


EXPECTED = {
    "train": (4201, 53),
    "validation": (646, 11),
    "test": (875, 15),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Verify selected-detector and RAFT flow CSV alignment.")
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}_selected_csv", required=True)
        parser.add_argument(f"--{split}_flow_csv", required=True)
    parser.add_argument("--output_json")
    return parser.parse_args()


def main():
    args = parse_args()
    reports = {}
    for split, (frames, swings) in EXPECTED.items():
        reports[split] = verify_selected_flow_alignment(
            getattr(args, f"{split}_selected_csv"),
            getattr(args, f"{split}_flow_csv"),
            expected_frames=frames,
            expected_swings=swings,
        )
    verify_disjoint_splits(reports)
    result = {
        "status": "passed",
        "split_overlap": False,
        "flow_features_generated_and_aligned_from_selected_detector_observations": True,
        "splits": reports,
    }
    text = json.dumps(result, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
