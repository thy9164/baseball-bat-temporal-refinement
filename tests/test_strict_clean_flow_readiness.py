import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REFINEMENT = ROOT / "refinement"
if str(REFINEMENT) not in sys.path:
    sys.path.insert(0, str(REFINEMENT))

from clean_input.adapters import FLOW_SOURCE_COLUMNS, build_from_wide_rows, normalize_flow_rows
from clean_input.checkpoint import checkpoint_contract, strict_load_model_state, validate_checkpoint_contract
from clean_input.flow_alignment import verify_disjoint_splits, verify_selected_flow_alignment
from clean_input.schema import CLEAN_NO_FLOW_SCHEMA, CLEAN_WITH_FLOW_SCHEMA, FLOW_FEATURES
from src.train_temporal_refinement import (
    EXPANDED_INPUT_WEIGHT_KEYS,
    EXPANSION_TYPE,
    expand_clean25_state_to_clean45,
    flip_flow_features,
)
from src.models import build_model


class StrictCleanFlowReadinessTests(unittest.TestCase):
    def _rows(self):
        rows = []
        for index in range(4):
            row = {
                "head_pred_x": 100 + index,
                "head_pred_y": 200 + index,
                "head_pred_conf": 0.9,
                "tail_pred_x": 400 + index,
                "tail_pred_y": 500 + index,
                "tail_pred_conf": 0.8,
                "head_gt_x": index * 1000,
                "head_gt_y": -index,
                "tail_gt_x": 999,
                "tail_gt_y": 888,
                "head_visibility": "fully_visible",
                "tail_visibility": "fully_occluded",
                "status": "matched",
            }
            row.update({name: float(index + 1) for name in FLOW_SOURCE_COLUMNS})
            rows.append(row)
        return rows

    def test_clean45_prefix_and_raft20_order(self):
        self.assertEqual(CLEAN_WITH_FLOW_SCHEMA[:25], CLEAN_NO_FLOW_SCHEMA)
        self.assertEqual(CLEAN_WITH_FLOW_SCHEMA[25:], FLOW_FEATURES)
        self.assertEqual(
            FLOW_SOURCE_COLUMNS,
            (
                "has_prev_flow", "head_valid", "tail_valid", "bbox_valid",
                "head_flow_x_mean", "head_flow_y_mean", "head_flow_mag_mean", "head_flow_mag_p95",
                "tail_flow_x_mean", "tail_flow_y_mean", "tail_flow_mag_mean", "tail_flow_mag_p95",
                "bbox_flow_x_mean", "bbox_flow_y_mean", "bbox_flow_mag_mean", "bbox_flow_mag_p95",
                "global_flow_x_mean", "global_flow_y_mean", "global_flow_mag_mean", "global_flow_mag_p95",
            ),
        )

    def test_clean45_is_invariant_to_gt_and_evaluation_labels(self):
        rows = self._rows()
        baseline = build_from_wide_rows(
            rows, image_width=1920, image_height=1080, view_label="front", use_flow=True
        ).features
        changed = [dict(row) for row in rows]
        for row in changed:
            row.update(
                head_gt_x=np.nan,
                head_gt_y=np.inf,
                tail_gt_x=-1e9,
                tail_gt_y=42,
                head_visibility="ignore",
                tail_visibility="fully_visible",
                status="missing_prediction",
            )
        actual = build_from_wide_rows(
            changed, image_width=1920, image_height=1080, view_label="front", use_flow=True
        ).features
        np.testing.assert_array_equal(baseline, actual)

    def test_missing_flow_and_horizontal_flip_contract(self):
        row = {name: "" for name in FLOW_SOURCE_COLUMNS}
        normalized = normalize_flow_rows([row], flow_clip_px=50.0)[0]
        np.testing.assert_array_equal(normalized, np.zeros(20, dtype=np.float32))

        values = np.arange(1, 21, dtype=np.float32)
        flipped = flip_flow_features(values)
        for index, name in enumerate(FLOW_SOURCE_COLUMNS):
            expected = -values[index] if "flow_x" in name else values[index]
            self.assertEqual(flipped[index], expected)

    def test_expansion_preserves_all_weights_and_zeroes_only_flow_columns(self):
        torch.manual_seed(7)
        parent = build_model("bigru", input_dim=25, hidden_dim=8, num_layers=2, dropout=0.15, output_dim=10)
        target = build_model("bigru", input_dim=45, hidden_dim=8, num_layers=2, dropout=0.15, output_dim=10)
        expand_clean25_state_to_clean45(parent, target)
        parent_state = parent.state_dict()
        target_state = target.state_dict()
        for key in parent_state:
            if key in EXPANDED_INPUT_WEIGHT_KEYS:
                torch.testing.assert_close(target_state[key][:, :25], parent_state[key], rtol=0, atol=0)
                torch.testing.assert_close(
                    target_state[key][:, 25:], torch.zeros_like(target_state[key][:, 25:]), rtol=0, atol=0
                )
            else:
                torch.testing.assert_close(target_state[key], parent_state[key], rtol=0, atol=0)

    def test_expanded_forward_is_equivalent_with_arbitrary_flow_input(self):
        torch.manual_seed(7)
        parent = build_model("bigru", input_dim=25, hidden_dim=8, num_layers=2, dropout=0.15, output_dim=10)
        target = build_model("bigru", input_dim=45, hidden_dim=8, num_layers=2, dropout=0.15, output_dim=10)
        expand_clean25_state_to_clean45(parent, target)
        parent.eval()
        target.eval()
        x25 = torch.randn(2, 31, 25)
        x45 = torch.cat([x25, torch.randn(2, 31, 20)], dim=-1)
        with torch.no_grad():
            expected = parent(x25)
            actual = target(x45)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_expansion_rejects_architecture_mismatch(self):
        parent = build_model("bigru", input_dim=25, hidden_dim=8, num_layers=2, dropout=0.15, output_dim=10)
        wrong = build_model("bigru", input_dim=45, hidden_dim=9, num_layers=2, dropout=0.15, output_dim=10)
        with self.assertRaisesRegex(ValueError, "input weight shape|architecture"):
            expand_clean25_state_to_clean45(parent, wrong)

    def _write_alignment_pair(self, directory):
        selected = Path(directory) / "selected.csv"
        flow = Path(directory) / "flow.csv"
        selected_fields = [
            "image_stem", "swing_id", "keypoint", "pred_x", "pred_y", "pred_conf",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        ]
        selected_rows = [
            dict(image_stem="frame_1", swing_id="s1", keypoint="head", pred_x=10, pred_y=20, pred_conf=.9,
                 bbox_x1=1, bbox_y1=2, bbox_x2=30, bbox_y2=40),
            dict(image_stem="frame_1", swing_id="s1", keypoint="tail", pred_x=25, pred_y=35, pred_conf=.8,
                 bbox_x1=1, bbox_y1=2, bbox_x2=30, bbox_y2=40),
        ]
        with selected.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=selected_fields)
            writer.writeheader()
            writer.writerows(selected_rows)
        flow_fields = [
            "image_stem", "swing_id", "head_pred_x", "head_pred_y", "head_pred_conf",
            "tail_pred_x", "tail_pred_y", "tail_pred_conf", *(
                name for name in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
            ), *FLOW_SOURCE_COLUMNS,
        ]
        flow_row = {
            "image_stem": "frame_1", "swing_id": "s1",
            "head_pred_x": 10, "head_pred_y": 20, "head_pred_conf": .9,
            "tail_pred_x": 25, "tail_pred_y": 35, "tail_pred_conf": .8,
            "bbox_x1": 1, "bbox_y1": 2, "bbox_x2": 30, "bbox_y2": 40,
            **{name: 0 for name in FLOW_SOURCE_COLUMNS},
        }
        with flow.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=flow_fields)
            writer.writeheader()
            writer.writerow(flow_row)
        return selected, flow

    def test_alignment_guard_records_selected_provenance_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected, flow = self._write_alignment_pair(tmp)
            report = verify_selected_flow_alignment(selected, flow, expected_frames=1, expected_swings=1)
            self.assertTrue(report["selected_detector_observations"])
            self.assertEqual(report["flow_columns"], list(FLOW_SOURCE_COLUMNS))
            verify_disjoint_splits({"train": report})
            text = flow.read_text(encoding="utf-8").replace("25,35,0.8", "25,99,0.8")
            flow.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "detector mismatch"):
                verify_selected_flow_alignment(selected, flow)

    def test_clean45_checkpoint_roundtrip_preserves_contract_lineage_and_output(self):
        torch.manual_seed(7)
        model = build_model("bigru", input_dim=45, hidden_dim=8, num_layers=2, dropout=0.15, output_dim=10)
        model.eval()
        x = torch.randn(1, 31, 45)
        with torch.no_grad():
            expected = model(x)
        contract = checkpoint_contract(
            use_flow_features=True,
            split_identifier="mysplit-v2",
            important_settings={"window_size": 31, "flow_clip_px": 50.0},
        )
        lineage = {
            "current_stage": "strict_clean_real_finetune_45d",
            "oracle_visibility_checkpoint_used": False,
            "expansion_type": EXPANSION_TYPE,
            "parent_checkpoint_sha256": "abc",
            "parent_input_dim": 25,
            "target_input_dim": 45,
            "raft20_feature_order": list(FLOW_FEATURES),
        }
        checkpoint = {
            **contract,
            "model_state": model.state_dict(),
            "model_type": "bigru",
            "hidden_dim": 8,
            "num_layers": 2,
            "dropout": .15,
            "output_dim": 10,
            "seed": 7,
            "lineage": lineage,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            torch.save(checkpoint, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            validate_checkpoint_contract(loaded, use_flow_features=True)
            restored = build_model("bigru", input_dim=45, hidden_dim=8, num_layers=2, dropout=.15, output_dim=10)
            strict_load_model_state(restored, loaded)
            restored.eval()
            with torch.no_grad():
                actual = restored(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(loaded["seed"], 7)
            self.assertEqual(loaded["lineage"], lineage)


if __name__ == "__main__":
    unittest.main()
