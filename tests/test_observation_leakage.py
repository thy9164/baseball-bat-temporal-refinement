import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refinement"))

from clean_input.adapters import build_from_wide_rows
from clean_input.schema import LEGACY_NO_FLOW_SCHEMA, LEGACY_VISIBILITY_FEATURES, CLEAN_NO_FLOW_SCHEMA
from clean_input.supervision import build_supervision


class ObservationLeakageTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"head_pred_x": 100, "head_pred_y": 200, "head_pred_conf": .9, "tail_pred_x": 400, "tail_pred_y": 500, "tail_pred_conf": .8,
             "head_gt_x": 101, "head_gt_y": 201, "tail_gt_x": 402, "tail_gt_y": 503, "head_visibility": "fully_visible", "tail_visibility": "fully_occluded", "status": "matched"},
            {"head_pred_x": 110, "head_pred_y": 210, "head_pred_conf": .85, "tail_pred_x": 410, "tail_pred_y": 510, "tail_pred_conf": .75,
             "head_gt_x": 999, "head_gt_y": 999, "tail_gt_x": 999, "tail_gt_y": 999, "head_visibility": "ignore", "tail_visibility": "ignore", "status": "ignore_with_prediction"},
        ]

    def _features(self, rows):
        return build_from_wide_rows(rows, image_width=1920, image_height=1080, view_label="front").features

    def test_visibility_delete_change_shuffle_does_not_change_observation(self):
        baseline = self._features(self.rows)
        changed = copy.deepcopy(self.rows)
        changed[0]["tail_visibility"] = "ignore"
        changed[1]["tail_visibility"] = "fully_visible"
        for row in changed:
            row.pop("head_visibility", None)
        np.testing.assert_array_equal(baseline, self._features(changed))

    def test_gt_coordinates_do_not_change_observation(self):
        baseline = self._features(self.rows)
        changed = copy.deepcopy(self.rows)
        for row in changed:
            row.update(head_gt_x=np.nan, head_gt_y=-1e9, tail_gt_x=42, tail_gt_y=np.inf)
        np.testing.assert_array_equal(baseline, self._features(changed))

    def test_evaluation_status_does_not_change_observation(self):
        baseline = self._features(self.rows)
        changed = copy.deepcopy(self.rows)
        changed[0]["status"] = "missing_prediction"
        changed[1].pop("status")
        np.testing.assert_array_equal(baseline, self._features(changed))

    def test_supervision_is_separate(self):
        baseline = self._features(self.rows)
        supervision = build_supervision(
            gt_head_px=[[1, 2], [3, 4]], gt_tail_px=[[5, 6], [7, 8]],
            head_visibility=["fully_visible", "ignore"], tail_visibility=["fully_occluded", "ignore"],
            image_width=1920, image_height=1080,
        )
        self.assertEqual(supervision.target_valid.tolist(), [1.0, 0.0])
        np.testing.assert_array_equal(baseline, self._features(self.rows))

    def test_clean_order_is_legacy_order_minus_visibility_only(self):
        expected = tuple(name for name in LEGACY_NO_FLOW_SCHEMA if name not in LEGACY_VISIBILITY_FEATURES)
        self.assertEqual(expected, CLEAN_NO_FLOW_SCHEMA)


if __name__ == "__main__":
    unittest.main()
