import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refinement"))

from clean_input.adapters import FLOW_SOURCE_COLUMNS
from clean_input.pipelines import (
    build_evaluation_observation,
    build_export_observation,
    build_inference_observation,
    build_train_observation,
    build_validation_observation,
)


class PathConsistencyTests(unittest.TestCase):
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
                "gt_x": index * 1000,
                "visibility": "ignore" if index % 2 else "fully_visible",
                "status": "missing_prediction",
            }
            row.update({name: index + 1 for name in FLOW_SOURCE_COLUMNS})
            rows.append(row)
        return rows

    def test_all_paths_produce_identical_no_flow_features(self):
        kwargs = dict(image_width=1920, image_height=1080, view_label="front")
        outputs = [fn(self._rows(), **kwargs).features for fn in (
            build_train_observation, build_validation_observation,
            build_evaluation_observation, build_export_observation,
            build_inference_observation,
        )]
        for output in outputs[1:]:
            np.testing.assert_array_equal(outputs[0], output)

    def test_all_paths_produce_identical_flow_features(self):
        kwargs = dict(image_width=1920, image_height=1080, view_label="3b_side", use_flow=True, flow_clip_px=50)
        outputs = [fn(self._rows(), **kwargs).features for fn in (
            build_train_observation, build_validation_observation,
            build_evaluation_observation, build_export_observation,
            build_inference_observation,
        )]
        self.assertEqual(outputs[0].shape[1], 45)
        for output in outputs[1:]:
            np.testing.assert_array_equal(outputs[0], output)


if __name__ == "__main__":
    unittest.main()
