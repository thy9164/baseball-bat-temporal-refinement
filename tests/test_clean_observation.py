import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refinement"))

from clean_input.observation import build_observation_features
from clean_input.schema import CLEAN_NO_FLOW_SCHEMA, CLEAN_WITH_FLOW_SCHEMA, FLOW_FEATURES


class CleanObservationTests(unittest.TestCase):
    def _build(self, **overrides):
        values = dict(
            head_xy_norm=np.asarray([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], np.float32),
            tail_xy_norm=np.asarray([[0.4, 0.5], [0.5, 0.6], [0.6, 0.7]], np.float32),
            head_confidence=np.asarray([0.9, 0.8, 0.7], np.float32),
            tail_confidence=np.asarray([0.7, 0.6, 0.5], np.float32),
            image_width=1920,
            image_height=1080,
            view_label="1b_side",
        )
        values.update(overrides)
        return build_observation_features(**values)

    def test_schema_widths(self):
        no_flow = self._build()
        self.assertEqual(no_flow.features.shape, (3, 25))
        self.assertEqual(no_flow.feature_schema, CLEAN_NO_FLOW_SCHEMA)
        flow = self._build(flow_features=np.zeros((3, len(FLOW_FEATURES)), np.float32))
        self.assertEqual(flow.features.shape, (3, 45))
        self.assertEqual(flow.feature_schema, CLEAN_WITH_FLOW_SCHEMA)

    def test_nonfinite_x_or_y_or_confidence_is_invalid(self):
        for bad_xy in ([np.nan, 0.2], [0.1, np.inf]):
            head = np.asarray([bad_xy, [0.2, 0.3], [0.3, 0.4]], np.float32)
            batch = self._build(head_xy_norm=head)
            self.assertFalse(batch.head_valid[0])
            self.assertTrue(batch.head_missing[0])
            self.assertEqual(batch.features[0, 6], 0.0)
        batch = self._build(tail_confidence=np.asarray([np.inf, 0.6, 0.5], np.float32))
        self.assertFalse(batch.tail_valid[0])

    def test_explicit_prediction_exists_is_required_when_supplied(self):
        batch = self._build(head_prediction_exists=np.asarray([False, True, True]))
        self.assertFalse(batch.head_valid[0])
        self.assertEqual(batch.features[0, 4], 0.0)

    def test_only_missing_coordinates_are_filled_and_mask_is_preserved(self):
        tail = np.asarray([[0.4, 0.5], [np.nan, np.nan], [0.6, 0.7]], np.float32)
        batch = self._build(tail_xy_norm=tail)
        np.testing.assert_array_equal(batch.tail_missing, [False, True, False])
        np.testing.assert_allclose(batch.features[0, 2:4], [0.4, 0.5])
        np.testing.assert_allclose(batch.features[1, 2:4], [0.4, 0.5])
        np.testing.assert_allclose(batch.features[2, 2:4], [0.6, 0.7])
        self.assertEqual(batch.features[1, 7], 0.0)


if __name__ == "__main__":
    unittest.main()
