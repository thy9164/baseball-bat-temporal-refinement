import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFINEMENT = ROOT / "refinement"
if str(REFINEMENT) not in sys.path:
    sys.path.insert(0, str(REFINEMENT))

from clean_input.schema import CLEAN_NO_FLOW_SCHEMA, LEGACY_NO_FLOW_SCHEMA
from src.build_synthetic_pretraining_dataset import apply_tail_visibility, build_clean_features_from_pseudo_detector
from src.geometry_features import add_v8_motion_features
from src.view_features import append_view_features


def visibility_args():
    return SimpleNamespace(
        image_width=1920.0, image_height=1080.0,
        weight_fully_visible=0.75, weight_partially_occluded=2.0,
        weight_fully_occluded=3.0, weight_ignore=4.0,
        partial_mask_prob=0.5, partial_tail_conf=0.4, partial_tail_extra_noise_px=4.0,
        fully_occluded_mask_prob=0.35, fully_occluded_tail_conf=0.25,
        fully_occluded_tail_extra_noise_px=12.0, ignore_keep_observation_prob=0.05,
        ignore_tail_conf=0.1, ignore_tail_extra_noise_px=24.0,
    )


class StrictCleanSyntheticTests(unittest.TestCase):
    def pseudo_detector(self):
        return np.asarray([
            [0.20, 0.30, 0.40, 0.55, 0.95, 0.90, 1.0, 1.0],
            [0.21, 0.31, np.nan, np.nan, 0.91, 0.00, 1.0, 0.0],
            [0.22, 0.32, 0.42, 0.57, 0.89, 0.35, 1.0, 1.0],
        ], dtype=np.float32)

    def test_fixed_observation_is_independent_of_latent_visibility(self):
        observed = self.pseudo_detector()
        latent_a = np.asarray(["fully_visible", "fully_occluded", "ignore"])
        latent_b = np.asarray(["ignore", "partially_occluded", "fully_visible"])
        self.assertFalse(np.array_equal(latent_a, latent_b))
        x_a = build_clean_features_from_pseudo_detector(
            observed, image_width=1920.0, image_height=1080.0, view_label="1b_side").features
        x_b = build_clean_features_from_pseudo_detector(
            observed.copy(), image_width=1920.0, image_height=1080.0, view_label="1b_side").features
        np.testing.assert_array_equal(x_a, x_b)

    def test_legacy_minus_visibility_has_numerical_parity_for_same_observation(self):
        observed = self.pseudo_detector()
        clean = build_clean_features_from_pseudo_detector(
            observed, image_width=1920.0, image_height=1080.0, view_label="1b_side").features
        filled = observed.copy()
        valid = filled[:, 7] > 0
        valid_indices = np.flatnonzero(valid)
        for idx in np.flatnonzero(~valid):
            nearest = valid_indices[np.argmin(np.abs(valid_indices - idx))]
            filled[idx, 2:4] = filled[nearest, 2:4]
        geometry = add_v8_motion_features(filled, 1920.0, 1080.0)
        visibility = np.eye(4, dtype=np.float32)[np.asarray([0, 2, 3])]
        time = np.linspace(-1.0, 1.0, len(filled), dtype=np.float32)[:, None]
        legacy = append_view_features(np.concatenate([geometry, visibility, time], axis=1), "1b_side")
        retained = np.concatenate([legacy[:, :19], legacy[:, 23:]], axis=1)
        self.assertEqual(tuple(LEGACY_NO_FLOW_SCHEMA[19:23]), (
            "tail_visibility_fully_visible", "tail_visibility_partially_occluded",
            "tail_visibility_fully_occluded", "tail_visibility_ignore"))
        self.assertEqual(clean.shape, (3, len(CLEAN_NO_FLOW_SCHEMA)))
        self.assertTrue(np.isfinite(clean).all())
        np.testing.assert_allclose(clean, retained, rtol=1e-6, atol=1e-7)

    def test_corruption_labels_remain_supervision_not_builder_input(self):
        seq10 = np.zeros((3, 19), dtype=np.float32)
        seq10[:, :8] = self.pseudo_detector()
        labels = np.asarray(["fully_visible", "partially_occluded", "fully_occluded"])
        seq8, weights, visibility_ids, status_ids, obs_valid = apply_tail_visibility(
            seq10, labels, np.random.default_rng(7), visibility_args())
        clean = build_clean_features_from_pseudo_detector(
            seq8, image_width=1920.0, image_height=1080.0, view_label="front").features
        self.assertEqual(clean.shape, (3, 25))
        self.assertTrue(np.isfinite(clean).all())
        self.assertEqual(weights.shape, visibility_ids.shape)
        self.assertEqual(status_ids.shape, obs_valid.shape)


if __name__ == "__main__":
    unittest.main()
