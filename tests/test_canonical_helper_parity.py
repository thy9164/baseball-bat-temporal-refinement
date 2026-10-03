import hashlib
import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REFINEMENT = Path(__file__).resolve().parents[1] / "refinement"
if str(REFINEMENT) not in sys.path:
    sys.path.insert(0, str(REFINEMENT))

from src import data_utils
from src import evaluation_metrics


def summary_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FixedSampler:
    def sample(self, rng):
        errors = rng.normal(0.0, 0.01, (5, 8)).astype(np.float32)
        errors[:, 2] = 0.8
        errors[:, 3] = [1, 1, 0, 1, 1]
        errors[:, 6] = 0.7
        errors[:, 7] = [1, 0, 1, 1, 1]
        return {
            "errors": errors,
            "head_status": ["matched"] * 5,
            "tail_status": ["matched"] * 5,
        }


class CanonicalDataHelperParityTests(unittest.TestCase):
    def test_scalar_row_and_window_edge_cases(self):
        self.assertTrue(math.isnan(data_utils.parse_float(None)))
        self.assertEqual(data_utils.parse_float("", -1.0), -1.0)
        self.assertEqual(data_utils.parse_float("bad", 7.0), 7.0)
        self.assertEqual(data_utils.clean_keypoint_columns(["head_u", "head_v", "tail_u", "tail_v"]),
                         ("head_u", "head_v", "tail_u", "tail_v"))
        row = {"sequence_id": "swing", "camera_id": "side", "mirror": "h", "image_stem": "frame_000007"}
        self.assertEqual(data_utils.row_swing_id(row), "swing")
        self.assertEqual(data_utils.row_view_id(row), "side__h")
        # Historical parser falls back to 0 for a non-numeric image_stem.
        self.assertEqual(data_utils.row_frame_idx(row), 0)
        self.assertEqual(data_utils.row_frame_idx({"frame": "7"}), 7)
        self.assertEqual(data_utils.tail_aligned_starts(3, 5, 2, False), [])
        self.assertEqual(data_utils.tail_aligned_starts(31, 31, 5, True), [0])
        self.assertEqual(data_utils.tail_aligned_starts(63, 31, 5, True), [32, 27, 22, 17, 12, 7, 2, 0])
        self.assertTrue(data_utils.usable_row({"x": "1", "y": "2"}, ("x", "y")))
        self.assertFalse(data_utils.usable_row({"x": "", "y": "2"}, ("x", "y")))

    def test_split_and_numeric_helpers_match_historical_golden_values(self):
        self.assertEqual(
            data_utils.make_splits(["a", "b", "c", "d", "e"], 0.6, 0.2, 7),
            {"train": ["e", "a", "d"], "val": ["b"], "test": ["c"]},
        )
        self.assertEqual(data_utils.split_lookup({"train": ["a"], "val": ["b"]}), {"a": "train", "b": "val"})
        self.assertEqual(data_utils.infer_downsample_step([0.0, 1 / 360, 2 / 360], 60, 360), 6)
        self.assertEqual(data_utils.normalize_xy(960, 540, 1920, 1080), (0.5, 0.5))
        np.testing.assert_array_equal(data_utils.clip01(np.asarray([-1.0, 0.5, 2.0])), [0.0, 0.5, 1.0])

        seq8 = np.asarray([[0.1, 0.2, 0.4, 0.6, 0.9, 0.8, 1.0, 1.0]], dtype=np.float32)
        expected = np.asarray([[0.1, 0.2, 0.4, 0.6, 0.9, 0.8, 1.0, 1.0, 0.3, 0.4]], dtype=np.float32)
        np.testing.assert_allclose(data_utils.add_bat_vector_features(seq8), expected, rtol=0.0, atol=1e-7)

    def test_fixed_seed_corruption_matches_historical_golden_hash(self):
        clean = np.asarray(
            [[0.2 + index * 0.01, 0.3, 0.6, 0.7 - index * 0.01] for index in range(5)],
            dtype=np.float32,
        )
        features, head_status, tail_status = data_utils.corrupt_clean_window(
            clean, FixedSampler(), np.random.default_rng(123)
        )
        digest = hashlib.sha256()
        digest.update(features.tobytes())
        digest.update(json.dumps([head_status, tail_status], separators=(",", ":")).encode("utf-8"))
        self.assertEqual(digest.hexdigest(), "cee77db7574f2d736897b2690f2399b7ed40357d3b75db2732b1863ba705afc9")


class CanonicalEvaluationHelperParityTests(unittest.TestCase):
    def assert_summary_matches(self, actual, expected, path="summary"):
        # Preserve structure, counts, and missing values exactly. Float32
        # percentile interpolation can vary slightly across NumPy versions.
        if isinstance(expected, dict):
            self.assertIsInstance(actual, dict, path)
            self.assertEqual(set(actual), set(expected), path)
            for key in expected:
                self.assert_summary_matches(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, float):
            self.assertIsInstance(actual, float, path)
            self.assertTrue(math.isfinite(actual), path)
            np.testing.assert_allclose(
                actual, expected, rtol=1e-7, atol=1e-7, equal_nan=False,
                err_msg=path,
            )
        else:
            self.assertIs(type(actual), type(expected), path)
            self.assertEqual(actual, expected, path)

    def setUp(self):
        self.masks = {
            "all": np.asarray([1, 1, 1, 1], dtype=bool),
            "vis": np.asarray([1, 1, 0, 0], dtype=bool),
        }
        self.target_valid = np.asarray([1, 1, 1, 0], dtype=bool)
        self.raw_valid = np.asarray([1, 1, 0, 1], dtype=bool)
        self.raw = np.asarray([2.0, 10.0, np.nan, 4.0])
        self.filled = np.asarray([2.0, 9.0, 12.0, 4.0])
        self.final = np.asarray([1.0, 12.0, 8.0, 3.0])
        self.args = SimpleNamespace(
            correction_gain_threshold_px=1.0,
            correction_shift_threshold_px=1.0,
            prior_gain_threshold_px=1.0,
        )

    def test_visibility_point_error_and_aggregate_stats(self):
        self.assertEqual(
            evaluation_metrics.VIS_NAMES,
            {0: "fully_visible", 1: "partially_occluded", 2: "fully_occluded", 3: "ignore"},
        )
        pred = np.asarray([[0.1, 0.2], [0.5, 0.7]], dtype=np.float32)
        target = np.asarray([[0.2, 0.2], [0.45, 0.65]], dtype=np.float32)
        expected = np.linalg.norm((pred - target) * np.asarray([1920, 1080], dtype=np.float32), axis=-1)
        np.testing.assert_array_equal(evaluation_metrics.point_errors_px(pred, target, 1920, 1080), expected)
        self.assertEqual(evaluation_metrics.stats([]),
                         {"count": 0, "rmse": None, "mae": None, "p90": None, "gt20": None, "gt40": None})
        self.assertEqual(evaluation_metrics.scalar_stats([]),
                         {"count": 0, "mean": None, "median": None, "p10": None, "p90": None})

    def test_effect_summaries_match_historical_golden_values(self):
        shift = np.asarray([1.2, 3.0, 5.0, np.nan])
        delta = np.asarray([1.0, 2.0, 4.0, 0.0])
        correction = evaluation_metrics.add_correction_effect(
            {}, self.masks, self.target_valid, self.raw_valid, self.raw,
            self.filled, self.final, shift, self.args, delta,
        )
        prior = evaluation_metrics.add_prior_effect(
            {}, self.masks, self.target_valid, self.raw_valid, self.raw,
            self.filled, np.asarray([1.5, 8.0, 9.0, 3.0]), self.args,
        )
        trust = evaluation_metrics.add_trust_effect(
            {}, self.masks, self.target_valid, self.raw_valid, self.raw,
            self.filled, self.final, np.asarray([0.1, 0.6, 0.9, np.nan]), self.args,
        )
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "historical_effect_summaries.json")
            .read_text(encoding="utf-8")
        )
        for name, actual in {"correction": correction, "prior": prior, "trust": trust}.items():
            with self.subTest(summary=name):
                expected = fixture["summaries"][name]
                # The numeric fixture reproduces the original historical hashes;
                # do not regenerate it from the environment running this test.
                self.assertEqual(summary_hash(expected),
                                 fixture["provenance"]["historical_summary_sha256"][name])
                self.assert_summary_matches(actual, expected, name)


if __name__ == "__main__":
    unittest.main()
