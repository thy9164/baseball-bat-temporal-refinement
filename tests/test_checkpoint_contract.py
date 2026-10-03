import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refinement"))

from clean_input.checkpoint import (
    checkpoint_contract,
    checkpoint_sha256,
    strict_load_model_state,
    validate_checkpoint_contract,
    verified_strict_clean_parent,
)


class FakeModel:
    def __init__(self):
        self.call = None

    def load_state_dict(self, state, strict):
        self.call = (state, strict)
        return "loaded"


class CheckpointContractTests(unittest.TestCase):
    def test_contract_records_schema_dimension_split_and_settings(self):
        contract = checkpoint_contract(use_flow_features=False, split_identifier="mysplit-v2", important_settings={"window_size": 31})
        self.assertEqual(contract["input_dim"], 25)
        self.assertEqual(len(contract["feature_schema"]), 25)
        self.assertEqual(contract["split_identifier"], "mysplit-v2")
        validate_checkpoint_contract(contract, use_flow_features=False, expected_split_identifier="mysplit-v2", important_settings={"window_size": 31})

    def test_legacy_and_schema_mismatches_are_rejected(self):
        for legacy_dim in (29, 49):
            with self.assertRaisesRegex(ValueError, "missing contract"):
                validate_checkpoint_contract({"input_dim": legacy_dim}, use_flow_features=False)
        contract = checkpoint_contract(use_flow_features=True, split_identifier="x", important_settings={})
        with self.assertRaisesRegex(ValueError, "schema/order"):
            validate_checkpoint_contract(contract, use_flow_features=False)

    def test_state_loading_is_strict(self):
        model = FakeModel()
        checkpoint = {"model_state": {"weight": 1}}
        self.assertEqual(strict_load_model_state(model, checkpoint), "loaded")
        self.assertEqual(model.call, ({"weight": 1}, True))

    def test_verified_synthetic_parent_records_generic_non_oracle_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            path.write_bytes(b"strict-clean-parent")
            checkpoint = checkpoint_contract(
                use_flow_features=False,
                split_identifier="synthetic:strict_clean25_synthetic_pretrain_seed7_shards",
                important_settings={"window_size": 31},
            )
            checkpoint.update({"task": "prior_guided_tail_residual_trust_refiner_v8", "epoch": 487, "seed": 7})
            summary = {
                "checkpoint_contract": "validated",
                "input_dim": 25,
                "feature_schema_version": checkpoint["feature_schema_version"],
                "best_epoch": 487,
                "seed": 7,
                "best_checkpoint_sha256": checkpoint_sha256(path),
            }
            path.with_name("final_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            lineage = verified_strict_clean_parent(checkpoint, path, use_flow_features=False)
            self.assertEqual(lineage["parent_stage"], "strict_clean_synthetic_pretrain_25d")
            self.assertEqual(lineage["current_stage"], "strict_clean_real_finetune_25d")
            self.assertFalse(lineage["oracle_visibility_checkpoint_used"])

    def test_oracle_or_unverified_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            path.write_bytes(b"unverified")
            checkpoint = checkpoint_contract(
                use_flow_features=False,
                split_identifier="train:legacy|val:legacy",
                important_settings={},
            )
            checkpoint["lineage"] = {
                "current_stage": "diagnostic_legacy_initialized_clean25",
                "oracle_visibility_checkpoint_used": True,
            }
            with self.assertRaisesRegex(ValueError, "missing adjacent final_summary"):
                verified_strict_clean_parent(checkpoint, path, use_flow_features=False)


if __name__ == "__main__":
    unittest.main()
