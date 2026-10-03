"""Strict metadata contract for clean-input checkpoints."""

import hashlib
import json
from pathlib import Path

from .schema import FEATURE_SCHEMA_VERSION, schema_for

REQUIRED_CONTRACT_KEYS = (
    "feature_schema_version",
    "feature_schema",
    "input_dim",
    "split_identifier",
    "important_settings",
)


def checkpoint_contract(*, use_flow_features, split_identifier, important_settings):
    if not split_identifier:
        raise ValueError("split_identifier is required")
    schema = schema_for(bool(use_flow_features))
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema": list(schema),
        "input_dim": len(schema),
        "split_identifier": str(split_identifier),
        "important_settings": dict(important_settings),
    }


def validate_checkpoint_contract(checkpoint, *, use_flow_features, expected_split_identifier=None, important_settings=None):
    missing = [key for key in REQUIRED_CONTRACT_KEYS if key not in checkpoint]
    if missing:
        raise ValueError(f"legacy or incomplete checkpoint; missing contract keys: {missing}")
    expected_schema = list(schema_for(bool(use_flow_features)))
    if checkpoint["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("checkpoint feature schema version mismatch")
    if checkpoint["feature_schema"] != expected_schema:
        raise ValueError("checkpoint feature schema/order mismatch")
    if checkpoint["input_dim"] != len(expected_schema):
        raise ValueError("checkpoint input_dim mismatch")
    if expected_split_identifier is not None and checkpoint["split_identifier"] != expected_split_identifier:
        raise ValueError("checkpoint split identifier mismatch")
    for key, expected in (important_settings or {}).items():
        actual = checkpoint["important_settings"].get(key)
        if actual != expected:
            raise ValueError(f"checkpoint setting mismatch for {key}: {actual!r} != {expected!r}")
    return checkpoint


def strict_load_model_state(model, checkpoint):
    if "model_state" not in checkpoint:
        raise ValueError("checkpoint is missing model_state")
    return model.load_state_dict(checkpoint["model_state"], strict=True)


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_strict_clean_parent(checkpoint, checkpoint_path, *, use_flow_features, current_stage=None):
    """Return lineage metadata only after strict checkpoint and run-record checks."""
    validate_checkpoint_contract(checkpoint, use_flow_features=use_flow_features)
    path = Path(checkpoint_path)
    digest = checkpoint_sha256(path)
    lineage = checkpoint.get("lineage") or {}
    inherited_stage = lineage.get("current_stage") or checkpoint.get("stage")
    inherited_oracle = lineage.get("oracle_visibility_checkpoint_used")

    if inherited_stage and str(inherited_stage).startswith("strict_clean_") and inherited_oracle is False:
        parent_stage = str(inherited_stage)
    else:
        summary_path = path.with_name("final_summary.json")
        if not summary_path.is_file():
            raise ValueError("strict clean parent is missing adjacent final_summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        split_identifier = str(checkpoint.get("split_identifier", ""))
        is_verified_synthetic = (
            split_identifier.startswith("synthetic:strict_clean25_synthetic_pretrain_seed7")
            and checkpoint.get("task") == "prior_guided_tail_residual_trust_refiner_v8"
            and summary.get("checkpoint_contract") == "validated"
            and summary.get("input_dim") == checkpoint.get("input_dim")
            and summary.get("feature_schema_version") == checkpoint.get("feature_schema_version")
            and summary.get("best_epoch") == checkpoint.get("epoch")
            and summary.get("seed") == checkpoint.get("seed")
            and summary.get("best_checkpoint_sha256") == digest
        )
        if not is_verified_synthetic:
            raise ValueError("parent checkpoint lacks verified strict-clean, non-oracle lineage")
        parent_stage = "strict_clean_synthetic_pretrain_25d"

    if current_stage is None:
        current_stage = "strict_clean_real_finetune_45d" if use_flow_features else "strict_clean_real_finetune_25d"
    return {
        "parent_stage": parent_stage,
        "current_stage": str(current_stage),
        "oracle_visibility_checkpoint_used": False,
        "parent_checkpoint_sha256": digest,
        "parent_checkpoint_epoch": checkpoint.get("epoch"),
        "parent_feature_schema": list(checkpoint["feature_schema"]),
        "parent_input_dim": int(checkpoint["input_dim"]),
    }
