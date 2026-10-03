"""Named execution-path entry points backed by one observation adapter."""

from .adapters import build_from_wide_rows


def _build(rows, **kwargs):
    return build_from_wide_rows(rows, **kwargs)


def build_train_observation(rows, **kwargs):
    return _build(rows, **kwargs)


def build_validation_observation(rows, **kwargs):
    return _build(rows, **kwargs)


def build_evaluation_observation(rows, **kwargs):
    return _build(rows, **kwargs)


def build_export_observation(rows, **kwargs):
    return _build(rows, **kwargs)


def build_inference_observation(rows, **kwargs):
    return _build(rows, **kwargs)
