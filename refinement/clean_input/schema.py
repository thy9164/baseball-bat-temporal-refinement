"""Canonical legacy and clean feature schemas.

The order here is an API. Checkpoints record the complete schema and loaders
must reject any mismatch.
"""

FEATURE_SCHEMA_VERSION = "clean-observation-v1"

DETECTOR_FEATURES = (
    "head_x_norm",
    "head_y_norm",
    "tail_x_norm",
    "tail_y_norm",
    "head_conf",
    "tail_conf",
    "head_valid",
    "tail_valid",
)

BAT_VECTOR_FEATURES = ("bat_dx_norm", "bat_dy_norm")

MOTION_FEATURES = (
    "bat_length_over_image_diagonal",
    "bat_length_change_over_image_diagonal",
    "bat_unit_x",
    "bat_unit_y",
    "bat_direction_change_1_minus_cos",
    "head_speed_over_image_diagonal",
    "tail_speed_over_image_diagonal",
    "head_acceleration_over_image_diagonal",
    "tail_acceleration_over_image_diagonal",
)

LEGACY_VISIBILITY_FEATURES = (
    "tail_visibility_fully_visible",
    "tail_visibility_partially_occluded",
    "tail_visibility_fully_occluded",
    "tail_visibility_ignore",
)

WINDOW_FEATURES = ("window_time_minus1_to_plus1",)

VIEW_FEATURES = (
    "view_1b_side",
    "view_front",
    "view_3b_side",
    "view_home_side",
    "view_random",
)

FLOW_FEATURES = (
    "has_prev_flow",
    "head_flow_region_valid",
    "tail_flow_region_valid",
    "bbox_flow_region_valid",
    "head_flow_x_mean_clipped_norm",
    "head_flow_y_mean_clipped_norm",
    "head_flow_mag_mean_clipped_norm",
    "head_flow_mag_p95_clipped_norm",
    "tail_flow_x_mean_clipped_norm",
    "tail_flow_y_mean_clipped_norm",
    "tail_flow_mag_mean_clipped_norm",
    "tail_flow_mag_p95_clipped_norm",
    "bbox_flow_x_mean_clipped_norm",
    "bbox_flow_y_mean_clipped_norm",
    "bbox_flow_mag_mean_clipped_norm",
    "bbox_flow_mag_p95_clipped_norm",
    "global_flow_x_mean_clipped_norm",
    "global_flow_y_mean_clipped_norm",
    "global_flow_mag_mean_clipped_norm",
    "global_flow_mag_p95_clipped_norm",
)

LEGACY_NO_FLOW_SCHEMA = (
    DETECTOR_FEATURES
    + BAT_VECTOR_FEATURES
    + MOTION_FEATURES
    + LEGACY_VISIBILITY_FEATURES
    + WINDOW_FEATURES
    + VIEW_FEATURES
)
LEGACY_WITH_FLOW_SCHEMA = LEGACY_NO_FLOW_SCHEMA + FLOW_FEATURES

CLEAN_NO_FLOW_SCHEMA = (
    DETECTOR_FEATURES
    + BAT_VECTOR_FEATURES
    + MOTION_FEATURES
    + WINDOW_FEATURES
    + VIEW_FEATURES
)
CLEAN_WITH_FLOW_SCHEMA = CLEAN_NO_FLOW_SCHEMA + FLOW_FEATURES

assert len(LEGACY_NO_FLOW_SCHEMA) == 29
assert len(LEGACY_WITH_FLOW_SCHEMA) == 49
assert len(CLEAN_NO_FLOW_SCHEMA) == 25
assert len(CLEAN_WITH_FLOW_SCHEMA) == 45


def schema_for(use_flow_features: bool) -> tuple[str, ...]:
    return CLEAN_WITH_FLOW_SCHEMA if use_flow_features else CLEAN_NO_FLOW_SCHEMA
