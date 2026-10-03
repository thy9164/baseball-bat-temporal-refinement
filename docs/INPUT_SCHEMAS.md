# Public input schemas

This document records the interfaces currently consumed by canonical code.
It does not imply that every historical CSV column will be published. Field
roles are classified as **INFERENCE INPUT**,
**SUPERVISION**, or **EVALUATION ONLY**. Supervision and evaluation fields must
never enter model `X` or candidate-selection decisions.

Evidence: `yolo/candidate_selection.py`,
`yolo/export_detector_evaluation_rows.py`,
`refinement/clean_input/adapters.py`, `refinement/clean_input/observation.py`,
`refinement/clean_input/supervision.py`,
`refinement/clean_input/flow_alignment.py`,
`refinement/src/real_data_pipeline.py`, and
`refinement/synthetic_data_by_c3d/generate_bat_2d_coordinates.py`.

## Shared conventions

- Real-video image coordinates use pixels with origin at the upper left: `x`
  increases rightward and `y` downward. Clean observation construction divides
  pixel coordinates by configured image width and height.
- Synthetic projected coordinates use `u/v` in pixels with the same image-plane
  convention. Projection depth and 3D coordinates are separate metadata.
- Real frames are identified by `(swing_id, image_stem)`; `frame` is accepted as
  a fallback. Numeric ordering currently extracts the last six digits.
- A detector point exists only when `x`, `y`, and confidence are finite and
  confidence is greater than zero. Missing coordinates are empty or non-finite;
  confidence is treated as zero. Canonical filling occurs only after this
  detector-derived validity decision, and the missing mask is retained.
- `visibility`, `status`, `ignore`, GT coordinates, detector error, and
  `target_valid` are never inference inputs.

## A. Detector export / candidate-selection input

The canonical selector consumes one record per frame, each containing a list of
YOLO detections. The exporter currently reads Ultralytics pose `save_txt
save_conf` rows.

| Field | Required | Units / semantics | Role |
|---|---:|---|---|
| `swing_id` | yes | Stable sequence identifier | INFERENCE INPUT metadata |
| `frame_no` | yes for temporal ordering | Integer frame index; `None` falls back to sort-last/gap 1 behavior | INFERENCE INPUT metadata |
| `frame_name` | yes | Deterministic secondary ordering key | INFERENCE INPUT metadata |
| `prediction_stem`, `image_stem` | yes | Detector/export identifiers | INFERENCE INPUT metadata |
| `detections[].bbox` | yes | Normalized `(xc, yc, width, height)` | INFERENCE INPUT |
| `detections[].bbox_conf` | yes | Detector bbox confidence | INFERENCE INPUT |
| `detections[].head` | yes | Normalized `(x, y, confidence)` | INFERENCE INPUT |
| `detections[].tail` | yes | Normalized `(x, y, confidence)` | INFERENCE INPUT |
| `detections[].detection_rank` | yes | One-based rank after descending bbox confidence | Derived inference metadata |

The selector derives swing-level bbox anchors, frame-to-frame jumps, bat length
and direction changes, bbox-center motion, scores, and local outlier decisions.
It rejects top-level GT/error/status fields. The following are forbidden:
`gt_head`, `gt_tail`, `gt_x`, `gt_y`, `visibility`, `status`, `error_px`, and
`detector_error`.

The exporter may join annotations only after selection to construct evaluation
rows. Current export discovery skips detector frames that lack a matching GT row
before forming its evaluation population; this is an evaluation-export boundary,
not part of the reusable selector. The repository does not provide a standalone
detector-only export CLI that enumerates all detector frames independently;
the reusable selector itself accepts annotation-free frame records.

## B. Selected detector CSV / clean observation input

The real-data loader expects long-form rows: one `head` and one `tail` row per
frame. The selected CSV combines inference fields with supervision/evaluation
fields, but `build_observation_features` receives only the inference subset.

| Field | Required | Units / semantics | Role |
|---|---:|---|---|
| `swing_id` | yes | Sequence identifier | INFERENCE INPUT metadata / split key |
| `image_stem` or `frame` | yes | Frame identifier; last six digits define ordering | INFERENCE INPUT metadata |
| `keypoint` | yes | `head` or `tail` | INFERENCE INPUT row identity |
| `pred_x`, `pred_y` | yes, may be missing | Pixel detector coordinate | INFERENCE INPUT |
| `pred_conf` | yes, may be zero/missing | Original detector keypoint confidence | INFERENCE INPUT |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | required for flow alignment; otherwise optional | Pixel bbox corners | INFERENCE INPUT / provenance |
| `detection_rank`, `bbox_conf` | optional for clean builder | Selected-candidate provenance | INFERENCE INPUT metadata |
| `view`, `batter_side` | optional in this CSV when supplied by clip metadata | View/hand metadata used to derive canonical view and horizontal flip | INFERENCE INPUT metadata |
| `gt_x`, `gt_y` | required for supervised train/evaluation | Pixel annotation | SUPERVISION |
| `visibility` | required by current loss/grouping path; defaults historically | `fully_visible`, `partially_occluded`, `fully_occluded`, `ignore` | SUPERVISION / EVALUATION ONLY |
| `status` | optional for canonical observation; retained in records | Evaluation status such as `matched` or `missing_prediction` | EVALUATION ONLY |
| `dx`, `dy`, `error_px` | not required by clean builder | Detector-versus-GT diagnostics | EVALUATION ONLY |

`target_valid` is derived only for supervision from finite GT head/tail and
`tail_visibility != ignore`. It MUST NOT affect detector validity, filling,
confidence, observation features, or inference gating.

## C. RAFT / flow feature CSV

The flow CSV is wide-form with exactly one row per `(swing_id, image_stem/frame)`.
It must align deterministically with the selected detector CSV. Duplicate,
missing, extra, or detector-mismatched frames are rejected.

Required identity and detector echo fields:

- `swing_id`, and `image_stem` or `frame`;
- `head_pred_x`, `head_pred_y`, `head_pred_conf`;
- `tail_pred_x`, `tail_pred_y`, `tail_pred_conf`;
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`.

Required RAFT20 source order:

1. `has_prev_flow`
2. `head_valid`
3. `tail_valid`
4. `bbox_valid`
5. `head_flow_x_mean`
6. `head_flow_y_mean`
7. `head_flow_mag_mean`
8. `head_flow_mag_p95`
9. `tail_flow_x_mean`
10. `tail_flow_y_mean`
11. `tail_flow_mag_mean`
12. `tail_flow_mag_p95`
13. `bbox_flow_x_mean`
14. `bbox_flow_y_mean`
15. `bbox_flow_mag_mean`
16. `bbox_flow_mag_p95`
17. `global_flow_x_mean`
18. `global_flow_y_mean`
19. `global_flow_mag_mean`
20. `global_flow_mag_p95`

The first four fields are binary inference-valid flags. Remaining values are
pixel displacement/statistics from RAFT and detector-defined regions. Canonical
normalization clips signed `x/y` components to `[-flow_clip_px,+flow_clip_px]`,
clips magnitudes to `[0,flow_clip_px]`, and divides by `flow_clip_px` (50 px in
the frozen protocol). Missing/non-numeric flow values currently default to zero.
No GT, visibility, status, or detector error is permitted in RAFT20.

The flow workflow uses separate extraction, summary, and merge scripts under
`raft/scripts/`, rather than one end-to-end export CLI. The extractor sorts source
image paths and processes consecutive pairs. Reproducers must record the ordered
source frames and extraction commands; the exact historical invocation was not
preserved. The CSV alignment requirements and RAFT20 schema above still apply.

## D. OBP projected 2D trajectory CSV

The projection generator writes one row per projected frame and camera view.
The strict-clean synthetic builder accepts either `head_u/head_v/tail_u/tail_v`
or the legacy alias `head_x/head_y/tail_x/tail_y`.

| Field | Required by downstream builder | Units / semantics | Role |
|---|---:|---|---|
| `source_c3d` or `sequence_id` | one stable identifier required | Source/sequence provenance | Dataset identity / split key |
| `camera_id` or `camera_name` | recommended | Camera/view identifier | INFERENCE-LIKE synthetic metadata |
| `mirror` | optional | `none` or horizontal-mirror label | Synthetic metadata |
| `frame_idx` | yes | Zero-based projected source-frame index | Temporal metadata |
| `time_sec` | optional | Seconds; used for source-rate inference when present | Temporal metadata |
| `image_width`, `image_height` | required by projection record and normalization context | Pixels | Inference-like metadata |
| `head_u`, `head_v`, `tail_u`, `tail_v` | yes | Projected pixels | Clean synthetic target/source trajectory |
| `head_in_frame`, `tail_in_frame` | generated, downstream use varies | Projection visibility inside image bounds | Synthetic generation metadata; not a real-inference annotation |
| camera pose/zoom/pan/distance fields | generated, optional downstream | Degrees, scale/pan, metres as named | Regeneration provenance |
| head/tail depth and 3D coordinates | generated, optional downstream | Generator coordinate units | Regeneration provenance |

Synthetic detector-like corruption creates inference-style coordinates,
confidence, and missingness. Latent synthetic visibility/corruption labels may
be retained for supervision/analysis but MUST NOT enter clean25/clean45 `X`.

`unit_mode` documents generator handling. The exact historical OBP snapshot and
automatic unit-conversion choice were not preserved. Reproducers must record
the acquired snapshot and effective unit-conversion settings with their generated
data rather than infer these from a CSV filename.

## E. Real swing metadata / annotation-compatible input

Clip metadata and keypoint annotations are separate logical inputs even when an
evaluation CSV later combines them.

### Clip metadata

| Field | Required | Semantics | Role |
|---|---:|---|---|
| `name` or `swing_id` (also legacy `clip`/`clip_name`) | yes | Swing identifier | INFERENCE INPUT metadata |
| `view` | recommended | Historical values include `side`/`back`; mapped to canonical view labels | INFERENCE INPUT metadata |
| `batter_side` | recommended | Left/right; controls canonical horizontal flip | INFERENCE INPUT metadata |

Missing metadata falls back to the configured `view_label` and `batter_hand`.
Reproducers must supply and record these settings when clip metadata is absent.

### Annotation-compatible frame data

Required for supervised training/evaluation: `swing_id`, frame identifier,
head/tail pixel GT coordinates, and head/tail visibility labels. Optional
evaluation fields include `status`, `dx`, `dy`, and `error_px`. `ignore` affects
only supervision validity and metric grouping.

Annotation fields MUST NOT enter candidate selection, detector confidence,
observation validity, missing-observation filling, clean25/clean45 features, or
inference gates. Ownership, rights, and redistribution of source images are
separate from this technical schema.
