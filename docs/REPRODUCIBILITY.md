# Reproducibility guide

This repository exposes the clean-input implementation and frozen protocol records. It does not contain the private videos, frame annotations, detector/flow exports, projected OBP trajectories, datasets, run directories, or model weights used for the reported experiment. Reproduction therefore requires separately obtained, rights-cleared inputs.

## Public and private boundary

Public code can:

- export deterministic detector candidates from Ultralytics pose text outputs;
- project OBP C3D bat markers into synthetic 2D trajectories;
- build strict-clean synthetic windows;
- train clean25 and clean45 temporal refiners;
- verify selected-detector/flow alignment;
- evaluate a schema-compatible checkpoint;
- regenerate the included self-recorded demo assets when the frozen private inputs are supplied.

Users must obtain or create separately:

- the OpenBiomechanics Project Baseball Hitting C3D data, directly from upstream;
- rights-cleared real swing frames and annotations;
- detector training/prediction outputs, including out-of-fold predictions for temporal-model training;
- the selected detector CSVs and, for flow, aligned RAFT feature CSVs;
- model checkpoints.

The configuration JSON files under `configs/` are frozen protocol records and path-independent templates. The training scripts do not automatically consume those JSON files; translate the recorded fields into CLI arguments and replace angle-bracket placeholders with local paths.

## Environment

Install the smallest relevant group described in [DEPENDENCIES.md](DEPENDENCIES.md). No one-command or cross-platform reproduction claim is made.

## Data preparation outline

1. Obtain OBP Baseball Hitting C3D data from its upstream project and review its current citation/license terms.
2. Generate 2D trajectories with `refinement/synthetic_data_by_c3d/generate_bat_2d_coordinates.py`.
3. Build synthetic clean25 windows with `refinement/src/build_synthetic_pretraining_dataset.py` and explicit input, output, and split paths.
4. Generate detector predictions for the real data. Use swing-disjoint folds so training CSV rows are out-of-fold detector predictions.
5. Export deterministic selected candidates with `yolo/export_detector_evaluation_rows.py`. Ground truth is joined after selection for supervised training/evaluation rows.
6. For flow, run the scripts in `raft/scripts/`, merge flow statistics, and validate deterministic selected/flow alignment with `refinement/src/verify_selected_flow_alignment.py`.

Representative detector export interface:

```bash
python yolo/export_detector_evaluation_rows.py \
  --pred-dirs <YOLO_PREDICTION_DIRECTORY> \
  --gt-csv <ANNOTATION_CSV> \
  --clips-list <CLIP_METADATA_CSV> \
  --output <SELECTED_DETAILS_CSV>
```

All material input/output paths are explicit. Omitting prediction directories, GT CSV, output path, trainer output directory, synthetic output NPZ, or synthetic split JSON fails instead of silently using historical `runs/` or `data/` locations.

The optional flow path is staged rather than a single command: `extract_raft_flow.py` writes pairwise flow files, `extract_local_flow_features.py` and `extract_global_flow_features.py` summarize them, and `merge_yolo_eval_with_flow_features.py` combines those summaries with selected detector rows. Run `verify_selected_flow_alignment.py` before training. Every stage requires explicit input/output locations.

## Frozen protocol sequence

### Strict-clean synthetic pretraining

Use `configs/synthetic_pretraining.json` with:

- clean25 schema;
- 31-frame windows, stride 5, include-first behavior;
- seed 7;
- a two-layer BiGRU with hidden size 256 and dropout 0.15;
- 500 maximum epochs and validation-total checkpoint selection.

The exact historical virtual-camera invocation was not preserved in the frozen training command. The projection section of the config therefore marks camera/projection choices as a provenance gap rather than inventing values.

### Real no-flow fine-tuning

Use `configs/real_finetuning_no_flow.json`. The frozen protocol starts from the strict-clean synthetic epoch-487 checkpoint, uses selected `my_split_v2` train/validation rows, clean25 input, 31-frame windows, batch 16, AdamW at `2e-6`, no scheduler, patience 20, and selects the minimum validation total loss. The test split is evaluation-only.

### Real flow fine-tuning

Use `configs/real_finetuning_flow.json`. It keeps the no-flow recipe and adds the verified clean25-to-clean45 expansion plus aligned RAFT20 features clipped/normalized at 50 px. Selected and flow CSV frame keys and echoed detector coordinates/confidences/bboxes must match exactly; mismatches fail rather than being inner-joined away.

### Evaluation

Use `configs/evaluation.json` with one frozen best checkpoint. The primary comparison is the same 840 target-valid detector-matched frames. PCK-Swing divides endpoint error by each swing's maximum valid projected GT bat length. The test split must not be used for checkpoint selection or tuning.

## Input and output paths

Training, evaluation, detector export, and RAFT feature scripts accept explicit input paths. Trainer output directories and synthetic dataset/split outputs must also be supplied explicitly. Evaluation output files are written only when requested.

The C3D generator has relative convenience defaults; choose explicit input/output locations for a controlled run. Clip metadata is optional, with documented fallbacks when omitted.

## Frozen results and demos

`results/final_metrics.json` contains aggregate metrics only. It does not expose per-frame predictions or annotations. The final demo renderer requires the private frozen prediction CSV, selected-detector CSV, and rights-cleared frames as explicit arguments. Its public manifest contains hashes and logical identifiers without private absolute paths.

Demo regeneration, when those private frozen inputs are available, uses:

```bash
python docs/demo_assets/generate_final_assets.py \
  --predictions <FROZEN_FORMAL_PREDICTIONS_CSV> \
  --selected-csv <SELECTED_DETECTOR_CSV> \
  --frames-root <RIGHTS_CLEARED_FRAMES_ROOT> \
  --output-dir <DEMO_OUTPUT_DIRECTORY> \
  --ffmpeg <FFMPEG_EXECUTABLE>
```

By default the renderer writes only the public-safe manifest. The private absolute-path provenance manifest requires the explicit `--write-private-manifest` option and must not be committed.

## Verification

From the repository root:

```bash
python -m pytest -q
python -m compileall -q refinement raft yolo tests docs/demo_assets/generate_final_assets.py
```

These checks validate code and contracts; they do not reproduce training or reported model accuracy without the private inputs and checkpoints.
