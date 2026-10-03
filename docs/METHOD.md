# Method

## Problem and evaluation unit

The project estimates baseball-bat head and tail keypoints from video. A frame-wise pose detector supplies candidate endpoints and confidence values. A deterministic selector chooses among multiple detections using detector confidence plus inference-available temporal/geometric consistency. Ground-truth coordinates, visibility/status labels, and detector error are excluded from selection and model input.

The final evaluation focuses on tail localization. Its primary population is the same 840 target-valid frames for which the selected detector, no-flow refiner, and flow refiner all have comparable predictions. This isolates localization quality; it is not a detector-coverage metric.

## Clean observation schema

The shared builder converts a detector sequence into an inference-valid feature tensor. A point is valid only if prediction `x`, `y`, and confidence are finite and confidence is positive. Only detector-missing observations are filled, and missing flags remain in the representation. Ground truth is held in a separate supervision structure.

The clean25 schema contains:

- normalized detector head/tail coordinates, original confidences, and validity flags;
- normalized bat vector, length, direction, speed, and acceleration features derived from observations;
- normalized window time;
- five view indicators.

The clean45 flow schema appends 20 RAFT features in the exact order enforced by `refinement/clean_input/schema.py`: four validity flags, 12 local head/tail/bbox flow statistics, and four global flow statistics. Local regions are defined by detector predictions and bounding boxes. Signed components and magnitudes are clipped and normalized with the frozen 50 px scale. See [INPUT_SCHEMAS.md](INPUT_SCHEMAS.md) for field contracts.

## Temporal model

The mature model is a two-layer bidirectional GRU with hidden size 256 and dropout 0.15. It operates on 31-frame windows. Its output represents a temporal prior, residual correction, learned trust, and auxiliary detector-quality estimates. The final position blends the observation and corrected prediction through learned trust. Overlapping inference windows are merged by arithmetic mean.

Because the model is bidirectional, it is an offline refiner: it uses both past and future frames and is not a causal real-time tracker.

## Training lineage

Strict-clean synthetic pretraining uses projected OBP bat trajectories corrupted into detector-like observations. The clean observation builder prevents latent synthetic visibility/corruption labels from entering the model input. Synthetic supervision and loss weighting remain separate from inference features.

Real fine-tuning starts from the strict-clean synthetic checkpoint and uses selected detector observations. Train, validation, and test swings are disjoint under `my_split_v2`. Training detector rows are generated out of fold. Validation total loss selects the checkpoint; test data does not participate in selection.

The loss implementation provides final-position, prior, correction/prior-guidance, trust, bat-length and direction, velocity/acceleration, line/end, observation-consistency, detector-quality, and damage-control terms across synthetic pretraining and real fine-tuning. Active weights differ by stage; not every term has a nonzero weight in every stage. For example, the frozen synthetic recipe disables good-detector damage loss, while the real fine-tuning recipe enables it and disables observation-consistency loss. These are structured exploratory objectives; individual terms were not independently proven beneficial by ablation.

## Flow ablation

The flow model holds the architecture and real-data recipe fixed while expanding clean25 to clean45. The first-layer weights for the original 25 inputs are copied from the strict-clean synthetic parent; 20 new flow columns start at zero. At initialization, the recurrent, deeper, and output weights are copied unchanged from the parent model. This provides a controlled test of RAFT20 information without using legacy oracle-trained checkpoints.

## Metrics

Reported pixel metrics are RMSE, mean, median, and 90th-percentile tail error. PCK-Swing normalizes each frame's pixel error by one fixed scale per swing: the maximum valid projected ground-truth bat length in that swing. Reported PCK thresholds are 2.5% and 5%; AUC integrates PCK from 0% to 10% with a 0.001 threshold step and divides by 0.10.

The aggregate frozen values are stored in `results/final_metrics.json`. The controlled experiment contains one seed and 15 test swings, so no statistical-significance claim is made.

## Interpretation boundary

The no-flow model reduces severe outliers and RMSE more than it changes PCK. It can make already-good detector estimates worse. Overall, the model behaves mainly as a temporal outlier corrector rather than a universal accuracy booster or an occlusion-specific solution. RAFT20 adds little incremental accuracy over the matched no-flow model.
