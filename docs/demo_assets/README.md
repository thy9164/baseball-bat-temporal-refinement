# Public demo assets

These assets use the frozen strict-clean no-flow formal predictions and self-recorded source footage. Rendering did not run model inference, retrain a model, select a checkpoint, or change any metric.

## Assets

- `swing074_hero.png` — final README/portfolio hero still. It compares the frame-wise selected detector with strict-clean temporal refinement at `swing_074`, frame 23.
- `swing074_detector_vs_temporal.mp4` — final short comparison video covering frames 18–30 at 6 fps with the same fixed crop and two-column layout.
- `swing074_detector_vs_temporal.gif` — inline README animation derived from the same frozen frames/predictions in the short MP4, with 960 px width, a generated 128-color palette, Lanczos scaling, Sierra2_4a dithering, and infinite looping.
- `swing074_full_swing_comparison.mp4` — complete-motion qualitative comparison covering every continuous source frame from 1 through 63 at 8 fps.
- `swing076_limitation.png` — final qualitative counterexample at `swing_076`, frame 43.
- `ASSET_MANIFEST.json` — optional private provenance manifest with exact local source paths and hashes; generated only when explicitly requested.
- `ASSET_MANIFEST_PUBLIC.json` — public-safe manifest with logical identifiers and hashes but no user or machine paths.
- `generate_final_assets.py` — canonical renderer for all five assets; it reads frozen CSV values and source images but does not load a model.

## Visual encoding

- Ground-truth tail: large white hollow ring
- Frame-wise selected detector tail: orange marker
- Strict-clean no-flow temporal-refined tail: cyan marker
- Bat-axis lines: faint references only

Flow is intentionally absent. Its limited incremental contribution belongs in the ablation results rather than the hero explanation.

## Video roles and playback

- The frames 18–30 video is a focused representative segment around the frame-23 improvement.
- The frames 1–63 video shows the complete recorded swing motion from start to end.

The 6 fps and 8 fps playback rates are presentation choices. They do not represent the original capture speed, model inference latency, or real-time throughput. The full-swing video uses the same frozen predictions and fixed crop on every frame; no interpolation, display-only smoothing, or hand-edited prediction was added.

The inline GIF presents the short segment at 6 fps, not original capture speed or inference throughput. GIF frame delays use centiseconds, so timing is quantized around the nominal 13/6-second duration. The short MP4 retains higher image quality.

## Crop policy

For each swing, the deterministic crop uses the valid GT tail trajectory envelope across the complete swing, adds 30% margin on every side, expands to 4:3, and clamps to the source image. One fixed crop is used for every frame in that swing.

## Rendering requirements

The renderer code and the five included demo assets are suitable for the curated public repository. The private inputs are not included. Exact regeneration requires rights-cleared source frames, a compatible frozen prediction CSV, and the selected-detector CSV used for provenance hashing.

The renderer requires Python with pandas and Pillow, plus an FFmpeg build with the CPU `libx264` encoder. It is self-contained within this directory and does not import prototype code. Font loading tries an optional `DEMO_FONT_REGULAR`/`DEMO_FONT_BOLD` override, DejaVu Sans, common system fonts, and finally Pillow's built-in fallback. No font file is bundled.

Pass the frozen prediction CSV, selected-detector CSV, rights-cleared frame root, output directory, and FFmpeg executable explicitly:

```text
python docs/demo_assets/generate_final_assets.py \
  --predictions <frozen-formal-predictions.csv> \
  --selected-csv <selected-detector.csv> \
  --frames-root <rights-cleared-frame-root> \
  --output-dir <demo-output-directory> \
  --ffmpeg <ffmpeg-executable>
```

One invocation renders the hero still, focused video, inline GIF derived from that video, full-swing video, and limitation still, then writes `ASSET_MANIFEST_PUBLIC.json`. Add `--write-private-manifest` only in a private workspace when an `ASSET_MANIFEST.json` containing absolute local provenance paths is required. Video encoding uses CPU `libx264`; the script contains no model loading or inference path.

The input files may remain private. A compatible frame root must contain the documented `swing_074` and `swing_076` frame identifiers, while the prediction CSV must provide the frozen head/tail fields consumed by the renderer. The public-safe manifest records logical identifiers and hashes without exposing local paths.

## Public claim boundary

The hero and full-swing video are frozen qualitative examples. The limitation image is one frozen qualitative counterexample. None of these assets replaces the aggregate performance measured over the complete 15-swing test set.
