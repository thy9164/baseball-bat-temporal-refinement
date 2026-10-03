# Third-party sources and dependencies

## OpenBiomechanics Project

Synthetic pretraining uses external Baseball Hitting data from **The OpenBiomechanics Project (OBP), Driveline Baseball Research & Development**. The project code reads C3D bat markers and projects trajectories into 2D.

- [Official repository](https://github.com/drivelineresearch/openbiomechanics)
- [Dataset citation](https://github.com/drivelineresearch/openbiomechanics/blob/main/CITATION.cff)
- [Data and documentation license](https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md): CC BY-NC-SA 4.0, with an additional professional-sports-organization / financial-analysis-firm exclusion.
- [Upstream code license](https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-CODE.md): MIT, separate from the data license.

Original C3D files and generated OBP-derived trajectory datasets are not included. Obtain data from upstream and review its current terms. These upstream licenses describe OBP materials, not this repository.

## PyTorch, Torchvision, and RAFT

The temporal model uses external PyTorch APIs. Flow extraction calls Torchvision's `raft_large`, `Raft_Large_Weights`, and `pil_to_tensor`. RAFT implementation source and pretrained weights are not bundled.

- [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE)
- [Torchvision BSD-3-Clause license](https://github.com/pytorch/vision/blob/main/LICENSE)
- [Original RAFT repository](https://github.com/princeton-vl/RAFT)
- [Original RAFT BSD-3-Clause license](https://github.com/princeton-vl/RAFT/blob/master/LICENSE)

RAFT execution may download pretrained weights. Package-source licenses should not be treated as a blanket license for every weight or training-data artifact.

## YOLO / Ultralytics output format

The detector exporter parses externally generated YOLO pose text outputs. This repository does not include or import Ultralytics implementation code, distribute YOLO weights, or run the detector itself.

[Ultralytics licensing](https://www.ultralytics.com/license) describes AGPL-3.0 and Enterprise options for its code/models and workflows. Parsing an external output format alone does not establish that this repository inherits AGPL. Users generating detector outputs should review the terms applicable to their separate detector workflow.

## FFmpeg / libx264

Demo rendering invokes a user-supplied FFmpeg executable with the CPU `libx264` encoder. No FFmpeg or libx264 executable/library binary is included.

[FFmpeg's licensing documentation](https://ffmpeg.org/legal.html) describes LGPL licensing and optional GPL components, including libx264. The license of an FFmpeg build depends on its enabled components.

## Other external Python dependencies

NumPy, pandas, Pillow, OpenCV, tqdm, ezc3d, and pytest are installed separately as needed. Their implementation source and package binaries are not bundled. See [Dependencies](DEPENDENCIES.md) for task-specific requirements.

Upstream license references:

- [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt)
- [pandas](https://github.com/pandas-dev/pandas/blob/main/LICENSE)
- [Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE)
- [OpenCV](https://github.com/opencv/opencv/blob/4.x/LICENSE)
- [tqdm](https://github.com/tqdm/tqdm/blob/master/LICENCE)
- [ezc3d](https://github.com/pyomeca/ezc3d/blob/master/LICENSE)
- [pytest](https://github.com/pytest-dev/pytest/blob/main/LICENSE)

No dependency license is presented as a repository-wide software license.

## Included and excluded materials

The four public demo assets use self-recorded footage with the recorded person's consent. They are frozen qualitative examples, not replacements for aggregate test-set results.

Web-sourced footage, raw extracted frames, private annotations, detector/flow exports, per-frame predictions, OBP original/derived datasets, run directories, checkpoints, model weights, and third-party binaries are not distributed.
