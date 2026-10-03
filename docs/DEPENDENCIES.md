# Dependencies

This inventory is derived from the imports, subprocess calls, and command-line entry points in the curated tree. The requirement files intentionally do not invent minimum versions: the project has only been verified in the local environments listed below, so minimum compatible versions remain unknown.

## Dependency classification

| Class | Dependencies | Scope |
|---|---|---|
| A. Core temporal refinement | Python, NumPy, PyTorch | clean observations, clean25/clean45 schemas, BiGRU training/evaluation, checkpoint contracts |
| B. Synthetic OBP preprocessing | core + OpenCV, ezc3d | C3D reading, 3D-to-2D projection, preview generation, synthetic-window construction |
| C. RAFT / flow path | core + Torchvision, Pillow, OpenCV, pandas, tqdm | pretrained RAFT execution, flow serialization, regional summaries, CSV merge/alignment |
| D. Demo rendering | pandas, Pillow, external FFmpeg with `libx264` | deterministic PNG/frame rendering and CPU MP4 encoding |
| E. Test / development | core + pytest | public test discovery and execution; tests use `unittest` assertions |
| F. Standard library | `argparse`, `collections`, `copy`, `csv`, `dataclasses`, `functools`, `glob`, `hashlib`, `json`, `math`, `os`, `pathlib`, `random`, `re`, `subprocess`, `sys`, `tempfile`, `types`, `unittest` | CLI, data I/O, hashing, process control, and tests |

## Install groups

| Group | Install file | Python packages | Purpose |
|---|---|---|---|
| Core | `requirements.txt` | `numpy`, `torch` | clean25/clean45 feature construction, BiGRU models, training, evaluation, checkpoint contracts |
| Synthetic / OBP | `requirements-synthetic.txt` | core + `opencv-python`, `ezc3d` | C3D reading, 3D-to-2D projection, optional preview rendering, synthetic dataset construction |
| RAFT | `requirements-flow.txt` | core + `torchvision`, `Pillow`, `opencv-python`, `pandas`, `tqdm` | torchvision RAFT inference, flow image I/O, feature extraction and CSV joins |
| Demo | `requirements-demo.txt` | `pandas`, `Pillow` | deterministic still/video frame rendering; also requires an external FFmpeg executable with `libx264` |
| Development / tests | `requirements-dev.txt` | core + `pytest` | run the complete public test suite; tests themselves use `unittest` and are pytest-discoverable |

Install only the groups required for the intended task, for example:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-synthetic.txt
python -m pip install -r requirements-flow.txt
python -m pip install -r requirements-demo.txt
python -m pip install -r requirements-dev.txt
```

`scipy` and `scikit-learn` are not imported by the current public tree and are not required.

## Specification choice

This is a scripts-oriented research repository rather than an installable Python package. Use the small, task-specific requirement files above. A fresh Python 3.11 environment passed all 36 public tests; this does not establish minimum supported versions or validate all optional dependency groups.

## Package-level evidence

| Dependency | Required or optional | Repository evidence and purpose | Version evidence |
|---|---|---|---|
| Python | required | all entry points | Frozen training used 3.14.2; the historical RAFT environment used 3.11.0. Syntax requires a modern Python, but the minimum supported release has not been established by a clean compatibility test. |
| NumPy | required | clean observation/schema adapters, datasets, metrics, geometry, RAFT feature summaries | Tested locally with 2.4.4; minimum unknown. |
| PyTorch | required for model work | BiGRU definition, dataset loaders, training, checkpoint loading, evaluation | Frozen training used 2.9.1+cu128; minimum unknown. CPU execution is supported by CLI selection but has not reproduced the published runs. |
| Torchvision | optional, RAFT only | `torchvision.models.optical_flow.raft_large` and `Raft_Large_Weights` | Historical flow environments contained 0.24.1+cu128 and 0.26.0+cu128; minimum unknown. Match it to the installed PyTorch build. |
| pandas | optional, flow/demo | flow CSV feature extraction/merge and demo renderer CSV loading | Tested in historical RAFT/OBP environments with 3.0.3; minimum unknown. |
| Pillow | optional, flow/demo | RAFT frame loading and public demo rendering | Tested locally with 12.2.0; minimum unknown. |
| opencv-python (`cv2`) | optional, synthetic/flow | OBP projection previews and RAFT flow serialization/visualization | Tested locally with 4.13.0.92; minimum unknown. `opencv-contrib-python` is not required by imports. |
| tqdm | optional, RAFT | progress display in RAFT extraction scripts | Tested in historical RAFT/OBP environments with 4.68.1; minimum unknown. |
| ezc3d | optional, synthetic/OBP | reads Baseball Hitting C3D files | Tested in the historical OBP environment with 1.7.0; minimum unknown. |
| pytest | development only | test discovery/runner for the public suite | Not present in the inspected historical environments; required version unknown. Tests use standard `unittest` APIs and are pytest-discoverable. |
| FFmpeg with `libx264` | optional external demo runtime | invoked by `generate_final_assets.py` through `subprocess` | One local development build is recorded below; minimum unknown. |
| SciPy | not required | no import found | no version requirement. |
| scikit-learn | not required | no import found | no version requirement. |

## Version evidence

The frozen strict-clean training manifests record **Python 3.14.2**, **PyTorch 2.9.1+cu128**, and **CUDA 12.8**. That is provenance for the reported runs, not a portable minimum requirement.

Read-only inspection of the surviving local environments found:

| Environment role | Python | Relevant installed packages |
|---|---|---|
| Frozen training environment | 3.14.2 | numpy 2.4.4; torch 2.9.1+cu128; torchvision 0.24.1+cu128; Pillow 12.2.0; opencv-python 4.13.0.92 |
| Historical RAFT environment | 3.11.0 | numpy 2.4.4; torch 2.11.0+cu128; torchvision 0.26.0+cu128; pandas 3.0.3; Pillow 12.2.0; opencv-python 4.13.0.92; tqdm 4.68.1 |
| Historical OBP environment | 3.14.2 | numpy 2.4.4; torch 2.9.1+cu128; torchvision 0.24.1+cu128; pandas 3.0.3; Pillow 12.2.0; opencv-python 4.13.0.92; tqdm 4.68.1; ezc3d 1.7.0 |

The historical environments were not jointly resolved from a lockfile, so these exact combinations should not be interpreted as tested cross-platform constraints.

## Standard-library and external runtime dependencies

The code also uses Python standard-library modules including `argparse`, `csv`, `json`, `pathlib`, `subprocess`, `tempfile`, `hashlib`, `random`, `re`, and `unittest`.

The demo renderer invokes FFmpeg by an explicit `--ffmpeg` path. FFmpeg is not installed through the Python requirement files. The renderer requires a build with the CPU `libx264` encoder. The recorded local renderer check used an FFmpeg development build identified as `N-124279-g0f6ba39122-20260430`; minimum compatible FFmpeg version is unknown.

Torchvision's RAFT entry point may acquire pretrained weights when run in a fresh environment. This repository does not vendor those weights. Network and cache behavior should be reviewed before running RAFT regeneration in a controlled environment.
