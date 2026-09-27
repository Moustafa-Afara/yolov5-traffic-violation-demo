# Changes relative to upstream YOLOv5

## Upstream

`yolov5/` is [ultralytics/yolov5](https://github.com/ultralytics/yolov5) at commit
**`724d5b2`** (24 June 2022, between releases v6.1 and v6.2), included unmodified as a git
submodule. It was identified by byte-comparing the files of the original 2024 upload of this
project against upstream history: 9 of 10 files matched commits `1156a32`…`724d5b2`
(18–24 June 2022) exactly.

## What this project adds

| File | Relation to upstream |
|---|---|
| `speed_violation_demo.py` | New. Written against upstream's `detect.py` (same loading, letterbox, NMS and box-scaling calls), restricted to vehicle classes, with tracking, a speed hook and Legal/Illegal annotation |
| `tracker.py` | New. IoU tracker |
| `README.md`, `CHANGES.md`, `requirements.txt`, `.gitignore` | New |
| `LICENSE` | Upstream's GPL-3.0 text, as required for a derived work |

Nothing inside `yolov5/` is changed. One compatibility shim lives in `speed_violation_demo.py`:
PyTorch ≥ 2.6 loads checkpoints with `weights_only=True` by default, which the 2022 code does not
expect, so the script restores the previous default (safe only for trusted weight files).

## History of this project

**2022 — original version (uploaded to GitHub 2024-02-12).** A copy of upstream's top-level
files (`detect.py`, `train.py`, `val.py`, `export.py`, …) without `models/` and `utils/`, the
v6.1 `yolov5s.pt` weights, and one modified script, `detect.1.py`. Its changes to `detect.py`:

- a speed was drawn with `randint(40, 150)` every third frame — a single value
  shared by all boxes in the frame, standing in for a roadside sensor;
- boxes were labelled `Illegal <speed> km/h` (red) outside 60–120 km/h and `Legal` (green) inside;
- the upstream per-class label drawing was commented out, which left `c` undefined and made
  `--save-crop` fail.

That upload was published under The Unlicense, which is incompatible with the GPL-3.0 upstream
code it contained, and it could not run on its own.

**2026-09-27 — restructure.** Upstream copies and weights removed; upstream added as a pinned
submodule; licence corrected to GPL-3.0; the modification rewritten as
`speed_violation_demo.py` with an IoU tracker so that each vehicle keeps its own ID and its own
simulated speed; the speed source moved behind `get_speed()` and labelled SIMULATED in the
output; the `--save-crop` bug removed with the rewrite.
