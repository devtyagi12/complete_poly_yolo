# YOLOv8-Extended — Polygon & Distance Detection

A PyTorch extension of YOLOv8 that adds **instance polygon segmentation** and **metric distance regression** to the standard bounding-box + classification outputs — all in a single forward pass with no external dependencies beyond PyTorch and OpenCV.

```
Input image
    │
    ▼
YOLOv8 Backbone + PAN-FPN Neck
    │
    ├─── Bounding box  (standard DFL + CIoU, unchanged)
    ├─── Class         (standard BCE, unchanged)
    ├─── Polygon       (star-shaped descriptor, 3 sub-heads per FPN scale)
    └─── Distance      (scalar metric depth in metres, per anchor)
```

---

## Table of Contents

1. [What's New vs Standard YOLOv8](#1-whats-new-vs-standard-yolov8)
2. [Architecture](#2-architecture)
3. [Star Polygon Representation](#3-star-polygon-representation)
4. [Dataset Format](#4-dataset-format)
5. [Project Structure](#5-project-structure)
6. [Installation](#6-installation)
7. [Configuration System](#7-configuration-system)
8. [Training](#8-training)
9. [Validation & Testing](#9-validation--testing)
10. [Inference](#10-inference)
11. [Dataloader Visualiser](#11-dataloader-visualiser)
12. [Training Visualisations](#12-training-visualisations)
13. [Logging](#13-logging)
14. [Loss Functions](#14-loss-functions)
15. [Post-Processing](#15-post-processing)
16. [GPU Compatibility](#16-gpu-compatibility)
17. [Extending the Codebase](#17-extending-the-codebase)
18. [Troubleshooting](#18-troubleshooting)

---

## 1. What's New vs Standard YOLOv8

| Feature | Standard YOLOv8 | YOLOv8-Extended |
|---------|----------------|-----------------|
| Bounding box | ✅ DFL + CIoU | ✅ Unchanged |
| Classification | ✅ BCE | ✅ Unchanged |
| Instance polygon | ❌ | ✅ Star-shaped descriptor |
| Metric distance | ❌ | ✅ Log-encoded regression |
| Multiple datasets | ❌ | ✅ Weighted sampling |
| Distance dataset | N/A | ✅ Optional, independent |
| Pretrained init | ✅ | ✅ Backbone + neck transferred |
| GroupNorm (AMD) | ❌ | ✅ `--gpu_type amd` |
| Auto experiment dirs | ❌ | ✅ `exp1/`, `exp2/`, ... |

---

## 2. Architecture

### Backbone and Neck

Standard YOLOv8 CSP backbone with SPPF and PAN-FPN neck, producing three FPN scales at strides 8, 16, and 32.

| Variant | Depth | Width | ~Parameters |
|---------|-------|-------|-------------|
| `n` | 0.33 | 0.25 | 3 M |
| `s` | 0.33 | 0.50 | 11 M |
| `m` | 0.67 | 0.75 | 26 M |
| `l` | 1.00 | 1.00 | 44 M |
| `x` | 1.33 | 1.25 | 69 M |

### Detection Head (unchanged)

Decoupled box branch (`2 × ConvBN(3×3) → Conv1×1 → 4 × REG_MAX channels`) and class branch (`2 × ConvBN(3×3) → Conv1×1 → num_classes channels`), applied independently at each FPN scale.

### Polygon Head (new — per FPN scale)

Three independent `Conv1×1` heads reading from the **penultimate box branch feature map**:

```
penultimate box feature map
    ├── poly_conf  → Conv1×1 → (B, num_angles, H, W)   raw logits
    ├── poly_angle → Conv1×1 → (B, num_angles, H, W)   raw logits
    └── poly_dist  → Conv1×1 → (B, num_angles, H, W)   pre-softplus
```

### Distance Head (new — per FPN scale)

`num_dist_blocks × ConvBN(3×3)` followed by `Conv1×1 → (B, 1, H, W)`, reading from the same neck feature as the detection head.

### Normalisation

| `--gpu_type` | Normalisation layer |
|---|---|
| `nvidia` (default) | `BatchNorm2d` |
| `amd` | `GroupNorm(32)` — avoids MIOpen JIT failures on ROCm |

---

## 3. Star Polygon Representation

Each object's polygon is encoded as a fixed-size **star-shaped descriptor** centred on the bounding box centre, eliminating variable-length vertex lists.

### Encoding algorithm

```
num_angles = 360 // angle_step          # default: 24 bins at 15 degrees each

for each polygon vertex (x, y):
    dx, dy   = vertex − bbox_centre
    angle    = atan2(dy, dx) mod 360
    bin_idx  = floor(angle / angle_step)
    keep the vertex with maximum distance from centre per bin

bins with no vertex → (x=0, y=0, conf=0)
bins with a vertex  → (x, y, conf=1)
```

### Wire format (per object)

```
[origin_x, origin_y,
 x0, y0, conf0,
 x1, y1, conf1,
 ...
 x_{N-1}, y_{N-1}, conf_{N-1}]

total length = 2 + num_angles × 3
```

All coordinates are **normalised to [0, 1]** in label files and target tensors. After letterbox resizing they are remapped to canvas-normalised space.

### Augmentation in polar form

Horizontal flip remaps angle bins analytically rather than recomputing from raw vertices:

```python
new_bin = int((180 - bin_angle_deg) % 360 / angle_step)
```

Mosaic translates origin and vertices proportionally in normalised space.

### Changing bin resolution

```yaml
# configs/my_exp.yaml
model:
  angle_step: 10    # 36 bins instead of 24 — finer polygon detail
```

All downstream code adapts automatically. The label cache filename encodes the `angle_step` so changing it generates a fresh cache.

---

## 4. Dataset Format

### Directory layout

```
data/
├── polygon/                        # polygon-only dataset
│   ├── images/
│   │   ├── train/   *.jpg / *.png
│   │   └── val/
│   └── labels/
│       ├── train/   *.txt
│       └── val/
│
└── polygon_distance/               # polygon + distance dataset (optional)
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

Image files and label files must share the same stem: `0001.jpg` ↔ `0001.txt`.

### Label file format

One object per line. Values are **normalised floats** in `[0, 1]`.

**Polygon-only:**
```
<class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>
```

**Polygon + distance:**
```
<class_id> <x1> <y1> <x2> <y2> ... <xN> <yN> <distance_metres>
```

`N` is the number of polygon vertices (can vary per object and per file). `distance_metres` must be `> 0` for a valid reading; non-positive values are treated as missing. Blank lines are ignored.

### Distance encoding

At parse time distances are stored as:
```python
stored = log(clip(dist_metres, min_distance, max_distance))
```

Objects without valid distance use the sentinel `INVALID_DISTANCE = -10.0`. The distance loss silently skips these entries. At inference:
```python
dist_metres = clip(exp(pred_distance), min_distance, max_distance)
```

### Label cache

The first time a label directory is parsed, a `.npz` cache file is written alongside it:

```
labels/train/.cache_as15_d0.5_200.0.npz
```

Subsequent runs load it in milliseconds. The cache is **automatically invalidated** when any `.txt` file's modification time or size changes, or when `angle_step`, `min_distance`, or `max_distance` change.

---

## 5. Project Structure

```
yolov8_extended/
│
├── configs/
│   ├── config.py               # Config dataclasses + YAML + argparse system
│   └── default.yaml            # All defaults — edit this first
│
├── data/
│   ├── parsers.py              # V8ParserExtended, V8DistanceParser, disk cache
│   └── dataset.py              # PolyDataset, PolyDistDataset, build_dataloader
│
├── models/
│   └── model.py                # YOLOv8Extended + load_pretrained_backbone
│
├── loss/
│   └── loss.py                 # YOLOv8ExtendedLoss + TaskAlignedAssigner
│
├── postprocess/
│   └── decode.py               # PostProcessor, Detection dataclass
│
├── utils/
│   ├── star_polygon.py         # Star encoding, augmentation helpers
│   ├── metrics.py              # BBoxF1Metric
│   ├── logger.py               # YOLOv8-style progress + CSV + TensorBoard
│   └── visualiser.py           # Per-epoch debug panels
│
├── train.py                    # Training entry point
├── test.py                     # Evaluation entry point
├── infer.py                    # Inference CLI + Predictor class
└── visualize_dataloader.py     # Standalone dataloader inspection tool
```

---

## 6. Installation

```bash
git clone <repo> && cd yolov8_extended

python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/macOS:  source venv/bin/activate

pip install -r requirements.txt

# recommended: enables pretrained weight download
pip install ultralytics

# optional: TensorBoard
pip install tensorboard
```

**requirements.txt**
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
numpy>=1.24.0
pyyaml>=6.0
matplotlib>=3.7.0
```

**AMD ROCm** — install the ROCm PyTorch wheel from [pytorch.org](https://pytorch.org), then pass `--gpu_type amd --no_amp`. See [Section 16](#16-gpu-compatibility).

---

## 7. Configuration System

All hyperparameters follow a strict three-tier priority:

```
configs/default.yaml  <--  --cfg my_exp.yaml  <--  CLI flags
      lowest                                          highest
```

Higher tiers override lower ones. Every parameter is individually addressable.

### Key fields in `default.yaml`

```yaml
model:
  model_size:      "s"       # n / s / m / l / x
  num_classes:     80
  angle_step:      15        # polygon bin width in degrees
  num_dist_blocks: 1         # ConvBN layers in distance head
  gpu_type:        "nvidia"  # "nvidia" → BN  |  "amd" → GN
  pretrained:      true      # transfer official YOLOv8 backbone+neck weights

data:
  poly_dataset_root: "data/polygon"   # single-root mode
  poly_datasets:     []               # multi-dataset mode (overrides above)
  poly_weights:      []               # per-dataset sampling weights
  dist_dataset_root: ""               # distance dataset (empty = disabled)
  class_names:       []               # for visualisation labels
  img_size:    640
  batch_size:  16

train:
  epochs:       300
  lr0:          0.01
  save_dir:     "runs/train"   # actual dir = runs/train/exp1, exp2, ...
  val_interval: 5
```

### Creating an experiment config

Copy and modify the defaults — only changed fields need to appear:

```bash
cp configs/default.yaml configs/my_experiment.yaml
python train.py --cfg configs/my_experiment.yaml
```

The **effective config** (merged result of all three tiers) is automatically saved to `{save_dir}/config.yaml` at the start of every run for exact reproducibility.

---

## 8. Training

### Quick start

```bash
python train.py \
    --poly_dataset_root data/polygon \
    --num_classes 10 \
    --model_size s \
    --epochs 300
```

The first run creates `runs/train/exp1/`. A second run creates `runs/train/exp2/` and so on — no overwriting.

### Multiple polygon datasets with sampling weights

```bash
python train.py \
    --poly_datasets \
        data/dataset_a/images/train:data/dataset_a/labels/train \
        data/dataset_b/images/train:data/dataset_b/labels/train \
        data/dataset_c/images/train:data/dataset_c/labels/train \
    --poly_weights 3.0 1.0 1.0 \
    --num_classes 10
```

Dataset A will be sampled 3× more frequently than B or C. Weights are normalised internally so only relative values matter. Uses `torch.utils.data.WeightedRandomSampler` under the hood.

Equivalently in YAML:
```yaml
data:
  poly_datasets:
    - "data/dataset_a/images/train:data/dataset_a/labels/train"
    - "data/dataset_b/images/train:data/dataset_b/labels/train"
  poly_weights: [3.0, 1.0]
```

### With an optional distance dataset

```bash
python train.py \
    --poly_dataset_root data/polygon \
    --dist_dataset_root data/polygon_distance \
    --num_classes 10
```

The distance dataset is independent and entirely optional. The distance loss only fires for objects that have a valid distance label; all other objects contribute only to box, class, and polygon losses.

### Pretrained weights

Enabled by default. The official ultralytics weights are downloaded automatically on first run (requires `pip install ultralytics` or internet).

```bash
# disable
python train.py --no_pretrained

# check what was transferred (logged at start)
# Pretrained weights loaded  matched=214  skipped_shape=12  ...
```

Only backbone and neck parameters are transferred. All three heads (detection, polygon, distance) are always randomly initialised.

### Resume training

```bash
python train.py --resume runs/train/exp1/last.pt
```

Resumes from the saved epoch and optimiser state. The save directory is inherited from the checkpoint — no new `expN` folder is created.

### Common recipes

**Fast debugging:**
```bash
python train.py \
    --poly_dataset_root data/polygon --num_classes 5 \
    --model_size n --epochs 20 --batch_size 4 --img_size 416 \
    --val_interval 2 --vis_interval 1 --mosaic_prob 0.0 --no_pretrained
```

**AMD GPU:**
```bash
python train.py \
    --poly_dataset_root data/polygon --num_classes 10 \
    --gpu_type amd --no_amp
```

### Full CLI reference

```
Config:
  --cfg PATH                      YAML config file

Model:
  --model_size {n,s,m,l,x}        default: s
  --num_classes INT                default: 80
  --angle_step INT                 polygon bin width in degrees, default: 15
  --num_dist_blocks INT            distance head depth, default: 1
  --min_distance METRES            default: 0.5
  --max_distance METRES            default: 200.0
  --gpu_type {nvidia,amd}          normalisation layer, default: nvidia
  --no_pretrained                  skip backbone weight transfer

Data:
  --poly_dataset_root PATH         single dataset root
  --poly_datasets IMG:LBL [...]    multiple datasets as img_dir:lbl_dir pairs
  --poly_weights W [...]           per-dataset sampling weights (parallel list)
  --dist_dataset_root PATH         distance dataset root (optional)
  --class_names NAME [...]         class name strings for visualisation
  --img_size INT                   default: 640
  --batch_size INT                 default: 16
  --num_workers INT                default: 4
  --mosaic_prob FLOAT              default: 1.0
  --flip_lr_prob FLOAT             default: 0.5
  --hsv_h / --hsv_s / --hsv_v     HSV jitter fractions

Training:
  --epochs INT                     default: 300
  --warmup_epochs INT              default: 3
  --lr0 FLOAT                      initial / peak LR, default: 0.01
  --lrf FLOAT                      final LR multiplier for cosine decay, default: 0.01
  --momentum FLOAT                 default: 0.937
  --weight_decay FLOAT             default: 0.0005
  --device STR                     default: cuda
  --save_dir PATH                  base dir, auto-incremented, default: runs/train
  --val_interval INT               default: 5
  --conf_thres FLOAT               default: 0.5
  --iou_thres FLOAT                default: 0.45
  --resume PATH                    resume from checkpoint
  --no_amp                         disable AMP (required for some AMD setups)

Loss gains:
  --box_gain / --cls_gain / --dfl_gain
  --poly_gain / --dist_gain
  --poly_dist_gain / --poly_conf_gain / --poly_angle_gain

Logging & Visualisation:
  --log_interval INT               log step loss every N steps, default: 10
  --vis_interval INT               save debug panels every N epochs, default: 1
  --vis_max_images INT             max images per panel, default: 8
  --no_tensorboard                 disable TensorBoard logging
```

### Output files

```
runs/train/
└── exp1/
    ├── best.pt             # checkpoint with highest val F1
    ├── last.pt             # most recent checkpoint
    ├── config.yaml         # effective merged config (reproducibility)
    ├── logs/
    │   ├── train.log       # full DEBUG trace
    │   ├── metrics.csv     # all scalars per step/epoch
    │   └── tb/             # TensorBoard event files
    └── vis/
        ├── epoch_0000/
        │   ├── train_batch.png
        │   ├── val_compare.png
        │   ├── loss_curves.png
        │   └── polygon_debug.png
        └── epoch_0005/
            └── ...
```

---

## 9. Validation & Testing

```bash
# evaluate on test split
python test.py \
    --weights runs/train/exp1/best.pt \
    --poly_img data/polygon/images/test \
    --poly_lbl data/polygon/labels/test

# load config from saved experiment (ensures matching hyperparameters)
python test.py \
    --cfg runs/train/exp1/config.yaml \
    --weights runs/train/exp1/best.pt \
    --poly_img data/polygon/images/test \
    --poly_lbl data/polygon/labels/test

# custom IoU matching threshold
python test.py \
    --weights runs/train/exp1/best.pt \
    --poly_img data/polygon/images/test \
    --poly_lbl data/polygon/labels/test \
    --iou_metric 0.75
```

### Metric: Bounding-box F1

The validation and test metric is **micro-averaged F1** computed via bounding-box IoU matching (threshold 0.5 by default). Polygon and distance quality are evaluated qualitatively via the `val_compare.png` visualisation panels.

Sample output:
```
── Per-class results (IoU@0.5) ──────────────────────
  class_0         P=0.8712  R=0.8103  F1=0.8396
  class_1         P=0.9105  R=0.8820  F1=0.8960
  class_2         P=0.7840  R=0.7590  F1=0.7713

  Micro avg       P=0.8552  R=0.8171  F1=0.8357
```

---

## 10. Inference

### CLI

```bash
# single image
python infer.py --weights runs/train/exp1/best.pt --source image.jpg

# directory
python infer.py \
    --weights runs/train/exp1/best.pt \
    --source images/ \
    --save_dir predictions/ \
    --class_names person car truck \
    --conf_thres 0.4

# disable overlays
python infer.py \
    --weights runs/train/exp1/best.pt \
    --source images/ \
    --no_polygon --no_distance
```

### Programmatic API

```python
import cv2
from infer import Predictor

predictor = Predictor("runs/train/exp1/best.pt")

img        = cv2.imread("image.jpg")
detections = predictor(img)

for d in detections:
    print(d)
    # Detection(cls=0, score=0.923, dist=12.4m, bbox=[120,80,340,310], poly_pts=18)

    d.bbox        # np.ndarray (4,)   x1,y1,x2,y2  in original pixel coords
    d.cls         # int
    d.score       # float
    d.distance    # float  metres
    d.polygon     # np.ndarray (K, 2)  pixel coords, conf-filtered
    d.poly_conf   # np.ndarray (num_angles,)  raw per-bin confidence

# draw and save
vis = predictor.draw(img, detections, class_names=["person", "car"])
cv2.imwrite("result.jpg", vis)
```

---

## 11. Dataloader Visualiser

Inspect **exactly what the model receives** — letterboxed images with remapped labels rendered back to pixel space.

```bash
# save panels, no augmentation
python visualize_dataloader.py \
    --poly_dataset_root data/polygon \
    --dist_dataset_root data/polygon_distance \
    --num_classes 10 \
    --class_names person car truck \
    --save_dir vis_check/

# interactive window
python visualize_dataloader.py \
    --poly_dataset_root data/polygon \
    --show --batch_size 4

# augmented training view
python visualize_dataloader.py \
    --poly_dataset_root data/polygon \
    --aug --n_batches 3 --save_dir vis_aug/
```

### Panel annotations

| Element | Appearance |
|---------|-----------|
| Bounding box | Per-object colour, labelled class + distance |
| Polygon ray | Line from origin to each active vertex |
| Active vertex | Filled circle |
| Inactive bin stub | Short dim grey line showing expected angle |
| Polygon outline | Connects active vertices in angle order |
| Padding boundary | Dark grey rectangle marking the letterbox image area |
| `OOB` label | Red text on any object whose centre falls outside the image area |

### Console sanity check output

```
Batch 0  —  4 images, 22 objects total
  classes   : [0, 1, 3, 7]
  bbox cx   : [0.183, 0.891]
  dist valid: 8/22  range [2.3, 45.1]m
  ✓  all 187 active polygon vertices within [0, 1]
```

Run this **before starting a long training run** to catch dataset or coordinate mapping issues early.

---

## 12. Training Visualisations

Four debug panels saved to `{save_dir}/vis/epoch_{N:04d}/` every `vis_interval` epochs, and forwarded to TensorBoard.

### `train_batch.png`

Grid of augmented training images with GT annotations: bounding boxes with class + distance labels, full star polygon rays, active vertex dots, inactive bin stubs, and a per-image stat bar showing object count and distance flag.

### `val_compare.png`

Side-by-side per-image comparison:
- **Left** — Ground truth in lime green with polygon rays
- **Right** — Model predictions in per-class colours with polygon outlines
- **Confidence strip** — Horizontal bar under predictions, green = high confidence, red = low

### `loss_curves.png`

Dark-theme matplotlib grid: one subplot per loss component (`total`, `box`, `cls`, `dfl`, `poly_dist`, `poly_conf`, `poly_ang`, `dist`) plus subplots for val F1, val Precision, and val Recall. Raw curve overlaid with EMA-smoothed line. Dashed vertical lines mark each validation epoch.

### `polygon_debug.png`

Per-object ray diagram: each active bin annotated with angle in degrees and pixel distance from origin; inactive bins shown as dim stubs.

---

## 13. Logging

### Console

YOLOv8-style progress display — overwrites the same line every step:

```
──────────────────────────────────────────────────────────────────────────────
  Epoch   GPU-mem       box       cls       dfl      poly      dist  Instances  ImgSize
──────────────────────────────────────────────────────────────────────────────
    1/300    2.14G    7.4321    0.5210    1.2300    0.0821    0.0000        42      640  ━━━━━━━━━━━━━━━━━━━━ 200/200  23s/epoch  eta 01:55:00

  Validation  P=0.7821  R=0.7103  F1=0.7445  ★ NEW BEST
```

### File log

`logs/train.log` — full DEBUG trace including every step's loss values, gradient norms, and checkpoint events.

### CSV

`logs/metrics.csv` — one row per step and per epoch:

```python
import pandas as pd
df    = pd.read_csv("runs/train/exp1/logs/metrics.csv")
train = df[df.mode == "train"]
val   = df[df.mode == "val"]
```

Columns: `timestamp, mode, epoch, step, loss, box, cls, dfl, poly_dist, poly_conf, poly_ang, dist, lr, precision, recall, f1`

### TensorBoard

```bash
tensorboard --logdir runs/train/exp1/logs/tb
# or watch all experiments
tensorboard --logdir runs/
```

Key tags: `train/loss`, `train/{component}`, `train/grad_norm`, `val/f1`, `val/precision`, `val/recall`, `val_per_class/{cls}_f1`, `vis/train_batch`, `vis/val_compare`, `vis/loss_curves`, `vis/polygon_debug`.

---

## 14. Loss Functions

### Total loss

```
L = box_gain  × L_box
  + cls_gain  × L_cls
  + dfl_gain  × L_dfl
  + poly_gain × (poly_dist_gain  × L_poly_dist
                + poly_conf_gain  × L_poly_conf
                + poly_angle_gain × L_poly_angle)
  + dist_gain × L_distance

L = L × batch_size
```

Default gains: `box=7.5, cls=0.5, dfl=1.5, poly=0.1, dist=0.1, poly_dist=2.0, poly_conf=0.2, poly_angle=0.5`.

### Assignment

All losses share the **TaskAlignedAssigner** (top-13, alignment metric = `cls_score^0.5 × IoU^6.0`).

### Polygon sub-losses

All three masked by both the foreground anchor mask and the per-bin `conf > 0` mask.

**Radial distance** — MSE between `softplus(pred_dist)` and Euclidean distance from origin to GT vertex. Normalised by `num_active_vertices × num_fg_anchors`.

**Angle fractional** — BCE between `sigmoid(pred_angle)` and the fractional GT angle within its bin: `frac = (angle_deg − bin × step) / step`.

**Confidence** — BCE across all `num_angles` bins (active target=1, inactive target=0).

### Scalar distance loss

L1 loss on foreground anchors with valid distance labels (`≠ INVALID_DISTANCE`). Objects from polygon-only datasets contribute zero gradient to this loss.

---

## 15. Post-Processing

1. **Box** — DFL softmax → LTRB offsets → `x1y1x2y2` normalised.
2. **Class** — sigmoid → max score and class id.
3. **Confidence filter** — `max_cls_score ≥ conf_thres`.
4. **NMS** — greedy IoU NMS, no external library.
5. **Polygon decode**:
   - `poly_dist  = softplus(raw_dist)`
   - `poly_angle = sigmoid(raw_angle)` — fractional bin offset in `[0, 1)`
   - `abs_angle  = (poly_angle + bin_offset) / num_angles × 360`
   - `dx = dist × cos(abs_angle)`,  `dy = dist × sin(abs_angle)`
   - `poly_x = origin_x − dx / img_size × stride`  (note: subtraction as specified)
   - `poly_y = origin_y − dy / img_size × stride`
   - Confidence-filter vertices with `poly_conf_thres = 0.5`
   - Scale to original image size
6. **Distance** — `clip(exp(pred_dist), min_distance, max_distance)`

---

## 16. GPU Compatibility

### NVIDIA (CUDA)

Fully supported. Mixed precision (AMP) enabled by default.

### AMD (ROCm)

ROCm maps its HIP/MIOpen stack onto the CUDA API. Two issues arise on newer GPUs (RX 9070/9080, `gfx1201`):

**MIOpen JIT failure (BatchNorm)**

```
MIOpen(HIP): Error [Compile] … 'type_traits' file not found
RuntimeError: miopenStatusUnknownError
```

Fix — use GroupNorm (no MIOpen JIT required):
```bash
python train.py --gpu_type amd
```

**AMP instability on older ROCm builds**
```bash
python train.py --gpu_type amd --no_amp
```

**Alternative workaround** (keep BatchNorm, disable cuDNN):
```bash
set MIOPEN_DEBUG_DISABLE_FIND_DB=1
set MIOPEN_FIND_MODE=1
python train.py --no_amp
```

### Windows

- `num_workers > 0` can hang; use `--num_workers 0` if you see freezing at startup.
- ANSI colour codes work in Windows Terminal and VS Code; fall back to plain text in `cmd.exe`.

---

## 17. Extending the Codebase

### Adding a new augmentation

Implement the transform in `utils/star_polygon.py` in polar/star form (see `flip_lr_star` as a template), then apply it in `data/dataset.py` inside `__getitem__`, updating both the image and the star target together.

### Adding a new output head

1. Define the `nn.Module` in `models/model.py`.
2. Register it as an `nn.ModuleList` over the 3 FPN scales in `YOLOv8Extended.__init__`.
3. Return its output from `forward()`.
4. Add the loss term in `loss/loss.py` and a gain to `TrainConfig`.
5. Add decoding in `postprocess/decode.py` and a field to `Detection`.

### Adding a dataset parser

Subclass `_BasePolyDataset` in `data/dataset.py` and override `_make_parser()` to return your parser. The parser must implement `parse_dir(label_dir) → dict[str, np.ndarray]`.

### Swapping in the real ultralytics backbone

Replace `YOLOv8Backbone` and `YOLOv8Neck` in `models/model.py` with the real ultralytics modules. The heads attach to the neck outputs unchanged — only channel counts need to match.

---

## 18. Troubleshooting

**`CUDA out of memory`**
Reduce `--batch_size` or `--img_size`. The `s` model at 640 px uses ~6 GB at batch 16.

**Boxes and polygons appear in the grey letterbox padding**
A `_flip_targets_lr` function was missing in an earlier version. Ensure you are on the current `data/dataset.py` and clear `__pycache__`:
```bash
# Windows
for /r %d in (__pycache__) do @rmdir /s /q "%d"
# Linux / macOS
find . -type d -name __pycache__ -exec rm -rf {} +
```

**`RuntimeError: shape '[N, -1, 3]' is invalid`** in `visualize_dataloader.py`
Delete `__pycache__` as above — stale `.pyc` bytecode from a previous bug.

**`No weights were matched` when loading pretrained weights**
The layer-index map in `load_pretrained_backbone` covers the standard YOLOv8 architecture. If you've modified channel widths, the shapes will not match and the count will be low. Pass `--no_pretrained` or add mappings to `LAYER_TO_MODULE` in `model.py`.

**Loss is NaN**
Ensure `min_distance > 0` (log of zero is −∞). Verify label coordinates are in `[0, 1]`. Try reducing `--lr0`.

**Val F1 stays at 0**
The default IoU matching threshold is 0.5. Pass `--iou_metric 0.3` in `test.py` to diagnose whether boxes are slightly offset. Confirm GT class indices match predictions.

**`num_workers > 0` hangs on Windows**
Pass `--num_workers 0`.

**TensorBoard shows no data**
Point `--logdir` to `{save_dir}/logs/tb/` not the experiment root. Verify `pip install tensorboard`.

**Cache not updating after editing labels**
The cache checks file `mtime` and `size`. If mtime is unchanged after an edit, delete the `.cache_*.npz` files from the label directories manually.

---

## Citing

If you build on this work, please also cite the original YOLOv8:

```bibtex
@software{yolov8_ultralytics,
  author  = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title   = {Ultralytics YOLOv8},
  year    = {2023},
  url     = {https://github.com/ultralytics/ultralytics}
}
```
