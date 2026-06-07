# YOLOv8-Extended — Polygon & Distance Detection

An extension of [YOLOv8s](https://docs.ultralytics.com/) that adds two new output modalities **alongside** the standard bounding-box and class outputs:

| Output | Description |
|--------|-------------|
| **Bounding box** | Standard YOLO box (cx, cy, w, h) decoded via DFL |
| **Class** | Standard softmax classification |
| **Polygon** | Star-shaped polygon representation — per-object shape mask |
| **Distance** | Scalar metric depth per anchor point (metres, log-encoded) |

All four outputs share a single forward pass. The box and class heads are **unchanged** from standard YOLOv8; the polygon and distance heads are additions that can be trained independently or jointly.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Star Polygon Representation](#2-star-polygon-representation)
3. [Dataset Format](#3-dataset-format)
4. [Project Structure](#4-project-structure)
5. [Installation](#5-installation)
6. [Configuration](#6-configuration)
7. [Training](#7-training)
8. [Validation & Testing](#8-validation--testing)
9. [Inference](#9-inference)
10. [Logging & Visualisation](#10-logging--visualisation)
11. [Loss Functions](#11-loss-functions)
12. [Post-Processing](#12-post-processing)
13. [Extending the Codebase](#13-extending-the-codebase)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Architecture Overview

```
Input Image (B, 3, 640, 640)
        │
        ▼
┌─────────────────┐
│  YOLOv8Backbone │  C2f blocks + SPPF, produces P3/P4/P5
└────────┬────────┘
         │  [P3, P4, P5]
         ▼
┌─────────────────┐
│  YOLOv8Neck     │  PAN-FPN, top-down + bottom-up
└────────┬────────┘
         │  [N3, N4, N5]  (strides 8, 16, 32)
         ▼
  For each FPN scale:
  ┌──────────────────────────────────────────────────────┐
  │  DetHead                                             │
  │    box_pre: ConvBN(3x3) → ConvBN(3x3)  ← penultimate│──┐
  │    box_out: Conv1x1 → (4 × REG_MAX) channels        │  │
  │    cls_pre: ConvBN(3x3) → ConvBN(3x3)               │  │
  │    cls_out: Conv1x1 → num_classes channels           │  │
  └──────────────────────────────────────────────────────┘  │ penultimate feat
                                                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  PolyHead  (3 independent Conv1x1, from penultimate box feat)│
  │    poly_conf  → (B, num_angles, H, W)  raw logits           │
  │    poly_angle → (B, num_angles, H, W)  raw logits           │
  │    poly_dist  → (B, num_angles, H, W)  pre-softplus         │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────┐
  │  DistHead  (from neck feature, same as det head)     │
  │    num_dist_blocks × ConvBN(3x3)                     │
  │    Conv1x1 → (B, 1, H, W)  scalar distance          │
  └──────────────────────────────────────────────────────┘
```

### Scale variants

| Variant | Depth mult | Width mult | Approx params |
|---------|-----------|-----------|---------------|
| `n` | 0.33 | 0.25 | ~3 M |
| `s` | 0.33 | 0.50 | ~11 M |
| `m` | 0.67 | 0.75 | ~26 M |
| `l` | 1.00 | 1.00 | ~44 M |
| `x` | 1.33 | 1.25 | ~69 M |

Set with `--model_size s` (default) or in `configs/default.yaml`.

---

## 2. Star Polygon Representation

Instead of storing a variable-length list of vertices, each polygon is encoded as a fixed-size **star-shaped** descriptor centred on the bounding box centre.

### Encoding

```
num_angles = 360 // angle_step          # default: 24 bins (angle_step=15°)

For each polygon vertex:
  1. Compute angle and distance from bbox centre
  2. Map to angle bin:  bin_idx = floor(angle_deg / angle_step)
  3. Keep only the farthest vertex per bin (max distance wins)

Bins with no vertex → conf = 0, xy = (0, 0)
Bins with a vertex  → conf = 1, xy = vertex coordinates
```

### Wire format

```
[origin_x, origin_y,
 x0, y0, conf0,
 x1, y1, conf1,
 ...
 x_{N-1}, y_{N-1}, conf_{N-1}]

Total length = 2 + num_angles * 3
```

All coordinates are **normalised** (0–1) in the dataset/target tensors.

### Augmentation in polar form

Horizontal flip remaps bins analytically without recomputing from raw vertices:

```
new_angle = (180 - angle_deg) % 360
new_bin   = new_angle // angle_step
```

Mosaic assembly translates origin and vertices in normalised space proportionally, then rescales back to [0, 1].

---

## 3. Dataset Format

### Directory layout

```
data/
├── polygon/                     # polygon-only dataset (V8ParserExtended)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
└── polygon_distance/            # polygon + distance (V8DistanceParser)
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

### Label file format

Each `.txt` file contains one object per line. Images and label files share the same stem name (`0001.jpg` ↔ `0001.txt`).

**Polygon-only** (paired with `poly_dataset_root`):
```
<class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>
```

**Polygon + distance** (paired with `dist_dataset_root`):
```
<class_id> <x1> <y1> <x2> <y2> ... <xN> <yN> <distance_metres>
```

- All `xi`, `yi` values are **normalised** floats in [0, 1] (YOLO convention).
- `N` is the number of polygon vertices (can vary per object and per file).
- `distance_metres` must be > 0 for a valid distance reading; non-positive values are treated as missing.
- Multiple objects on separate lines; blank lines are skipped.

### Example label file (polygon-only)

```
0 0.512 0.310 0.601 0.298 0.635 0.401 0.590 0.455 0.501 0.440
1 0.120 0.550 0.180 0.540 0.200 0.620 0.115 0.630
```

### Distance encoding

At parse time, distances are log-clipped and stored as:

```python
dist_stored = log(clip(dist_metres, min_distance, max_distance))
```

Objects without a valid distance (polygon-only dataset, or distance ≤ 0) are stored with `dist = -10.0` (the sentinel `INVALID_DISTANCE`). The distance loss ignores these entries automatically.

At inference time, distance is recovered as:

```python
dist_metres = clip(exp(pred_distance), min_distance, max_distance)
```

---

## 4. Project Structure

```
yolov8_extended/
│
├── configs/
│   ├── config.py           # Config dataclasses + YAML + argparse system
│   └── default.yaml        # All hyperparameter defaults (edit this first)
│
├── data/
│   ├── parsers.py          # V8ParserExtended, V8DistanceParser
│   └── dataset.py          # PolyDataset, PolyDistDataset, DataLoader factory
│
├── models/
│   └── model.py            # YOLOv8Extended (backbone, neck, all heads)
│
├── loss/
│   └── loss.py             # YOLOv8ExtendedLoss + TaskAlignedAssigner
│
├── postprocess/
│   └── decode.py           # PostProcessor, Detection dataclass
│
├── utils/
│   ├── star_polygon.py     # Star encoding / augmentation helpers
│   ├── metrics.py          # BBoxF1Metric
│   ├── logger.py           # Console + CSV + TensorBoard logger
│   └── visualiser.py       # Per-epoch debug panel generator
│
├── train.py                # Training entry point
├── test.py                 # Evaluation entry point
├── infer.py                # Inference entry point + Predictor class
└── requirements.txt
```

---

## 5. Installation

```bash
# clone / unzip the project
cd yolov8_extended

# create a virtual environment (recommended)
python -m venv venv && source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# optional: TensorBoard support
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

### GPU check

```python
import torch
print(torch.cuda.is_available())   # should print True
print(torch.cuda.get_device_name(0))
```

---

## 6. Configuration

The configuration system follows a strict three-tier priority:

```
defaults (default.yaml)  ←  custom YAML (--cfg)  ←  CLI flags
```

Higher tiers override lower ones. Every parameter is individually addressable.

### default.yaml — annotated

```yaml
model:
  model_size:      "s"      # backbone scale: n / s / m / l / x
  num_classes:     80       # number of object classes
  angle_step:      15       # polygon bin width in degrees
                            # num_angles = 360 // angle_step = 24
  num_dist_blocks: 1        # ConvBN layers in distance head before final 1×1
  min_distance:    0.5      # metres — log-distance lower clip
  max_distance:    200.0    # metres — log-distance upper clip

data:
  poly_dataset_root: "data/polygon"
  dist_dataset_root: "data/polygon_distance"
  img_size:    640          # square resolution for training and inference
  mosaic_prob: 1.0          # probability of using mosaic augmentation
  hsv_h:       0.015        # hue jitter fraction
  hsv_s:       0.7          # saturation jitter fraction
  hsv_v:       0.4          # value jitter fraction
  flip_lr_prob: 0.5         # horizontal flip probability
  batch_size:  16
  num_workers: 4

train:
  epochs:        300
  warmup_epochs: 3          # linear LR warm-up for first N epochs
  lr0:           0.01       # initial LR (peak after warm-up)
  lrf:           0.01       # final LR = lr0 * lrf (cosine decay target)
  momentum:      0.937
  weight_decay:  0.0005
  # loss gains
  box_gain:        7.5
  cls_gain:        0.5
  dfl_gain:        1.5
  poly_gain:       0.1      # total polygon loss multiplier
  dist_gain:       0.1      # scalar distance loss multiplier
  poly_dist_gain:  2.0      # polygon radial distance sub-loss gain
  poly_conf_gain:  0.2      # polygon confidence sub-loss gain
  poly_angle_gain: 0.5      # polygon angle sub-loss gain
  # runtime
  device:       "cuda"
  save_dir:     "runs/train"
  val_interval: 5           # validate every N epochs
  conf_thres:   0.5
  iou_thres:    0.45
  # logging
  log_interval:   10        # print step loss every N steps
  vis_interval:   1         # save debug panels every N epochs
  vis_max_images: 8
  tensorboard:    true
```

### Creating an experiment config

Copy and edit the defaults for your experiment:

```bash
cp configs/default.yaml configs/my_experiment.yaml
# edit configs/my_experiment.yaml
python train.py --cfg configs/my_experiment.yaml
```

Only the fields you change need to appear in your YAML — all others inherit from `default.yaml`.

---

## 7. Training

### Quickstart

```bash
python train.py \
    --poly_dataset_root data/polygon \
    --dist_dataset_root data/polygon_distance \
    --num_classes 10 \
    --model_size s \
    --epochs 300 \
    --save_dir runs/exp1
```

### Full CLI reference — `train.py`

```
Model arguments:
  --model_size {n,s,m,l,x}    Backbone scale (default: s)
  --num_classes INT            Number of object classes (default: 80)
  --angle_step INT             Polygon bin width in degrees (default: 15)
  --num_dist_blocks INT        ConvBN layers in distance head (default: 1)
  --min_distance FLOAT         Distance clip lower bound, metres (default: 0.5)
  --max_distance FLOAT         Distance clip upper bound, metres (default: 200.0)

Data arguments:
  --poly_dataset_root PATH     Root directory for polygon dataset
  --dist_dataset_root PATH     Root directory for polygon+distance dataset
  --img_size INT               Input resolution (default: 640)
  --batch_size INT             (default: 16)
  --num_workers INT            DataLoader workers (default: 4)
  --mosaic_prob FLOAT          Mosaic augmentation probability (default: 1.0)
  --flip_lr_prob FLOAT         Horizontal flip probability (default: 0.5)
  --hsv_h / --hsv_s / --hsv_v  HSV colour jitter fractions

Training arguments:
  --epochs INT                 (default: 300)
  --warmup_epochs INT          Linear LR warm-up length (default: 3)
  --lr0 FLOAT                  Initial / peak learning rate (default: 0.01)
  --lrf FLOAT                  Final LR multiplier for cosine decay (default: 0.01)
  --momentum FLOAT             SGD momentum (default: 0.937)
  --weight_decay FLOAT         (default: 0.0005)
  --device STR                 "cuda" or "cpu" (default: cuda)
  --save_dir PATH              Experiment output directory (default: runs/train)
  --val_interval INT           Validate every N epochs (default: 5)
  --conf_thres FLOAT           Object confidence threshold (default: 0.5)
  --iou_thres FLOAT            NMS IoU threshold (default: 0.45)
  --resume PATH                Resume from a checkpoint file

Loss gains:
  --box_gain FLOAT             (default: 7.5)
  --cls_gain FLOAT             (default: 0.5)
  --dfl_gain FLOAT             (default: 1.5)
  --poly_gain FLOAT            Total polygon loss multiplier (default: 0.1)
  --dist_gain FLOAT            Scalar distance loss multiplier (default: 0.1)
  --poly_dist_gain FLOAT       (default: 2.0)
  --poly_conf_gain FLOAT       (default: 0.2)
  --poly_angle_gain FLOAT      (default: 0.5)

Logging & Visualisation:
  --cfg PATH                   Path to YAML config (overrides defaults)
  --log_interval INT           Print step loss every N steps (default: 10)
  --vis_interval INT           Save debug panels every N epochs (default: 1)
  --vis_max_images INT         Max images per panel grid (default: 8)
  --no_tensorboard             Disable TensorBoard logging
```

### Resume from checkpoint

```bash
python train.py --resume runs/exp1/last.pt
```

The checkpoint stores epoch number, model weights, optimiser state, and best F1. Training resumes from the next epoch automatically.

### Common training recipes

**Small dataset, fast iteration:**
```bash
python train.py \
    --epochs 100 --batch_size 8 --img_size 416 \
    --val_interval 2 --vis_interval 2 \
    --model_size n
```

**Production run, medium model:**
```bash
python train.py \
    --cfg configs/default.yaml \
    --model_size m --num_classes 20 \
    --epochs 300 --lr0 0.01 --batch_size 32 \
    --save_dir runs/production
```

**Fine-tuning with a lower LR:**
```bash
python train.py \
    --resume runs/production/best.pt \
    --lr0 0.001 --lrf 0.001 \
    --epochs 50 --save_dir runs/finetune
```

**Disable mosaic for debugging:**
```bash
python train.py --mosaic_prob 0.0 --vis_interval 1
```

### Output files

```
runs/exp1/
├── best.pt             # checkpoint with highest val F1
├── last.pt             # checkpoint from last validation epoch
├── config.yaml         # effective config (exact reproducibility)
├── logs/
│   ├── train.log       # full training log (DEBUG level)
│   ├── metrics.csv     # all scalars, one row per step/epoch
│   └── tb/             # TensorBoard event files
└── vis/
    ├── epoch_0000/
    │   ├── train_batch.png
    │   ├── val_pred.png
    │   ├── loss_curves.png
    │   └── polygon_debug.png
    ├── epoch_0005/
    │   └── ...
    └── ...
```

---

## 8. Validation & Testing

### Running evaluation

```bash
python test.py \
    --weights runs/exp1/best.pt \
    --poly_img data/polygon/images/test \
    --poly_lbl data/polygon/labels/test
```

**With distance dataset:**
```bash
python test.py \
    --weights runs/exp1/best.pt \
    --poly_img data/polygon/images/test \
    --poly_lbl data/polygon/labels/test \
    --dist_img data/polygon_distance/images/test \
    --dist_lbl data/polygon_distance/labels/test
```

**Load config from saved experiment:**
```bash
python test.py \
    --cfg runs/exp1/config.yaml \
    --weights runs/exp1/best.pt \
    --poly_img data/polygon/images/test \
    --poly_lbl data/polygon/labels/test
```

### CLI reference — `test.py`

```
  --weights PATH       Required. Checkpoint to evaluate
  --poly_img PATH      Required. Test polygon images directory
  --poly_lbl PATH      Required. Test polygon labels directory
  --dist_img PATH      Test distance images (falls back to poly_img)
  --dist_lbl PATH      Test distance labels (falls back to poly_lbl)
  --iou_metric FLOAT   IoU threshold for TP/FP matching (default: 0.5)
  --conf_thres FLOAT   Detection confidence threshold
  --cfg PATH           Load settings from YAML
```

### Metric: F1 score

The validation metric is **bbox-only F1** computed via IoU matching (default IoU threshold = 0.5). Polygon and distance are not included in the metric — they are monitored via the visualisation panels during training.

The metric reports:
- **Per-class** precision, recall, F1
- **Micro-averaged** precision, recall, F1 (all classes combined)

Example output:
```
── Per-class results (IoU@0.5) ──────────────────────────────
  class_0         P=0.8712  R=0.8103  F1=0.8396
  class_1         P=0.9105  R=0.8820  F1=0.8960
  class_2         P=0.7840  R=0.7590  F1=0.7713

  Micro avg       P=0.8552  R=0.8171  F1=0.8357
```

---

## 9. Inference

### CLI

```bash
# single image
python infer.py --weights runs/exp1/best.pt --source image.jpg

# directory of images
python infer.py \
    --weights runs/exp1/best.pt \
    --source images/ \
    --save_dir predictions/ \
    --conf_thres 0.4 \
    --class_names person car truck bicycle

# disable polygon / distance overlay
python infer.py \
    --weights runs/exp1/best.pt \
    --source images/ \
    --no_polygon --no_distance

# load full config from saved experiment
python infer.py \
    --cfg runs/exp1/config.yaml \
    --weights runs/exp1/best.pt \
    --source images/
```

### CLI reference — `infer.py`

```
  --weights PATH           Required. Checkpoint to run
  --source PATH            Required. Image file or directory
  --save_dir PATH          Output directory for annotated images (default: out)
  --class_names STR [...]  Optional list of class name strings
  --no_polygon             Suppress polygon drawing
  --no_distance            Suppress distance label drawing
  --conf_thres FLOAT       Detection confidence threshold
  --iou_thres FLOAT        NMS IoU threshold
  --cfg PATH               Load settings from YAML
```

### Programmatic API

Use the `Predictor` class to integrate inference into a pipeline:

```python
import cv2
from infer import Predictor

# load once
predictor = Predictor("runs/exp1/best.pt")

# run on any BGR numpy image
img = cv2.imread("image.jpg")
detections = predictor(img)

for d in detections:
    print(d)
    # Detection(cls=0, score=0.923, dist=12.4m, bbox=[120, 80, 340, 310], poly_pts=18)

    # attributes
    d.bbox        # np.ndarray (4,)  x1,y1,x2,y2 in original pixel coords
    d.cls         # int
    d.score       # float   overall confidence
    d.distance    # float   metres
    d.polygon     # np.ndarray (K, 2)  pixel coords, conf-filtered vertices
    d.poly_conf   # np.ndarray (num_angles,)  per-bin confidence

# draw and save
vis = predictor.draw(img, detections, class_names=["person", "car"])
cv2.imwrite("result.jpg", vis)
```

### Output format

Each `Detection` object carries:

| Attribute | Type | Description |
|-----------|------|-------------|
| `bbox` | `np.ndarray (4,)` | `[x1, y1, x2, y2]` in original pixel coordinates |
| `cls` | `int` | Class index |
| `score` | `float` | Object confidence (max class score) |
| `distance` | `float` | Predicted metric depth in metres |
| `polygon` | `np.ndarray (K, 2)` | Polygon vertices in original pixel coords (K ≤ num_angles, conf-filtered) |
| `poly_conf` | `np.ndarray (num_angles,)` | Raw per-bin confidence values |

---

## 10. Logging & Visualisation

### TensorBoard

```bash
# start TensorBoard pointing at the experiment logs
tensorboard --logdir runs/exp1/logs/tb

# or watch all experiments
tensorboard --logdir runs/
```

TensorBoard tracks:

| Tag | Description |
|-----|-------------|
| `step/loss_total` | Per-step total loss |
| `step/{component}` | Per-step individual loss components |
| `step/lr` | Learning rate at each step |
| `train/loss` | Epoch-averaged total loss |
| `train/{component}` | Epoch-averaged component losses |
| `train/lr` | LR at epoch end |
| `train/grad_norm` | Gradient norm (after clipping) |
| `val/precision` | Validation precision |
| `val/recall` | Validation recall |
| `val/f1` | Validation F1 (primary metric) |
| `val_per_class/{cls}_f1` | Per-class F1 |
| `vis/train_batch` | GT-annotated training images |
| `vis/val_pred` | GT vs predicted overlay |
| `vis/loss_curves` | Loss curve grid |
| `vis/polygon_debug` | Star polygon ray visualisation |

### CSV metrics

`logs/metrics.csv` is appended every step and every epoch:

```
timestamp, mode, epoch, step, loss, box, cls, dfl, poly_dist, poly_conf, poly_ang, dist, lr, precision, recall, f1
2024-01-15 10:23:41, train, 0, 0, 12.3401, 3.2100, 1.0800, 0.9200, 2.1000, 0.3200, 0.8100, 0.0500, 0.003333, 0, 0, 0
2024-01-15 10:24:55, val, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7200, 0.6900, 0.7047
```

Plot with pandas:
```python
import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("runs/exp1/logs/metrics.csv")
train = df[df.mode == "train"]
val   = df[df.mode == "val"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
train.groupby("epoch")["loss"].mean().plot(ax=axes[0], title="Train Loss")
val.set_index("epoch")["f1"].plot(ax=axes[1], title="Val F1", marker="o")
plt.tight_layout(); plt.show()
```

### Visualisation panels

Saved to `vis/epoch_NNNN/` every `vis_interval` epochs:

**`train_batch.png`** — Augmented training images annotated with ground-truth boxes and star polygon rays. Useful for verifying augmentation correctness.

**`val_pred.png`** — Ground truth overlaid in green, model predictions in red. Lets you visually track precision/recall improvement epoch-by-epoch without waiting for full metric computation.

**`loss_curves.png`** — One subplot per loss component (box, cls, dfl, poly_dist, poly_conf, poly_angle, dist) plus a val-F1 subplot, all plotted from epoch 0 to the current epoch.

**`polygon_debug.png`** — Single-image close-up showing each angle bin's ray from the polygon origin. Active bins (conf ≥ threshold) are coloured; inactive bins show a short dim stub. The angle label at each ray midpoint shows the bin's starting angle in degrees.

---

## 11. Loss Functions

### Total loss

```
L_total = box_gain  × L_box
        + cls_gain  × L_cls
        + dfl_gain  × L_dfl
        + poly_gain × (poly_dist_gain  × L_poly_dist
                      + poly_conf_gain  × L_poly_conf
                      + poly_angle_gain × L_poly_angle)
        + dist_gain × L_distance

L_total = L_total × batch_size
```

### Assignment

All losses use the **TaskAlignedAssigner** (top-13 selection by `cls_score^0.5 × iou^6.0`). The assigner is shared between the box/cls and polygon branches — the polygon head targets are derived from the same assigned GT.

### Box loss

- **CIoU** between predicted and GT bounding boxes
- **DFL** (Distribution Focal Loss) on the LTRB offsets from anchor centres
- Both weighted by the per-anchor IoU alignment score

### Polygon loss (3 sub-components)

All three are masked by the foreground anchor mask AND the per-bin `target_conf` mask.

**Radial distance loss** (`L_poly_dist`): MSE between the softplus-activated predicted distance and the GT Euclidean distance from the origin to each active vertex.

**Angle fractional loss** (`L_poly_angle`): BCE between `sigmoid(pred_angle)` and the fractional part of the ground-truth angle within its bin `frac = (angle - bin * step) / step ∈ [0, 1)`.

**Confidence loss** (`L_poly_conf`): BCE with logits across all bins (both active and inactive).

### Scalar distance loss

L1 loss, computed only on foreground anchors whose assigned GT has a valid distance (≠ `INVALID_DISTANCE = -10.0`). Normalised by the count of valid objects in the batch.

---

## 12. Post-Processing

### Decode pipeline

1. **Box**: DFL softmax over REG_MAX=16 bins → LTRB offsets → x1y1x2y2 in normalised coords.
2. **Class**: sigmoid → max class score and class id.
3. **Confidence filter**: keep anchors with `max_cls_score ≥ conf_thres`.
4. **NMS**: greedy IoU-based NMS per class.
5. **Polygon decode** (for surviving anchors):
   - `poly_dist  = softplus(pred_dist)`   → radial distance per bin
   - `poly_angle = sigmoid(pred_angle)`   → fractional bin offset
   - `abs_angle  = (poly_angle + bin_offset) / num_angles × 360`
   - `dx = dist × cos(abs_angle)`,  `dy = dist × sin(abs_angle)`
   - `poly_x = origin_x − dx / img_size × stride`
   - `poly_y = origin_y − dy / img_size × stride`
   - Confidence-filter with `poly_conf_thres = 0.5`
   - Scale to original image size
6. **Distance decode**: `dist_metres = clip(exp(pred_distance), min, max)`

### NMS behaviour

NMS operates on the bounding box; polygon and distance are carried as attributes and are **not** used in suppression. Only one NMS call per image (across all classes simultaneously).

---

## 13. Extending the Codebase

### Adding a new augmentation

1. Implement the spatial transform in `utils/star_polygon.py`, following `flip_lr_star` as a template — operate in polar form where possible.
2. Apply it in `data/dataset.py` inside `_BasePolyDataset.__getitem__`, updating both image and star targets together.

### Adding a new head

1. Define the `nn.Module` in `models/model.py`.
2. Register it in `YOLOv8Extended.__init__` as an `nn.ModuleList` over the 3 FPN scales.
3. Return its output from `forward()`.
4. Add the corresponding loss in `loss/loss.py` and a gain in `TrainConfig`.
5. Add decoding in `postprocess/decode.py` and store the result in `Detection`.

### Changing the polygon bin resolution

```yaml
# configs/my_exp.yaml
model:
  angle_step: 10    # 36 bins instead of 24
```

`num_angles` is derived automatically. No other changes needed — the star format and all downstream code adapt to the new bin count.

### Using with ultralytics backbone

Replace `YOLOv8Backbone` and `YOLOv8Neck` in `models/model.py` with the real ultralytics modules. The `DetHead`, `PolyHead`, and `DistHead` attach to the neck outputs unchanged, requiring only that the input channel counts match.

---

## 14. Troubleshooting

**`CUDA out of memory`**
Reduce `--batch_size` or `--img_size`. As a guideline: `s` model at 640px uses ~6 GB at batch 16.

**`No images found`**
The dataloader expects `<dataset_root>/images/train/` and `<dataset_root>/labels/train/`. Check the directory structure matches exactly and that image extensions are `.jpg`, `.jpeg`, `.png`, or `.bmp`.

**`KeyError: 'model'` when loading checkpoint**
Pass the raw state dict: the checkpoint saves `{"model": state_dict, ...}`. If you're loading a raw `state_dict` file, it will be detected automatically.

**Polygon rays all pointing in the same direction**
Usually means the `angle_step` used at training differs from the value used at inference. Always load the experiment's `config.yaml` with `--cfg` when running `test.py` or `infer.py`.

**Loss is NaN after a few steps**
Check that `min_distance > 0` (log of zero is −∞) and that label coordinates are truly normalised to [0, 1]. Try reducing `lr0` and confirming gradient clipping is active (`max_norm=10.0` in the training loop).

**TensorBoard not showing up**
Install with `pip install tensorboard` and ensure you point `--logdir` at `runs/<exp>/logs/tb/`, not the experiment root.

**Val F1 = 0 despite visible detections**
The default IoU matching threshold for F1 is 0.5. If predicted boxes are slightly offset (common early in training), try `--iou_metric 0.3` in `test.py` to diagnose. Also confirm GT labels have the correct class index.

---

## Citing

If you build on this work, please also cite the original YOLOv8:

```
@software{yolov8_ultralytics,
  author  = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title   = {Ultralytics YOLOv8},
  year    = {2023},
  url     = {https://github.com/ultralytics/ultralytics}
}
```
