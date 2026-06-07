"""
visualize_dataloader.py
───────────────────────
Interactive inspection of exactly what gets sent to the model:
tensor images + collated target tensors rendered back to pixel space.

Shows per-image panels with:
  • The letterboxed / augmented image as the model sees it
  • Bounding box  (yellow)
  • Star polygon rays from origin to each active vertex  (colour per object)
  • Polygon outline connecting active vertices  (same colour, dashed feel)
  • Origin dot
  • Class label + distance tag (INVALID shown as "—")
  • Padding region boundary  (dim grey rectangle)

Usage
─────
  # quick check — one batch, no augmentation
  python visualize_dataloader.py \\
      --poly_dataset_root data/polygon \\
      --dist_dataset_root data/polygon_distance \\
      --num_classes 10 --no_aug

  # augmented training view, save to disk
  python visualize_dataloader.py \\
      --poly_dataset_root data/polygon \\
      --dist_dataset_root data/polygon_distance \\
      --num_classes 10 --aug \\
      --n_batches 3 --save_dir vis_check/

  # interactive window (press any key to advance, q to quit)
  python visualize_dataloader.py \\
      --poly_dataset_root data/polygon \\
      --dist_dataset_root data/polygon_distance \\
      --show

Keys (--show mode)
──────────────────
  any key  → next batch
  q / Esc  → quit
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ── allow running from the project root without installing ───────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import (
    COL_CLS, COL_CX, COL_CY, COL_DIST, COL_H, COL_STAR, COL_W,
    build_dataloader,
)
from data.parsers import INVALID_DISTANCE


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (BGR for OpenCV)
# ─────────────────────────────────────────────────────────────────────────────

_PAL_BGR = [
    (56,  56,  255),   # red
    (51, 157,  255),   # orange
    (249, 180,  74),   # sky-blue
    (65,  255, 149),   # green
    (255, 111, 221),   # pink
    (30,  153,  63),   # dark-green
    (170,  85, 255),   # purple
    (255, 165,   0),   # cyan-ish
    (0,   255, 255),   # yellow
    (255,   0, 128),   # magenta
]

def _col(i: int):
    return _PAL_BGR[i % len(_PAL_BGR)]


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_bbox(img: np.ndarray, cx, cy, w, h, color, label: str = "") -> None:
    """Draw a normalised cxcywh box."""
    H, W = img.shape[:2]
    x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)


def _draw_star(
    img: np.ndarray,
    star: np.ndarray,      # (2 + num_angles*3,)  normalised
    num_angles: int,
    angle_step: int,
    color,
    conf_thresh: float = 0.5,
) -> None:
    """
    Draw:
      • a ray from origin → each active vertex
      • a filled circle at each active vertex
      • a polyline connecting active vertices in angle order
      • a white+colour ring at the origin
      • dim stubs for inactive bins (so missing directions are visible)
    """
    H, W = img.shape[:2]

    ox = int(star[0] * W)
    oy = int(star[1] * H)
    verts = star[2:].reshape(num_angles, 3)   # (N, 3)  x, y, conf

    active_pts = []

    for i, (vx, vy, vc) in enumerate(verts):
        if vc >= conf_thresh:
            px, py = int(vx * W), int(vy * H)
            # ray
            cv2.line(img, (ox, oy), (px, py), color, 1, cv2.LINE_AA)
            # vertex dot
            cv2.circle(img, (px, py), 4, color, -1, cv2.LINE_AA)
            active_pts.append((px, py))
        else:
            # dim stub showing expected direction
            angle_rad = math.radians(i * angle_step)
            ex = int(ox + 18 * math.cos(angle_rad))
            ey = int(oy + 18 * math.sin(angle_rad))
            cv2.line(img, (ox, oy), (ex, ey), (70, 70, 70), 1)

    # polygon outline
    if len(active_pts) >= 2:
        pts = np.array(active_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=color,
                      thickness=1, lineType=cv2.LINE_AA)

    # origin ring
    cv2.circle(img, (ox, oy), 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (ox, oy), 6, color, 2, cv2.LINE_AA)


def _detect_letterbox_pad(img_rgb: np.ndarray) -> tuple[int, int, int, int]:
    """
    Precisely detect letterbox padding by finding the outermost rows/columns
    where ALL pixels are exactly the letterbox fill value (114).
    Returns (pad_x, pad_y, pad_x_right, pad_y_bottom).
    """
    H, W = img_rgb.shape[:2]
    PAD_VAL = 114

    # Check rows from top
    pad_y_top = 0
    for r in range(H):
        if np.all(img_rgb[r, :, :] == PAD_VAL):
            pad_y_top = r + 1
        else:
            break

    # Check rows from bottom
    pad_y_bot = 0
    for r in range(H - 1, -1, -1):
        if np.all(img_rgb[r, :, :] == PAD_VAL):
            pad_y_bot = H - r
        else:
            break

    # Check columns from left
    pad_x_left = 0
    for c in range(W):
        if np.all(img_rgb[:, c, :] == PAD_VAL):
            pad_x_left = c + 1
        else:
            break

    # Check columns from right
    pad_x_right = 0
    for c in range(W - 1, -1, -1):
        if np.all(img_rgb[:, c, :] == PAD_VAL):
            pad_x_right = W - c
        else:
            break

    return pad_x_left, pad_y_top, pad_x_right, pad_y_bot


# ─────────────────────────────────────────────────────────────────────────────
# Per-image panel builder
# ─────────────────────────────────────────────────────────────────────────────

def render_batch_panel(
    imgs: torch.Tensor,      # (B, 3, H, W)  float32 [0,1]
    targets: torch.Tensor,   # (K, 1+1+4+1+star)  — collate prepends batch_idx
    num_angles: int,
    angle_step: int,
    class_names: list[str] | None = None,
    conf_thresh: float = 0.5,
    max_images: int = 16,
    ncols: int = 4,
) -> np.ndarray:
    """
    Build a grid image showing every sample in the batch with full annotation.
    Returns a BGR numpy array suitable for cv2.imshow / cv2.imwrite.
    """
    B    = min(imgs.shape[0], max_images)
    H, W = imgs.shape[2], imgs.shape[3]

    panels = []

    for bi in range(B):
        # ── recover BGR uint8 image ───────────────────────────────────────────
        img_np = (imgs[bi].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        panel  = img_np[:, :, ::-1].copy()   # RGB → BGR

        # ── draw padding boundary ─────────────────────────────────────────────
        pad_x, pad_y, pad_x_r, pad_y_b = _detect_letterbox_pad(img_np)
        cv2.rectangle(panel,
                      (pad_x, pad_y),
                      (W - pad_x_r, H - pad_y_b),
                      (60, 60, 60), 1)

        # ── draw each object ──────────────────────────────────────────────────
        mask = targets[:, 0] == bi
        bt   = targets[mask].cpu().numpy()    # (M, 1+1+4+1+star)
        # cols after batch_idx: cls(1) cx(2) cy(3) w(4) h(5) dist(6) star(7..)
        # note: targets from collate have batch_idx prepended as col 0

        for obj_idx, row in enumerate(bt):
            # col offsets relative to the collated tensor
            cls_  = int(row[1 + COL_CLS])
            cx    = row[1 + COL_CX]
            cy    = row[1 + COL_CY]
            w_    = row[1 + COL_W]
            h_    = row[1 + COL_H]
            dist  = row[1 + COL_DIST]
            star  = row[1 + COL_STAR:]

            color = _col(obj_idx)

            # warn if annotation centre is outside image region
            cx_px = int(cx * W); cy_px = int(cy * H)
            in_image = (pad_x <= cx_px <= W - pad_x_r and
                        pad_y <= cy_px <= H - pad_y_b)
            if not in_image:
                cv2.putText(panel, "OOB", (cx_px - 10, cy_px),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # label string
            name     = class_names[cls_] if class_names else f"cls{cls_}"
            dist_str = f" {math.exp(dist):.1f}m" if dist > INVALID_DISTANCE + 1 else " no dist"
            label    = f"{name}{dist_str}"

            _draw_bbox(panel, cx, cy, w_, h_, color, label)
            _draw_star(panel, star, num_angles, angle_step, color, conf_thresh)

        # ── image index watermark ─────────────────────────────────────────────
        cv2.putText(panel, f"img {bi}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                    cv2.LINE_AA)

        panels.append(panel)

    # ── grid ─────────────────────────────────────────────────────────────────
    nrows = math.ceil(len(panels) / ncols)
    grid  = np.full((nrows * H, ncols * W, 3), 30, dtype=np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, ncols)
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = p

    # ── legend ────────────────────────────────────────────────────────────────
    legend_h = 28
    legend   = np.zeros((legend_h, grid.shape[1], 3), dtype=np.uint8)
    items = [
        (( 56,  56, 255), "bbox"),
        ((255, 255, 255), "poly ray"),
        (( 70,  70,  70), "inactive bin stub"),
        (( 60,  60,  60), "padding boundary"),
    ]
    x = 8
    for col, text in items:
        cv2.rectangle(legend, (x, 7), (x + 14, 21), col, -1)
        x += 18
        cv2.putText(legend, text, (x, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        x += int(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]) + 20

    return np.vstack([grid, legend])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Visualise dataloader output — exactly what the model sees",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--poly_dataset_root", required=True)
    ap.add_argument("--dist_dataset_root", required=True)
    ap.add_argument("--num_classes",  type=int, default=80)
    ap.add_argument("--angle_step",   type=int, default=15)
    ap.add_argument("--img_size",     type=int, default=640)
    ap.add_argument("--batch_size",   type=int, default=4)
    ap.add_argument("--num_workers",  type=int, default=0)
    ap.add_argument("--n_batches",    type=int, default=1,
                    help="Number of batches to visualise")
    ap.add_argument("--max_images",   type=int, default=16,
                    help="Max images shown per batch grid")
    ap.add_argument("--ncols",        type=int, default=4,
                    help="Grid columns")
    ap.add_argument("--conf_thresh",  type=float, default=0.5,
                    help="Polygon vertex confidence threshold")
    ap.add_argument("--min_distance", type=float, default=0.5)
    ap.add_argument("--max_distance", type=float, default=200.0)
    ap.add_argument("--class_names",  nargs="*", default=None,
                    help="Optional class name list")
    ap.add_argument("--split",        default="train",
                    choices=["train", "val", "test"],
                    help="Dataset split subfolder to load")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--aug",    action="store_true", default=False,
                      help="Enable augmentation (HSV, flip, mosaic)")
    mode.add_argument("--no_aug", action="store_true", default=False,
                      help="Disable all augmentation (default)")

    ap.add_argument("--show",     action="store_true",
                    help="Show interactive OpenCV window")
    ap.add_argument("--save_dir", default=None,
                    help="Directory to save PNG panels (skipped if not set)")
    args = ap.parse_args()

    augment  = args.aug
    split    = args.split
    num_angles = 360 // args.angle_step

    print(f"Loading '{split}' split  |  augment={augment}")

    loader = build_dataloader(
        poly_img_dir  = os.path.join(args.poly_dataset_root, f"images/{split}"),
        poly_lbl_dir  = os.path.join(args.poly_dataset_root, f"labels/{split}"),
        dist_img_dir  = os.path.join(args.dist_dataset_root, f"images/{split}"),
        dist_lbl_dir  = os.path.join(args.dist_dataset_root, f"labels/{split}"),
        img_size      = args.img_size,
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
        angle_step    = args.angle_step,
        min_dist      = args.min_distance,
        max_dist      = args.max_distance,
        augment       = augment,
        mosaic_prob   = 1.0 if augment else 0.0,
        flip_lr_prob  = 0.5 if augment else 0.0,
    )

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for batch_idx, (imgs, targets) in enumerate(loader):
        if batch_idx >= args.n_batches:
            break

        n_objs = targets.shape[0]
        print(f"\nBatch {batch_idx}  —  "
              f"{imgs.shape[0]} images, {n_objs} objects total")

        # ── per-object stats ──────────────────────────────────────────────────
        if n_objs > 0:
            # cols: batch_idx(0) cls(1) cx(2) cy(3) w(4) h(5) dist(6) star(7..)
            dist_col = targets[:, 1 + COL_DIST]
            valid_d  = dist_col[dist_col > INVALID_DISTANCE + 1]
            print(f"  classes   : {targets[:, 1 + COL_CLS].long().unique().tolist()}")
            print(f"  bbox cx   : [{targets[:, 1 + COL_CX].min():.3f}, "
                  f"{targets[:, 1 + COL_CX].max():.3f}]")
            print(f"  bbox cy   : [{targets[:, 1 + COL_CY].min():.3f}, "
                  f"{targets[:, 1 + COL_CY].max():.3f}]")
            print(f"  bbox w    : [{targets[:, 1 + COL_W].min():.3f}, "
                  f"{targets[:, 1 + COL_W].max():.3f}]")
            print(f"  bbox h    : [{targets[:, 1 + COL_H].min():.3f}, "
                  f"{targets[:, 1 + COL_H].max():.3f}]")
            print(f"  dist valid: {len(valid_d)}/{n_objs}  "
                  + (f"range [{math.exp(valid_d.min()):.1f}, "
                     f"{math.exp(valid_d.max()):.1f}]m" if len(valid_d) else ""))

            # star sanity: vertices start at col (1 + COL_STAR + 2) — skip origin xy
            star_verts = targets[:, 1 + COL_STAR + 2:]   # (N, num_angles*3)
            star_verts = star_verts.reshape(n_objs, num_angles, 3)
            conf_mask  = star_verts[:, :, 2] > 0
            if conf_mask.any():
                active_xy = star_verts[:, :, :2][conf_mask]
                oob = ((active_xy < 0) | (active_xy > 1)).any(dim=1).sum().item()
                if oob:
                    print(f"  ⚠  {oob} polygon vertices OUTSIDE [0,1] "
                          f"— letterbox mapping may still be off")
                else:
                    print(f"  ✓  all {conf_mask.sum().item()} active polygon "
                          f"vertices within [0, 1]")

        # ── render ────────────────────────────────────────────────────────────
        panel = render_batch_panel(
            imgs, targets,
            num_angles  = num_angles,
            angle_step  = args.angle_step,
            class_names = args.class_names,
            conf_thresh = args.conf_thresh,
            max_images  = args.max_images,
            ncols       = args.ncols,
        )

        # ── save ──────────────────────────────────────────────────────────────
        if args.save_dir:
            fname = os.path.join(args.save_dir, f"batch_{batch_idx:04d}.png")
            cv2.imwrite(fname, panel)
            print(f"  saved → {fname}")

        # ── show ──────────────────────────────────────────────────────────────
        if args.show:
            win = "Dataloader visualiser  [any key = next | q = quit]"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            # fit to screen (max 1400px wide)
            disp_w = min(panel.shape[1], 1400)
            disp_h = int(panel.shape[0] * disp_w / panel.shape[1])
            cv2.resizeWindow(win, disp_w, disp_h)
            cv2.imshow(win, panel)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):   # q or Esc
                print("Quit.")
                cv2.destroyAllWindows()
                return

    if args.show:
        cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()