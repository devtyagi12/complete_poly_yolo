"""
Per-epoch visualisation for YOLOv8-Extended.

Panels saved each vis_interval epochs
──────────────────────────────────────
1. train_batch_grid   – augmented training images with GT boxes + polygons
2. val_pred_grid      – val images with GT (green) vs predicted (red) overlays
3. loss_curves        – matplotlib plot of all loss components over epochs
4. polygon_debug      – star polygon ray visualisation (one image, close-up)

All images are saved to {save_dir}/vis/epoch_{N:04d}/ as PNGs, and also
forwarded to the Logger (→ TensorBoard) as numpy arrays.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from postprocess.decode import Detection
from utils.star_polygon import star_to_vertices

# matplotlib is optional (loss curves only)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

_PAL = [
    (255,  56,  56), (255, 157,  51), ( 74, 180, 249), (149, 255,  65),
    (221, 111, 255), ( 63, 153,  30), (255,  85, 170), (  0, 165, 255),
    (255, 255,   0), (128,   0, 255),
]


def _col(cls: int):
    return _PAL[cls % len(_PAL)]


# ─────────────────────────────────────────────────────────────────────────────
# Drawing primitives
# ─────────────────────────────────────────────────────────────────────────────

def _draw_box(img: np.ndarray, x1, y1, x2, y2, color, label: str = "") -> None:
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img,
                      (int(x1), int(y1) - th - 4),
                      (int(x1) + tw, int(y1)), color, -1)
        cv2.putText(img, label, (int(x1), int(y1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)


def _draw_star_polygon(
    img: np.ndarray,
    star: np.ndarray,     # (2 + num_angles*3,)  normalised or pixel coords
    img_w: int,
    img_h: int,
    num_angles: int,
    color,
    conf_thresh: float = 0.5,
    normalised: bool = True,
) -> None:
    """Draw polygon vertices and rays from origin."""
    ox = star[0] * img_w if normalised else star[0]
    oy = star[1] * img_h if normalised else star[1]
    verts = star[2:].reshape(num_angles, 3)

    pts = []
    for dx, dy, conf in verts:
        if conf < conf_thresh:
            continue
        px = (dx * img_w if normalised else dx)
        py = (dy * img_h if normalised else dy)
        pts.append((int(px), int(py)))
        # ray from origin
        cv2.line(img, (int(ox), int(oy)), (int(px), int(py)), color, 1,
                 cv2.LINE_AA)
        cv2.circle(img, (int(px), int(py)), 3, color, -1)

    if len(pts) >= 2:
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [arr], isClosed=True, color=color, thickness=1,
                      lineType=cv2.LINE_AA)

    # origin dot
    cv2.circle(img, (int(ox), int(oy)), 4, (255, 255, 255), -1)
    cv2.circle(img, (int(ox), int(oy)), 4, color, 2)


def _make_grid(imgs: List[np.ndarray], ncols: int = 4) -> np.ndarray:
    """Tile a list of same-size BGR images into a grid."""
    if not imgs:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    h, w = imgs[0].shape[:2]
    nrows = math.ceil(len(imgs) / ncols)
    grid  = np.full((nrows * h, ncols * w, 3), 50, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, ncols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = im
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Main visualiser class
# ─────────────────────────────────────────────────────────────────────────────

class Visualiser:
    """
    Accumulates history and saves per-epoch debug panels.

    Parameters
    ----------
    save_dir        : experiment root (panels go to save_dir/vis/)
    num_angles      : polygon angle bins
    img_size        : inference resolution
    vis_max_images  : max images in each grid panel
    conf_thresh     : polygon vertex confidence threshold
    logger          : Logger instance (optional, forwards images to TB)
    """

    def __init__(
        self,
        save_dir: str | Path,
        num_angles: int,
        img_size: int = 640,
        vis_max_images: int = 8,
        conf_thresh: float = 0.5,
        logger=None,
    ):
        self.save_dir       = Path(save_dir) / "vis"
        self.num_angles     = num_angles
        self.img_size       = img_size
        self.vis_max_images = vis_max_images
        self.conf_thresh    = conf_thresh
        self.logger         = logger

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # history for loss curves
        self.loss_history: dict[str, list[float]] = {}
        self.val_history:  dict[str, list[float]] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def update_loss_history(
        self,
        epoch: int,
        loss: float,
        loss_dict: dict[str, float],
        val_f1: Optional[float] = None,
    ):
        self.loss_history.setdefault("total", []).append(loss)
        for k, v in loss_dict.items():
            self.loss_history.setdefault(k, []).append(v)
        if val_f1 is not None:
            self.val_history.setdefault("f1", []).append((epoch, val_f1))

    def save_train_batch(
        self,
        epoch: int,
        imgs: "torch.Tensor",      # (B, 3, H, W) float [0,1]
        targets: "torch.Tensor",   # (K, 1+1+4+1+star)
        class_names: List[str] | None = None,
    ) -> None:
        """Panel 1: GT-annotated training batch."""
        import torch
        imgs_np = (imgs[:self.vis_max_images].detach().cpu()
                   .permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        # RGB → BGR for cv2 drawing
        imgs_np = imgs_np[..., ::-1].copy()

        panels = []
        B = imgs_np.shape[0]
        for bi in range(B):
            im = imgs_np[bi].copy()
            H, W = im.shape[:2]
            mask = targets[:, 0] == bi
            bt   = targets[mask].cpu().numpy()

            for row in bt:
                cls  = int(row[1])
                cx   = row[2] * W; cy = row[3] * H
                w_   = row[4] * W; h_ = row[5] * H
                x1   = cx - w_/2; y1 = cy - h_/2
                x2   = cx + w_/2; y2 = cy + h_/2
                col  = _col(cls)
                lbl  = class_names[cls] if class_names else str(cls)
                _draw_box(im, x1, y1, x2, y2, col, lbl)

                star = row[7:]   # normalised
                _draw_star_polygon(im, star, W, H, self.num_angles, col,
                                   self.conf_thresh, normalised=True)

            # epoch watermark
            cv2.putText(im, f"ep{epoch}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            panels.append(im)

        grid = _make_grid(panels, ncols=min(4, B))
        self._save_and_log(grid, f"train_batch", epoch)

    def save_val_predictions(
        self,
        epoch: int,
        imgs: "torch.Tensor",              # (B, 3, H, W) float [0,1]
        targets: "torch.Tensor",           # (K, 1+1+4+1+star)
        detections: List[List[Detection]],
        class_names: List[str] | None = None,
    ) -> None:
        """Panel 2: GT (green) vs predicted (red) overlay."""
        imgs_np = (imgs[:self.vis_max_images].detach().cpu()
                   .permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        imgs_np = imgs_np[..., ::-1].copy()

        panels = []
        B = imgs_np.shape[0]
        for bi in range(B):
            im = imgs_np[bi].copy()
            H, W = im.shape[:2]

            # ── ground truth (green) ─────────────────────────────────────────
            mask = targets[:, 0] == bi
            for row in targets[mask].cpu().numpy():
                cls = int(row[1])
                cx  = row[2] * W; cy = row[3] * H
                w_  = row[4] * W; h_ = row[5] * H
                _draw_box(im, cx-w_/2, cy-h_/2, cx+w_/2, cy+h_/2,
                          (0, 200, 0), f"GT:{cls}")
                star = row[7:]
                _draw_star_polygon(im, star, W, H, self.num_angles,
                                   (0, 200, 0), self.conf_thresh, normalised=True)

            # ── predictions (red) ─────────────────────────────────────────────
            if bi < len(detections):
                for d in detections[bi]:
                    x1, y1, x2, y2 = d.bbox
                    lbl = (f"{class_names[d.cls] if class_names else d.cls}"
                           f" {d.score:.2f}")
                    _draw_box(im, x1, y1, x2, y2, (0, 0, 220), lbl)
                    if d.polygon is not None and len(d.polygon) > 0:
                        pts = d.polygon.astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(im, [pts], isClosed=True,
                                      color=(0, 0, 220), thickness=1)

            # distance label
            if bi < len(detections) and detections[bi]:
                for d in detections[bi]:
                    cx_ = int((d.bbox[0] + d.bbox[2]) / 2)
                    cv2.putText(im, f"{d.distance:.1f}m",
                                (cx_, int(d.bbox[1]) - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

            panels.append(im)

        grid = _make_grid(panels, ncols=min(4, B))
        self._save_and_log(grid, "val_pred", epoch)

    def save_loss_curves(self, epoch: int) -> None:
        """Panel 3: matplotlib loss curves (all components)."""
        if not _MPL or not self.loss_history:
            return

        keys    = list(self.loss_history.keys())
        n_plots = len(keys) + 1   # +1 for val F1
        ncols   = min(4, n_plots)
        nrows   = math.ceil(n_plots / ncols)

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 3.5 * nrows),
                                 squeeze=False)
        axes_flat = axes.flatten()

        for i, key in enumerate(keys):
            ax = axes_flat[i]
            vals = self.loss_history[key]
            ax.plot(range(len(vals)), vals, linewidth=1.2, color="#4db3ff")
            ax.set_title(key, fontsize=10)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.set_ylim(bottom=0)

        # val F1
        if self.val_history.get("f1"):
            ax = axes_flat[len(keys)]
            ep, f1s = zip(*self.val_history["f1"])
            ax.plot(ep, f1s, "o-", linewidth=1.2, color="#80ff80", markersize=4)
            ax.set_title("val F1", fontsize=10)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1)

        # hide unused axes
        for j in range(len(keys) + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(f"Loss curves  —  epoch {epoch}", fontsize=12)
        fig.tight_layout()

        # save
        ep_dir = self.save_dir / f"epoch_{epoch:04d}"
        ep_dir.mkdir(exist_ok=True)
        path = ep_dir / "loss_curves.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)

        # log to TB
        if self.logger and self.logger._tb:
            img = cv2.imread(str(path))
            if img is not None:
                self.logger.log_image("vis/loss_curves", img[:, :, ::-1], epoch)

    def save_polygon_debug(
        self,
        epoch: int,
        img_bgr: np.ndarray,
        star: np.ndarray,       # (2 + num_angles*3,)  in pixel coords
        angle_step: int,
    ) -> None:
        """
        Panel 4: close-up of star polygon rays on a single image.
        Shows each bin's ray, confidence bar, and angle annotation.
        """
        out = img_bgr.copy()
        H, W = out.shape[:2]
        ox = int(star[0]); oy = int(star[1])
        verts = star[2:].reshape(self.num_angles, 3)

        for i, (x, y, conf) in enumerate(verts):
            if conf < self.conf_thresh:
                # draw dim expected direction
                angle_deg = i * angle_step
                rad = math.radians(angle_deg)
                ex  = int(ox + 30 * math.cos(rad))
                ey  = int(oy + 30 * math.sin(rad))
                cv2.line(out, (ox, oy), (ex, ey), (60, 60, 60), 1)
                continue

            col   = _col(i)
            angle_deg = i * angle_step
            cv2.line(out, (ox, oy), (int(x), int(y)), col, 1, cv2.LINE_AA)
            cv2.circle(out, (int(x), int(y)), 4, col, -1)

            # angle label
            mid_x = int((ox + x) / 2); mid_y = int((oy + y) / 2)
            cv2.putText(out, f"{angle_deg}°", (mid_x + 2, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)

        # origin
        cv2.circle(out, (ox, oy), 6, (255, 255, 255), 2)

        self._save_and_log(out, "polygon_debug", epoch)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _save_and_log(self, img_bgr: np.ndarray, tag: str, epoch: int) -> None:
        ep_dir = self.save_dir / f"epoch_{epoch:04d}"
        ep_dir.mkdir(exist_ok=True)
        path = ep_dir / f"{tag}.png"
        cv2.imwrite(str(path), img_bgr)

        # forward to logger (RGB for TB)
        if self.logger:
            self.logger.log_image(f"vis/{tag}", img_bgr[:, :, ::-1], epoch)
