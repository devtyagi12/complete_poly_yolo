"""
Per-epoch visualisation for YOLOv8-Extended.

Panels saved each vis_interval epochs
──────────────────────────────────────
1. train_batch   – labelled grid: augmented images with GT boxes + polygons,
                   per-image stat bar (n_objects, has_distance flag)
2. val_compare   – side-by-side split: left half = GT, right half = prediction
                   with colour-coded class boxes, polygon rays, distance tags,
                   and a per-image confidence histogram strip
3. loss_curves   – dark-theme matplotlib grid: all loss components + val F1,
                   with epoch markers at each validation point
4. polygon_debug – annotated ray diagram: per-bin angle label, dist value,
                   inactive bins shown as dim stubs, active bins colour-coded
                   by confidence

All panels are saved to  {save_dir}/vis/epoch_{N:04d}/
and forwarded to TensorBoard when a Logger is attached.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Optional

import torch
import cv2
import numpy as np

from postprocess.decode import Detection

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL = True
except ImportError:
    _MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Palette  (BGR for OpenCV)
# ─────────────────────────────────────────────────────────────────────────────

_PAL = [
    ( 56,  56, 255), ( 51, 157, 255), (249, 180,  74), ( 65, 255, 149),
    (255, 111, 221), ( 30, 153,  63), (170,  85, 255), (255, 165,   0),
    (  0, 255, 255), (255,   0, 128), (200, 200,   0), (  0, 200, 200),
]

_GT_COLOR   = ( 50, 205,  50)   # lime green  — ground truth
_PRED_COLOR = ( 50,  50, 220)   # red-ish     — predictions
_FONT       = cv2.FONT_HERSHEY_SIMPLEX


def _col(idx: int):
    return _PAL[idx % len(_PAL)]


# ─────────────────────────────────────────────────────────────────────────────
# Low-level drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _label_box(
    img: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    color: tuple,
    text: str,
    thickness: int = 2,
    font_scale: float = 0.42,
) -> None:
    """Draw a filled-label bounding box."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if text:
        (tw, th), bl = cv2.getTextSize(text, _FONT, font_scale, 1)
        ty = max(y1 - 2, th + 2)
        cv2.rectangle(img, (x1, ty - th - bl - 1), (x1 + tw + 2, ty + 1),
                      color, -1)
        cv2.putText(img, text, (x1 + 1, ty - bl),
                    _FONT, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_polygon(
    img: np.ndarray,
    star: np.ndarray,       # (2 + num_angles*3,)  normalised [0,1]
    W: int, H: int,
    num_angles: int,
    color: tuple,
    conf_thresh: float = 0.5,
    draw_rays: bool = True,
    draw_stubs: bool = True,
) -> None:
    """Draw star polygon: rays, vertex dots, outline, origin ring."""
    ox = int(star[0] * W)
    oy = int(star[1] * H)
    verts = star[2:].reshape(num_angles, 3)
    active_pts = []

    for i, (vx, vy, vc) in enumerate(verts):
        if vc >= conf_thresh:
            px, py = int(vx * W), int(vy * H)
            active_pts.append((px, py))
            if draw_rays:
                cv2.line(img, (ox, oy), (px, py), color, 1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 3, color, -1, cv2.LINE_AA)
        elif draw_stubs:
            # dim directional stub
            rad = math.radians(i * (360 // num_angles))
            ex, ey = int(ox + 14 * math.cos(rad)), int(oy + 14 * math.sin(rad))
            cv2.line(img, (ox, oy), (ex, ey), (60, 60, 60), 1)

    if len(active_pts) >= 2:
        pts = np.array(active_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=color,
                      thickness=1, lineType=cv2.LINE_AA)

    cv2.circle(img, (ox, oy), 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (ox, oy), 5, color, 2, cv2.LINE_AA)


def _stat_bar(
    img: np.ndarray,
    n_obj: int,
    has_dist: bool,
    epoch: int,
    img_idx: int,
) -> np.ndarray:
    """
    Append a thin dark stat strip below the image showing
    object count, distance flag, and image index.
    """
    H, W = img.shape[:2]
    bar_h = 18
    bar = np.full((bar_h, W, 3), 28, dtype=np.uint8)
    txt = f"  img {img_idx}   objs:{n_obj}   {'dist:✓' if has_dist else 'dist:—'}   ep{epoch}"
    cv2.putText(bar, txt, (4, 13), _FONT, 0.36,
                (180, 180, 180), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def _make_grid(
    panels: List[np.ndarray],
    ncols: int = 4,
    gap: int = 2,
    bg: int = 28,
) -> np.ndarray:
    """Tile panels into a grid with a thin gap between cells."""
    if not panels:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    h, w = panels[0].shape[:2]
    nrows = math.ceil(len(panels) / ncols)
    gh = nrows * h + (nrows + 1) * gap
    gw = ncols * w + (ncols + 1) * gap
    grid = np.full((gh, gw, 3), bg, dtype=np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, ncols)
        y0 = gap + r * (h + gap)
        x0 = gap + c * (w + gap)
        grid[y0:y0 + h, x0:x0 + w] = p
    return grid


def _confidence_strip(scores: List[float], W: int, height: int = 14) -> np.ndarray:
    """
    Horizontal bar showing predicted confidence scores as coloured segments.
    Green = high, yellow = mid, red = low.
    """
    strip = np.full((height, W, 3), 20, dtype=np.uint8)
    if not scores:
        return strip
    bw = max(1, W // max(len(scores), 1))
    for i, s in enumerate(scores):
        g = int(s * 255); r = int((1 - s) * 255)
        color = (0, g, r)
        x0 = i * bw; x1 = min(x0 + bw, W)
        cv2.rectangle(strip, (x0, 1), (x1, height - 1), color, -1)
        cv2.putText(strip, f"{s:.2f}", (x0 + 1, height - 2),
                    _FONT, 0.28, (255, 255, 255), 1)
    return strip


# ─────────────────────────────────────────────────────────────────────────────
# Tensor → numpy helper
# ─────────────────────────────────────────────────────────────────────────────

def _to_bgr(imgs_tensor, max_n: int) -> np.ndarray:
    """(B,3,H,W) float32 → (N,H,W,3) uint8 BGR."""
    arr = (imgs_tensor[:max_n].detach().cpu()
           .permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    return arr[..., ::-1].copy()  # RGB→BGR


# ─────────────────────────────────────────────────────────────────────────────
# Visualiser
# ─────────────────────────────────────────────────────────────────────────────

class Visualiser:
    """
    Accumulates training history and saves per-epoch debug panels.

    Parameters
    ----------
    save_dir        : experiment root  (panels → save_dir/vis/)
    num_angles      : polygon angle bins
    angle_step      : degrees per bin
    img_size        : square canvas resolution
    vis_max_images  : max images per panel grid
    conf_thresh     : polygon vertex confidence threshold
    class_names     : optional list of class name strings
    logger          : Logger instance (forwards panels to TensorBoard)
    """

    def __init__(
        self,
        save_dir: str | Path,
        num_angles: int,
        angle_step: int = 15,
        img_size: int = 640,
        vis_max_images: int = 8,
        conf_thresh: float = 0.5,
        class_names: List[str] | None = None,
        logger=None,
    ):
        self.save_dir       = Path(save_dir) / "vis"
        self.num_angles     = num_angles
        self.angle_step     = angle_step
        self.img_size       = img_size
        self.vis_max_images = vis_max_images
        self.conf_thresh    = conf_thresh
        self.class_names    = class_names
        self.logger         = logger

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # loss / metric history
        self.loss_history: dict[str, list[float]] = {}
        self.val_epochs:   list[int]   = []
        self.val_f1s:      list[float] = []
        self.val_prec:     list[float] = []
        self.val_rec:      list[float] = []

    # ── history accumulation ──────────────────────────────────────────────────

    def update_loss_history(
        self,
        epoch: int,
        loss: float,
        loss_dict: dict[str, float],
        val_f1:   Optional[float] = None,
        val_prec: Optional[float] = None,
        val_rec:  Optional[float] = None,
    ) -> None:
        self.loss_history.setdefault("total", []).append(loss)
        for k, v in loss_dict.items():
            self.loss_history.setdefault(k, []).append(v)
        if val_f1 is not None:
            self.val_epochs.append(epoch)
            self.val_f1s.append(val_f1)
            self.val_prec.append(val_prec or 0.0)
            self.val_rec.append(val_rec  or 0.0)

    # ── Panel 1: training batch ───────────────────────────────────────────────

    def save_train_batch(
        self,
        epoch: int,
        imgs: "torch.Tensor",      # (B, 3, H, W)
        targets: "torch.Tensor",   # (K, 1+target_cols)  collated
    ) -> None:
        """
        Grid of training images annotated with GT boxes + polygon rays.
        Each cell has a stat bar showing object count and distance flag.
        """
        imgs_np = _to_bgr(imgs, self.vis_max_images)
        B = imgs_np.shape[0]
        panels = []

        for bi in range(B):
            im   = imgs_np[bi].copy()
            H, W = im.shape[:2]
            mask = targets[:, 0] == bi
            bt   = targets[mask].cpu().numpy()   # (M, 1+cols)

            has_dist = False
            for oi, row in enumerate(bt):
                cls  = int(row[1])
                cx, cy, w_, h_ = row[2]*W, row[3]*H, row[4]*W, row[5]*H
                dist = row[6]
                star = row[7:]

                has_dist = has_dist or (dist > -9.0)
                col  = _col(cls)
                name = self.class_names[cls] if self.class_names else f"cls{cls}"
                dist_str = f" {math.exp(dist):.1f}m" if dist > -9.0 else ""
                label = f"{name}{dist_str}"

                _label_box(im, cx - w_/2, cy - h_/2, cx + w_/2, cy + h_/2,
                           col, label)
                _draw_polygon(im, star, W, H, self.num_angles, col,
                              self.conf_thresh, draw_rays=True, draw_stubs=True)

            panel = _stat_bar(im, len(bt), has_dist, epoch, bi)
            panels.append(panel)

        title = self._title_bar(
            f"Train batch — epoch {epoch + 1}",
            panels[0].shape[1] * min(4, B) + 10 if panels else 640,
        )
        grid = _make_grid(panels, ncols=min(4, B))
        out  = np.vstack([title, grid])
        self._save_and_log(out, "train_batch", epoch)

    # ── Panel 2: validation comparison ───────────────────────────────────────

    def save_val_predictions(
        self,
        epoch: int,
        imgs: "torch.Tensor",
        targets: "torch.Tensor",
        detections: List[List[Detection]],
    ) -> None:
        """
        Side-by-side comparison for each val image:
          LEFT  half → ground truth  (lime green boxes + polygons)
          RIGHT half → model output  (colour per class, confidence strip)

        Panels are arranged in a 2×N grid where each row is one image.
        """
        imgs_np = _to_bgr(imgs, self.vis_max_images)
        B       = imgs_np.shape[0]
        rows    = []

        for bi in range(B):
            base = imgs_np[bi].copy()
            H, W = base.shape[:2]

            # ── left: ground truth ────────────────────────────────────────────
            gt_im = base.copy()
            mask  = targets[:, 0] == bi
            for row in targets[mask].cpu().numpy():
                cls  = int(row[1])
                cx, cy, w_, h_ = row[2]*W, row[3]*H, row[4]*W, row[5]*H
                dist = row[6]; star = row[7:]
                name = self.class_names[cls] if self.class_names else f"cls{cls}"
                dist_str = f" {math.exp(dist):.1f}m" if dist > -9.0 else ""
                _label_box(gt_im,
                           cx - w_/2, cy - h_/2, cx + w_/2, cy + h_/2,
                           _GT_COLOR, f"GT {name}{dist_str}")
                _draw_polygon(gt_im, star, W, H, self.num_angles,
                              _GT_COLOR, self.conf_thresh,
                              draw_rays=True, draw_stubs=False)

            _watermark(gt_im, "Ground Truth")

            # ── right: predictions ────────────────────────────────────────────
            pred_im = base.copy()
            scores  = []
            if bi < len(detections):
                for d in detections[bi]:
                    col  = _col(d.cls)
                    name = (self.class_names[d.cls]
                            if self.class_names else f"cls{d.cls}")
                    dist_str = f" {d.distance:.1f}m"
                    lbl  = f"{name} {d.score:.2f}{dist_str}"
                    x1, y1, x2, y2 = d.bbox
                    _label_box(pred_im, x1, y1, x2, y2, col, lbl)
                    if d.polygon is not None and len(d.polygon) >= 2:
                        pts = d.polygon.astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(pred_im, [pts], isClosed=True,
                                      color=col, thickness=1,
                                      lineType=cv2.LINE_AA)
                        for pt in d.polygon:
                            cv2.circle(pred_im, tuple(pt.astype(int)), 2, col, -1)
                    scores.append(d.score)

            _watermark(pred_im, "Prediction")

            # confidence strip under prediction panel
            conf_strip = _confidence_strip(scores, W)
            pred_with_strip = np.vstack([pred_im, conf_strip])
            # pad GT panel to same height
            pad_h = pred_with_strip.shape[0] - gt_im.shape[0]
            gt_padded = np.vstack([gt_im,
                                   np.full((pad_h, W, 3), 28, dtype=np.uint8)])

            # divider line
            divider = np.full((pred_with_strip.shape[0], 3, 3), 80,
                              dtype=np.uint8)

            row_img = np.hstack([gt_padded, divider, pred_with_strip])

            # image index label on the left edge
            idx_bar = np.full((row_img.shape[0], 22, 3), 20, dtype=np.uint8)
            cv2.putText(idx_bar, f"{bi}", (2, row_img.shape[0] // 2),
                        _FONT, 0.38, (150, 150, 150), 1)
            row_img = np.hstack([idx_bar, row_img])

            rows.append(row_img)

        if not rows:
            return

        # stack all image rows vertically
        max_w = max(r.shape[1] for r in rows)
        padded = []
        for r in rows:
            dw = max_w - r.shape[1]
            if dw > 0:
                r = np.hstack([r, np.full((r.shape[0], dw, 3), 28, dtype=np.uint8)])
            padded.append(r)

        title = self._title_bar(
            f"Validation — epoch {epoch + 1}   "
            f"[green = GT  |  coloured = prediction]",
            max_w,
        )
        out = np.vstack([title] + padded)
        self._save_and_log(out, "val_compare", epoch)

    # ── Panel 3: loss curves ──────────────────────────────────────────────────

    def save_loss_curves(self, epoch: int) -> None:
        """
        Dark-theme matplotlib grid of all loss components + val metrics.
        Vertical dashed lines mark validation epochs.
        """
        if not _MPL or not self.loss_history:
            return

        # order: total first, then component losses, then val metrics
        ordered_keys = ["total"] + [k for k in self.loss_history if k != "total"]
        n_loss  = len(ordered_keys)
        n_val   = 3   # F1, precision, recall
        n_plots = n_loss + n_val
        ncols   = min(4, n_plots)
        nrows   = math.ceil(n_plots / ncols)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5.5 * ncols, 3.2 * nrows),
            squeeze=False,
        )
        axes_flat = axes.flatten()
        colors = ["#4db3ff", "#ff7043", "#66bb6a", "#ffa726",
                  "#ab47bc", "#26c6da", "#ef5350", "#d4e157",
                  "#8d6e63", "#78909c"]

        def _vlines(ax):
            for ve in self.val_epochs:
                ax.axvline(x=ve, color="#555555", linewidth=0.6,
                           linestyle="--", alpha=0.7)

        for i, key in enumerate(ordered_keys):
            ax   = axes_flat[i]
            vals = self.loss_history[key]
            xs   = list(range(len(vals)))
            ax.plot(xs, vals, linewidth=1.4,
                    color=colors[i % len(colors)], alpha=0.9)
            # smoothed overlay (exponential moving average)
            if len(vals) > 5:
                ema = _ema_smooth(vals, alpha=0.3)
                ax.plot(xs, ema, linewidth=2.0,
                        color=colors[i % len(colors)])
            _vlines(ax)
            ax.set_title(key, fontsize=9, color="white")
            ax.set_xlabel("epoch", fontsize=8, color="#aaaaaa")
            ax.tick_params(colors="#888888", labelsize=7)
            ax.set_facecolor("#1a1a2e")
            ax.grid(alpha=0.2, color="#444444")
            ax.spines[:].set_color("#333333")
            ax.set_ylim(bottom=0)

        # val metric subplots
        val_series = [
            ("val F1",        self.val_f1s,   "#80ff80"),
            ("val Precision", self.val_prec,  "#80c0ff"),
            ("val Recall",    self.val_rec,   "#ffb080"),
        ]
        for j, (title, vals, col) in enumerate(val_series):
            idx = n_loss + j
            if idx >= len(axes_flat):
                break
            ax = axes_flat[idx]
            if self.val_epochs and vals:
                ax.plot(self.val_epochs, vals, "o-",
                        linewidth=1.4, color=col, markersize=4, alpha=0.9)
            _vlines(ax)
            ax.set_title(title, fontsize=9, color="white")
            ax.set_xlabel("epoch", fontsize=8, color="#aaaaaa")
            ax.tick_params(colors="#888888", labelsize=7)
            ax.set_facecolor("#1a1a2e")
            ax.grid(alpha=0.2, color="#444444")
            ax.spines[:].set_color("#333333")
            ax.set_ylim(0, 1)

        for j in range(n_plots, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.patch.set_facecolor("#0f0f1a")
        fig.suptitle(
            f"YOLOv8-Extended  —  training curves  (epoch {epoch + 1})",
            fontsize=11, color="white", y=1.01,
        )
        fig.tight_layout()

        ep_dir = self._ep_dir(epoch)
        path   = ep_dir / "loss_curves.png"
        fig.savefig(path, dpi=110, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        plt.close(fig)
        plt.style.use("default")

        if self.logger:
            img = cv2.imread(str(path))
            if img is not None:
                self.logger.log_image("vis/loss_curves", img[:, :, ::-1], epoch)

    # ── Panel 4: polygon debug ────────────────────────────────────────────────

    def save_polygon_debug(
        self,
        epoch: int,
        img_bgr: np.ndarray,
        star: np.ndarray,   # (2 + num_angles*3,)  pixel-space coords
    ) -> None:
        """
        Annotated ray diagram for one object:
          • Each active bin: coloured ray, vertex dot, angle° + dist label
          • Inactive bins: dim grey stub
          • Confidence shown as circle radius
        """
        out = img_bgr.copy()
        ox, oy = int(star[0]), int(star[1])
        verts  = star[2:].reshape(self.num_angles, 3)

        for i, (vx, vy, vc) in enumerate(verts):
            angle_deg = i * self.angle_step
            rad       = math.radians(angle_deg)

            if vc < self.conf_thresh:
                ex = int(ox + 22 * math.cos(rad))
                ey = int(oy + 22 * math.sin(rad))
                cv2.line(out, (ox, oy), (ex, ey), (55, 55, 55), 1)
                continue

            col  = _col(i)
            px, py = int(vx), int(vy)
            dist_px = math.hypot(px - ox, py - oy)

            # ray
            cv2.line(out, (ox, oy), (px, py), col, 1, cv2.LINE_AA)
            # vertex dot (radius proportional to conf)
            r = max(3, int(vc * 7))
            cv2.circle(out, (px, py), r, col, -1, cv2.LINE_AA)

            # label: angle + distance
            mid_x = int(ox + (px - ox) * 0.55)
            mid_y = int(oy + (py - oy) * 0.55)
            cv2.putText(out, f"{angle_deg}° {dist_px:.0f}px",
                        (mid_x + 2, mid_y),
                        _FONT, 0.28, col, 1, cv2.LINE_AA)

        # origin
        cv2.circle(out, (ox, oy), 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (ox, oy), 7, (100, 100, 100), 2, cv2.LINE_AA)

        self._save_and_log(out, "polygon_debug", epoch)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ep_dir(self, epoch: int) -> Path:
        d = self.save_dir / f"epoch_{epoch:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_and_log(
        self, img_bgr: np.ndarray, tag: str, epoch: int
    ) -> None:
        path = self._ep_dir(epoch) / f"{tag}.png"
        cv2.imwrite(str(path), img_bgr)
        if self.logger:
            self.logger.log_image(f"vis/{tag}", img_bgr[:, :, ::-1], epoch)

    @staticmethod
    def _title_bar(text: str, width: int, height: int = 28) -> np.ndarray:
        bar = np.full((height, max(width, 1), 3), 18, dtype=np.uint8)
        cv2.putText(bar, text, (8, height - 7),
                    _FONT, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
        return bar


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _watermark(img: np.ndarray, text: str) -> None:
    """Bottom-right corner watermark."""
    H, W = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.45, 1)
    cv2.putText(img, text, (W - tw - 6, H - 6),
                _FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def _ema_smooth(values: list, alpha: float = 0.3) -> list:
    """Exponential moving average for smooth curve overlay."""
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out