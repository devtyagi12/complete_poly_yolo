"""
PyTorch Dataset for YOLOv8-Extended.

Datasets
────────
- PolyDataset       : polygon-only labels (no distance)
- PolyDistDataset   : polygon + distance labels
- WeightedPolyDataset : wraps multiple PolyDataset instances with per-dataset
                        sampling weights / probabilities

Distance dataset is entirely optional.  build_dataloader() merges:
    1. One or more polygon datasets (required), each with an optional weight
    2. An optional distance dataset

Augmentations (no rotation, no vertical flip):
    • Mosaic 4-image
    • HSV colour jitter
    • Horizontal flip  (bbox + star polygon transformed in polar form)
"""
from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from data.parsers import INVALID_DISTANCE, V8DistanceParser, V8ParserExtended
from utils.star_polygon import flip_lr_star


# ─────────────────────────────────────────────────────────────────────────────
# Constants / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_image_path(img_dir: str, stem: str) -> Optional[str]:
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def _augment_hsv(img: np.ndarray, h: float, s: float, v: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-h * 180, h * 180)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(1 - s, 1 + s), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(1 - v, 1 + v), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _resize_pad(img: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Letterbox resize to (size, size). Returns (img, scale, (pad_x, pad_y))."""
    h, w  = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img    = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_y  = (size - nh) // 2
    pad_x  = (size - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img
    return canvas, scale, (pad_x, pad_y)


# ─────────────────────────────────────────────────────────────────────────────
# Target column layout
# ─────────────────────────────────────────────────────────────────────────────
# Row: [cls(0), cx(1), cy(2), w(3), h(4), dist(5), ox(6), oy(7),
#       x0(8), y0(9), c0(10), ...,  xN-1, yN-1, cN-1]

COL_CLS  = 0
COL_CX   = 1
COL_CY   = 2
COL_W    = 3
COL_H    = 4
COL_DIST = 5
COL_STAR = 6   # star starts here


def _adjust_targets_letterbox(
    targets: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    src_w: int,
    src_h: int,
    canvas_size: int,
    num_angles: int,
) -> np.ndarray:
    """
    Remap normalised targets from original-image space into
    letterboxed-canvas space.
    """
    if targets.shape[0] == 0:
        return targets

    out = targets.copy()
    s   = canvas_size

    def _rx(x):  return (x * src_w * scale + pad_x) / s
    def _ry(y):  return (y * src_h * scale + pad_y) / s
    def _rw(w):  return w * src_w * scale / s
    def _rh(h):  return h * src_h * scale / s

    out[:, COL_CX] = _rx(targets[:, COL_CX])
    out[:, COL_CY] = _ry(targets[:, COL_CY])
    out[:, COL_W]  = _rw(targets[:, COL_W])
    out[:, COL_H]  = _rh(targets[:, COL_H])

    out[:, COL_STAR + 0] = _rx(targets[:, COL_STAR + 0])
    out[:, COL_STAR + 1] = _ry(targets[:, COL_STAR + 1])

    verts_src = targets[:, COL_STAR + 2:].reshape(-1, num_angles, 3)
    verts_dst = out[:,    COL_STAR + 2:].reshape(-1, num_angles, 3)
    mask = verts_src[:, :, 2] > 0
    verts_dst[:, :, 0] = np.where(mask, _rx(verts_src[:, :, 0]), 0.0)
    verts_dst[:, :, 1] = np.where(mask, _ry(verts_src[:, :, 1]), 0.0)
    out[:, COL_STAR + 2:] = verts_dst.reshape(targets.shape[0], -1)

    return out


def _flip_targets_lr(targets: np.ndarray, num_angles: int, angle_step: int) -> np.ndarray:
    if targets.shape[0] == 0:
        return targets
    out = targets.copy()
    out[:, COL_CX] = 1.0 - targets[:, COL_CX]
    for i in range(len(out)):
        star = out[i, COL_STAR:]
        out[i, COL_STAR:] = flip_lr_star(star, num_angles, angle_step)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Base dataset
# ─────────────────────────────────────────────────────────────────────────────

class _BasePolyDataset(Dataset):
    """Abstract base; subclasses provide _make_parser()."""

    def __init__(
        self,
        img_dir:      str,
        label_dir:    str,
        img_size:     int   = 640,
        angle_step:   int   = 15,
        min_dist:     float = 0.5,
        max_dist:     float = 200.0,
        hsv_h:        float = 0.015,
        hsv_s:        float = 0.7,
        hsv_v:        float = 0.4,
        flip_lr_prob: float = 0.5,
        mosaic_prob:  float = 1.0,
        augment:      bool  = True,
    ):
        self.img_dir      = img_dir
        self.label_dir    = label_dir
        self.img_size     = img_size
        self.angle_step   = angle_step
        self.num_angles   = 360 // angle_step
        self.min_dist     = min_dist
        self.max_dist     = max_dist
        self.hsv_h        = hsv_h
        self.hsv_s        = hsv_s
        self.hsv_v        = hsv_v
        self.flip_lr_prob = flip_lr_prob
        self.mosaic_prob  = mosaic_prob
        self.augment      = augment
        self._build_index()

    def _make_parser(self):
        raise NotImplementedError

    def _build_index(self):
        parser         = self._make_parser()
        self.label_map = parser.parse_dir(self.label_dir)
        self.stems     = sorted(self.label_map.keys())
        self.stems     = [
            s for s in self.stems
            if _get_image_path(self.img_dir, s) is not None
        ]

    def __len__(self):
        return len(self.stems)

    def _load(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        stem = self.stems[idx]
        img  = cv2.imread(_get_image_path(self.img_dir, stem))
        tgts = self.label_map[stem].copy()
        return img, tgts

    def _mosaic(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        s  = self.img_size
        yc = random.randint(s // 4, 3 * s // 4)
        xc = random.randint(s // 4, 3 * s // 4)

        indices  = [idx] + random.choices(range(len(self)), k=3)
        canvas   = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        all_tgts = []

        placements = [
            (s - yc, s - xc, s,         s        ),
            (s - yc, s,      s,         s + (s - xc)),
            (s,      s - xc, s + (s - yc), s      ),
            (s,      s,      s + (s - yc), s + (s - xc)),
        ]

        for i, (r1, c1, r2, c2) in enumerate(placements):
            img_i, tgts_i = self._load(indices[i])
            if img_i is None:
                continue
            h_i, w_i = img_i.shape[:2]
            ph, pw   = r2 - r1, c2 - c1
            img_i    = cv2.resize(img_i, (pw, ph), interpolation=cv2.INTER_LINEAR)
            canvas[r1:r2, c1:c2] = img_i

            if tgts_i.shape[0] > 0:
                tgts_c = tgts_i.copy()
                tgts_c[:, COL_CX] = (tgts_i[:, COL_CX] * pw + c1) / (s * 2)
                tgts_c[:, COL_CY] = (tgts_i[:, COL_CY] * ph + r1) / (s * 2)
                tgts_c[:, COL_W]  =  tgts_i[:, COL_W]  * pw / (s * 2)
                tgts_c[:, COL_H]  =  tgts_i[:, COL_H]  * ph / (s * 2)

                for j in range(len(tgts_c)):
                    star        = tgts_c[j, COL_STAR:].copy()
                    star[0]     = (tgts_i[j, COL_STAR + 0] * pw + c1) / (s * 2)
                    star[1]     = (tgts_i[j, COL_STAR + 1] * ph + r1) / (s * 2)
                    verts       = star[2:].reshape(self.num_angles, 3)
                    orig_verts  = tgts_i[j, COL_STAR + 2:].reshape(self.num_angles, 3)
                    mask        = verts[:, 2] > 0
                    verts[mask, 0] = (orig_verts[mask, 0] * pw + c1) / (s * 2)
                    verts[mask, 1] = (orig_verts[mask, 1] * ph + r1) / (s * 2)
                    star[2:]    = verts.reshape(-1)
                    tgts_c[j, COL_STAR:] = star
                all_tgts.append(tgts_c)

        img = canvas[yc:yc + s, xc:xc + s]

        if all_tgts:
            targets = np.concatenate(all_tgts, axis=0)
            shift_x = xc / (2 * s)
            shift_y = yc / (2 * s)

            targets[:, COL_CX]     -= shift_x
            targets[:, COL_CY]     -= shift_y
            targets[:, COL_STAR+0] -= shift_x
            targets[:, COL_STAR+1] -= shift_y

            for j in range(len(targets)):
                verts        = targets[j, COL_STAR+2:].reshape(self.num_angles, 3)
                mask         = verts[:, 2] > 0
                verts[mask, 0] -= shift_x
                verts[mask, 1] -= shift_y
                targets[j, COL_STAR+2:] = verts.reshape(-1)

            keep = (targets[:, COL_CX] > 0) & (targets[:, COL_CY] > 0)
            targets = targets[keep]

            # scale from 2s-canvas-normalised → crop-normalised
            for col in (COL_CX, COL_CY, COL_W, COL_H):
                targets[:, col] *= 2
            targets[:, COL_STAR+0] *= 2
            targets[:, COL_STAR+1] *= 2
            for j in range(len(targets)):
                verts        = targets[j, COL_STAR+2:].reshape(self.num_angles, 3)
                mask         = verts[:, 2] > 0
                verts[mask, 0] *= 2
                verts[mask, 1] *= 2
                targets[j, COL_STAR+2:] = verts.reshape(-1)

            targets = np.clip(targets, 0, None)
            targets[:, COL_CX:COL_CX+4] = np.clip(
                targets[:, COL_CX:COL_CX+4], 0, 1
            )
        else:
            dim     = self.label_map[self.stems[0]].shape[1]
            targets = np.zeros((0, dim), dtype=np.float32)

        return img, targets

    def __getitem__(self, idx: int):
        if self.augment and random.random() < self.mosaic_prob:
            img, targets = self._mosaic(idx)
        else:
            img, targets = self._load(idx)
            src_h, src_w = img.shape[:2]
            img, scale, (pad_x, pad_y) = _resize_pad(img, self.img_size)
            targets = _adjust_targets_letterbox(
                targets, scale, pad_x, pad_y,
                src_w, src_h, self.img_size, self.num_angles,
            )

        if self.augment:
            img = _augment_hsv(img, self.hsv_h, self.hsv_s, self.hsv_v)
            if random.random() < self.flip_lr_prob:
                img     = img[:, ::-1].copy()
                targets = _flip_targets_lr(targets, self.num_angles, self.angle_step)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img)

        if targets.shape[0] > 0:
            tgt_t = torch.from_numpy(targets)
        else:
            dim   = (targets.shape[1] if targets.ndim > 1
                     else 6 + 2 + self.num_angles * 3)
            tgt_t = torch.zeros((0, dim), dtype=torch.float32)

        return img_t, tgt_t


# ─────────────────────────────────────────────────────────────────────────────
# Concrete datasets
# ─────────────────────────────────────────────────────────────────────────────

class PolyDataset(_BasePolyDataset):
    """Polygon-only dataset (distance column = INVALID_DISTANCE)."""
    def _make_parser(self):
        return V8ParserExtended(
            angle_step=self.angle_step,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
        )


class PolyDistDataset(_BasePolyDataset):
    """Polygon + metric distance dataset."""
    def _make_parser(self):
        return V8DistanceParser(
            angle_step=self.angle_step,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    tgt_list = []
    for i, t in enumerate(targets):
        if t.shape[0] > 0:
            bi = torch.full((t.shape[0], 1), i, dtype=torch.float32)
            tgt_list.append(torch.cat([bi, t], dim=1))
    if tgt_list:
        targets_out = torch.cat(tgt_list, 0)
    else:
        tgt_dim     = targets[0].shape[1] if targets[0].ndim > 1 else 1
        targets_out = torch.zeros((0, 1 + tgt_dim))
    return imgs, targets_out


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(
    # ── required: one or more polygon datasets ────────────────────────────────
    poly_datasets: List[Tuple[str, str]],
    # [(img_dir, lbl_dir), ...]  — required, at least one entry

    # ── optional: per-dataset sampling weights ────────────────────────────────
    poly_weights: Optional[List[float]] = None,
    # If provided, len must match poly_datasets.
    # Values are relative (they are normalised internally).
    # e.g. [2.0, 1.0] samples the first dataset twice as often.

    # ── optional: distance dataset ────────────────────────────────────────────
    dist_img_dir: Optional[str] = None,
    dist_lbl_dir: Optional[str] = None,
    # Distance dataset is included only when both are provided and the
    # directories exist.  Its weight is always 1.0.

    # ── common settings ───────────────────────────────────────────────────────
    img_size:    int   = 640,
    batch_size:  int   = 16,
    num_workers: int   = 4,
    angle_step:  int   = 15,
    min_dist:    float = 0.5,
    max_dist:    float = 200.0,
    augment:     bool  = True,
    **aug_kwargs,
) -> DataLoader:
    """
    Build a DataLoader from one or more polygon datasets plus an optional
    distance dataset.

    Parameters
    ----------
    poly_datasets : list of (img_dir, lbl_dir) pairs.  At least one required.
    poly_weights  : optional per-dataset sampling weights (uniform if None).
    dist_img_dir  : path to distance dataset images (optional).
    dist_lbl_dir  : path to distance dataset labels (optional).
    """
    if not poly_datasets:
        raise ValueError("At least one polygon dataset must be provided.")

    common = dict(
        img_size=img_size, angle_step=angle_step,
        min_dist=min_dist, max_dist=max_dist,
        augment=augment, **aug_kwargs,
    )

    # ── build polygon datasets ────────────────────────────────────────────────
    ds_list: List[Dataset] = []
    for img_dir, lbl_dir in poly_datasets:
        ds_list.append(PolyDataset(img_dir, lbl_dir, **common))

    # ── optional distance dataset ─────────────────────────────────────────────
    has_dist = (
        dist_img_dir is not None and
        dist_lbl_dir is not None and
        os.path.isdir(dist_img_dir) and
        os.path.isdir(dist_lbl_dir)
    )
    if has_dist:
        ds_list.append(PolyDistDataset(dist_img_dir, dist_lbl_dir, **common))

    # ── sampling weights ──────────────────────────────────────────────────────
    use_weighted = (poly_weights is not None and len(poly_weights) > 0)

    if use_weighted:
        if len(poly_weights) != len(poly_datasets):
            raise ValueError(
                f"poly_weights length ({len(poly_weights)}) must match "
                f"poly_datasets length ({len(poly_datasets)})."
            )
        # normalise weights
        w_poly = [float(w) for w in poly_weights]
        w_sum  = sum(w_poly)
        w_poly = [w / w_sum for w in w_poly]

        # distance dataset gets weight = average of poly weights (neutral)
        if has_dist:
            w_poly.append(1.0 / (len(w_poly) + 1))
            # re-normalise
            w_s    = sum(w_poly)
            w_poly = [w / w_s for w in w_poly]

        # build per-sample weights
        sample_weights: List[float] = []
        for ds, w in zip(ds_list, w_poly):
            n = len(ds)
            if n == 0:
                continue
            sample_weights.extend([w / n] * n)

        sampler = WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(sample_weights),
            replacement = True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = augment

    merged = ConcatDataset(ds_list)

    return DataLoader(
        merged,
        batch_size  = batch_size,
        shuffle     = shuffle,
        sampler     = sampler,
        num_workers = num_workers,
        pin_memory  = True,
        collate_fn  = collate_fn,
        drop_last   = augment,
    )
