"""
PyTorch Dataset for YOLOv8-Extended.

Two concrete datasets are merged at construction time:
    - PolyDataset       (V8ParserExtended, distance = INVALID)
    - PolyDistDataset   (V8DistanceParser)

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
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from data.parsers import INVALID_DISTANCE, V8DistanceParser, V8ParserExtended
from utils.star_polygon import flip_lr_star, scale_star, translate_star


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


def _resize_pad(img: np.ndarray, size: int) -> tuple[np.ndarray, float, float]:
    """Letterbox resize to (size, size). Returns (img, scale, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_y = (size - nh) // 2
    pad_x = (size - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img
    return canvas, scale, (pad_x, pad_y)


# ─────────────────────────────────────────────────────────────────────────────
# Target column layout helpers
# ─────────────────────────────────────────────────────────────────────────────
# Row: [cls(0), cx(1), cy(2), w(3), h(4), dist(5), ox(6), oy(7),
#       x0(8), y0(9), c0(10), ...,  xN-1, yN-1, cN-1]

COL_CLS   = 0
COL_CX    = 1
COL_CY    = 2
COL_W     = 3
COL_H     = 4
COL_DIST  = 5
COL_STAR  = 6   # star starts here


def _flip_targets_lr(targets: np.ndarray, num_angles: int, angle_step: int) -> np.ndarray:
    if targets.shape[0] == 0:
        return targets
    out = targets.copy()
    out[:, COL_CX] = 1.0 - targets[:, COL_CX]        # flip bbox cx
    for i in range(len(out)):
        star = out[i, COL_STAR:]
        out[i, COL_STAR:] = flip_lr_star(star, num_angles, angle_step)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Single-image base dataset
# ─────────────────────────────────────────────────────────────────────────────

class _BasePolyDataset(Dataset):
    """Abstract base; subclasses provide (parser, has_distance)."""

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        img_size: int = 640,
        angle_step: int = 15,
        min_dist: float = 0.5,
        max_dist: float = 200.0,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        flip_lr_prob: float = 0.5,
        mosaic_prob: float = 1.0,
        augment: bool = True,
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

    # ── subclass contract ─────────────────────────────────────────────────────
    def _make_parser(self):
        raise NotImplementedError

    # ── index ─────────────────────────────────────────────────────────────────
    def _build_index(self):
        parser = self._make_parser()
        self.label_map = parser.parse_dir(self.label_dir)
        self.stems = sorted(self.label_map.keys())
        # filter stems without an image file
        self.stems = [
            s for s in self.stems
            if _get_image_path(self.img_dir, s) is not None
        ]

    def __len__(self):
        return len(self.stems)

    # ── load one image + labels ───────────────────────────────────────────────
    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        stem  = self.stems[idx]
        img   = cv2.imread(_get_image_path(self.img_dir, stem))
        tgts  = self.label_map[stem].copy()   # (N, target_dim)
        return img, tgts

    # ── mosaic ────────────────────────────────────────────────────────────────
    def _mosaic(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        s  = self.img_size
        yc = random.randint(s // 4, 3 * s // 4)
        xc = random.randint(s // 4, 3 * s // 4)

        indices = [idx] + random.choices(range(len(self)), k=3)
        canvas  = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        all_tgts = []

        placements = [
            (s - yc, s - xc, s, s),          # top-left
            (s - yc, s, s, s + (s - xc)),     # top-right
            (s, s - xc, s + (s - yc), s),     # bottom-left
            (s, s, s + (s - yc), s + (s - xc)),
        ]

        for i, (r1, c1, r2, c2) in enumerate(placements):
            img_i, tgts_i = self._load(indices[i])
            if img_i is None:
                continue
            h_i, w_i = img_i.shape[:2]
            ph = r2 - r1
            pw = c2 - c1
            # scale to fit the slot
            scale_y = ph / h_i
            scale_x = pw / w_i
            img_i = cv2.resize(img_i, (pw, ph), interpolation=cv2.INTER_LINEAR)
            canvas[r1:r2, c1:c2] = img_i

            if tgts_i.shape[0] > 0:
                # adjust bbox: convert norm → absolute in 2s canvas, then back to norm
                tgts_c = tgts_i.copy()
                # bbox
                tgts_c[:, COL_CX] = (tgts_i[:, COL_CX] * pw + c1) / (s * 2)
                tgts_c[:, COL_CY] = (tgts_i[:, COL_CY] * ph + r1) / (s * 2)
                tgts_c[:, COL_W]  = tgts_i[:, COL_W]  * pw / (s * 2)
                tgts_c[:, COL_H]  = tgts_i[:, COL_H]  * ph / (s * 2)
                # star polygon
                for j in range(len(tgts_c)):
                    star = tgts_c[j, COL_STAR:].copy()
                    # origin
                    star[0] = (tgts_i[j, COL_STAR + 0] * pw + c1) / (s * 2)
                    star[1] = (tgts_i[j, COL_STAR + 1] * ph + r1) / (s * 2)
                    # vertices
                    verts = star[2:].reshape(self.num_angles, 3)
                    mask = verts[:, 2] > 0
                    verts[mask, 0] = (tgts_i[j, COL_STAR + 2:].reshape(
                        self.num_angles, 3)[mask, 0] * pw + c1) / (s * 2)
                    verts[mask, 1] = (tgts_i[j, COL_STAR + 2:].reshape(
                        self.num_angles, 3)[mask, 1] * ph + r1) / (s * 2)
                    star[2:] = verts.reshape(-1)
                    tgts_c[j, COL_STAR:] = star
                all_tgts.append(tgts_c)

        # crop from centre of 2s canvas
        img = canvas[yc:yc + s, xc:xc + s]
        if all_tgts:
            targets = np.concatenate(all_tgts, axis=0)
            # shift coords by (-xc/(2s), -yc/(2s)) and clip
            targets[:, COL_CX] = targets[:, COL_CX] - xc / (2 * s)
            targets[:, COL_CY] = targets[:, COL_CY] - yc / (2 * s)
            targets[:, COL_STAR + 0] -= xc / (2 * s)
            targets[:, COL_STAR + 1] -= yc / (2 * s)
            for j in range(len(targets)):
                verts = targets[j, COL_STAR + 2:].reshape(self.num_angles, 3)
                mask = verts[:, 2] > 0
                verts[mask, 0] -= xc / (2 * s)
                verts[mask, 1] -= yc / (2 * s)
                targets[j, COL_STAR + 2:] = verts.reshape(-1)
            # keep only objects fully or mostly inside crop (cx, cy > 0)
            keep = (targets[:, COL_CX] > 0) & (targets[:, COL_CY] > 0)
            targets = targets[keep]
            # scale back to [0, 1] (crop is s from a 2s canvas)
            targets[:, COL_CX] *= 2
            targets[:, COL_CY] *= 2
            targets[:, COL_W]  *= 2
            targets[:, COL_H]  *= 2
            targets[:, COL_STAR + 0] *= 2
            targets[:, COL_STAR + 1] *= 2
            for j in range(len(targets)):
                verts = targets[j, COL_STAR + 2:].reshape(self.num_angles, 3)
                mask = verts[:, 2] > 0
                verts[mask, 0] *= 2
                verts[mask, 1] *= 2
                targets[j, COL_STAR + 2:] = verts.reshape(-1)
            targets = np.clip(targets, 0, None)
            # clip bbox and star to [0, 1]
            targets[:, COL_CX:COL_CX + 4] = np.clip(targets[:, COL_CX:COL_CX + 4], 0, 1)
        else:
            targets = np.zeros((0, self.label_map[self.stems[0]].shape[1]),
                               dtype=np.float32)
        return img, targets

    # ── __getitem__ ───────────────────────────────────────────────────────────
    def __getitem__(self, idx: int):
        if self.augment and random.random() < self.mosaic_prob:
            img, targets = self._mosaic(idx)
        else:
            img, targets = self._load(idx)
            img, _, _ = _resize_pad(img, self.img_size)

        if self.augment:
            img = _augment_hsv(img, self.hsv_h, self.hsv_s, self.hsv_v)
            if random.random() < self.flip_lr_prob:
                img = img[:, ::-1].copy()
                targets = _flip_targets_lr(targets, self.num_angles, self.angle_step)

        # ── normalise image ───────────────────────────────────────────────────
        img = img[:, :, ::-1].transpose(2, 0, 1)   # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img)

        # ── targets ────────────────────────────────────────────────────────────
        if targets.shape[0] > 0:
            tgt_t = torch.from_numpy(targets)
        else:
            tgt_t = torch.zeros((0, targets.shape[1] if targets.ndim > 1
                                 else 6 + 2 + self.num_angles * 3), dtype=torch.float32)

        return img_t, tgt_t


# ─────────────────────────────────────────────────────────────────────────────
# Concrete datasets
# ─────────────────────────────────────────────────────────────────────────────

class PolyDataset(_BasePolyDataset):
    """Polygon-only dataset (no distance)."""
    def _make_parser(self):
        return V8ParserExtended(
            angle_step=self.angle_step,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
        )


class PolyDistDataset(_BasePolyDataset):
    """Polygon + distance dataset."""
    def _make_parser(self):
        return V8DistanceParser(
            angle_step=self.angle_step,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Collate & DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    # prepend batch index to each target row
    tgt_list = []
    for i, t in enumerate(targets):
        if t.shape[0] > 0:
            bi = torch.full((t.shape[0], 1), i, dtype=torch.float32)
            tgt_list.append(torch.cat([bi, t], dim=1))
    if tgt_list:
        targets_out = torch.cat(tgt_list, 0)
    else:
        tgt_dim = targets[0].shape[1] if targets[0].ndim > 1 else 1
        targets_out = torch.zeros((0, 1 + tgt_dim))
    return imgs, targets_out


def build_dataloader(
    poly_img_dir: str,
    poly_lbl_dir: str,
    dist_img_dir: str,
    dist_lbl_dir: str,
    img_size: int = 640,
    batch_size: int = 16,
    num_workers: int = 4,
    angle_step: int = 15,
    min_dist: float = 0.5,
    max_dist: float = 200.0,
    augment: bool = True,
    **aug_kwargs,
) -> DataLoader:
    """
    Build a merged DataLoader from both datasets.
    """
    ds_poly = PolyDataset(
        poly_img_dir, poly_lbl_dir, img_size=img_size,
        angle_step=angle_step, min_dist=min_dist, max_dist=max_dist,
        augment=augment, **aug_kwargs,
    )
    ds_dist = PolyDistDataset(
        dist_img_dir, dist_lbl_dir, img_size=img_size,
        angle_step=angle_step, min_dist=min_dist, max_dist=max_dist,
        augment=augment, **aug_kwargs,
    )
    merged = ConcatDataset([ds_poly, ds_dist])
    return DataLoader(
        merged,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=augment,
    )
