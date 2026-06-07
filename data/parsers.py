"""
Dataset parsers for YOLOv8-Extended.

Label file formats
──────────────────
Polygon-only  : <cls> <x1> <y1> <x2> <y2> ... <xN> <yN>
Polygon+dist  : <cls> <x1> <y1> <x2> <y2> ... <xN> <yN> <distance>

All xy are normalised YOLO floats (0-1).

Parsed target tensor columns (per object row):
    0          : class label
    1-4        : bbox  [cx, cy, w, h]  (derived from polygon vertices)
    5          : distance  (log-clipped or INVALID_DISTANCE=-10)
    6 .. end   : star polygon  [ox, oy, x0,y0,c0, ..., xN-1,yN-1,cN-1]
                 length = 2 + num_angles * 3

Disk cache
──────────
parse_dir() saves a single .npy file alongside the label directory:
    <label_dir>/.cache_as{angle_step}_d{min_dist}_{max_dist}.npy

The cache stores a structured array with columns:
    [stem_hash (uint64), row_start (int64), row_end (int64)]
plus all target rows concatenated in a second array.

Cache is invalidated automatically when any .txt file's mtime changes,
the angle_step changes, or the distance bounds change.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from utils.star_polygon import polygon_to_star

INVALID_DISTANCE = -10.0


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _vertices_to_bbox(vertices: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1 = vertices[:, 0].min(), vertices[:, 1].min()
    x2, y2 = vertices[:, 0].max(), vertices[:, 1].max()
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def _clip_log_distance(dist: float, min_dist: float, max_dist: float) -> float:
    return math.log(max(min(dist, max_dist), min_dist))


def _parse_line(
    line: str,
    has_distance: bool,
    angle_step: int,
    min_dist: float,
    max_dist: float,
) -> Optional[np.ndarray]:
    tokens = line.strip().split()
    if len(tokens) < 7:
        return None
    try:
        cls    = int(tokens[0])
        coords = [float(t) for t in tokens[1:]]
    except ValueError:
        return None

    if has_distance:
        distance_raw = coords[-1]
        coords = coords[:-1]
    else:
        distance_raw = None

    if len(coords) < 6 or len(coords) % 2 != 0:
        return None

    vertices     = np.array(coords, dtype=np.float32).reshape(-1, 2)
    cx, cy, w, h = _vertices_to_bbox(vertices)

    dist_val = (
        _clip_log_distance(distance_raw, min_dist, max_dist)
        if (distance_raw is not None and distance_raw > 0)
        else INVALID_DISTANCE
    )

    star = polygon_to_star(vertices, cx, cy, angle_step=angle_step)
    return np.concatenate(
        [np.array([cls, cx, cy, w, h, dist_val], dtype=np.float32), star]
    )


def _parse_label_file(
    path: str,
    has_distance: bool,
    angle_step: int,
    min_dist: float,
    max_dist: float,
) -> np.ndarray:
    num_angles = 360 // angle_step
    target_dim = 6 + 2 + num_angles * 3
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = _parse_line(line, has_distance, angle_step, min_dist, max_dist)
            if row is not None:
                rows.append(row)
    if rows:
        return np.stack(rows).astype(np.float32)
    return np.zeros((0, target_dim), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Disk cache
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(label_dir: str, angle_step: int,
                min_dist: float, max_dist: float) -> Path:
    """Deterministic cache filename encoding the parser parameters."""
    tag = f"as{angle_step}_d{min_dist}_{max_dist}"
    return Path(label_dir) / f".cache_{tag}.npz"


def _dir_mtime_hash(label_dir: str) -> str:
    """
    A fast fingerprint of all .txt files in label_dir:
    sha1 of sorted (filename, mtime, size) tuples.
    """
    entries = []
    for p in sorted(Path(label_dir).glob("*.txt")):
        st = p.stat()
        entries.append(f"{p.name}:{st.st_mtime_ns}:{st.st_size}")
    return hashlib.sha1("\n".join(entries).encode()).hexdigest()


def _load_cache(
    cache_file: Path,
    label_dir: str,
) -> Optional[dict[str, np.ndarray]]:
    """
    Return the cached label_map if it is valid, else None.
    The cache stores:
        meta["mtime_hash"]  – sha1 of directory state at write time
        data["stems"]       – array of stem strings
        data["rows"]        – concatenated target rows  (R, target_dim)
        data["offsets"]     – (N+1,) int64 row-start offsets per stem
    """
    if not cache_file.exists():
        return None
    try:
        data      = np.load(cache_file, allow_pickle=True)
        saved_hash = str(data["mtime_hash"])
        current   = _dir_mtime_hash(label_dir)
        if saved_hash != current:
            return None   # stale — files changed

        stems   = data["stems"].tolist()          # list[str]
        rows    = data["rows"]                    # (R, D)
        offsets = data["offsets"]                 # (N+1,)

        label_map = {}
        for i, stem in enumerate(stems):
            r0, r1 = int(offsets[i]), int(offsets[i + 1])
            label_map[stem] = rows[r0:r1]
        return label_map

    except Exception:
        return None   # corrupt cache — will rebuild


def _save_cache(
    cache_file: Path,
    label_dir: str,
    label_map: dict[str, np.ndarray],
) -> None:
    stems   = sorted(label_map.keys())
    arrays  = [label_map[s] for s in stems]
    offsets = np.zeros(len(stems) + 1, dtype=np.int64)
    for i, a in enumerate(arrays):
        offsets[i + 1] = offsets[i] + len(a)

    # concatenate — handle empty arrays (0-row)
    if any(len(a) > 0 for a in arrays):
        rows = np.concatenate([a for a in arrays if len(a) > 0])
        # recompute offsets over non-empty slices only if all empty share dim
        # simpler: keep per-stem even if empty by using a dummy row shape
        target_dim = next(a.shape[1] for a in arrays if len(a) > 0)
    else:
        target_dim = arrays[0].shape[1] if arrays else 1
        rows = np.zeros((0, target_dim), dtype=np.float32)

    # rebuild offsets including empty stems correctly
    rows_list: list[np.ndarray] = []
    offsets = np.zeros(len(stems) + 1, dtype=np.int64)
    for i, a in enumerate(arrays):
        offsets[i + 1] = offsets[i] + len(a)
        if len(a) > 0:
            rows_list.append(a)
    rows = np.concatenate(rows_list) if rows_list else np.zeros(
        (0, target_dim), dtype=np.float32
    )

    np.savez(
        cache_file,
        mtime_hash=np.array(_dir_mtime_hash(label_dir)),
        stems=np.array(stems),
        rows=rows,
        offsets=offsets,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core parse_dir with caching
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dir_cached(
    label_dir: str,
    has_distance: bool,
    angle_step: int,
    min_dist: float,
    max_dist: float,
    use_cache: bool = True,
) -> dict[str, np.ndarray]:
    cache_file = _cache_path(label_dir, angle_step, min_dist, max_dist)

    if use_cache:
        cached = _load_cache(cache_file, label_dir)
        if cached is not None:
            return cached

    # cache miss — parse from scratch
    label_map: dict[str, np.ndarray] = {}
    for p in Path(label_dir).glob("*.txt"):
        label_map[p.stem] = _parse_label_file(
            str(p), has_distance, angle_step, min_dist, max_dist
        )

    if use_cache and label_map:
        try:
            _save_cache(cache_file, label_dir, label_map)
        except OSError:
            pass   # read-only filesystem — silently skip

    return label_map


# ─────────────────────────────────────────────────────────────────────────────
# Public parsers
# ─────────────────────────────────────────────────────────────────────────────

class V8ParserExtended:
    """
    Parser for the polygon-only dataset.
    Distance column is filled with INVALID_DISTANCE (-10.0).
    """

    def __init__(
        self,
        angle_step: int = 15,
        min_dist: float = 0.5,
        max_dist: float = 200.0,
        use_cache: bool = True,
    ):
        self.angle_step = angle_step
        self.min_dist   = min_dist
        self.max_dist   = max_dist
        self.use_cache  = use_cache

    def parse_file(self, label_path: str) -> np.ndarray:
        return _parse_label_file(
            label_path, False, self.angle_step, self.min_dist, self.max_dist
        )

    def parse_dir(self, label_dir: str) -> dict[str, np.ndarray]:
        return _parse_dir_cached(
            label_dir, False,
            self.angle_step, self.min_dist, self.max_dist,
            self.use_cache,
        )


class V8DistanceParser:
    """Parser for the polygon + distance dataset."""

    def __init__(
        self,
        angle_step: int = 15,
        min_dist: float = 0.5,
        max_dist: float = 200.0,
        use_cache: bool = True,
    ):
        self.angle_step = angle_step
        self.min_dist   = min_dist
        self.max_dist   = max_dist
        self.use_cache  = use_cache

    def parse_file(self, label_path: str) -> np.ndarray:
        return _parse_label_file(
            label_path, True, self.angle_step, self.min_dist, self.max_dist
        )

    def parse_dir(self, label_dir: str) -> dict[str, np.ndarray]:
        return _parse_dir_cached(
            label_dir, True,
            self.angle_step, self.min_dist, self.max_dist,
            self.use_cache,
        )