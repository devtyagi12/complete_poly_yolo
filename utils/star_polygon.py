"""
Utility helpers for star-shaped polygon representation.

Star polygon format (per object):
    [origin_x, origin_y, x0, y0, conf0, ..., xN-1, yN-1, confN-1]
where N = num_angles = 360 // angle_step.

All coordinates are normalised (0-1) unless noted.
"""
from __future__ import annotations

import math
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Conversion: raw polygon vertices → star representation
# ─────────────────────────────────────────────────────────────────────────────

def polygon_to_star(
    vertices: np.ndarray,       # (V, 2)  normalised xy
    bbox_cx: float,
    bbox_cy: float,
    angle_step: int = 15,
) -> np.ndarray:
    """
    Convert a polygon to a star-shaped representation centred on the bbox centre.

    Returns
    -------
    star : np.ndarray  shape (2 + num_angles * 3,)
        [origin_x, origin_y, x0, y0, conf0, ..., xN-1, yN-1, confN-1]
    """
    num_angles = 360 // angle_step
    origin_x, origin_y = bbox_cx, bbox_cy

    # bins[i] = (best_dist², best_vertex) or None
    bins: list[tuple[float, np.ndarray] | None] = [None] * num_angles

    for vx, vy in vertices:
        dx = vx - origin_x
        dy = vy - origin_y
        dist2 = dx * dx + dy * dy
        angle_deg = math.degrees(math.atan2(dy, dx)) % 360
        idx = int(angle_deg / angle_step) % num_angles

        if bins[idx] is None or dist2 > bins[idx][0]:
            bins[idx] = (dist2, np.array([vx, vy], dtype=np.float32))

    parts: list[float] = [origin_x, origin_y]
    for b in bins:
        if b is None:
            parts += [0.0, 0.0, 0.0]
        else:
            parts += [b[1][0], b[1][1], 1.0]

    return np.array(parts, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers (operate on the star representation directly)
# ─────────────────────────────────────────────────────────────────────────────

def _unpack_star(star: np.ndarray, num_angles: int):
    """Return origin_xy, xy (N,2), conf (N,)."""
    origin = star[:2]
    data   = star[2:].reshape(num_angles, 3)
    xy     = data[:, :2]
    conf   = data[:, 2]
    return origin, xy, conf


def _pack_star(origin: np.ndarray, xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    N = xy.shape[0]
    out = np.empty(2 + N * 3, dtype=np.float32)
    out[:2] = origin
    out[2:].reshape(N, 3)[:, :2] = xy
    out[2:].reshape(N, 3)[:, 2]  = conf
    return out


def flip_lr_star(star: np.ndarray, num_angles: int, angle_step: int) -> np.ndarray:
    """
    Horizontal flip applied directly in polar form.

    Flipping x → (1 - x) mirrors across the vertical axis.
    In angle space: angle → (180 - angle) mod 360.
    We remap the bins accordingly and flip the x-coordinate of each vertex.
    """
    origin, xy, conf = _unpack_star(star, num_angles)

    new_origin = np.array([1.0 - origin[0], origin[1]], dtype=np.float32)

    # flip vertex x coordinates (conf-masked only)
    new_xy = xy.copy()
    mask = conf > 0
    new_xy[mask, 0] = 1.0 - xy[mask, 0]

    # remap bins: bin i → bin corresponding to (180 - angle) % 360
    new_xy_out  = np.zeros_like(new_xy)
    new_conf_out = np.zeros_like(conf)
    for i in range(num_angles):
        angle_deg = i * angle_step
        new_angle = (180 - angle_deg) % 360
        new_idx   = int(new_angle / angle_step) % num_angles
        new_xy_out[new_idx]   = new_xy[i]
        new_conf_out[new_idx] = conf[i]

    return _pack_star(new_origin, new_xy_out, new_conf_out)


def translate_star(
    star: np.ndarray,
    num_angles: int,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Translate a star polygon by (dx, dy) in normalised coordinates.
    Vertices and origin shift; angular bin mapping is unchanged.
    """
    origin, xy, conf = _unpack_star(star, num_angles)
    new_origin = origin + np.array([dx, dy], dtype=np.float32)
    mask = conf > 0
    new_xy = xy.copy()
    new_xy[mask] += np.array([dx, dy], dtype=np.float32)
    # clip to [0, 1]
    new_xy = np.clip(new_xy, 0.0, 1.0)
    new_origin = np.clip(new_origin, 0.0, 1.0)
    return _pack_star(new_origin, new_xy, new_conf=conf)


def scale_star(
    star: np.ndarray,
    num_angles: int,
    sx: float,
    sy: float,
    canvas_w: float = 1.0,
    canvas_h: float = 1.0,
) -> np.ndarray:
    """
    Scale a star polygon (used in mosaic assembly).
    sx, sy are scale factors; canvas_w/h normalise back to [0, 1].
    """
    origin, xy, conf = _unpack_star(star, num_angles)
    new_origin = origin * np.array([sx / canvas_w, sy / canvas_h])
    mask = conf > 0
    new_xy = xy.copy()
    new_xy[mask] *= np.array([sx / canvas_w, sy / canvas_h])
    new_xy = np.clip(new_xy, 0.0, 1.0)
    new_origin = np.clip(new_origin, 0.0, 1.0)
    return _pack_star(new_origin, new_xy, conf)


# ─────────────────────────────────────────────────────────────────────────────
# Star → cartesian vertices (for visualisation / post-processing)
# ─────────────────────────────────────────────────────────────────────────────

def star_to_vertices(
    star: np.ndarray,
    num_angles: int,
    conf_thresh: float = 0.5,
) -> np.ndarray:
    """
    Convert star representation back to a list of (x, y) vertices.
    Returns array of shape (K, 2) where K ≤ num_angles.
    """
    origin, xy, conf = _unpack_star(star, num_angles)
    mask = conf >= conf_thresh
    return xy[mask]
