"""
Post-processing for YOLOv8-Extended.

Pipeline
────────
1. Decode box (DFL softmax + anchor offset)
2. Decode cls (sigmoid)
3. Decode polygon  (softplus dist, sigmoid angle/conf → polar → cartesian)
4. Decode scalar distance (exp + clip)
5. NMS on bbox
6. Scale polygon back to original image size + conf threshold
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_anchor_grid(preds: list, strides: List[int], device: torch.device):
    """(total_anchors, 2) normalised + (total_anchors,) stride."""
    aps, sts = [], []
    for (box_raw, *_), stride in zip(preds, strides):
        _, _, H, W = box_raw.shape
        xs = (torch.arange(W, device=device) + 0.5) / W
        ys = (torch.arange(H, device=device) + 0.5) / H
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        aps.append(torch.stack([gx, gy], -1).reshape(-1, 2))
        sts.append(torch.full((H * W,), stride, dtype=torch.float32, device=device))
    return torch.cat(aps), torch.cat(sts)


def _xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(*, 4) cx,cy,w,h → x1,y1,x2,y2."""
    out = boxes.clone()
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return out


def _nms(
    boxes: torch.Tensor,   # (N, 4) x1y1x2y2
    scores: torch.Tensor,  # (N,)
    iou_thres: float = 0.45,
) -> torch.Tensor:
    """Torchvision-free greedy NMS. Returns kept indices."""
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    order = scores.argsort(descending=True)
    keep  = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ix1  = x1[rest].clamp(min=x1[i])
        iy1  = y1[rest].clamp(min=y1[i])
        ix2  = x2[rest].clamp(max=x2[i])
        iy2  = y2[rest].clamp(max=y2[i])
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        iou   = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou <= iou_thres]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# ─────────────────────────────────────────────────────────────────────────────
# Box decoder
# ─────────────────────────────────────────────────────────────────────────────

def _decode_boxes(
    box_raw: torch.Tensor,    # (N, 4*REG_MAX)
    anchor_xy: torch.Tensor,  # (N, 2)  normalised
    stride: torch.Tensor,     # (N,)
    img_size: int,
    reg_max: int = 16,
) -> torch.Tensor:
    """
    Returns (N, 4) normalised x1y1x2y2.
    """
    proj = torch.arange(reg_max, dtype=torch.float32, device=box_raw.device)
    dist = (
        F.softmax(box_raw.reshape(-1, 4, reg_max), dim=-1) @ proj
    )   # (N, 4) in grid cells

    dist_px = dist * stride.unsqueeze(-1)     # in pixels
    ax_px = anchor_xy[:, 0] * img_size
    ay_px = anchor_xy[:, 1] * img_size

    x1 = (ax_px - dist_px[:, 0]) / img_size
    y1 = (ay_px - dist_px[:, 1]) / img_size
    x2 = (ax_px + dist_px[:, 2]) / img_size
    y2 = (ay_px + dist_px[:, 3]) / img_size

    return torch.stack([x1, y1, x2, y2], -1).clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Polygon decoder
# ─────────────────────────────────────────────────────────────────────────────

def _decode_polygon(
    poly_dist: torch.Tensor,   # (N, num_angles)  raw pre-softplus
    poly_angle: torch.Tensor,  # (N, num_angles)  raw logits
    poly_conf: torch.Tensor,   # (N, num_angles)  raw logits
    pred_bbox: torch.Tensor,   # (N, 4)  normalised x1y1x2y2
    stride: torch.Tensor,      # (N,)
    img_size: int,
    num_angles: int,
    conf_thresh: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    poly_xy   : (N, num_angles, 2)  normalised xy (0 if conf < thresh)
    poly_conf : (N, num_angles)     confidence in [0, 1]
    """
    # activate
    p_dist  = F.softplus(poly_dist)                      # (N, A) > 0
    p_angle = torch.sigmoid(poly_angle)                  # (N, A) ∈ (0,1)  fractional
    p_conf  = torch.sigmoid(poly_conf)                   # (N, A)

    # origin = centre of predicted bbox (normalised)
    origin_x = ((pred_bbox[:, 0] + pred_bbox[:, 2]) / 2)   # (N,)
    origin_y = ((pred_bbox[:, 1] + pred_bbox[:, 3]) / 2)

    # absolute angle for each bin
    offsets   = torch.arange(num_angles, dtype=torch.float32, device=poly_dist.device)
    # p_angle + offsets → absolute angle in [0, 360)
    abs_angle = (p_angle + offsets.unsqueeze(0)) / num_angles * 360   # (N, A) degrees

    # polar → cartesian delta
    rad    = torch.deg2rad(abs_angle)
    dx     = p_dist * torch.cos(rad)    # (N, A)
    dy     = p_dist * torch.sin(rad)

    # convert delta (in stride units) back to normalised image coords
    # delta is in the same unit as the polygon distance (stride units)
    dx_norm = dx * stride.unsqueeze(1) / img_size
    dy_norm = dy * stride.unsqueeze(1) / img_size

    # absolute polygon coords: origin - delta  (as specified)
    poly_x = origin_x.unsqueeze(1) - dx_norm    # (N, A)
    poly_y = origin_y.unsqueeze(1) - dy_norm

    poly_xy = torch.stack([poly_x, poly_y], -1).clamp(0, 1)   # (N, A, 2)

    return poly_xy, p_conf


# ─────────────────────────────────────────────────────────────────────────────
# Distance decoder
# ─────────────────────────────────────────────────────────────────────────────

def _decode_distance(
    dist_raw: torch.Tensor,   # (N, 1)
    min_dist: float,
    max_dist: float,
) -> torch.Tensor:
    """Returns (N,) clipped distance in metres."""
    return torch.exp(dist_raw.squeeze(-1)).clamp(min_dist, max_dist)


# ─────────────────────────────────────────────────────────────────────────────
# Detection result dataclass
# ─────────────────────────────────────────────────────────────────────────────

class Detection:
    """Single detection result."""
    __slots__ = ("bbox", "cls", "score", "distance", "polygon", "poly_conf")

    def __init__(
        self,
        bbox: np.ndarray,       # (4,)  x1y1x2y2  in original pixel coords
        cls: int,
        score: float,
        distance: float,        # metres (or exp(INVALID) if no distance)
        polygon: np.ndarray,    # (K, 2)  pixel coords, conf-filtered
        poly_conf: np.ndarray,  # (num_angles,)
    ):
        self.bbox      = bbox
        self.cls       = cls
        self.score     = score
        self.distance  = distance
        self.polygon   = polygon
        self.poly_conf = poly_conf

    def __repr__(self):
        return (f"Detection(cls={self.cls}, score={self.score:.3f}, "
                f"dist={self.distance:.2f}m, "
                f"bbox={np.round(self.bbox).astype(int).tolist()}, "
                f"poly_pts={len(self.polygon)})")


# ─────────────────────────────────────────────────────────────────────────────
# Main post-processor
# ─────────────────────────────────────────────────────────────────────────────

class PostProcessor:
    """
    Stateless post-processor for YOLOv8-Extended.

    Parameters
    ----------
    num_classes   : number of classes
    num_angles    : polygon angle bins
    strides       : FPN strides
    img_size      : square inference resolution
    conf_thres    : object confidence threshold
    iou_thres     : NMS IoU threshold
    poly_conf_thres: polygon vertex confidence threshold
    min_distance  : metres
    max_distance  : metres
    """

    REG_MAX = 16

    def __init__(
        self,
        num_classes: int,
        num_angles: int,
        strides: List[int] | None = None,
        img_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        poly_conf_thres: float = 0.5,
        min_distance: float = 0.5,
        max_distance: float = 200.0,
    ):
        self.nc              = num_classes
        self.num_angles      = num_angles
        self.strides         = strides or [8, 16, 32]
        self.img_size        = img_size
        self.conf_thres      = conf_thres
        self.iou_thres       = iou_thres
        self.poly_conf_thres = poly_conf_thres
        self.min_distance    = min_distance
        self.max_distance    = max_distance

    @torch.no_grad()
    def __call__(
        self,
        preds: list,
        orig_shapes: List[Tuple[int, int]],   # [(H, W), ...] per image
    ) -> List[List[Detection]]:
        """
        Parameters
        ----------
        preds       : model output list (batch)
        orig_shapes : original image dimensions before letterbox

        Returns
        -------
        List of detections per image in the batch.
        """
        device     = preds[0][0].device
        batch_size = preds[0][0].shape[0]

        anchor_pts, stride_tensor = _make_anchor_grid(preds, self.strides, device)

        # flatten per-scale outputs
        all_box_raw = torch.cat([
            p[0].permute(0,2,3,1).reshape(batch_size, -1, 4 * self.REG_MAX)
            for p in preds], 1)   # (B, A, 4*R)
        all_cls = torch.cat([
            p[1].permute(0,2,3,1).reshape(batch_size, -1, self.nc)
            for p in preds], 1)   # (B, A, nc)
        all_pc = torch.cat([
            p[2].permute(0,2,3,1).reshape(batch_size, -1, self.num_angles)
            for p in preds], 1)
        all_pa = torch.cat([
            p[3].permute(0,2,3,1).reshape(batch_size, -1, self.num_angles)
            for p in preds], 1)
        all_pd = torch.cat([
            p[4].permute(0,2,3,1).reshape(batch_size, -1, self.num_angles)
            for p in preds], 1)
        all_dist_raw = torch.cat([
            p[5].permute(0,2,3,1).reshape(batch_size, -1, 1)
            for p in preds], 1)   # (B, A, 1)

        results = []

        for bi in range(batch_size):
            orig_h, orig_w = orig_shapes[bi]
            scale_x = orig_w / self.img_size
            scale_y = orig_h / self.img_size

            # ── decode box ───────────────────────────────────────────────────
            boxes_norm = _decode_boxes(
                all_box_raw[bi], anchor_pts, stride_tensor,
                self.img_size, self.REG_MAX
            )   # (A, 4) normalised x1y1x2y2

            # ── class scores + object score ──────────────────────────────────
            cls_scores = torch.sigmoid(all_cls[bi])   # (A, nc)
            obj_scores, cls_ids = cls_scores.max(-1)  # (A,)

            # ── filter by confidence ─────────────────────────────────────────
            keep_conf = obj_scores >= self.conf_thres
            if not keep_conf.any():
                results.append([])
                continue

            boxes_f    = boxes_norm[keep_conf]
            scores_f   = obj_scores[keep_conf]
            cls_ids_f  = cls_ids[keep_conf]
            pc_f       = all_pc[bi][keep_conf]
            pa_f       = all_pa[bi][keep_conf]
            pd_f       = all_pd[bi][keep_conf]
            dist_raw_f = all_dist_raw[bi][keep_conf]
            anc_f      = anchor_pts[keep_conf]
            str_f      = stride_tensor[keep_conf]

            # ── NMS ──────────────────────────────────────────────────────────
            kept = _nms(boxes_f * self.img_size, scores_f, self.iou_thres)

            detections = []
            for ki in kept:
                ki = ki.item()
                box_n   = boxes_f[ki]                 # (4,) normalised
                score   = scores_f[ki].item()
                cls_id  = cls_ids_f[ki].item()

                # ── polygon ──────────────────────────────────────────────────
                poly_xy_n, pconf = _decode_polygon(
                    pd_f[ki:ki+1],
                    pa_f[ki:ki+1],
                    pc_f[ki:ki+1],
                    box_n.unsqueeze(0),
                    str_f[ki:ki+1],
                    self.img_size,
                    self.num_angles,
                    self.poly_conf_thres,
                )   # (1, A, 2), (1, A)
                poly_xy_n = poly_xy_n[0]    # (A, 2)
                pconf     = pconf[0]        # (A,)

                # scale to original image
                poly_xy_px = poly_xy_n.cpu().numpy() * np.array([[orig_w, orig_h]])
                pconf_np   = pconf.cpu().numpy()

                # conf threshold polygon vertices
                valid_mask = pconf_np >= self.poly_conf_thres
                polygon    = poly_xy_px[valid_mask]

                # ── scalar distance ───────────────────────────────────────────
                dist_metres = _decode_distance(
                    dist_raw_f[ki:ki+1], self.min_distance, self.max_distance
                )[0].item()

                # ── scale bbox to original image ─────────────────────────────
                box_px = np.array([
                    box_n[0].item() * orig_w,
                    box_n[1].item() * orig_h,
                    box_n[2].item() * orig_w,
                    box_n[3].item() * orig_h,
                ])

                detections.append(Detection(
                    bbox=box_px,
                    cls=cls_id,
                    score=score,
                    distance=dist_metres,
                    polygon=polygon,
                    poly_conf=pconf_np,
                ))

            results.append(detections)

        return results
