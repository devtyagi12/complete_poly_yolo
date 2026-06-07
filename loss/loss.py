"""
Loss functions for YOLOv8-Extended.

Components
──────────
• Standard YOLOv8 loss (box via DFL+CIoU, class via BCE) with
  TaskAlignedAssigner.
• Polygon loss  (dist MSE + angle BCE + conf BCE), poly_gain=0.1
  — poly_dist_gain=2.0,  poly_conf_gain=0.2,  poly_angle_gain=0.5
• Distance loss (L1, dist_gain=0.1)

Total loss per batch:
    loss = box_loss * box_gain
         + cls_loss * cls_gain
         + dfl_loss * dfl_gain
         + poly_gain * (poly_dist_gain * poly_dist_loss
                       + poly_conf_gain * poly_conf_loss
                       + poly_angle_gain * poly_angle_loss)
         + dist_gain * distance_loss
    loss = loss * batch_size
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

INVALID_DISTANCE = -10.0


# ─────────────────────────────────────────────────────────────────────────────
# Task-Aligned Assigner (simplified, following YOLOv8 paper)
# ─────────────────────────────────────────────────────────────────────────────

def _make_anchors(preds: list, strides: List[int], offset: float = 0.5):
    """
    Build anchor centre grid for each FPN scale.

    Returns
    -------
    anchor_points : (total_anchors, 2)  normalised to [0,1]
    stride_tensor : (total_anchors,)
    """
    ap, st = [], []
    for (box_raw, *_), stride in zip(preds, strides):
        B, _, H, W = box_raw.shape
        xs = (torch.arange(W, device=box_raw.device) + offset) / W
        ys = (torch.arange(H, device=box_raw.device) + offset) / H
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        ap.append(torch.stack([gx, gy], -1).reshape(-1, 2))
        st.append(torch.full((H * W,), stride, dtype=torch.float32,
                             device=box_raw.device))
    return torch.cat(ap), torch.cat(st)


def _decode_box_raw(box_raw: torch.Tensor, anchor_xy: torch.Tensor,
                    stride: int, reg_max: int = 16) -> torch.Tensor:
    """
    DFL decode: (B, 4*reg_max, H, W) → (B*H*W, 4) in normalised coords.
    Returns x1y1x2y2.
    """
    B, _, H, W = box_raw.shape
    proj = torch.arange(reg_max, dtype=torch.float32, device=box_raw.device)
    raw  = box_raw.permute(0, 2, 3, 1).reshape(-1, 4, reg_max)
    dist = F.softmax(raw, dim=-1) @ proj            # (B*H*W, 4)  in grid cells
    dist = dist * stride                             # to pixels  (not used normalised here)
    # convert to normalised (divide by img size is handled in caller)
    ltrb = dist                                      # left,top,right,bottom offsets
    ax   = anchor_xy[:, 0].unsqueeze(1)
    ay   = anchor_xy[:, 1].unsqueeze(1)
    # anchor_xy normalised; convert to pixel equivalent via stride
    # For assignment we just need ordering, keep in stride units
    return ltrb                                      # (N_anchors, 4)  left/top/right/bottom


class TaskAlignedAssigner(nn.Module):
    """
    YOLOv8 Task-Aligned Assigner.
    Simplified: top-k selection by alignment metric = cls_score^alpha * iou^beta.
    """
    def __init__(self, topk: int = 13, alpha: float = 0.5, beta: float = 6.0):
        super().__init__()
        self.topk  = topk
        self.alpha = alpha
        self.beta  = beta

    @torch.no_grad()
    def forward(
        self,
        pred_cls: torch.Tensor,    # (N_anchors, nc)  sigmoid scores
        pred_box: torch.Tensor,    # (N_anchors, 4)   x1y1x2y2 in pixel units
        anchor_xy: torch.Tensor,   # (N_anchors, 2)   normalised
        gt_cls: torch.Tensor,      # (M,)             long
        gt_box: torch.Tensor,      # (M, 4)           cx,cy,w,h  normalised [0,1]
        stride: float = 1.0,
        img_size: int = 640,
    ):
        """
        Returns
        -------
        fg_mask      : (N_anchors,) bool
        assigned_gt  : (N_anchors,) long  index into GTs (-1 if bg)
        target_cls   : (N_anchors, nc)
        target_box   : (N_anchors, 4)  x1y1x2y2 pixel
        iou_weights  : (N_anchors,)
        """
        N = anchor_xy.shape[0]
        M = gt_box.shape[0]

        if M == 0:
            fg = anchor_xy.new_zeros(N, dtype=torch.bool)
            return (fg, fg.long() - 1,
                    anchor_xy.new_zeros(N, pred_cls.shape[-1]),
                    anchor_xy.new_zeros(N, 4),
                    anchor_xy.new_zeros(N))

        # convert gt from cxcywh norm → x1y1x2y2 pixel
        gt_cx  = gt_box[:, 0] * img_size
        gt_cy  = gt_box[:, 1] * img_size
        gt_w   = gt_box[:, 2] * img_size
        gt_h   = gt_box[:, 3] * img_size
        gt_x1  = gt_cx - gt_w / 2
        gt_y1  = gt_cy - gt_h / 2
        gt_x2  = gt_cx + gt_w / 2
        gt_y2  = gt_cy + gt_h / 2
        gt_xyxy = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], -1)   # (M, 4) pixel

        anc_px = anchor_xy * img_size   # (N, 2) pixel

        # anchor inside gt?
        ax = anc_px[:, 0].unsqueeze(0)   # (1, N)
        ay = anc_px[:, 1].unsqueeze(0)
        inside = (
            (ax > gt_xyxy[:, 0:1]) &
            (ax < gt_xyxy[:, 2:3]) &
            (ay > gt_xyxy[:, 1:2]) &
            (ay < gt_xyxy[:, 3:4])
        )                                           # (M, N)

        # IoU (M, N) between pred boxes and gt
        pb_x1 = anc_px[:, 0] - pred_box[:, 0]   # (N,) pixel x1
        pb_y1 = anc_px[:, 1] - pred_box[:, 1]
        pb_x2 = anc_px[:, 0] + pred_box[:, 2]
        pb_y2 = anc_px[:, 1] + pred_box[:, 3]
        pred_xyxy = torch.stack([pb_x1, pb_y1, pb_x2, pb_y2], -1)  # (N, 4)

        ix1 = torch.max(gt_xyxy[:, 0:1], pred_xyxy[:, 0].unsqueeze(0))
        iy1 = torch.max(gt_xyxy[:, 1:2], pred_xyxy[:, 1].unsqueeze(0))
        ix2 = torch.min(gt_xyxy[:, 2:3], pred_xyxy[:, 2].unsqueeze(0))
        iy2 = torch.min(gt_xyxy[:, 3:4], pred_xyxy[:, 3].unsqueeze(0))
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        area_g = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
        area_p = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
        iou = inter / (area_g.unsqueeze(1) + area_p.unsqueeze(0) - inter + 1e-7)  # (M, N)
        iou = iou.clamp(0)

        # cls score at target class
        cls_prob = pred_cls[:, gt_cls].T   # (M, N)

        align_metric = cls_prob.pow(self.alpha) * iou.pow(self.beta) * inside.float()

        # top-k per gt
        topk_vals, topk_idx = align_metric.topk(
            min(self.topk, N), dim=1, largest=True
        )
        candidate = torch.zeros_like(align_metric, dtype=torch.bool)
        candidate.scatter_(1, topk_idx, True)
        candidate &= inside

        # resolve conflicts: assign anchor to gt with highest metric
        max_metric, best_gt = (align_metric * candidate.float()).max(0)   # (N,)
        fg_mask = max_metric > 0

        assigned_gt = best_gt.clone()
        assigned_gt[~fg_mask] = -1

        target_cls = torch.zeros(N, pred_cls.shape[-1],
                                 device=pred_cls.device)
        target_cls[fg_mask, gt_cls[assigned_gt[fg_mask]]] = 1.0

        target_box = gt_xyxy[assigned_gt.clamp(0)]   # (N, 4); bg rows irrelevant

        iou_weights = iou[assigned_gt.clamp(0), torch.arange(N, device=iou.device)]
        iou_weights[~fg_mask] = 0.0

        return fg_mask, assigned_gt, target_cls, target_box, iou_weights


# ─────────────────────────────────────────────────────────────────────────────
# DFL + CIoU helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dfl_loss(pred: torch.Tensor, target: torch.Tensor, reg_max: int = 16):
    """pred: (N, 4*reg_max), target: (N, 4) in grid-cell units."""
    N = pred.shape[0]
    pred   = pred.reshape(-1, reg_max)
    tgt    = target.reshape(-1)
    tgt_l  = tgt.long().clamp(0, reg_max - 1)
    tgt_r  = (tgt_l + 1).clamp(0, reg_max - 1)
    w_r    = tgt - tgt_l.float()
    w_l    = 1.0 - w_r
    loss   = (F.cross_entropy(pred, tgt_l, reduction="none") * w_l +
              F.cross_entropy(pred, tgt_r, reduction="none") * w_r)
    return loss.reshape(N, 4).mean(-1)


def _ciou_loss(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor) -> torch.Tensor:
    """Element-wise CIoU loss. Both (N, 4) in any consistent unit."""
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    tx1, ty1, tx2, ty2 = target_xyxy.unbind(-1)

    # intersection
    ix1 = torch.max(px1, tx1); iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2); iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    ap = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    ag = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    union = ap + ag - inter + 1e-7
    iou   = inter / union

    # enclosing box
    ex1 = torch.min(px1, tx1); ey1 = torch.min(py1, ty1)
    ex2 = torch.max(px2, tx2); ey2 = torch.max(py2, ty2)
    c2  = (ex2 - ex1).pow(2) + (ey2 - ey1).pow(2) + 1e-7

    # centre distance
    pcx = (px1 + px2) / 2; pcy = (py1 + py2) / 2
    tcx = (tx1 + tx2) / 2; tcy = (ty1 + ty2) / 2
    d2  = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)

    # aspect ratio
    pw = (px2 - px1).clamp(0); ph = (py2 - py1).clamp(0)
    tw = (tx2 - tx1).clamp(0); th = (ty2 - ty1).clamp(0)
    v  = (4 / math.pi ** 2) * (
        torch.atan(tw / (th + 1e-7)) - torch.atan(pw / (ph + 1e-7))
    ).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    return 1 - iou + d2 / c2 + alpha * v


# ─────────────────────────────────────────────────────────────────────────────
# Main loss class
# ─────────────────────────────────────────────────────────────────────────────

class YOLOv8ExtendedLoss(nn.Module):
    REG_MAX = 16

    def __init__(
        self,
        num_classes: int,
        num_angles: int,
        angle_step: int = 15,
        strides: List[int] | None = None,
        img_size: int = 640,
        # gains
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
        poly_gain: float = 0.1,
        dist_gain: float = 0.1,
        poly_dist_gain: float = 2.0,
        poly_conf_gain: float = 0.2,
        poly_angle_gain: float = 0.5,
    ):
        super().__init__()
        self.nc          = num_classes
        self.num_angles  = num_angles
        self.angle_step  = angle_step
        self.strides     = strides or [8, 16, 32]
        self.img_size    = img_size

        self.box_gain       = box_gain
        self.cls_gain       = cls_gain
        self.dfl_gain       = dfl_gain
        self.poly_gain      = poly_gain
        self.dist_gain      = dist_gain
        self.poly_dist_gain = poly_dist_gain
        self.poly_conf_gain = poly_conf_gain
        self.poly_angle_gain = poly_angle_gain

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=13)

    # ── target preprocessing ─────────────────────────────────────────────────

    def _preprocess_poly(
        self,
        star: torch.Tensor,   # (M, 2 + num_angles * 3)  normalised
        stride: float,
    ) -> torch.Tensor:
        """
        Denorm xy by img_size, divide by stride, convert to delta from origin.
        Returns same shape with (ox, oy, dx0, dy0, c0, ...) in stride units.
        """
        out = star.clone()
        # origin
        out[:, 0] = star[:, 0] * self.img_size / stride
        out[:, 1] = star[:, 1] * self.img_size / stride
        # vertices
        ox = out[:, 0].unsqueeze(1)   # (M, 1)
        oy = out[:, 1].unsqueeze(1)
        verts = out[:, 2:].reshape(-1, self.num_angles, 3)   # (M, N, 3)
        verts[..., 0] = verts[..., 0] * self.img_size / stride - ox
        verts[..., 1] = verts[..., 1] * self.img_size / stride - oy
        out[:, 2:] = verts.reshape(out.shape[0], -1)
        return out

    # ── polygon loss ─────────────────────────────────────────────────────────

    def _poly_loss(
        self,
        pred_pc: torch.Tensor,   # (Nfg, num_angles)  raw logits
        pred_pa: torch.Tensor,   # (Nfg, num_angles)
        pred_pd: torch.Tensor,   # (Nfg, num_angles)  pre-softplus
        tgt_star: torch.Tensor,  # (Nfg, 2 + num_angles*3)  delta representation
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (dist_loss, angle_loss, conf_loss) each scalar.
        """
        Nfg = pred_pc.shape[0]
        if Nfg == 0:
            z = pred_pc.sum() * 0
            return z, z, z

        verts = tgt_star[:, 2:].reshape(Nfg, self.num_angles, 3)
        dx    = verts[..., 0]          # (Nfg, N)
        dy    = verts[..., 1]
        tconf = verts[..., 2]          # (Nfg, N)  0 or 1

        poly_mask = tconf > 0          # foreground vertex mask

        num_vertices = poly_mask.sum().clamp(min=1)

        # ── distance loss ─────────────────────────────────────────────────────
        tgt_dist = torch.sqrt(dx.pow(2) + dy.pow(2))            # (Nfg, N)
        pred_dist = F.softplus(pred_pd)                          # (Nfg, N) > 0
        dist_loss = (F.mse_loss(pred_dist, tgt_dist, reduction="none") * poly_mask)
        dist_loss = dist_loss.sum() / num_vertices / max(Nfg, 1)

        # ── angle loss ────────────────────────────────────────────────────────
        angle_rad = torch.atan2(dy, dx)
        angle_deg = torch.rad2deg(angle_rad) % 360               # (Nfg, N) [0, 360)
        idx       = (angle_deg / self.angle_step).floor().long()
        frac      = (angle_deg - idx * self.angle_step) / self.angle_step  # [0, 1)

        angle_loss = (
            self.bce(pred_pa, frac).clamp(0) * poly_mask
        )
        angle_loss = angle_loss.sum() / num_vertices / max(Nfg, 1)

        # ── confidence loss ───────────────────────────────────────────────────
        conf_loss = (self.bce(pred_pc, tconf) * poly_mask.float())
        # also include negative bins in conf
        conf_loss_all = self.bce(pred_pc, tconf)
        conf_loss = conf_loss_all.sum() / (self.num_angles * max(Nfg, 1))

        return dist_loss, angle_loss, conf_loss

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        preds: list,                   # from model.forward()
        targets: torch.Tensor,         # (K, 1+1+4+1+star)
        # col: batch_idx, cls, cx,cy,w,h, dist, *star
    ):
        """
        targets columns:
            0  : batch_idx
            1  : class
            2-5: bbox  cx,cy,w,h  normalised
            6  : distance (log or -10)
            7+ : star (2 + num_angles*3 floats)
        """
        device    = targets.device if targets.numel() else preds[0][0].device
        batch_size = preds[0][0].shape[0]

        # build anchor grid
        anchor_pts, stride_tensor = _make_anchors(preds, self.strides)
        anchor_pts   = anchor_pts.to(device)
        stride_tensor = stride_tensor.to(device)

        # flatten all scale predictions
        # box: (total, 4*REG_MAX), cls: (total, nc), pc/pa/pd: (total, num_angles)
        # dist: (total, 1)
        all_box = torch.cat([
            p[0].permute(0,2,3,1).reshape(batch_size, -1, 4 * self.REG_MAX)
            for p in preds], dim=1)   # (B, total, 4*REG_MAX)
        all_cls = torch.cat([
            p[1].permute(0,2,3,1).reshape(batch_size, -1, self.nc)
            for p in preds], dim=1)   # (B, total, nc)
        all_pc = torch.cat([
            p[2].permute(0,2,3,1).reshape(batch_size, -1, self.num_angles)
            for p in preds], dim=1)
        all_pa = torch.cat([
            p[3].permute(0,2,3,1).reshape(batch_size, -1, self.num_angles)
            for p in preds], dim=1)
        all_pd = torch.cat([
            p[4].permute(0,2,3,1).reshape(batch_size, -1, self.num_angles)
            for p in preds], dim=1)
        all_dist = torch.cat([
            p[5].permute(0,2,3,1).reshape(batch_size, -1, 1)
            for p in preds], dim=1)   # (B, total, 1)

        # decode box for assignment (in pixel space, per anchor)
        proj = torch.arange(self.REG_MAX, dtype=torch.float32, device=device)
        box_decode = (
            F.softmax(all_box.reshape(batch_size, -1, 4, self.REG_MAX), dim=-1)
            @ proj
        )   # (B, total, 4)  left/top/right/bottom in stride units

        # accumulate losses across batch
        total_box  = torch.tensor(0., device=device)
        total_cls  = torch.tensor(0., device=device)
        total_dfl  = torch.tensor(0., device=device)
        total_pd   = torch.tensor(0., device=device)
        total_pa   = torch.tensor(0., device=device)
        total_pc   = torch.tensor(0., device=device)
        total_sdist = torch.tensor(0., device=device)
        num_fg_total = 0

        for bi in range(batch_size):
            bt_mask = (targets[:, 0] == bi)
            bt      = targets[bt_mask]    # (M, ...)

            pred_cls_bi  = torch.sigmoid(all_cls[bi])    # (total, nc)
            pred_box_bi  = box_decode[bi]                 # (total, 4)  stride units
            pred_box_pix = pred_box_bi * stride_tensor.unsqueeze(1)  # pixel

            if bt.shape[0] == 0:
                total_cls += (self.bce(all_cls[bi],
                    torch.zeros_like(all_cls[bi]))).mean()
                continue

            gt_cls  = bt[:, 1].long()
            gt_bbox = bt[:, 2:6]              # normalised cxcywh
            gt_dist = bt[:, 6]                # log-dist or -10
            gt_star = bt[:, 7:]               # (M, 2+num_angles*3)

            fg_mask, assigned_gt, tgt_cls, tgt_box, iou_w = self.assigner(
                pred_cls_bi,
                pred_box_pix,
                anchor_pts,
                gt_cls,
                gt_bbox,
                img_size=self.img_size,
            )

            nfg = fg_mask.sum().item()
            num_fg_total += nfg

            # ── cls loss ─────────────────────────────────────────────────────
            total_cls += self.bce(all_cls[bi], tgt_cls).sum() / max(nfg, 1)

            if nfg == 0:
                continue

            fg_idx = fg_mask.nonzero(as_tuple=True)[0]
            asgn   = assigned_gt[fg_mask]

            # ── box loss (CIoU + DFL) ────────────────────────────────────────
            # strides for fg anchors
            fg_stride = stride_tensor[fg_idx]

            # target ltrb in stride units
            tgt_xyxy_pix = tgt_box[fg_idx]      # (Nfg, 4) pixel
            ax_pix = anchor_pts[fg_idx, 0] * self.img_size
            ay_pix = anchor_pts[fg_idx, 1] * self.img_size

            tgt_l = (ax_pix - tgt_xyxy_pix[:, 0]) / fg_stride
            tgt_t = (ay_pix - tgt_xyxy_pix[:, 1]) / fg_stride
            tgt_r = (tgt_xyxy_pix[:, 2] - ax_pix) / fg_stride
            tgt_b = (tgt_xyxy_pix[:, 3] - ay_pix) / fg_stride
            tgt_ltrb = torch.stack([tgt_l, tgt_t, tgt_r, tgt_b], -1).clamp(0, self.REG_MAX - 1 - 1e-3)

            pred_box_fg = all_box[bi][fg_idx]    # (Nfg, 4*REG_MAX)
            dfl = _dfl_loss(pred_box_fg, tgt_ltrb, self.REG_MAX)
            total_dfl += (dfl * iou_w[fg_idx]).sum() / max(nfg, 1)

            # CIoU
            pred_dec_fg = pred_box_pix[fg_idx]   # (Nfg, 4) ltrb pixel
            pred_xyxy = torch.stack([
                ax_pix - pred_dec_fg[:, 0],
                ay_pix - pred_dec_fg[:, 1],
                ax_pix + pred_dec_fg[:, 2],
                ay_pix + pred_dec_fg[:, 3],
            ], -1)
            ciou = _ciou_loss(pred_xyxy, tgt_xyxy_pix)
            total_box += (ciou * iou_w[fg_idx]).sum() / max(nfg, 1)

            # ── polygon loss ─────────────────────────────────────────────────
            # We need to find the stride for each fg anchor for poly preprocessing
            # Use the scale that produced each anchor
            fg_pc = all_pc[bi][fg_idx]    # (Nfg, num_angles)
            fg_pa = all_pa[bi][fg_idx]
            fg_pd = all_pd[bi][fg_idx]

            # preprocess star targets per fg anchor (per stride)
            # group by stride for efficiency
            tgt_star_asgn = gt_star[asgn]   # (Nfg, star_dim)
            tgt_star_proc = torch.zeros_like(tgt_star_asgn)
            for stride_val in self.strides:
                smask = (fg_stride == stride_val)
                if smask.any():
                    tgt_star_proc[smask] = torch.from_numpy(
                        self._preprocess_poly_numpy(
                            tgt_star_asgn[smask].cpu().numpy(), stride_val
                        )
                    ).to(device)

            pd_l, pa_l, pc_l = self._poly_loss(fg_pc, fg_pa, fg_pd, tgt_star_proc)
            total_pd += pd_l
            total_pa += pa_l
            total_pc += pc_l

            # ── distance loss ─────────────────────────────────────────────────
            fg_dist_pred = all_dist[bi][fg_idx].squeeze(-1)   # (Nfg,)
            fg_dist_tgt  = gt_dist[asgn]                      # (Nfg,)
            valid_dist   = fg_dist_tgt > (INVALID_DISTANCE + 1)   # != -10
            if valid_dist.any():
                l1 = F.l1_loss(
                    fg_dist_pred[valid_dist],
                    fg_dist_tgt[valid_dist],
                    reduction="sum",
                )
                total_sdist += l1 / max(valid_dist.sum().item(), 1)

        # ── combine ───────────────────────────────────────────────────────────
        poly_loss = (
            self.poly_dist_gain  * total_pd +
            self.poly_conf_gain  * total_pc +
            self.poly_angle_gain * total_pa
        )
        loss = (
            self.box_gain  * total_box +
            self.cls_gain  * total_cls +
            self.dfl_gain  * total_dfl +
            self.poly_gain * poly_loss +
            self.dist_gain * total_sdist
        ) * batch_size

        return loss, {
            "box":      (self.box_gain  * total_box).item(),
            "cls":      (self.cls_gain  * total_cls).item(),
            "dfl":      (self.dfl_gain  * total_dfl).item(),
            "poly_dist":(self.poly_dist_gain * total_pd).item(),
            "poly_conf":(self.poly_conf_gain * total_pc).item(),
            "poly_ang": (self.poly_angle_gain * total_pa).item(),
            "dist":     (self.dist_gain * total_sdist).item(),
        }

    # ── numpy helper (avoids re-importing star_polygon in loss) ──────────────
    def _preprocess_poly_numpy(self, star: "np.ndarray", stride: float) -> "np.ndarray":
        import numpy as np
        out = star.copy()
        out[:, 0] = star[:, 0] * self.img_size / stride
        out[:, 1] = star[:, 1] * self.img_size / stride
        ox = out[:, 0:1]
        oy = out[:, 1:2]
        N  = self.num_angles
        verts = out[:, 2:].reshape(-1, N, 3)
        orig_v = star[:, 2:].reshape(-1, N, 3)
        verts[..., 0] = orig_v[..., 0] * self.img_size / stride - ox
        verts[..., 1] = orig_v[..., 1] * self.img_size / stride - oy
        out[:, 2:] = verts.reshape(out.shape[0], -1)
        return out
