"""
YOLOv8-Extended model.

Architecture additions over standard YOLOv8s:
  • Polygon branch  (3 independent Conv1x1 heads, per FPN scale)
      - poly_conf  : (B, num_angles, H, W)
      - poly_angle : (B, num_angles, H, W)
      - poly_dist  : (B, num_angles, H, W)
      Input: penultimate box feature map (before the box-head's final 1×1).

  • Distance head  (num_dist_blocks ConvBN + 1×1, per FPN scale)
      Input: same as box/cls head.
      Output: (B, 1, H, W)

The backbone/neck mirrors YOLOv8s channel widths; swap in the real
`ultralytics` backbone if available.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation factory
# ─────────────────────────────────────────────────────────────────────────────

# GroupNorm group count — 32 is standard; falls back to out_ch if out_ch < 32.
_GN_GROUPS = 32

def _make_norm(out_ch: int, use_gn: bool) -> nn.Module:
    """Return GroupNorm (AMD) or BatchNorm2d (NVIDIA) for out_ch channels."""
    if use_gn:
        groups = min(_GN_GROUPS, out_ch)
        # out_ch must be divisible by groups; reduce groups until it is
        while out_ch % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, out_ch, eps=1e-3)
    return nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03)


# ─────────────────────────────────────────────────────────────────────────────
# Basic building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 1,
        s: int = 1,
        p: int = 0,
        act: bool = True,
        use_gn: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = _make_norm(out_ch, use_gn)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, ch: int, shortcut: bool = True, e: float = 0.5,
                 use_gn: bool = False):
        super().__init__()
        hid = int(ch * e)
        self.cv1 = ConvBN(ch, hid, 3, 1, 1, use_gn=use_gn)
        self.cv2 = ConvBN(hid, ch, 3, 1, 1, use_gn=use_gn)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP-style fused bottleneck (YOLOv8 C2f)."""
    def __init__(self, in_ch: int, out_ch: int, n: int = 1, shortcut: bool = True,
                 use_gn: bool = False):
        super().__init__()
        self.hid  = out_ch // 2
        self.cv1  = ConvBN(in_ch, out_ch, 1, use_gn=use_gn)
        self.cv2  = ConvBN((2 + n) * self.hid, out_ch, 1, use_gn=use_gn)
        self.bots = nn.ModuleList(
            [Bottleneck(self.hid, shortcut, use_gn=use_gn) for _ in range(n)]
        )

    def forward(self, x):
        y  = list(self.cv1(x).split(self.hid, 1))
        y += [b(y[-1]) for b in self.bots]
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 5, use_gn: bool = False):
        super().__init__()
        hid      = in_ch // 2
        self.cv1 = ConvBN(in_ch, hid, 1, use_gn=use_gn)
        self.cv2 = ConvBN(hid * 4, out_ch, 1, use_gn=use_gn)
        self.mp  = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x  = self.cv1(x)
        y1 = self.mp(x)
        y2 = self.mp(y1)
        return self.cv2(torch.cat([x, y1, y2, self.mp(y2)], 1))


# ─────────────────────────────────────────────────────────────────────────────
# Backbone (YOLOv8s channel widths: d=0.33, w=0.50)
# ─────────────────────────────────────────────────────────────────────────────

class YOLOv8Backbone(nn.Module):
    """
    Produces three FPN feature maps: P3 (stride 8), P4 (16), P5 (32).
    Channels: [128, 256, 512] for size 's'.
    """
    def __init__(self, depth: int = 1, width_mult: float = 0.5,
                 use_gn: bool = False):
        super().__init__()
        def ch(c):
            return max(round(c * width_mult), 1)

        def reps(n):
            return max(round(n * depth), 1)

        # stem
        self.stem = nn.Sequential(
            ConvBN(3, ch(64), 3, 2, 1, use_gn=use_gn),
            ConvBN(ch(64), ch(128), 3, 2, 1, use_gn=use_gn),
            C2f(ch(128), ch(128), reps(3), True, use_gn=use_gn),
        )
        # P3
        self.stage1 = nn.Sequential(
            ConvBN(ch(128), ch(256), 3, 2, 1, use_gn=use_gn),
            C2f(ch(256), ch(256), reps(6), True, use_gn=use_gn),
        )
        # P4
        self.stage2 = nn.Sequential(
            ConvBN(ch(256), ch(512), 3, 2, 1, use_gn=use_gn),
            C2f(ch(512), ch(512), reps(6), True, use_gn=use_gn),
        )
        # P5
        self.stage3 = nn.Sequential(
            ConvBN(ch(512), ch(512), 3, 2, 1, use_gn=use_gn),
            C2f(ch(512), ch(512), reps(3), True, use_gn=use_gn),
            SPPF(ch(512), ch(512), use_gn=use_gn),
        )

        self.out_channels = [ch(256), ch(512), ch(512)]

    def forward(self, x):
        x  = self.stem(x)
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)
        return p3, p4, p5


# ─────────────────────────────────────────────────────────────────────────────
# Neck (PAN-FPN, YOLOv8 style)
# ─────────────────────────────────────────────────────────────────────────────

class YOLOv8Neck(nn.Module):
    def __init__(self, in_channels: List[int], depth: int = 1,
                 use_gn: bool = False):
        super().__init__()
        c3, c4, c5 = in_channels

        def reps(n):
            return max(round(n * depth), 1)

        # top-down
        self.up5      = nn.Upsample(scale_factor=2)
        self.c2f_p4   = C2f(c4 + c5, c4, reps(3), use_gn=use_gn)

        self.up4      = nn.Upsample(scale_factor=2)
        self.c2f_p3   = C2f(c3 + c4, c3, reps(3), use_gn=use_gn)

        # bottom-up
        self.down_p3  = ConvBN(c3, c3, 3, 2, 1, use_gn=use_gn)
        self.c2f_n4   = C2f(c3 + c4, c4, reps(3), use_gn=use_gn)

        self.down_n4  = ConvBN(c4, c4, 3, 2, 1, use_gn=use_gn)
        self.c2f_n5   = C2f(c4 + c5, c5, reps(3), use_gn=use_gn)

        self.out_channels = [c3, c4, c5]

    def forward(self, feats):
        p3, p4, p5 = feats

        x  = self.c2f_p4(torch.cat([p4, self.up5(p5)], 1))
        n3 = self.c2f_p3(torch.cat([p3, self.up4(x)], 1))

        n4 = self.c2f_n4(torch.cat([x,  self.down_p3(n3)], 1))
        n5 = self.c2f_n5(torch.cat([p5, self.down_n4(n4)], 1))

        return n3, n4, n5   # strides 8, 16, 32


# ─────────────────────────────────────────────────────────────────────────────
# Detection head (standard YOLOv8 decoupled head)
# ─────────────────────────────────────────────────────────────────────────────

class DetHead(nn.Module):
    """
    Decoupled box + class heads per FPN scale.
    Also exposes `penultimate_feat` (before the final box 1×1)
    for the polygon branch.
    """
    REG_MAX = 16   # DFL bins

    def __init__(self, in_ch: int, num_classes: int, use_gn: bool = False):
        super().__init__()
        self.nc = num_classes
        mid     = max(in_ch, 256)

        # ── box branch ────────────────────────────────────────────────────────
        self.box_pre = nn.Sequential(
            ConvBN(in_ch, mid, 3, 1, 1, use_gn=use_gn),
            ConvBN(mid,   mid, 3, 1, 1, use_gn=use_gn),
        )
        self.box_out = nn.Conv2d(mid, 4 * self.REG_MAX, 1)

        # ── class branch ─────────────────────────────────────────────────────
        self.cls_pre = nn.Sequential(
            ConvBN(in_ch, mid, 3, 1, 1, use_gn=use_gn),
            ConvBN(mid,   mid, 3, 1, 1, use_gn=use_gn),
        )
        self.cls_out = nn.Conv2d(mid, num_classes, 1)

    def forward(self, x):
        # box
        box_feat       = self.box_pre(x)
        self._pen_feat = box_feat          # ← penultimate feature map
        box_out        = self.box_out(box_feat)   # (B, 4*REG_MAX, H, W)
        # cls
        cls_out = self.cls_out(self.cls_pre(x))   # (B, nc, H, W)
        return box_out, cls_out, box_feat          # expose penultimate


# ─────────────────────────────────────────────────────────────────────────────
# Polygon branch (3 independent Conv1x1 heads)
# ─────────────────────────────────────────────────────────────────────────────

class PolyHead(nn.Module):
    """Three independent Conv1x1 heads from the penultimate box feature map."""
    def __init__(self, in_ch: int, num_angles: int):
        super().__init__()
        self.poly_conf  = ConvBN(in_ch, num_angles, 1, act=False)
        self.poly_angle = ConvBN(in_ch, num_angles, 1, act=False)
        self.poly_dist  = ConvBN(in_ch, num_angles, 1, act=False)

    def forward(self, pen_feat):
        return (
            self.poly_conf(pen_feat),    # (B, num_angles, H, W)  raw logits
            self.poly_angle(pen_feat),   # (B, num_angles, H, W)  raw logits
            self.poly_dist(pen_feat),    # (B, num_angles, H, W)  raw pre-softplus
        )


# ─────────────────────────────────────────────────────────────────────────────
# Distance head
# ─────────────────────────────────────────────────────────────────────────────

class DistHead(nn.Module):
    """num_dist_blocks × ConvBN(3×3) + Conv1×1 → scalar per anchor."""
    def __init__(self, in_ch: int, num_dist_blocks: int = 1,
                 use_gn: bool = False):
        super().__init__()
        mid = max(in_ch // 2, 64)
        layers: list[nn.Module] = []
        ch_in = in_ch
        for _ in range(num_dist_blocks):
            layers += [ConvBN(ch_in, mid, 3, 1, 1, use_gn=use_gn)]
            ch_in = mid
        layers += [ConvBN(ch_in, 1, 1, act=False, use_gn=use_gn)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)    # (B, 1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

DEPTH_MULT = {"n": 0.33, "s": 0.33, "m": 0.67, "l": 1.00, "x": 1.33}
WIDTH_MULT = {"n": 0.25, "s": 0.50, "m": 0.75, "l": 1.00, "x": 1.25}


class YOLOv8Extended(nn.Module):
    """
    YOLOv8-Extended: bbox + class + polygon + distance.

    Forward returns (in training mode):
        preds: list of per-scale tuples
            (box_raw, cls_raw, poly_conf, poly_angle, poly_dist_raw, dist_raw)
            each tensor shape: (B, C, H_i, W_i)

    In eval mode the same tuple is returned; post-processing is done externally.
    """

    def __init__(
        self,
        model_size: str = "s",
        num_classes: int = 80,
        num_angles: int = 24,
        num_dist_blocks: int = 1,
        strides: List[int] | None = None,
        gpu_type: str = "nvidia",   # "nvidia" → BN, "amd" → GN
    ):
        super().__init__()
        self.num_classes     = num_classes
        self.num_angles      = num_angles
        self.num_dist_blocks = num_dist_blocks
        self.strides         = strides or [8, 16, 32]

        use_gn = (gpu_type.lower() == "amd")

        d = DEPTH_MULT[model_size]
        w = WIDTH_MULT[model_size]

        self.backbone = YOLOv8Backbone(depth=d, width_mult=w, use_gn=use_gn)
        self.neck     = YOLOv8Neck(self.backbone.out_channels, depth=d,
                                   use_gn=use_gn)

        neck_chs = self.neck.out_channels   # [c3, c4, c5]

        self.det_heads  = nn.ModuleList([
            DetHead(c, num_classes, use_gn=use_gn) for c in neck_chs
        ])
        self.poly_heads = nn.ModuleList([
            PolyHead(max(c, 256), num_angles) for c in neck_chs
        ])
        self.dist_heads = nn.ModuleList([
            DistHead(c, num_dist_blocks, use_gn=use_gn) for c in neck_chs
        ])

        self._init_weights(use_gn)

    # ── pretrained weights ────────────────────────────────────────────────────

    def load_pretrained_backbone(self, model_size: str) -> dict:
        """
        Download official ultralytics YOLOv8 weights and copy backbone + neck
        parameters into this model by matching parameter names and shapes.

        Strategy
        ────────
        Ultralytics uses numeric layer indices: model.0.*, model.1.*, …
        We map those indices to our named submodules using the known YOLOv8
        architecture layout, then match parameter suffixes exactly.

        Only backbone.* and neck.* parameters are transferred.
        All head weights (det, poly, dist) stay randomly initialised.

        Returns a summary dict: matched / skipped_shape / skipped_name / total_dst.
        """
        import re

        # ── official YOLOv8 layer-index → our submodule prefix ────────────────
        # YOLOv8s backbone: layers 0-9, neck: 10-22 (approximate; shape-match
        # handles slight version differences automatically).
        LAYER_TO_MODULE = {
            # backbone
            0:  "backbone.stem.0",      # Conv  (stem[0])
            1:  "backbone.stem.1",      # Conv  (stem[1])
            2:  "backbone.stem.2",      # C2f   (stem[2])
            3:  "backbone.stage1.0",    # Conv
            4:  "backbone.stage1.1",    # C2f
            5:  "backbone.stage2.0",    # Conv
            6:  "backbone.stage2.1",    # C2f
            7:  "backbone.stage3.0",    # Conv
            8:  "backbone.stage3.1",    # C2f
            9:  "backbone.stage3.2",    # SPPF
            # neck (PAN-FPN)
            10: "neck.up5",             # Upsample (no params)
            11: "neck.c2f_p4",          # C2f
            12: "neck.up4",             # Upsample (no params)
            13: "neck.c2f_p3",          # C2f
            14: "neck.down_p3",         # Conv
            15: "neck.c2f_n4",          # C2f
            16: "neck.down_n4",         # Conv
            17: "neck.c2f_n5",          # C2f
        }

        # ── download weights ──────────────────────────────────────────────────
        hub_names = {
            "n": "yolov8n.pt", "s": "yolov8s.pt",
            "m": "yolov8m.pt", "l": "yolov8l.pt", "x": "yolov8x.pt",
        }
        pt_name = hub_names[model_size]

        src_sd = None

        # Method 1: ultralytics package (preferred — handles download + caching)
        try:
            from ultralytics import YOLO as _YOLO
            _ult  = _YOLO(pt_name)
            src_sd = _ult.model.state_dict()
        except Exception:
            pass

        # Method 2: torch.hub
        if src_sd is None:
            try:
                _hub = torch.hub.load(
                    "ultralytics/ultralytics",
                    "yolov8" + model_size,
                    pretrained=True, verbose=False,
                )
                src_sd = _hub.model.state_dict()
            except Exception:
                pass

        if src_sd is None:
            raise RuntimeError(
                "Could not download pretrained YOLOv8 weights.\n"
                "  • Install ultralytics:  pip install ultralytics\n"
                "  • Or pass --no_pretrained to train from scratch."
            )

        # ── build source lookup: our_submodule_prefix.param_suffix → tensor ───
        # e.g. "backbone.stem.0.conv.weight" → tensor
        _idx_re = re.compile(r"^model\.(\d+)\.(.*)")
        src_lookup: dict[str, torch.Tensor] = {}

        for k, v in src_sd.items():
            m = _idx_re.match(k)
            if not m:
                continue
            layer_idx  = int(m.group(1))
            param_suf  = m.group(2)          # e.g. "cv1.conv.weight"
            our_prefix = LAYER_TO_MODULE.get(layer_idx)
            if our_prefix is None:
                continue
            our_key = f"{our_prefix}.{param_suf}"
            src_lookup[our_key] = v

        # ── transfer matching parameters ──────────────────────────────────────
        dst_sd        = self.state_dict()
        update_sd:    dict[str, torch.Tensor] = {}
        matched       = 0
        skipped_shape = 0
        skipped_name  = 0

        for dst_key, dst_val in dst_sd.items():
            if not (dst_key.startswith("backbone.") or
                    dst_key.startswith("neck.")):
                skipped_name += 1
                continue

            if dst_key in src_lookup:
                src_val = src_lookup[dst_key]
                if src_val.shape == dst_val.shape:
                    update_sd[dst_key] = src_val
                    matched += 1
                else:
                    skipped_shape += 1
            else:
                skipped_name += 1

        self.load_state_dict(update_sd, strict=False)

        return {
            "matched":       matched,
            "skipped_shape": skipped_shape,
            "skipped_name":  skipped_name,
            "total_dst":     len(dst_sd),
        }

    # ── weight init ───────────────────────────────────────────────────────────
    def _init_weights(self, use_gn: bool = False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        backbone_feats = self.backbone(x)
        neck_feats     = self.neck(backbone_feats)

        preds = []
        for feat, det_h, poly_h, dist_h in zip(
            neck_feats, self.det_heads, self.poly_heads, self.dist_heads
        ):
            box_raw, cls_raw, pen_feat = det_h(feat)
            pc, pa, pd                 = poly_h(pen_feat)
            dist_raw                   = dist_h(feat)
            preds.append((box_raw, cls_raw, pc, pa, pd, dist_raw))

        return preds   # list[scale]  len=3
