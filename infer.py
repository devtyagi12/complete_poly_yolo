"""
Inference script for YOLOv8-Extended.

Usage
─────
  python infer.py --weights runs/train/best.pt --source images/

  # with custom thresholds and output dir
  python infer.py --weights runs/train/best.pt \\
                  --source images/ \\
                  --save_dir out/preds \\
                  --conf_thres 0.3 --iou_thres 0.5 \\
                  --class_names person car truck

  # load config from saved experiment
  python infer.py --cfg runs/train/exp1/config.yaml \\
                  --weights runs/train/exp1/best.pt \\
                  --source test_images/
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from configs.config import load_config
from models.model import YOLOv8Extended
from postprocess.decode import Detection, PostProcessor
from utils.logger import Logger


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing
# ─────────────────────────────────────────────────────────────────────────────

def _letterbox(img: np.ndarray, size: int):
    h, w   = img.shape[:2]
    scale  = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_r  = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_y  = (size - nh) // 2
    pad_x  = (size - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img_r
    return canvas, scale, (pad_x, pad_y)


def _preprocess(img_bgr: np.ndarray, size: int) -> torch.Tensor:
    lb, _, _ = _letterbox(img_bgr, size)
    t = torch.from_numpy(
        np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
    ) / 255.0
    return t.unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

_PAL = [
    (255,  56,  56), (255, 157,  51), ( 74, 180, 249), (149, 255,  65),
    (221, 111, 255), ( 63, 153,  30), (255,  85, 170), (  0, 165, 255),
    (255, 255,   0), (128,   0, 255),
]

def _col(cls: int): return _PAL[cls % len(_PAL)]


def draw_detections(
    img: np.ndarray,
    detections: List[Detection],
    class_names: List[str] | None = None,
    show_distance: bool = True,
    show_polygon: bool = True,
) -> np.ndarray:
    out = img.copy()
    for d in detections:
        col = _col(d.cls)
        x1, y1, x2, y2 = d.bbox.astype(int)

        # bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)

        # label
        name  = class_names[d.cls] if class_names else str(d.cls)
        dist  = f" {d.distance:.1f}m" if show_distance else ""
        label = f"{name} {d.score:.2f}{dist}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), col, -1)
        cv2.putText(out, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # polygon
        if show_polygon and d.polygon is not None and len(d.polygon) >= 2:
            pts = d.polygon.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=True, color=col, thickness=1,
                          lineType=cv2.LINE_AA)
            for pt in d.polygon:
                cv2.circle(out, tuple(pt.astype(int)), 2, col, -1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Predictor class
# ─────────────────────────────────────────────────────────────────────────────

class Predictor:
    """
    Programmatic inference API.

        p = Predictor("best.pt")
        dets = p(img_bgr)          # list[Detection]
        vis  = p.draw(img_bgr, dets)
    """

    def __init__(self, weights: str, cfg=None, device_str: str | None = None):
        from configs.config import Config
        self.cfg    = cfg or Config()
        mc          = self.cfg.model
        dc          = self.cfg.data
        tc          = self.cfg.train
        self.device = torch.device(
            (device_str or tc.device) if torch.cuda.is_available() else "cpu"
        )
        self.img_size = dc.img_size

        self.model = YOLOv8Extended(
            model_size      = mc.model_size,
            num_classes     = mc.num_classes,
            num_angles      = mc.num_angles,
            num_dist_blocks = mc.num_dist_blocks,
            strides         = mc.strides,
        ).to(self.device)

        ckpt = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        self.model.eval()

        self.post_proc = PostProcessor(
            num_classes     = mc.num_classes,
            num_angles      = mc.num_angles,
            strides         = mc.strides,
            img_size        = dc.img_size,
            conf_thres      = tc.conf_thres,
            iou_thres       = tc.iou_thres,
            poly_conf_thres = tc.conf_thres,
            min_distance    = mc.min_distance,
            max_distance    = mc.max_distance,
        )

    @torch.no_grad()
    def __call__(self, img_bgr: np.ndarray) -> List[Detection]:
        orig_h, orig_w = img_bgr.shape[:2]
        t = _preprocess(img_bgr, self.img_size).to(self.device)
        return self.post_proc(self.model(t), [(orig_h, orig_w)])[0]

    def draw(self, img_bgr: np.ndarray, dets: List[Detection],
             class_names=None) -> np.ndarray:
        return draw_detections(img_bgr, dets, class_names=class_names)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def infer():
    extra = [
        (["--weights"],       dict(required=True)),
        (["--source"],        dict(required=True, help="Image file or directory")),
        (["--save_dir"],      dict(default="out")),
        (["--class_names"],   dict(nargs="*", default=None,
                                  help="Optional list of class name strings")),
        (["--no_polygon"],    dict(action="store_true")),
        (["--no_distance"],   dict(action="store_true")),
    ]
    cfg, args = load_config(description="YOLOv8-Extended inference", extra_args=extra)
    mc = cfg.model; dc = cfg.data; tc = cfg.train

    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(save_dir=args.save_dir, use_tb=False)

    device = torch.device(tc.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = YOLOv8Extended(
        model_size      = mc.model_size,
        num_classes     = mc.num_classes,
        num_angles      = mc.num_angles,
        num_dist_blocks = mc.num_dist_blocks,
        strides         = mc.strides,
    ).to(device)

    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    logger.info(f"Weights: {args.weights}")

    post_proc = PostProcessor(
        num_classes     = mc.num_classes,
        num_angles      = mc.num_angles,
        strides         = mc.strides,
        img_size        = dc.img_size,
        conf_thres      = tc.conf_thres,
        iou_thres       = tc.iou_thres,
        poly_conf_thres = tc.conf_thres,
        min_distance    = mc.min_distance,
        max_distance    = mc.max_distance,
    )

    # ── collect images ────────────────────────────────────────────────────────
    src  = Path(args.source)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted(
        [src] if src.is_file() else
        [p for p in src.rglob("*") if p.suffix.lower() in exts]
    )
    logger.info(f"Found {len(img_paths)} images → {args.save_dir}")

    # ── run ───────────────────────────────────────────────────────────────────
    all_results = {}
    with torch.no_grad():
        for img_path in img_paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning(f"Skipping unreadable: {img_path}")
                continue

            orig_h, orig_w = img_bgr.shape[:2]
            t = _preprocess(img_bgr, dc.img_size).to(device)
            preds = model(t)
            dets  = post_proc(preds, [(orig_h, orig_w)])[0]

            vis = draw_detections(
                img_bgr, dets,
                class_names   = args.class_names,
                show_distance = not args.no_distance,
                show_polygon  = not args.no_polygon,
            )
            out_path = os.path.join(args.save_dir, img_path.name)
            cv2.imwrite(out_path, vis)

            all_results[str(img_path)] = dets
            logger.info(
                f"  {img_path.name:<40}  "
                f"{len(dets)} detection(s)"
            )

    logger.close()
    return all_results


if __name__ == "__main__":
    infer()
