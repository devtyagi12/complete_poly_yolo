"""
Evaluation script for YOLOv8-Extended.

Usage
─────
  python test.py --weights runs/train/best.pt \\
                 --poly_img data/polygon/images/test \\
                 --poly_lbl data/polygon/labels/test

  # also evaluate distance dataset
  python test.py --weights runs/train/best.pt \\
                 --poly_img data/polygon/images/test \\
                 --poly_lbl data/polygon/labels/test \\
                 --dist_img data/polygon_distance/images/test \\
                 --dist_lbl data/polygon_distance/labels/test \\
                 --conf_thres 0.4 --iou_thres 0.5

  # load settings from a saved experiment config
  python test.py --cfg runs/train/exp1/config.yaml \\
                 --weights runs/train/exp1/best.pt \\
                 --poly_img data/polygon/images/test \\
                 --poly_lbl data/polygon/labels/test
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from configs.config import load_config
from data.dataset import build_dataloader
from models.model import YOLOv8Extended
from postprocess.decode import PostProcessor
from utils.logger import Logger
from utils.metrics import BBoxF1Metric


def test():
    # ── args ──────────────────────────────────────────────────────────────────
    extra = [
        (["--weights"],  dict(required=True, help="Path to checkpoint (.pt)")),
        (["--poly_img"], dict(required=True, help="Path to polygon test images")),
        (["--poly_lbl"], dict(required=True, help="Path to polygon test labels")),
        (["--dist_img"], dict(default=None,  help="Path to dist test images (optional)")),
        (["--dist_lbl"], dict(default=None,  help="Path to dist test labels (optional)")),
        (["--iou_metric"], dict(default=0.5, type=float,
                               help="IoU threshold for F1 matching")),
    ]
    cfg, args = load_config(description="YOLOv8-Extended evaluation", extra_args=extra)
    mc = cfg.model; dc = cfg.data; tc = cfg.train

    save_dir = Path(tc.save_dir) / "test_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(save_dir=save_dir, use_tb=False)

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
    logger.info(f"Weights loaded from {args.weights}")

    # ── data ──────────────────────────────────────────────────────────────────
    dist_img = args.dist_img or args.poly_img
    dist_lbl = args.dist_lbl or args.poly_lbl

    test_loader = build_dataloader(
        poly_img_dir = args.poly_img,
        poly_lbl_dir = args.poly_lbl,
        dist_img_dir = dist_img,
        dist_lbl_dir = dist_lbl,
        img_size     = dc.img_size,
        batch_size   = dc.batch_size,
        num_workers  = dc.num_workers,
        angle_step   = mc.angle_step,
        min_dist     = mc.min_distance,
        max_dist     = mc.max_distance,
        augment      = False,
    )
    logger.info(f"Test batches: {len(test_loader)}")

    # ── post-processor ────────────────────────────────────────────────────────
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

    metric = BBoxF1Metric(num_classes=mc.num_classes, iou_thres=args.iou_metric)

    # ── inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            B    = imgs.shape[0]

            preds      = model(imgs)
            orig_shapes = [(dc.img_size, dc.img_size)] * B
            detections  = post_proc(preds, orig_shapes)

            for bi in range(B):
                mask = targets[:, 0] == bi
                bt   = targets[mask]
                gts  = []
                for row in bt:
                    cls = int(row[1].item())
                    cx  = row[2].item() * dc.img_size
                    cy  = row[3].item() * dc.img_size
                    w_  = row[4].item() * dc.img_size
                    h_  = row[5].item() * dc.img_size
                    gts.append((cls, cx - w_/2, cy - h_/2, cx + w_/2, cy + h_/2))
                metric.update(detections[bi], gts)

    # ── results ───────────────────────────────────────────────────────────────
    results = metric.compute()
    logger.info(f"\n── Per-class results (IoU@{args.iou_metric}) " + "─" * 30)
    for cls_key, vals in results.items():
        if cls_key == "micro":
            continue
        logger.info(
            f"  {cls_key:<14}  "
            f"P={vals['precision']:.4f}  "
            f"R={vals['recall']:.4f}  "
            f"F1={vals['f1']:.4f}"
        )
    m = results["micro"]
    logger.info(
        f"\n  Micro avg      "
        f"P={m['precision']:.4f}  "
        f"R={m['recall']:.4f}  "
        f"F1={m['f1']:.4f}"
    )
    logger.close()
    return results


if __name__ == "__main__":
    test()
