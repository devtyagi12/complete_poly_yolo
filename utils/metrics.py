"""
Validation metrics for YOLOv8-Extended.

F1 score is computed using bounding boxes only (IoU-based matching).
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np

from postprocess.decode import Detection


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two boxes (x1,y1,x2,y2)."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a_area = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    b_area = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


class BBoxF1Metric:
    """
    Accumulates predictions and ground-truths over a validation set,
    then computes per-class and micro-averaged F1.

    Ground-truth format per image: list of (cls, x1, y1, x2, y2) in pixel coords.
    Predictions: list of Detection objects.
    """

    def __init__(self, num_classes: int, iou_thres: float = 0.5):
        self.nc       = num_classes
        self.iou_thres = iou_thres
        self.reset()

    def reset(self):
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)

    def update(
        self,
        preds: List[Detection],
        gts: List[tuple],      # [(cls, x1, y1, x2, y2), ...]
    ):
        """Update TP/FP/FN for one image."""
        # group by class
        pred_by_cls: dict[int, list] = defaultdict(list)
        for d in preds:
            pred_by_cls[d.cls].append(d)

        gt_by_cls: dict[int, list] = defaultdict(list)
        for g in gts:
            gt_by_cls[int(g[0])].append(np.array(g[1:], dtype=float))

        all_cls = set(list(pred_by_cls.keys()) + list(gt_by_cls.keys()))
        for cls in all_cls:
            p_list = sorted(pred_by_cls[cls], key=lambda d: -d.score)
            g_list = list(gt_by_cls[cls])
            matched = [False] * len(g_list)

            for det in p_list:
                best_iou  = 0.0
                best_gi   = -1
                for gi, gt_box in enumerate(g_list):
                    if matched[gi]:
                        continue
                    iou = _iou_xyxy(det.bbox, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gi  = gi
                if best_iou >= self.iou_thres and best_gi >= 0:
                    self.tp[cls] += 1
                    matched[best_gi] = True
                else:
                    self.fp[cls] += 1

            self.fn[cls] += matched.count(False)

    def compute(self) -> dict:
        """Returns per-class and micro-averaged precision, recall, F1."""
        all_cls = set(list(self.tp.keys()) + list(self.fp.keys()) + list(self.fn.keys()))
        results = {}
        total_tp = total_fp = total_fn = 0

        for cls in sorted(all_cls):
            tp = self.tp[cls]; fp = self.fp[cls]; fn = self.fn[cls]
            prec = tp / (tp + fp + 1e-7)
            rec  = tp / (tp + fn + 1e-7)
            f1   = 2 * prec * rec / (prec + rec + 1e-7)
            results[f"class_{cls}"] = {"precision": prec, "recall": rec, "f1": f1}
            total_tp += tp; total_fp += fp; total_fn += fn

        prec = total_tp / (total_tp + total_fp + 1e-7)
        rec  = total_tp / (total_tp + total_fn + 1e-7)
        f1   = 2 * prec * rec / (prec + rec + 1e-7)
        results["micro"] = {"precision": prec, "recall": rec, "f1": f1}
        return results
