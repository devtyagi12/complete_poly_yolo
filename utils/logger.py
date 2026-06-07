"""
Structured logger for YOLOv8-Extended training.

Outputs
───────
• Console  – colour-coded, human-readable lines
• CSV      – {save_dir}/logs/metrics.csv  (append-mode, one row per step/epoch)
• TensorBoard – {save_dir}/logs/tb/  (optional, requires tensorboard package)

Usage
─────
    logger = Logger(save_dir="runs/train/exp1", use_tb=True)
    logger.log_config(cfg)
    logger.log_step(epoch=0, step=10, n_steps=100, loss=1.23, loss_dict={...}, lr=0.01)
    logger.log_epoch(epoch=0, loss=1.1, loss_dict={...}, lr=0.01, elapsed=42.3)
    logger.log_val(epoch=0, precision=0.8, recall=0.75, f1=0.77, per_class={...})
    logger.log_image("vis/train", img_numpy_rgb, epoch=0)
    logger.close()
"""
from __future__ import annotations

import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np

# ── optional tensorboard ──────────────────────────────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers
# ─────────────────────────────────────────────────────────────────────────────

_NO_COLOR = not sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    if _NO_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(t):   return _c(t, "1")
def _green(t):  return _c(t, "32")
def _yellow(t): return _c(t, "33")
def _cyan(t):   return _c(t, "36")
def _red(t):    return _c(t, "31")
def _dim(t):    return _c(t, "2")


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    """
    Unified logger: console + CSV + TensorBoard.

    Parameters
    ----------
    save_dir    : experiment root directory
    use_tb      : enable TensorBoard writer
    rank        : DDP rank (only rank-0 writes)
    """

    def __init__(
        self,
        save_dir: str | Path,
        use_tb: bool = True,
        rank: int = 0,
    ):
        self.save_dir = Path(save_dir)
        self.rank     = rank
        self._active  = (rank == 0)

        if not self._active:
            return

        log_dir = self.save_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # ── Python logging ────────────────────────────────────────────────────
        fmt = logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._py_logger = logging.getLogger(f"yolov8x.{id(self)}")
        self._py_logger.setLevel(logging.DEBUG)
        self._py_logger.propagate = False

        # file handler (DEBUG+)
        fh = logging.FileHandler(log_dir / "train.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self._py_logger.addHandler(fh)

        # console handler (INFO+)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        self._py_logger.addHandler(ch)

        # ── CSV ───────────────────────────────────────────────────────────────
        self._csv_path  = log_dir / "metrics.csv"
        self._csv_file  = open(self._csv_path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if os.path.getsize(self._csv_path) == 0:
            self._csv_writer.writerow([
                "timestamp", "mode", "epoch", "step",
                "loss", "box", "cls", "dfl",
                "poly_dist", "poly_conf", "poly_ang", "dist",
                "lr", "precision", "recall", "f1",
            ])
            self._csv_file.flush()

        # ── TensorBoard ───────────────────────────────────────────────────────
        self._tb: SummaryWriter | None = None
        if use_tb and _TB_AVAILABLE:
            tb_dir = log_dir / "tb"
            self._tb = SummaryWriter(str(tb_dir))
            self._py_logger.info(_dim(f"TensorBoard → {tb_dir}"))
        elif use_tb and not _TB_AVAILABLE:
            self._py_logger.warning(
                _yellow("TensorBoard requested but not installed.  "
                        "Run: pip install tensorboard")
            )

        self._train_start = time.time()

    # ── config dump ───────────────────────────────────────────────────────────

    def log_config(self, cfg) -> None:
        if not self._active:
            return
        from configs.config import save_config
        save_config(cfg, self.save_dir / "config.yaml")

        lines = [_bold("─" * 60), _bold("  YOLOv8-Extended  —  effective config")]
        flat = cfg.to_flat_dict()
        section = None
        for k, v in flat.items():
            sec, name = k.split(".", 1)
            if sec != section:
                section = sec
                lines.append(_cyan(f"\n  [{section}]"))
            lines.append(f"    {name:<22} {_green(str(v))}")
        lines.append(_bold("─" * 60))
        self._py_logger.info("\n".join(lines))

    # ── step-level (within epoch) ─────────────────────────────────────────────

    def log_step(
        self,
        epoch: int,
        step: int,
        n_steps: int,
        loss: float,
        loss_dict: dict[str, float],
        lr: float,
    ) -> None:
        if not self._active:
            return

        global_step = epoch * n_steps + step

        # TensorBoard
        if self._tb:
            self._tb.add_scalar("step/loss_total", loss, global_step)
            for k, v in loss_dict.items():
                self._tb.add_scalar(f"step/{k}", v, global_step)
            self._tb.add_scalar("step/lr", lr, global_step)

        # console (compact)
        pct    = f"{step}/{n_steps}"
        parts  = "  ".join(f"{k}={v:.3f}" for k, v in loss_dict.items())
        msg = (
            f"  step [{_yellow(pct):>12}]  "
            f"loss={_bold(f'{loss:.4f}')}  "
            f"{_dim(parts)}  lr={lr:.2e}"
        )
        self._py_logger.debug(msg)   # debug → file only; INFO → also console

    # ── epoch-level ──────────────────────────────────────────────────────────

    def log_epoch(
        self,
        epoch: int,
        loss: float,
        loss_dict: dict[str, float],
        lr: float,
        elapsed: float,
    ) -> None:
        if not self._active:
            return

        # TensorBoard
        if self._tb:
            self._tb.add_scalar("train/loss", loss, epoch)
            for k, v in loss_dict.items():
                self._tb.add_scalar(f"train/{k}", v, epoch)
            self._tb.add_scalar("train/lr", lr, epoch)

        # CSV
        self._csv_row(mode="train", epoch=epoch, step=-1,
                      loss=loss, loss_dict=loss_dict, lr=lr)

        # console
        total_elapsed = time.time() - self._train_start
        parts = "  ".join(
            f"{_dim(k)}={_green(f'{v:.4f}')}" for k, v in loss_dict.items()
        )
        msg = (
            f"\n{_bold(f'[Epoch {epoch:04d}]')}  "
            f"loss={_bold(f'{loss:.4f}')}  "
            f"{parts}  "
            f"lr={_cyan(f'{lr:.2e}')}  "
            f"{_dim(f'{elapsed:.1f}s / {self._fmt_elapsed(total_elapsed)} total')}"
        )
        self._py_logger.info(msg)

    # ── validation ────────────────────────────────────────────────────────────

    def log_val(
        self,
        epoch: int,
        precision: float,
        recall: float,
        f1: float,
        per_class: dict[str, dict] | None = None,
        is_best: bool = False,
    ) -> None:
        if not self._active:
            return

        # TensorBoard
        if self._tb:
            self._tb.add_scalar("val/precision", precision, epoch)
            self._tb.add_scalar("val/recall",    recall,    epoch)
            self._tb.add_scalar("val/f1",        f1,        epoch)
            if per_class:
                for cls_key, vals in per_class.items():
                    if cls_key == "micro":
                        continue
                    self._tb.add_scalar(f"val_per_class/{cls_key}_f1",
                                        vals["f1"], epoch)

        # CSV
        self._csv_row(mode="val", epoch=epoch, step=-1,
                      loss=0, loss_dict={},
                      lr=0, precision=precision, recall=recall, f1=f1)

        # console
        star = f"  {_bold(_yellow('★ NEW BEST'))}" if is_best else ""
        self._py_logger.info(
            f"  {_bold('Val')}  "
            f"P={_green(f'{precision:.4f}')}  "
            f"R={_green(f'{recall:.4f}')}  "
            f"F1={_bold(_green(f'{f1:.4f}'))}"
            f"{star}"
        )
        if per_class:
            for cls_key, vals in per_class.items():
                if cls_key == "micro":
                    continue
                self._py_logger.info(
                    f"    {cls_key:<14}  "
                    f"P={vals['precision']:.3f}  "
                    f"R={vals['recall']:.3f}  "
                    f"F1={vals['f1']:.3f}"
                )

    # ── image / visualisation ─────────────────────────────────────────────────

    def log_image(
        self,
        tag: str,
        img_rgb: "np.ndarray",   # HWC uint8
        epoch: int,
    ) -> None:
        """Send a numpy HWC-RGB image to TensorBoard."""
        if not self._active or self._tb is None:
            return
        import numpy as np
        img = img_rgb.astype(np.uint8)
        # TensorBoard expects (C, H, W) or (H, W, C) — use CHW
        self._tb.add_image(tag, img.transpose(2, 0, 1), epoch)

    # ── gradient norm ─────────────────────────────────────────────────────────

    def log_grad_norm(self, norm: float, global_step: int) -> None:
        if not self._active or self._tb is None:
            return
        self._tb.add_scalar("train/grad_norm", norm, global_step)

    # ── custom scalar ─────────────────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if not self._active:
            return
        if self._tb:
            self._tb.add_scalar(tag, value, step)
        self._py_logger.debug(f"  {tag}={value:.6f} @ step {step}")

    # ── info / warning / error ────────────────────────────────────────────────

    def info(self, msg: str)    -> None:
        if self._active: self._py_logger.info(msg)

    def warning(self, msg: str) -> None:
        if self._active: self._py_logger.warning(_yellow(f"WARNING  {msg}"))

    def error(self, msg: str)   -> None:
        if self._active: self._py_logger.error(_red(f"ERROR    {msg}"))

    def debug(self, msg: str)   -> None:
        if self._active: self._py_logger.debug(msg)

    # ── cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        if not self._active:
            return
        if self._tb:
            self._tb.close()
        if hasattr(self, "_csv_file"):
            self._csv_file.close()
        total = time.time() - self._train_start
        self._py_logger.info(
            _bold(f"\n✓ Training finished  —  total time {self._fmt_elapsed(total)}")
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _csv_row(
        self,
        mode: str,
        epoch: int,
        step: int,
        loss: float,
        loss_dict: dict,
        lr: float,
        precision: float = 0,
        recall: float = 0,
        f1: float = 0,
    ):
        row = [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            mode, epoch, step,
            round(loss, 6),
            round(loss_dict.get("box",       0), 6),
            round(loss_dict.get("cls",       0), 6),
            round(loss_dict.get("dfl",       0), 6),
            round(loss_dict.get("poly_dist", 0), 6),
            round(loss_dict.get("poly_conf", 0), 6),
            round(loss_dict.get("poly_ang",  0), 6),
            round(loss_dict.get("dist",      0), 6),
            round(lr, 8),
            round(precision, 6),
            round(recall,    6),
            round(f1,        6),
        ]
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    @staticmethod
    def _fmt_elapsed(secs: float) -> str:
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = int(secs % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
