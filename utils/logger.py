"""
Structured logger for YOLOv8-Extended training.

Outputs
───────
• Console  – YOLOv8-style inline progress bar with live loss columns
• File     – {save_dir}/logs/train.log  (full DEBUG trace)
• CSV      – {save_dir}/logs/metrics.csv
• TensorBoard – {save_dir}/logs/tb/  (optional)

Console format mirrors official YOLOv8:

  Epoch   GPU-mem   box      cls      dfl    poly   dist   Instances   Size
  1/300    2.14G   7.4321   0.5210   1.2300  0.0821  0.0000    42        640

followed by a progress bar that overwrites itself each step:

  1/300  ━━━━━━━━━━━━━━━━━━━━  23/200  loss=9.2341  lr=1.00e-02
"""
from __future__ import annotations

import csv
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# ANSI helpers  (gracefully disabled when stdout is not a tty)
# ─────────────────────────────────────────────────────────────────────────────

_NO_COLOR = not sys.stdout.isatty()
_IS_WIN   = sys.platform == "win32"

# Enable ANSI on Windows terminals that support it
if _IS_WIN:
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32          # type: ignore[attr-defined]
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        _NO_COLOR = True


def _c(text: str, code: str) -> str:
    return text if _NO_COLOR else f"\033[{code}m{text}\033[0m"

def _bold(t):    return _c(t, "1")
def _green(t):   return _c(t, "32")
def _yellow(t):  return _c(t, "33")
def _cyan(t):    return _c(t, "36")
def _red(t):     return _c(t, "31")
def _dim(t):     return _c(t, "2")
def _white(t):   return _c(t, "97")
def _magenta(t): return _c(t, "35")


# ─────────────────────────────────────────────────────────────────────────────
# Progress bar
# ─────────────────────────────────────────────────────────────────────────────

_BAR_FULL  = "━"
_BAR_EMPTY = "─"
_BAR_WIDTH = 20


def _bar(step: int, total: int) -> str:
    filled = int(_BAR_WIDTH * step / max(total, 1))
    bar    = _BAR_FULL * filled + _BAR_EMPTY * (_BAR_WIDTH - filled)
    return _green(bar) if not _NO_COLOR else bar


def _fmt_mem() -> str:
    """GPU memory in GiB, or '  —  ' if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.memory_reserved() / 1024 ** 3
            return f"{mem:.2f}G"
    except Exception:
        pass
    return "  —  "


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    """
    Unified logger: rich console progress + file log + CSV + TensorBoard.
    """

    _HEADER = (
        f"{'Epoch':>10}  {'GPU-mem':>8}  "
        f"{'box':>8}  {'cls':>8}  {'dfl':>8}  "
        f"{'poly':>8}  {'dist':>8}  "
        f"{'Instances':>10}  {'ImgSize':>7}"
    )
    _SEP = "─" * len(_HEADER)

    def __init__(
        self,
        save_dir: str | Path,
        use_tb:   bool = True,
        rank:     int  = 0,
    ):
        self.save_dir = Path(save_dir)
        self.rank     = rank
        self._active  = (rank == 0)

        # track current epoch header state
        self._header_epoch: int = -1
        self._n_steps:      int = 1

        if not self._active:
            return

        log_dir = self.save_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # ── file logger ───────────────────────────────────────────────────────
        fmt = logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._py_logger = logging.getLogger(f"yolov8x.{id(self)}")
        self._py_logger.setLevel(logging.DEBUG)
        self._py_logger.propagate = False

        fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self._py_logger.addHandler(fh)

        # ── CSV ───────────────────────────────────────────────────────────────
        self._csv_path   = log_dir / "metrics.csv"
        self._csv_file   = open(self._csv_path, "a", newline="", encoding="utf-8")
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
        self._tb: "SummaryWriter | None" = None
        if use_tb and _TB_AVAILABLE:
            tb_dir = log_dir / "tb"
            self._tb = SummaryWriter(str(tb_dir))
            self._print(f"  TensorBoard → {tb_dir}")
        elif use_tb and not _TB_AVAILABLE:
            self._print(
                _yellow("  TensorBoard not installed — run: pip install tensorboard")
            )

        self._train_start = time.time()

    # ── config ────────────────────────────────────────────────────────────────

    def log_config(self, cfg) -> None:
        if not self._active:
            return
        from configs.config import save_config
        save_config(cfg, self.save_dir / "config.yaml")

        term_w = shutil.get_terminal_size((100, 24)).columns
        sep    = "─" * min(term_w, 72)
        lines  = [
            _bold(sep),
            _bold("  YOLOv8-Extended"),
            "",
        ]
        flat    = cfg.to_flat_dict()
        section = None
        for k, v in flat.items():
            sec, name = k.split(".", 1)
            if sec != section:
                section = sec
                lines.append(_cyan(f"\n  [{section}]"))
            lines.append(f"    {_dim(name):<30} {_white(str(v))}")
        lines += ["", _bold(sep)]
        self._print("\n".join(lines))

    # ── epoch header  (printed once per epoch before the progress bar) ────────

    def _print_epoch_header(
        self,
        epoch:    int,
        n_epochs: int,
        n_steps:  int,
        img_size: int,
    ) -> None:
        self._header_epoch = epoch
        self._n_steps      = n_steps
        ep_str = _bold(f"{epoch + 1}/{n_epochs}")
        self._print("")
        self._print(_bold(_dim(self._SEP)))
        self._print(_bold(self._HEADER))
        self._print(_bold(_dim(self._SEP)))

    # ── step-level progress  (overwrites current line) ────────────────────────

    def log_step(
        self,
        epoch:     int,
        n_epochs:  int,
        step:      int,
        n_steps:   int,
        loss:      float,
        loss_dict: dict[str, float],
        lr:        float,
        n_instances: int  = 0,
        img_size:    int  = 640,
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

        # file log
        parts = "  ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
        self._py_logger.debug(
            f"ep{epoch+1} step{step}/{n_steps}  loss={loss:.4f}  {parts}  lr={lr:.2e}"
        )

        # ── inline progress bar (overwrites the same line) ────────────────────
        ep_str  = _bold(f"{epoch + 1}/{n_epochs}")
        mem_str = _fmt_mem()
        bar     = _bar(step + 1, n_steps)

        # build the loss columns matching the header
        box  = loss_dict.get("box",       0.0)
        cls  = loss_dict.get("cls",       0.0)
        dfl  = loss_dict.get("dfl",       0.0)
        poly = (loss_dict.get("poly_dist", 0.0) +
                loss_dict.get("poly_conf",  0.0) +
                loss_dict.get("poly_ang",   0.0))
        dist = loss_dict.get("dist",      0.0)

        # step fraction
        step_str = _dim(f"{step + 1}/{n_steps}")

        line = (
            f"\r  {ep_str:>10}  {_cyan(mem_str):>8}  "
            f"{box:>8.4f}  {cls:>8.4f}  {dfl:>8.4f}  "
            f"{poly:>8.4f}  {dist:>8.4f}  "
            f"{n_instances:>10}  {img_size:>7}  "
            f"{bar} {step_str}"
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    # ── epoch summary  (replaces the last progress line) ─────────────────────

    def log_epoch(
        self,
        epoch:     int,
        n_epochs:  int,
        loss:      float,
        loss_dict: dict[str, float],
        lr:        float,
        elapsed:   float,
        n_instances: int = 0,
        img_size:    int = 640,
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

        # overwrite the progress bar line with the final epoch summary row
        box  = loss_dict.get("box",       0.0)
        cls  = loss_dict.get("cls",       0.0)
        dfl  = loss_dict.get("dfl",       0.0)
        poly = (loss_dict.get("poly_dist", 0.0) +
                loss_dict.get("poly_conf",  0.0) +
                loss_dict.get("poly_ang",   0.0))
        dist = loss_dict.get("dist",      0.0)

        ep_str  = _bold(f"{epoch + 1}/{n_epochs}")
        mem_str = _fmt_mem()
        total   = time.time() - self._train_start
        eta_s   = (elapsed * (n_epochs - epoch - 1))
        timing  = _dim(f"  {elapsed:.0f}s/epoch  eta {self._fmt_elapsed(eta_s)}")

        # print the filled summary row (newline at end to lock it in)
        summary = (
            f"\r  {ep_str:>10}  {_cyan(mem_str):>8}  "
            f"{_green(f'{box:.4f}'):>8}  {_green(f'{cls:.4f}'):>8}  "
            f"{_green(f'{dfl:.4f}'):>8}  {_green(f'{poly:.4f}'):>8}  "
            f"{_green(f'{dist:.4f}'):>8}  "
            f"{n_instances:>10}  {img_size:>7}"
            f"{timing}\n"
        )
        sys.stdout.write(summary)
        sys.stdout.flush()

        # file log
        self._py_logger.debug(
            f"[Epoch {epoch+1}/{n_epochs}] "
            f"loss={loss:.4f}  lr={lr:.2e}  {elapsed:.1f}s"
        )

    # ── validation ────────────────────────────────────────────────────────────

    def log_val(
        self,
        epoch:     int,
        n_epochs:  int,
        precision: float,
        recall:    float,
        f1:        float,
        per_class: dict | None = None,
        is_best:   bool = False,
    ) -> None:
        if not self._active:
            return

        if self._tb:
            self._tb.add_scalar("val/precision", precision, epoch)
            self._tb.add_scalar("val/recall",    recall,    epoch)
            self._tb.add_scalar("val/f1",        f1,        epoch)
            if per_class:
                for cls_key, vals in per_class.items():
                    if cls_key == "micro":
                        continue
                    self._tb.add_scalar(
                        f"val_per_class/{cls_key}_f1", vals["f1"], epoch
                    )

        self._csv_row(mode="val", epoch=epoch, step=-1,
                      loss=0, loss_dict={}, lr=0,
                      precision=precision, recall=recall, f1=f1)

        star = f"  {_bold(_yellow('★ NEW BEST'))}" if is_best else ""
        self._print(
            f"  {_bold('Validation')}  "
            f"P={_green(f'{precision:.4f}')}  "
            f"R={_green(f'{recall:.4f}')}  "
            f"F1={_bold(_green(f'{f1:.4f}'))}"
            f"{star}"
        )

        if per_class:
            for cls_key, vals in sorted(per_class.items()):
                if cls_key == "micro":
                    continue
                self._py_logger.debug(
                    f"    {cls_key:<14}  "
                    f"P={vals['precision']:.3f}  "
                    f"R={vals['recall']:.3f}  "
                    f"F1={vals['f1']:.3f}"
                )

    # ── gradient norm ─────────────────────────────────────────────────────────

    def log_grad_norm(self, norm: float, global_step: int) -> None:
        if not self._active or self._tb is None:
            return
        self._tb.add_scalar("train/grad_norm", norm, global_step)

    # ── image ─────────────────────────────────────────────────────────────────

    def log_image(self, tag: str, img_rgb: "np.ndarray", epoch: int) -> None:
        if not self._active or self._tb is None:
            return
        import numpy as np
        img = img_rgb.astype(np.uint8)
        self._tb.add_image(tag, img.transpose(2, 0, 1), epoch)

    # ── generic scalar ────────────────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if not self._active:
            return
        if self._tb:
            self._tb.add_scalar(tag, value, step)

    # ── plain text helpers ────────────────────────────────────────────────────

    def info(self, msg: str)    -> None:
        if self._active: self._print(msg)

    def warning(self, msg: str) -> None:
        if self._active: self._print(_yellow(f"  WARNING  {msg}"))

    def error(self, msg: str)   -> None:
        if self._active: self._print(_red(f"  ERROR    {msg}"))

    def debug(self, msg: str)   -> None:
        if self._active and hasattr(self, "_py_logger"):
            self._py_logger.debug(msg)

    # ── close ─────────────────────────────────────────────────────────────────

    def close(self) -> None:
        if not self._active:
            return
        if self._tb:
            self._tb.close()
        if hasattr(self, "_csv_file"):
            self._csv_file.close()
        total = time.time() - self._train_start
        self._print(
            f"\n{_bold('Training complete')}  —  "
            f"total time {self._fmt_elapsed(total)}"
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _print(self, msg: str) -> None:
        """Print to stdout and file."""
        print(msg)
        if hasattr(self, "_py_logger"):
            clean = msg.replace("\r", "")
            if clean.strip():
                self._py_logger.info(clean)

    def _csv_row(
        self,
        mode:      str,
        epoch:     int,
        step:      int,
        loss:      float,
        loss_dict: dict,
        lr:        float,
        precision: float = 0,
        recall:    float = 0,
        f1:        float = 0,
    ) -> None:
        row = [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            mode, epoch, step,
            round(loss, 6),
            round(loss_dict.get("box",        0), 6),
            round(loss_dict.get("cls",        0), 6),
            round(loss_dict.get("dfl",        0), 6),
            round(loss_dict.get("poly_dist",  0), 6),
            round(loss_dict.get("poly_conf",  0), 6),
            round(loss_dict.get("poly_ang",   0), 6),
            round(loss_dict.get("dist",       0), 6),
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