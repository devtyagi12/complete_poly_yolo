"""
Configuration system for YOLOv8-Extended.

Priority (highest → lowest):
  1. CLI arguments (--key value)
  2. Custom YAML  (--cfg path/to/config.yaml)
  3. Default YAML  (configs/default.yaml)

Usage
─────
    from configs.config import load_config
    cfg = load_config()                     # uses defaults + CLI
    cfg = load_config("configs/exp.yaml")   # YAML override + CLI

    # programmatic override
    cfg.train.epochs = 50
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    model_size:      str   = "s"
    num_classes:     int   = 80
    angle_step:      int   = 15
    num_dist_blocks: int   = 1
    min_distance:    float = 0.5
    max_distance:    float = 200.0

    # derived — always recomputed from angle_step
    @property
    def num_angles(self) -> int:
        return 360 // self.angle_step

    @property
    def strides(self) -> list[int]:
        return [8, 16, 32]


@dataclass
class DataConfig:
    poly_dataset_root: str   = "data/polygon"
    dist_dataset_root: str   = "data/polygon_distance"
    img_size:          int   = 640
    max_labels:        int   = 100
    mosaic_prob:       float = 1.0
    hsv_h:             float = 0.015
    hsv_s:             float = 0.7
    hsv_v:             float = 0.4
    flip_lr_prob:      float = 0.5
    batch_size:        int   = 16
    num_workers:       int   = 4

    # ── derived ───────────────────────────────────────────────────────────────
    @property
    def angle_step(self) -> int:
        # kept in sync with model; access via cfg.model.angle_step
        return 15

    @property
    def min_distance(self) -> float:
        return 0.5

    @property
    def max_distance(self) -> float:
        return 200.0

    invalid_distance: float = -10.0


@dataclass
class TrainConfig:
    epochs:        int   = 300
    warmup_epochs: int   = 3
    lr0:           float = 0.01
    lrf:           float = 0.01
    momentum:      float = 0.937
    weight_decay:  float = 5e-4
    # loss gains
    box_gain:        float = 7.5
    cls_gain:        float = 0.5
    dfl_gain:        float = 1.5
    poly_gain:       float = 0.1
    dist_gain:       float = 0.1
    poly_dist_gain:  float = 2.0
    poly_conf_gain:  float = 0.2
    poly_angle_gain: float = 0.5
    # runtime
    device:       str   = "cuda"
    save_dir:     str   = "runs/train"
    val_interval: int   = 5
    conf_thres:   float = 0.5
    iou_thres:    float = 0.45
    # logging / vis
    log_interval:   int  = 10
    vis_interval:   int  = 1
    vis_max_images: int  = 8
    tensorboard:    bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data:  DataConfig  = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_flat_dict(self) -> dict[str, Any]:
        """Flat {key: value} for logging."""
        out = {}
        for section in ("model", "data", "train"):
            obj = getattr(self, section)
            for f in fields(obj):
                out[f"{section}.{f.name}"] = getattr(obj, f.name)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# YAML helpers
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_YAML = Path(__file__).parent / "default.yaml"


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _apply_yaml(cfg: Config, d: dict):
    """Overwrite cfg fields from a nested YAML dict."""
    section_map = {"model": cfg.model, "data": cfg.data, "train": cfg.train}
    for section, obj in section_map.items():
        if section not in d:
            continue
        for k, v in d[section].items():
            if hasattr(obj, k):
                setattr(obj, k, v)


def _save_yaml(cfg: Config, path: str | Path):
    """Persist the effective config next to checkpoints."""
    d = {
        "model": {f.name: getattr(cfg.model, f.name) for f in fields(cfg.model)},
        "data":  {f.name: getattr(cfg.data,  f.name) for f in fields(cfg.data)},
        "train": {f.name: getattr(cfg.train, f.name) for f in fields(cfg.train)},
    }
    with open(path, "w") as f:
        yaml.dump(d, f, default_flow_style=False, sort_keys=False)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser(description: str = "YOLOv8-Extended") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── config file ───────────────────────────────────────────────────────────
    ap.add_argument(
        "--cfg", default=None, metavar="PATH",
        help="Path to a YAML config file (overrides defaults, overridden by CLI args)",
    )

    # ── model ─────────────────────────────────────────────────────────────────
    g = ap.add_argument_group("Model")
    g.add_argument("--model_size",      default=None, choices=["n","s","m","l","x"])
    g.add_argument("--num_classes",     default=None, type=int)
    g.add_argument("--angle_step",      default=None, type=int,
                   help="Degrees per polygon bin (num_angles = 360 // angle_step)")
    g.add_argument("--num_dist_blocks", default=None, type=int)
    g.add_argument("--min_distance",    default=None, type=float, metavar="METRES")
    g.add_argument("--max_distance",    default=None, type=float, metavar="METRES")

    # ── data ──────────────────────────────────────────────────────────────────
    g = ap.add_argument_group("Data")
    g.add_argument("--poly_dataset_root", default=None)
    g.add_argument("--dist_dataset_root", default=None)
    g.add_argument("--img_size",    default=None, type=int)
    g.add_argument("--batch_size",  default=None, type=int)
    g.add_argument("--num_workers", default=None, type=int)
    g.add_argument("--mosaic_prob", default=None, type=float)
    g.add_argument("--flip_lr_prob",default=None, type=float)
    g.add_argument("--hsv_h",       default=None, type=float)
    g.add_argument("--hsv_s",       default=None, type=float)
    g.add_argument("--hsv_v",       default=None, type=float)

    # ── training ──────────────────────────────────────────────────────────────
    g = ap.add_argument_group("Training")
    g.add_argument("--epochs",        default=None, type=int)
    g.add_argument("--warmup_epochs", default=None, type=int)
    g.add_argument("--lr0",           default=None, type=float)
    g.add_argument("--lrf",           default=None, type=float)
    g.add_argument("--momentum",      default=None, type=float)
    g.add_argument("--weight_decay",  default=None, type=float)
    g.add_argument("--device",        default=None)
    g.add_argument("--save_dir",      default=None)
    g.add_argument("--val_interval",  default=None, type=int)
    g.add_argument("--conf_thres",    default=None, type=float)
    g.add_argument("--iou_thres",     default=None, type=float)
    g.add_argument("--resume",        default=None, metavar="CKPT",
                   help="Resume from checkpoint path")

    # ── loss gains ────────────────────────────────────────────────────────────
    g = ap.add_argument_group("Loss gains")
    g.add_argument("--box_gain",        default=None, type=float)
    g.add_argument("--cls_gain",        default=None, type=float)
    g.add_argument("--dfl_gain",        default=None, type=float)
    g.add_argument("--poly_gain",       default=None, type=float)
    g.add_argument("--dist_gain",       default=None, type=float)
    g.add_argument("--poly_dist_gain",  default=None, type=float)
    g.add_argument("--poly_conf_gain",  default=None, type=float)
    g.add_argument("--poly_angle_gain", default=None, type=float)

    # ── logging / vis ─────────────────────────────────────────────────────────
    g = ap.add_argument_group("Logging & Visualisation")
    g.add_argument("--log_interval",   default=None, type=int,
                   help="Log training loss every N steps")
    g.add_argument("--vis_interval",   default=None, type=int,
                   help="Save visualisation grid every N epochs")
    g.add_argument("--vis_max_images", default=None, type=int)
    g.add_argument("--no_tensorboard", action="store_true",
                   help="Disable TensorBoard logging")

    return ap


def _apply_args(cfg: Config, args: argparse.Namespace):
    """Write non-None CLI args back into the config."""
    # model
    _maybe(cfg.model, "model_size",      args.model_size)
    _maybe(cfg.model, "num_classes",     args.num_classes)
    _maybe(cfg.model, "angle_step",      args.angle_step)
    _maybe(cfg.model, "num_dist_blocks", args.num_dist_blocks)
    _maybe(cfg.model, "min_distance",    args.min_distance)
    _maybe(cfg.model, "max_distance",    args.max_distance)
    # data
    _maybe(cfg.data, "poly_dataset_root", args.poly_dataset_root)
    _maybe(cfg.data, "dist_dataset_root", args.dist_dataset_root)
    _maybe(cfg.data, "img_size",          args.img_size)
    _maybe(cfg.data, "batch_size",        args.batch_size)
    _maybe(cfg.data, "num_workers",       args.num_workers)
    _maybe(cfg.data, "mosaic_prob",       args.mosaic_prob)
    _maybe(cfg.data, "flip_lr_prob",      args.flip_lr_prob)
    _maybe(cfg.data, "hsv_h",            args.hsv_h)
    _maybe(cfg.data, "hsv_s",            args.hsv_s)
    _maybe(cfg.data, "hsv_v",            args.hsv_v)
    # train
    _maybe(cfg.train, "epochs",        args.epochs)
    _maybe(cfg.train, "warmup_epochs", args.warmup_epochs)
    _maybe(cfg.train, "lr0",           args.lr0)
    _maybe(cfg.train, "lrf",           args.lrf)
    _maybe(cfg.train, "momentum",      args.momentum)
    _maybe(cfg.train, "weight_decay",  args.weight_decay)
    _maybe(cfg.train, "device",        args.device)
    _maybe(cfg.train, "save_dir",      args.save_dir)
    _maybe(cfg.train, "val_interval",  args.val_interval)
    _maybe(cfg.train, "conf_thres",    args.conf_thres)
    _maybe(cfg.train, "iou_thres",     args.iou_thres)
    _maybe(cfg.train, "box_gain",        args.box_gain)
    _maybe(cfg.train, "cls_gain",        args.cls_gain)
    _maybe(cfg.train, "dfl_gain",        args.dfl_gain)
    _maybe(cfg.train, "poly_gain",       args.poly_gain)
    _maybe(cfg.train, "dist_gain",       args.dist_gain)
    _maybe(cfg.train, "poly_dist_gain",  args.poly_dist_gain)
    _maybe(cfg.train, "poly_conf_gain",  args.poly_conf_gain)
    _maybe(cfg.train, "poly_angle_gain", args.poly_angle_gain)
    _maybe(cfg.train, "log_interval",    args.log_interval)
    _maybe(cfg.train, "vis_interval",    args.vis_interval)
    _maybe(cfg.train, "vis_max_images",  args.vis_max_images)
    if args.no_tensorboard:
        cfg.train.tensorboard = False


def _maybe(obj, attr: str, val):
    if val is not None:
        setattr(obj, attr, val)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_config(
    yaml_path: str | Path | None = None,
    argv: list[str] | None = None,
    description: str = "YOLOv8-Extended",
    extra_args: list[tuple] | None = None,   # [(flags, kwargs), ...]
) -> tuple[Config, argparse.Namespace]:
    """
    Build the effective Config from defaults → YAML → CLI.

    Returns (cfg, args) so callers can inspect raw args (e.g. --resume).
    """
    # 1. start from defaults
    cfg = Config()
    _apply_yaml(cfg, _load_yaml(_DEFAULT_YAML))

    # 2. build parser and parse CLI
    ap = _build_parser(description)
    if extra_args:
        for flags, kwargs in extra_args:
            ap.add_argument(*flags, **kwargs)
    args = ap.parse_args(argv)

    # 3. user-supplied YAML (--cfg) overrides defaults
    if args.cfg:
        _apply_yaml(cfg, _load_yaml(args.cfg))

    # 4. direct YAML path argument overrides --cfg
    if yaml_path:
        _apply_yaml(cfg, _load_yaml(yaml_path))

    # 5. CLI flags override everything
    _apply_args(cfg, args)

    return cfg, args


def save_config(cfg: Config, path: str | Path):
    _save_yaml(cfg, path)
