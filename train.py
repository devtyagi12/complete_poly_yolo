"""
Training script for YOLOv8-Extended.

Usage examples
──────────────
  # single polygon dataset
  python train.py --poly_dataset_root data/polygon --num_classes 10

  # multiple polygon datasets with weights + optional distance
  python train.py \\
      --poly_datasets data/poly1/images/train:data/poly1/labels/train \\
                      data/poly2/images/train:data/poly2/labels/train \\
      --poly_weights 2.0 1.0 \\
      --dist_dataset_root data/polygon_distance \\
      --num_classes 10

  # custom YAML + CLI override
  python train.py --cfg configs/my_exp.yaml --epochs 100 --batch_size 8

  # resume
  python train.py --resume runs/train/exp1/last.pt

  # AMD GPU (GroupNorm + no AMP)
  python train.py --gpu_type amd --no_amp
"""
from __future__ import annotations

import copy
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from configs.config import Config, load_config, save_config
from data.dataset import build_dataloader
from loss.loss import YOLOv8ExtendedLoss
from models.model import YOLOv8Extended
from postprocess.decode import PostProcessor
from utils.logger import Logger
from utils.metrics import BBoxF1Metric
from utils.visualiser import Visualiser


# ─────────────────────────────────────────────────────────────────────────────
# Save-dir: auto-increment to avoid overwriting previous runs
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_save_dir(base: str, resume_path: Optional[str]) -> Path:
    """
    If resume_path is given return the directory that checkpoint lives in.
    Otherwise, find the next available numbered sub-directory:
        runs/train → runs/train/exp1  (if exp1 exists → exp2, exp3, …)
    """
    if resume_path:
        return Path(resume_path).parent

    base_path = Path(base)
    for i in range(1, 10_000):
        candidate = base_path / f"exp{i}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"Could not find a free experiment directory under {base}")


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, lr0, lrf):
    if epoch < warmup_epochs:
        lr = lr0 * (epoch + 1) / max(warmup_epochs, 1)
    else:
        t  = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = lrf + 0.5 * (lr0 - lrf) * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.ema     = copy.deepcopy(model)
        self.decay   = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.updates += 1
        d   = self.decay * (1 - math.exp(-self.updates / 2000))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_((1 - d) * msd[k].detach())


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_ckpt(path, model, optimizer, scaler, epoch, best_f1, logger):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
        "best_f1":   best_f1,
    }, path)
    logger.info(f"  ✓ checkpoint → {path}")


def _load_ckpt(path, model, optimizer, scaler, device, logger):
    logger.info(f"Resuming from {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("best_f1", 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_loader(dc, mc, split: str, augment: bool) -> "DataLoader":
    """
    Build a DataLoader for one split using the DataConfig.
    Supports single dataset root, multi-dataset list, and optional distance.
    """
    # ── resolve polygon datasets ──────────────────────────────────────────────
    if dc.poly_datasets:
        # explicit list of img:lbl pairs (already resolved by config)
        poly_ds = []
        for entry in dc.poly_datasets:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                img_d, lbl_d = entry
            elif isinstance(entry, str) and ":" in entry:
                img_d, lbl_d = entry.split(":", 1)
            else:
                continue
            # allow {split} placeholder
            poly_ds.append((
                img_d.replace("{split}", split),
                lbl_d.replace("{split}", split),
            ))
        weights = list(dc.poly_weights) if dc.poly_weights else None
    else:
        # single root
        poly_ds = [(
            os.path.join(dc.poly_dataset_root, f"images/{split}"),
            os.path.join(dc.poly_dataset_root, f"labels/{split}"),
        )]
        weights = None

    # ── resolve distance dataset ──────────────────────────────────────────────
    dist_img = dist_lbl = None
    if dc.dist_dataset_root:
        dist_img = os.path.join(dc.dist_dataset_root, f"images/{split}")
        dist_lbl = os.path.join(dc.dist_dataset_root, f"labels/{split}")

    return build_dataloader(
        poly_datasets = poly_ds,
        poly_weights  = weights,
        dist_img_dir  = dist_img,
        dist_lbl_dir  = dist_lbl,
        img_size      = dc.img_size,
        batch_size    = dc.batch_size,
        num_workers   = dc.num_workers,
        angle_step    = mc.angle_step,
        min_dist      = mc.min_distance,
        max_dist      = mc.max_distance,
        augment       = augment,
        hsv_h         = dc.hsv_h,
        hsv_s         = dc.hsv_s,
        hsv_v         = dc.hsv_v,
        flip_lr_prob  = dc.flip_lr_prob,
        mosaic_prob   = dc.mosaic_prob,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model,
    val_loader,
    post_proc,
    cfg: Config,
    device,
    logger: Logger,
    vis: Visualiser,
    epoch: int,
    n_epochs: int = 0,
) -> Tuple[float, float, float, dict]:
    model.eval()
    mc = cfg.model
    dc = cfg.data
    tc = cfg.train

    metric = BBoxF1Metric(num_classes=mc.num_classes, iou_thres=0.5)

    vis_imgs_list:    List[torch.Tensor] = []
    vis_targets_list: List[torch.Tensor] = []
    vis_dets_list:    list               = []
    vis_collected = 0

    for imgs, targets in val_loader:
        imgs    = imgs.to(device)
        targets = targets.to(device)
        B       = imgs.shape[0]

        preds       = model(imgs)
        orig_shapes = [(dc.img_size, dc.img_size)] * B
        detections  = post_proc(preds, orig_shapes)

        # collect images for visualisation
        if vis_collected < tc.vis_max_images:
            take = min(B, tc.vis_max_images - vis_collected)
            bt   = targets[targets[:, 0] < take].cpu().clone()
            bt[:, 0] += vis_collected          # rebase batch-idx for concat
            vis_imgs_list.append(imgs[:take].cpu())
            vis_targets_list.append(bt)
            vis_dets_list.extend(detections[:take])
            vis_collected += take

        for bi in range(B):
            mask = targets[:, 0] == bi
            gts  = []
            for row in targets[mask]:
                cls = int(row[1].item())
                cx  = row[2].item() * dc.img_size
                cy  = row[3].item() * dc.img_size
                w_  = row[4].item() * dc.img_size
                h_  = row[5].item() * dc.img_size
                gts.append((cls, cx - w_/2, cy - h_/2, cx + w_/2, cy + h_/2))
            metric.update(detections[bi], gts)

    result = metric.compute()
    micro  = result["micro"]

    logger.log_val(
        epoch=epoch, n_epochs=n_epochs,
        precision=micro["precision"],
        recall=micro["recall"],
        f1=micro["f1"],
        per_class=result,
        is_best=False,
    )

    if vis_imgs_list:
        all_imgs    = torch.cat(vis_imgs_list,    dim=0)
        all_targets = torch.cat(vis_targets_list, dim=0)
        vis.save_val_predictions(epoch, all_imgs, all_targets, vis_dets_list)

    model.train()
    return micro["f1"], micro["precision"], micro["recall"], result


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Config | None = None):
    if cfg is None:
        cfg, args = load_config(
            description="YOLOv8-Extended training",
            extra_args=[(
                ["--no_amp"],
                dict(action="store_true",
                     help="Disable AMP (GradScaler + autocast). "
                          "Required for ROCm on Windows / unsupported GPUs."),
            )],
        )
        resume_path = getattr(args, "resume", None)
        use_amp     = not args.no_amp
    else:
        resume_path = None
        use_amp     = True

    mc = cfg.model
    dc = cfg.data
    tc = cfg.train

    # ── resolve experiment directory (auto-increment) ─────────────────────────
    save_dir = _resolve_save_dir(tc.save_dir, resume_path)
    # write back so logger and checkpoints use the resolved path
    tc.save_dir = str(save_dir)

    device = torch.device(tc.device if torch.cuda.is_available() else "cpu")

    # ── logger ────────────────────────────────────────────────────────────────
    logger = Logger(save_dir=save_dir, use_tb=tc.tensorboard)
    logger.log_config(cfg)
    logger.info(f"Experiment → {save_dir}")
    logger.info(f"Device     → {device}")

    # ── visualiser ────────────────────────────────────────────────────────────
    class_names = dc.class_names if dc.class_names else None
    vis = Visualiser(
        save_dir       = save_dir,
        num_angles     = mc.num_angles,
        angle_step     = mc.angle_step,
        img_size       = dc.img_size,
        vis_max_images = tc.vis_max_images,
        conf_thresh    = tc.conf_thres,
        class_names    = class_names,
        logger         = logger,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = YOLOv8Extended(
        model_size      = mc.model_size,
        num_classes     = mc.num_classes,
        num_angles      = mc.num_angles,
        num_dist_blocks = mc.num_dist_blocks,
        strides         = mc.strides,
        gpu_type        = mc.gpu_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model      → YOLOv8{mc.model_size}  |  {n_params:.1f}M parameters")

    # ── pretrained backbone/neck weights ──────────────────────────────────────
    if mc.pretrained and resume_path is None:
        logger.info("Loading pretrained YOLOv8 backbone + neck weights …")
        try:
            stats = model.load_pretrained_backbone(mc.model_size)
            logger.info(
                f"  Pretrained weights loaded  "
                f"matched={stats['matched']}  "
                f"skipped_shape={stats['skipped_shape']}  "
                f"skipped_name={stats['skipped_name']}  "
                f"total_dst={stats['total_dst']}"
            )
            if stats["matched"] == 0:
                logger.warning(
                    "No weights were matched. The backbone/neck may have "
                    "incompatible channel widths.  Training from scratch."
                )
        except Exception as e:
            logger.warning(
                f"Could not load pretrained weights ({e}). "
                f"Training from scratch.  Pass --no_pretrained to suppress."
            )

    # ── loss ──────────────────────────────────────────────────────────────────
    criterion = YOLOv8ExtendedLoss(
        num_classes     = mc.num_classes,
        num_angles      = mc.num_angles,
        angle_step      = mc.angle_step,
        strides         = mc.strides,
        img_size        = dc.img_size,
        box_gain        = tc.box_gain,
        cls_gain        = tc.cls_gain,
        dfl_gain        = tc.dfl_gain,
        poly_gain       = tc.poly_gain,
        dist_gain       = tc.dist_gain,
        poly_dist_gain  = tc.poly_dist_gain,
        poly_conf_gain  = tc.poly_conf_gain,
        poly_angle_gain = tc.poly_angle_gain,
    ).to(device)

    # ── optimiser ─────────────────────────────────────────────────────────────
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if "bn" not in n and p.requires_grad],
         "weight_decay": tc.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if "bn" in n and p.requires_grad],
         "weight_decay": 0.0},
    ]
    optimizer = optim.SGD(
        param_groups, lr=tc.lr0, momentum=tc.momentum, nesterov=True
    )
    scaler = GradScaler(enabled=use_amp)

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema = ModelEMA(model)

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader = _build_loader(dc, mc, "train", augment=True)
    val_loader   = _build_loader(dc, mc, "val",   augment=False)

    n_poly_ds = len(dc.poly_datasets) if dc.poly_datasets else 1
    dist_info = (f"+ dist dataset" if dc.dist_dataset_root else "no dist dataset")
    logger.info(
        f"Data       → {n_poly_ds} polygon dataset(s)  {dist_info}  "
        f"| train={len(train_loader)} batches  val={len(val_loader)} batches"
    )

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

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_f1     = 0.0
    last_ckpt   = save_dir / "last.pt"
    if resume_path and Path(resume_path).exists():
        start_epoch, best_f1 = _load_ckpt(
            resume_path, model, optimizer, scaler, device, logger
        )
        start_epoch += 1
    elif last_ckpt.exists():
        start_epoch, best_f1 = _load_ckpt(
            str(last_ckpt), model, optimizer, scaler, device, logger
        )
        start_epoch += 1

    # ── training loop ─────────────────────────────────────────────────────────
    model.train()
    n_steps  = len(train_loader)
    n_epochs = tc.epochs

    for epoch in range(start_epoch, n_epochs):
        lr = _cosine_lr(
            optimizer, epoch, n_epochs, tc.warmup_epochs, tc.lr0, tc.lrf
        )

        epoch_loss:       float = 0.0
        epoch_loss_dict:  dict  = {}
        epoch_instances:  int   = 0
        t0 = time.time()

        vis_batch_imgs    = None
        vis_batch_targets = None

        logger._print_epoch_header(epoch, n_epochs, n_steps, dc.img_size)

        for step, (imgs, targets) in enumerate(train_loader):
            imgs    = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            n_instances      = targets.shape[0]
            epoch_instances += n_instances

            if step == 0 and vis_batch_imgs is None:
                vis_batch_imgs    = imgs.detach().cpu()
                vis_batch_targets = targets.detach().cpu()

            with autocast(enabled=use_amp):
                preds           = model(imgs)
                loss, loss_dict = criterion(preds, targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0
            ).item()

            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            epoch_loss += loss.item()
            for k, v in loss_dict.items():
                epoch_loss_dict[k] = epoch_loss_dict.get(k, 0.0) + v

            global_step = epoch * n_steps + step

            logger.log_step(
                epoch=epoch,   n_epochs=n_epochs,
                step=step,     n_steps=n_steps,
                loss=loss.item(), loss_dict=loss_dict, lr=lr,
                n_instances=n_instances, img_size=dc.img_size,
            )
            if step % tc.log_interval == 0:
                logger.log_grad_norm(grad_norm, global_step)

        # ── epoch summary ─────────────────────────────────────────────────────
        avg_loss = epoch_loss / max(n_steps, 1)
        avg_dict = {k: v / max(n_steps, 1) for k, v in epoch_loss_dict.items()}
        elapsed  = time.time() - t0

        logger.log_epoch(
            epoch=epoch, n_epochs=n_epochs,
            loss=avg_loss, loss_dict=avg_dict,
            lr=lr, elapsed=elapsed,
            n_instances=epoch_instances // max(n_steps, 1),
            img_size=dc.img_size,
        )

        # ── vis: training batch ────────────────────────────────────────────────
        if (epoch % tc.vis_interval == 0) and vis_batch_imgs is not None:
            vis.save_train_batch(epoch, vis_batch_imgs, vis_batch_targets)

        vis.update_loss_history(epoch, avg_loss, avg_dict)

        # ── validation ────────────────────────────────────────────────────────
        if (epoch + 1) % tc.val_interval == 0 or epoch == n_epochs - 1:
            f1, prec, rec, val_result = validate(
                ema.ema, val_loader, post_proc, cfg, device,
                logger, vis, epoch, n_epochs=n_epochs,
            )

            vis.update_loss_history(
                epoch, avg_loss, avg_dict,
                val_f1=f1, val_prec=prec, val_rec=rec,
            )

            is_best = f1 > best_f1
            if is_best:
                best_f1 = f1
                logger.log_val(
                    epoch=epoch, n_epochs=n_epochs,
                    precision=prec, recall=rec, f1=f1,
                    per_class=val_result, is_best=True,
                )
                _save_ckpt(str(save_dir / "best.pt"),
                           model, optimizer, scaler, epoch, best_f1, logger)

            _save_ckpt(str(save_dir / "last.pt"),
                       model, optimizer, scaler, epoch, best_f1, logger)

        if epoch % tc.vis_interval == 0:
            vis.save_loss_curves(epoch)

    logger.info(f"\nBest F1 = {best_f1:.4f}  |  saved to {save_dir}")
    logger.close()


if __name__ == "__main__":
    train()
