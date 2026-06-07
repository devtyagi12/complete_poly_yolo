"""
Training script for YOLOv8-Extended.

Usage examples
──────────────
  # defaults from configs/default.yaml
  python train.py

  # custom YAML + CLI override
  python train.py --cfg configs/my_exp.yaml --epochs 100 --batch_size 8

  # full CLI (no YAML needed)
  python train.py \\
      --model_size m --num_classes 10 \\
      --poly_dataset_root data/poly --dist_dataset_root data/dist \\
      --epochs 200 --lr0 0.005 --batch_size 16 \\
      --save_dir runs/exp1 --vis_interval 5 --no_tensorboard

  # resume
  python train.py --resume runs/exp1/last.pt
"""
from __future__ import annotations

import copy
import math
import os
import time
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast

from configs.config import Config, load_config, save_config
from data.dataset import build_dataloader
from loss.loss import YOLOv8ExtendedLoss
from models.model import YOLOv8Extended
from postprocess.decode import PostProcessor
from utils.logger import Logger
from utils.metrics import BBoxF1Metric
from utils.visualiser import Visualiser


torch.backends.cudnn.enabled = False

# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, lr0, lrf):
    if epoch < warmup_epochs:
        lr = lr0 * (epoch + 1) / max(warmup_epochs, 1)
    else:
        t = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
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
) -> float:
    model.eval()
    mc = cfg.model
    dc = cfg.data
    tc = cfg.train

    metric = BBoxF1Metric(num_classes=mc.num_classes, iou_thres=0.5)

    # collect one batch for visualisation
    vis_imgs    = None
    vis_targets = None
    vis_dets    = None

    for batch_idx, (imgs, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):
        imgs    = imgs.to(device)
        targets = targets.to(device)
        B       = imgs.shape[0]

        preds      = model(imgs)
        orig_shapes = [(dc.img_size, dc.img_size)] * B
        detections  = post_proc(preds, orig_shapes)

        if vis_imgs is None:
            vis_imgs    = imgs.cpu()
            vis_targets = targets.cpu()
            vis_dets    = detections

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

    result   = metric.compute()
    micro    = result["micro"]
    per_cls  = {k: v for k, v in result.items() if k != "micro"}
    is_best  = False   # caller sets this

    logger.log_val(
        epoch=epoch,
        precision=micro["precision"],
        recall=micro["recall"],
        f1=micro["f1"],
        per_class=result,
        is_best=False,
    )

    # vis panel 2: GT vs pred
    if vis_imgs is not None:
        vis.save_val_predictions(epoch, vis_imgs, vis_targets, vis_dets)

    model.train()
    return micro["f1"], result


# ─────────────────────────────────────────────────────────────────────────────
# Main
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

    device   = torch.device(tc.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(tc.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── logger ────────────────────────────────────────────────────────────────
    logger = Logger(save_dir=save_dir, use_tb=tc.tensorboard)
    logger.log_config(cfg)
    logger.info(f"Device: {device}  |  save_dir: {save_dir}")

    # ── visualiser ────────────────────────────────────────────────────────────
    vis = Visualiser(
        save_dir       = save_dir,
        num_angles     = mc.num_angles,
        img_size       = dc.img_size,
        vis_max_images = tc.vis_max_images,
        conf_thresh    = tc.conf_thres,
        logger         = logger,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = YOLOv8Extended(
        model_size      = mc.model_size,
        num_classes     = mc.num_classes,
        num_angles      = mc.num_angles,
        num_dist_blocks = mc.num_dist_blocks,
        strides         = mc.strides,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model: YOLOv8{mc.model_size}  |  {n_params:.1f}M parameters")

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
    scaler = GradScaler(tc.device, enabled=use_amp)

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema = ModelEMA(model)

    # ── data ──────────────────────────────────────────────────────────────────
    def _loader(split: str, augment: bool):
        return build_dataloader(
            poly_img_dir = os.path.join(dc.poly_dataset_root, f"images/{split}"),
            poly_lbl_dir = os.path.join(dc.poly_dataset_root, f"labels/{split}"),
            dist_img_dir = os.path.join(dc.dist_dataset_root, f"images/{split}"),
            dist_lbl_dir = os.path.join(dc.dist_dataset_root, f"labels/{split}"),
            img_size     = dc.img_size,
            batch_size   = dc.batch_size,
            num_workers  = dc.num_workers,
            angle_step   = mc.angle_step,
            min_dist     = mc.min_distance,
            max_dist     = mc.max_distance,
            augment      = augment,
            hsv_h        = dc.hsv_h,
            hsv_s        = dc.hsv_s,
            hsv_v        = dc.hsv_v,
            flip_lr_prob = dc.flip_lr_prob,
            mosaic_prob  = dc.mosaic_prob,
        )

    train_loader = _loader("train", augment=True)
    val_loader   = _loader("val",   augment=False)
    logger.info(f"Train batches: {len(train_loader)}  |  "
                f"Val batches: {len(val_loader)}")

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
    n_steps = len(train_loader)

    for epoch in range(start_epoch, tc.epochs):
        lr = _cosine_lr(
            optimizer, epoch, tc.epochs, tc.warmup_epochs, tc.lr0, tc.lrf
        )

        epoch_loss            = 0.0
        epoch_loss_dict: dict = {}
        t0                    = time.time()

        # keep one batch for visualisation
        vis_batch_imgs    = None
        vis_batch_targets = None

        for step, (imgs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            imgs    = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # store first batch for vis
            if step == 0 and vis_batch_imgs is None:
                vis_batch_imgs    = imgs.detach().cpu()
                vis_batch_targets = targets.detach().cpu()

            with autocast(tc.device, enabled=use_amp):
                preds = model(imgs)
                loss, loss_dict = criterion(preds, targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # gradient norm
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

            # ── step-level log (every log_interval steps) ─────────────────────
            if step % tc.log_interval == 0:
                logger.log_step(
                    epoch=epoch, step=step, n_steps=n_steps,
                    loss=loss.item(), loss_dict=loss_dict, lr=lr,
                )
                logger.log_grad_norm(grad_norm, global_step)

        # ── epoch summary ─────────────────────────────────────────────────────
        avg_loss = epoch_loss / max(n_steps, 1)
        avg_dict = {k: v / max(n_steps, 1) for k, v in epoch_loss_dict.items()}
        elapsed  = time.time() - t0

        logger.log_epoch(
            epoch=epoch, loss=avg_loss, loss_dict=avg_dict,
            lr=lr, elapsed=elapsed,
        )

        # ── visualisation panel 1: training batch ─────────────────────────────
        if (epoch % tc.vis_interval == 0) and vis_batch_imgs is not None:
            vis.save_train_batch(epoch, vis_batch_imgs, vis_batch_targets)

        # ── loss history (for curves) ─────────────────────────────────────────
        vis.update_loss_history(epoch, avg_loss, avg_dict)

        # ── validation ────────────────────────────────────────────────────────
        if (epoch + 1) % tc.val_interval == 0 or epoch == tc.epochs - 1:
            f1, val_result = validate(
                ema.ema, val_loader, post_proc, cfg, device, logger, vis, epoch
            )

            vis.update_loss_history(epoch, avg_loss, avg_dict, val_f1=f1)

            is_best = f1 > best_f1
            if is_best:
                best_f1 = f1
                logger.log_val(epoch, **{
                    "precision": val_result["micro"]["precision"],
                    "recall":    val_result["micro"]["recall"],
                    "f1":        f1,
                    "per_class": val_result,
                    "is_best":   True,
                })
                _save_ckpt(str(save_dir / "best.pt"),
                           model, optimizer, scaler, epoch, best_f1, logger)

            _save_ckpt(str(save_dir / "last.pt"),
                       model, optimizer, scaler, epoch, best_f1, logger)

        # ── loss curves (every vis_interval epochs) ───────────────────────────
        if epoch % tc.vis_interval == 0:
            vis.save_loss_curves(epoch)

    # ── done ──────────────────────────────────────────────────────────────────
    logger.info(f"\nBest F1 = {best_f1:.4f}")
    logger.close()


if __name__ == "__main__":
    train()