import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.augment import RobustAugment
from src.dataset import CIFAKEDataset
from src.engine import evaluate, train_one_epoch
from src.model import build_model
from src.transforms import test_transform, train_transform
from src.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data_root",
        default="/kaggle/input/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images"
    )

    p.add_argument("--model", default="efficientnet_b0")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument(
        "--subset",
        type=int,
        default=0,
        help="0 = dùng toàn bộ train; N = chỉ dùng N ảnh để chạy thử"
    )

    p.add_argument("--aug", choices=["none", "jpeg", "resize", "both"], default="none")
    p.add_argument("--p_jpeg", type=float, default=0.5)
    p.add_argument("--p_resize", type=float, default=0.5)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="/kaggle/working/run_baseline")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run_name", default="run")
    p.add_argument("--project", default="ai-image-detector-thesis")

    return p.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    device = get_device()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Device:", device)

    # --- Augmentation ---
    robust_aug = None
    if args.aug == "none":
        train_transform_with_aug = train_transform
    else:
        pj = args.p_jpeg if args.aug in ("jpeg", "both") else 0.0
        pr = args.p_resize if args.aug in ("resize", "both") else 0.0
        robust_aug = RobustAugment(jpeg_prob=pj, resize_prob=pr)

        def train_transform_with_aug(img):
            return train_transform(robust_aug(img))

    # --- Ghi cấu hình (kèm tham số augmentation) ngay từ đầu ---
    cfg = vars(args).copy()
    if robust_aug is not None:
        cfg["aug_params"] = {
            "jpeg_prob": robust_aug.jpeg_prob,
            "resize_prob": robust_aug.resize_prob,
            "jpeg_quality_range": list(robust_aug.jpeg_quality_range),
            "resize_scale_range": list(robust_aug.resize_scale_range),
        }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    if args.wandb:
        import wandb
        wandb.init(project=args.project, name=args.run_name, config=cfg)

    train_dir = os.path.join(args.data_root, "train")

    ds_train = CIFAKEDataset(train_dir, transform=train_transform_with_aug)
    ds_val = CIFAKEDataset(train_dir, transform=test_transform)

    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds_train), generator=g).tolist()
    n_val = int(0.1 * len(idx))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    if args.subset > 0:
        train_idx = train_idx[:args.subset]
        val_idx = val_idx[:max(args.subset // 5, 200)]

    train_set = Subset(ds_train, train_idx)
    val_set = Subset(ds_val, val_idx)

    print(f"train={len(train_set)}  val={len(val_set)}")

    kw = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kw
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size * 2, shuffle=False, **kw
    )

    model = build_model(args.model, pretrained=True).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_steps = args.epochs * len(train_loader)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = []
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, criterion
        )

        val = evaluate(model, val_loader, device, criterion)

        row = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "val_loss": round(val["loss"], 4),
            "val_acc": round(val["acc"], 4),
            "val_auc": round(float(val["auc"]), 4),
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": round(time.time() - t0, 1),
        }

        history.append(row)
        print(row)

        if args.wandb:
            wandb.log(row)

        torch.save(model.state_dict(), os.path.join(args.out_dir, "last.pth"))

        if val["acc"] > best_acc:
            best_acc = val["acc"]
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))

        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"Xong. Best val_acc = {best_acc:.4f}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()