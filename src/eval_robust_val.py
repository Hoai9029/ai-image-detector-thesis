import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from src.dataset import CIFAKEDataset
from src.degrade import jpeg_compress, resize_down_up
from src.engine import evaluate
from src.model import build_model
from src.transforms import test_transform
from src.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",
                   default="/kaggle/input/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
    p.add_argument("--baseline_ckpt", required=True)
    p.add_argument("--robust_ckpt", required=True)
    p.add_argument("--out_file", default="/kaggle/working/robust_val_results.json")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42, help="PHẢI bằng seed lúc train")
    return p.parse_args()


def degrade(img, mode, value):
    if mode is None:
        return img
    if mode == "jpeg":                      # PIL - cùng hàm với lúc huấn luyện
        return jpeg_compress(img, value)
    if mode == "resize":                    # bilinear - cùng hàm với lúc huấn luyện
        return resize_down_up(img, value)
    if mode == "resize_nearest":            # kiểu nội suy khác
        return resize_down_up(img, value, down=Image.NEAREST, up=Image.NEAREST)
    if mode == "jpeg_cv2":                  # bộ nén JPEG của thư viện khác
        import cv2
        arr = np.asarray(img)[:, :, ::-1]
        ok, enc = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, int(value)])
        return Image.fromarray(cv2.imdecode(enc, 1)[:, :, ::-1])
    if mode == "resize_then_jpeg":          # thu nhỏ rồi mới nén (thứ tự mạng xã hội hay làm)
        scale, q = value
        return jpeg_compress(resize_down_up(img, scale), q)
    raise ValueError(mode)


class DegradedDataset(Dataset):
    def __init__(self, base, indices, mode, value):
        self.base, self.indices, self.mode, self.value = base, indices, mode, value

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, label = self.base.samples[self.indices[i]]
        img = Image.open(path).convert("RGB")
        img = degrade(img, self.mode, self.value)
        return test_transform(img), label


def load_model(path, device):
    m = build_model("efficientnet_b0", pretrained=False).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval()


def run(model, dataset, device, args):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    r = evaluate(model, loader, device, torch.nn.BCEWithLogitsLoss())
    probs, labels = np.asarray(r["probs"]), np.asarray(r["labels"])
    preds = (probs > 0.5).astype(int)
    real = labels == 0
    return {
        "acc": float(r["acc"]),
        "auc": float(r["auc"]),
        "loss": float(r["loss"]),
        "fpr_real": float((preds[real] == 1).mean()),       # ảnh THẬT bị báo là AI
        "recall_fake": float((preds[~real] == 1).mean()),   # ảnh AI bị bắt
    }


# RobustAugment của bạn huấn luyện với: JPEG quality 30-70, thu nhỏ scale 0.5-0.75
# (mỗi phép áp dụng với xác suất 0.5, thứ tự JPEG rồi resize).
CONDITIONS = [
    # --- nằm TRONG dải đã huấn luyện (resize 0.75 và 0.5 là hai đầu mút của dải) ---
    ("clean", None, None),
    ("jpeg_q70", "jpeg", 70), ("jpeg_q50", "jpeg", 50), ("jpeg_q30", "jpeg", 30),
    ("resize_075", "resize", 0.75), ("resize_050", "resize", 0.50),
    # --- NGOÀI dải (kiểm tra tổng quát hóa) ---
    ("jpeg_q90", "jpeg", 90),                       # nhẹ hơn dải huấn luyện
    ("jpeg_q10", "jpeg", 10),                       # nặng hơn dải huấn luyện
    ("resize_025", "resize", 0.25),                 # nặng hơn dải huấn luyện
    ("resize_050_nearest", "resize_nearest", 0.50), # kiểu nội suy khác
    ("jpeg_cv2_q30", "jpeg_cv2", 30),               # bộ nén JPEG khác
    ("resize050_then_jpeg50", "resize_then_jpeg", (0.50, 50)),  # thứ tự ngoài đời thật
]


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    train_dir = os.path.join(args.data_root, "train")

    base = CIFAKEDataset(train_dir, transform=None)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(base), generator=g).tolist()   # đúng công thức chia của train.py
    val_idx = idx[: int(0.1 * len(idx))]
    print("Validation size:", len(val_idx))

    models = {"baseline": load_model(args.baseline_ckpt, device),
              "robust": load_model(args.robust_ckpt, device)}

    rows = []
    for name, mode, value in CONDITIONS:
        if mode is None:
            ds = Subset(CIFAKEDataset(train_dir, transform=test_transform), val_idx)
        else:
            ds = DegradedDataset(base, val_idx, mode, value)
        row = {"condition": name}
        for mname, m in models.items():
            for k, v in run(m, ds, device, args).items():
                row[f"{mname}_{k}"] = v
        rows.append(row)
        print(row)

    with open(args.out_file, "w") as f:
        json.dump(rows, f, indent=2)
    pd.DataFrame(rows).to_csv(args.out_file.replace(".json", ".csv"), index=False)
    print("Saved:", args.out_file)


if __name__ == "__main__":
    main()