import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from torch.utils.data import DataLoader

from src.dataset import CIFAKEDataset
from src.engine import evaluate
from src.model import build_model
from src.transforms import test_transform
from src.utils import get_device


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data_root",
        default="/kaggle/input/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images"
    )

    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="efficientnet_b0")
    p.add_argument(
        "--out_dir",
        default="/kaggle/working/eval_baseline"
    )

    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = get_device()

    print("Device:", device)
    print("Checkpoint:", args.ckpt)

    test_ds = CIFAKEDataset(
        os.path.join(args.data_root, "test"),
        transform=test_transform
    )

    print("Số ảnh test:", len(test_ds))

    loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = build_model(
        args.model,
        pretrained=False
    ).to(device)

    model.load_state_dict(
        torch.load(
            args.ckpt,
            map_location=device
        )
    )

    out = evaluate(
        model,
        loader,
        device
    )

    probs = out["probs"]
    labels = out["labels"]

    preds = (probs > 0.5).astype(int)

    cm = confusion_matrix(
        labels,
        preds
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "n_test": int(len(labels)),
        "accuracy": float((preds == labels).mean()),
        "precision_fake": float(
            precision_score(labels, preds)
        ),
        "recall_fake": float(
            recall_score(labels, preds)
        ),
        "f1_fake": float(
            f1_score(labels, preds)
        ),
        "auc": float(
            roc_auc_score(labels, probs)
        ),
        "false_positive_rate_real": float(
            fp / (fp + tn)
        ),
        "false_negative_rate_fake": float(
            fn / (fn + tp)
        ),
        "confusion_matrix": cm.tolist()
    }

    print("\n===== TEST RESULTS =====")
    print(json.dumps(
        metrics,
        indent=2
    ))

    with open(
        os.path.join(
            args.out_dir,
            "test_metrics.json"
        ),
        "w"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2
        )

    # Confusion Matrix
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["REAL", "FAKE"]
    ).plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Confusion Matrix - Test")
    plt.savefig(
        os.path.join(
            args.out_dir,
            "confusion_matrix.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    # ROC
    RocCurveDisplay.from_predictions(
        labels,
        probs
    )

    plt.title("ROC Curve - Test")
    plt.savefig(
        os.path.join(
            args.out_dir,
            "roc_curve.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    # Misclassified images
    wrong = np.where(
        preds != labels
    )[0][:16]

    fig, axes = plt.subplots(
        2,
        8,
        figsize=(16, 4.5)
    )

    for ax in axes.ravel():
        ax.axis("off")

    for ax, i in zip(
        axes.ravel(),
        wrong
    ):
        path, lab = test_ds.samples[i]

        ax.imshow(
            Image.open(path).convert("RGB")
        )

        ax.set_title(
            f"true={'FAKE' if lab else 'REAL'}\n"
            f"p(fake)={probs[i]:.2f}",
            fontsize=8
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            args.out_dir,
            "misclassified.png"
        ),
        dpi=150
    )

    plt.close()

    print("\nĐã lưu kết quả vào:")
    print(args.out_dir)


if __name__ == "__main__":
    main()