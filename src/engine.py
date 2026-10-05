import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def _autocast(device):
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(device.type == "cuda")
    )


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, criterion):
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with _autocast(device):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += ((logits > 0).float() == labels).sum().item()
        n += bs

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()

    all_logits, all_labels = [], []
    total_loss, n = 0.0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        with _autocast(device):
            logits = model(images).squeeze(1)

        logits = logits.float().cpu()

        if criterion is not None:
            total_loss += criterion(
                logits,
                labels.float()
            ).item() * labels.size(0)

        n += labels.size(0)

        all_logits.append(logits)
        all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    probs = torch.sigmoid(logits)

    acc = ((probs > 0.5).long() == labels).float().mean().item()
    auc = roc_auc_score(
        labels.numpy(),
        probs.numpy()
    )

    return {
        "loss": (total_loss / n) if criterion is not None else None,
        "acc": acc,
        "auc": auc,
        "probs": probs.numpy(),
        "labels": labels.numpy(),
    }