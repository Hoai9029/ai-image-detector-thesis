import os
from torch.utils.data import Dataset
from PIL import Image


class CIFAKEDataset(Dataset):
    """Dataset đọc ảnh CIFAKE từ thư mục REAL/FAKE."""

    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        # 0 = REAL, 1 = FAKE
        for label, cls in enumerate(["REAL", "FAKE"]):
            cls_dir = os.path.join(root_dir, cls)

            for fname in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label