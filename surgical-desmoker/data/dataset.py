from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class CleanFrameDataset(Dataset):
    def __init__(self, clean_dir, size=(512, 640)):
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.files = sorted(p for p in Path(clean_dir).rglob("*") if p.suffix.lower() in exts)
        assert len(self.files) > 0, f"No clean images in {clean_dir}"
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with Image.open(self.files[idx]) as img:
            img = img.convert("RGB").resize(self.size[::-1], Image.Resampling.BILINEAR)
            clean = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255.0)

        if random.random() < 0.5:
            clean = clean.flip(-1)
        return clean
