import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=(512, 512), augment=False):
        self.images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')])
        self.masks = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.png')])
        self.img_size = img_size
        self.augment = augment
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=20, p=0.5),
        ]) if augment else None
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]).convert('RGB').resize(self.img_size))
        mask = np.array(Image.open(self.masks[idx]).convert('L').resize(self.img_size))
        if self.augment and self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        img = img.transpose(2, 0, 1) / 255.0  # (C, H, W)
        mask = (mask > 127).astype(np.float32)[None, ...]  # (1, H, W)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32) 