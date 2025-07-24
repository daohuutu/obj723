import os
import torch
from torch.utils.data import DataLoader
from .unet import UNet
from .dataset import SegmentationDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import cv2
import datetime

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth
    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1 - ((2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth))

def train_segmentation(
    images_train, masks_train, images_val, masks_val,
    img_size=(512, 512), batch_size=8, epochs=20, lr=1e-3, save_dir='output/models', results_base_dir='output/results', device=None, augment=True, loss_type='dice'
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_base_dir, exist_ok=True)
    model = UNet().to(device)
    train_ds = SegmentationDataset(images_train, masks_train, img_size, augment=augment)
    val_ds = SegmentationDataset(images_val, masks_val, img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    if loss_type == 'dice':
        criterion = DiceLoss()
    else:
        criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'unet_best.pth'))
    # Save val predictions for this run
    run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    results_dir = os.path.join(results_base_dir, run_name)
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, (img, _) in enumerate(val_loader):
            img = img.to(device)
            pred = model(img)[0,0].cpu().numpy()
            pred_mask = (pred > 0.5).astype(np.uint8) * 255
            # Lưu mask
            Image.fromarray(pred_mask).save(os.path.join(results_dir, f'val_{i:04d}_pred.png'))
            # Overlay mask lên ảnh gốc
            orig_img = np.array(Image.open(val_ds.images[i]).convert('RGB').resize(img_size))
            mask_color = np.zeros_like(orig_img)
            mask_color[..., 0] = pred_mask
            overlay = cv2.addWeighted(orig_img, 0.7, mask_color, 0.3, 0)
            Image.fromarray(overlay).save(os.path.join(results_dir, f'val_{i:04d}_overlay.png'))
    print(f"Kết quả val đã lưu vào: {results_dir}")
    return model, results_dir 