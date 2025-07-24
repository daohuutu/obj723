import torch
import numpy as np
from PIL import Image
import cv2
import os
import config
from .unet import UNet

def get_all_bounding_boxes(mask, min_area=None):
    min_area = min_area if min_area is not None else config.MIN_BOX_AREA
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > min_area:
            boxes.append((x, y, x+w, y+h))
    return boxes

def draw_boxes_on_image(image, boxes, color=None, thickness=None):
    color = color if color is not None else config.BOX_COLOR
    thickness = thickness if thickness is not None else config.BOX_THICKNESS
    img = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

def predict_mask(model_path, image_path, out_mask_path=None, out_overlay_path=None, out_box_path=None, img_size=None, device=None, box=False, mask_threshold=None, min_box_area=None, box_color=None, box_thickness=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    img_size = img_size if img_size is not None else config.IMG_SIZE
    img = Image.open(image_path).convert('RGB').resize(img_size)
    img_np = np.array(img).transpose(2, 0, 1) / 255.0
    x = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)[0,0].cpu().numpy()
    threshold = mask_threshold if mask_threshold is not None else config.MASK_THRESHOLD
    mask = (pred > threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask)
    boxes = get_all_bounding_boxes(mask, min_area=min_box_area) if box or out_box_path else []
    if out_mask_path:
        mask_img.save(out_mask_path)
    if out_overlay_path:
        orig_img = np.array(img)
        mask_color = np.zeros_like(orig_img)
        mask_color[..., 0] = mask
        overlay = cv2.addWeighted(orig_img, 0.7, mask_color, 0.3, 0)
        if box and boxes:
            overlay = draw_boxes_on_image(overlay, boxes, color=box_color, thickness=box_thickness)
        Image.fromarray(overlay).save(out_overlay_path)
    if out_box_path and boxes:
        with open(out_box_path, 'w') as f:
            for b in boxes:
                f.write(f"{b[0]},{b[1]},{b[2]},{b[3]}\n")
    return mask_img, boxes 