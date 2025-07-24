import cv2
import numpy as np
from PIL import Image
from src.segmenter.segmenter import Segmenter

video_path = "input/video.mp4"  # Đường dẫn video gốc
output_video_path = "output/results/overlay_mask_video.mp4"  # Đường dẫn video overlay đầu ra
model_path = "output/models/segmenter_unet.h5"
img_size = (512, 512)

def main():
    segmenter = Segmenter(img_size=img_size)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Dự đoán mask trực tiếp từ frame (numpy array BGR)
        mask_img = segmenter.predict_from_array(frame, model_path=model_path)
        mask_np = np.array(mask_img)
        # Resize mask về kích thước gốc
        mask_resized = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
        # Tạo mask màu đỏ overlay lên frame gốc
        mask_color = np.zeros_like(frame)
        mask_color[..., 2] = mask_resized  # Kênh đỏ
        overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)
        out.write(overlay)
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Đã xử lý {frame_idx}/{frame_count} frames...")
    cap.release()
    out.release()
    print("Đã tạo xong video overlay mask:", output_video_path)

if __name__ == "__main__":
    main() 