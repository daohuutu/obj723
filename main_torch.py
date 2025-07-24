import os
import config
from src.torch_segmenter.train import train_segmentation
from src.torch_segmenter.predict import predict_mask

if __name__ == "__main__":
    model, val_results_dir = train_segmentation(
        config.IMAGES_TRAIN, config.MASKS_TRAIN, config.IMAGES_VAL, config.MASKS_VAL,
        img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS,
        save_dir=os.path.dirname(config.TORCH_MODEL_PATH),
        results_base_dir=config.RESULTS_DIR, augment=True
    )

    val_images = [f for f in os.listdir(config.IMAGES_VAL) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if val_images:
        test_image = os.path.join(config.IMAGES_VAL, val_images[0])
        out_mask = os.path.join(config.RESULTS_DIR, "test_pred_mask.png")
        out_overlay = os.path.join(config.RESULTS_DIR, "test_pred_overlay.png")
        out_box = os.path.join(config.RESULTS_DIR, "test_pred_box.txt")
        predict_mask(
            model_path=config.TORCH_MODEL_PATH,
            image_path=test_image,
            out_mask_path=out_mask,
            out_overlay_path=out_overlay,
            out_box_path=out_box,
            img_size=config.IMG_SIZE,
            box=True,
            mask_threshold=config.MASK_THRESHOLD,
            min_box_area=config.MIN_BOX_AREA,
            box_color=config.BOX_COLOR,
            box_thickness=config.BOX_THICKNESS
        )
        print(f"Dự đoán mask cho ảnh: {test_image} -> {out_mask}, overlay: {out_overlay}")
    else:
        print("Không tìm thấy ảnh nào trong thư mục val để dự đoán.") 