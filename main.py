from src.segmenter.segmenter import Segmenter
import os
import config

if __name__ == "__main__":
    segmenter = Segmenter(img_size=config.IMG_SIZE)

    # 1. Chuyển đổi txt polygon sang mask PNG cho train và val
    segmenter.convert_txt_to_mask(config.IMAGES_TRAIN, config.LABELS_TRAIN, config.MASKS_TRAIN)
    segmenter.convert_txt_to_mask(config.IMAGES_VAL, config.LABELS_VAL, config.MASKS_VAL)

    # 2. Train model với val riêng
    results_dir = segmenter.train(
        images_dir=config.IMAGES_TRAIN,
        masks_dir=config.MASKS_TRAIN,
        val_images_dir=config.IMAGES_VAL,
        val_masks_dir=config.MASKS_VAL,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        save_path=config.TF_MODEL_PATH,
        results_base_dir=config.RESULTS_DIR  # Tự động tạo thư mục run mới
    )
    print(f"Kết quả val đã lưu vào: {results_dir}")

    # 3. Dự đoán mask cho 1 ảnh val bất kỳ
    val_images = [f for f in os.listdir(config.IMAGES_VAL) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if val_images:
        test_image = os.path.join(config.IMAGES_VAL, val_images[0])  # Lấy ảnh đầu tiên (hoặc random.choice(val_images))
        pred_mask_out = os.path.join(config.RESULTS_DIR, f"{os.path.splitext(val_images[0])[0]}_mask.png")
        segmenter.predict(test_image, model_path=config.TF_MODEL_PATH, out_path=pred_mask_out)
        print(f"Dự đoán mask cho ảnh: {test_image} -> {pred_mask_out}")
    else:
        print("Không tìm thấy ảnh nào trong thư mục val để dự đoán.") 