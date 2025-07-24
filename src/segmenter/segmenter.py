import os
import glob
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import cv2
import config

class Segmenter:
    """
    Module segmentation tích hợp:
    - Chuyển đổi mask YOLO txt (polygon) sang mask PNG nhị phân
    - Train segmentation (U-Net, TensorFlow)
    - Predict mask từ ảnh
    """
    def __init__(self, img_size=(512, 512)):
        self.img_size = img_size
        self.model = None

    def convert_txt_to_mask(self, images_dir, labels_dir, masks_dir, img_ext='.png'):
        """
        Chuyển đổi tất cả file .txt (YOLO polygon) sang mask nhị phân PNG.
        images_dir: thư mục chứa ảnh gốc
        labels_dir: thư mục chứa file txt mask
        masks_dir: thư mục output mask PNG
        img_ext: đuôi ảnh gốc (mặc định .png)
        """
        os.makedirs(masks_dir, exist_ok=True)
        img_files = glob.glob(os.path.join(images_dir, f'*{img_ext}'))
        for img_path in img_files:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, img_name + '.txt')
            mask_path = os.path.join(masks_dir, img_name + '.png')
            img = Image.open(img_path)
            w, h = img.size
            mask = Image.new('L', (w, h), 0)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 7:  # class + at least 3 points
                            continue
                        # Bỏ class_id, lấy các cặp toạ độ
                        points = [float(x) for x in parts[1:]]
                        xy = [(points[i]*w, points[i+1]*h) for i in range(0, len(points), 2)]
                        ImageDraw.Draw(mask).polygon(xy, outline=255, fill=255)
            mask.save(mask_path)

    def build_unet(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        # Encoder
        c1 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
        c1 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D()(c1)
        c2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
        c2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D()(c2)
        c3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
        c3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D()(c3)
        # Bottleneck
        bn = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p3)
        bn = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(bn)
        # Decoder
        u1 = tf.keras.layers.UpSampling2D()(bn)
        u1 = tf.keras.layers.Concatenate()([u1, c3])
        c4 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
        c4 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c4)
        u2 = tf.keras.layers.UpSampling2D()(c4)
        u2 = tf.keras.layers.Concatenate()([u2, c2])
        c5 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(u2)
        c5 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c5)
        u3 = tf.keras.layers.UpSampling2D()(c5)
        u3 = tf.keras.layers.Concatenate()([u3, c1])
        c6 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(u3)
        c6 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(c6)
        outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c6)
        model = tf.keras.Model(inputs, outputs)
        return model

    def train(self, images_dir, masks_dir, epochs=20, batch_size=8, val_split=0.1, save_path='segmenter_unet.h5', val_images_dir=None, val_masks_dir=None, results_base_dir=None, augment=True):
        """
        Train segmentation model (U-Net) với ảnh và mask PNG.
        images_dir: thư mục ảnh gốc train
        masks_dir: thư mục mask PNG train
        val_images_dir: thư mục ảnh gốc val (nếu có)
        val_masks_dir: thư mục mask PNG val (nếu có)
        results_base_dir: nếu truyền, sẽ tự động lưu kết quả val (mask, overlay) vào thư mục run mới sau khi train
        augment: nếu True, áp dụng augmentation động cho batch train
        Nếu không truyền val_images_dir/val_masks_dir sẽ tự chia theo val_split.
        Trả về đường dẫn thư mục kết quả val nếu có, ngược lại trả None.
        """
        import os
        import albumentations as A
        img_paths = sorted(glob.glob(os.path.join(images_dir, '*.png')))
        mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.png')))
        X = []
        Y = []
        for img_path, mask_path in zip(img_paths, mask_paths):
            img = Image.open(img_path).convert('RGB').resize(self.img_size)
            mask = Image.open(mask_path).convert('L').resize(self.img_size)
            X.append(np.array(img)/255.0)
            Y.append(np.expand_dims(np.array(mask)/255.0, axis=-1))
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        if val_images_dir and val_masks_dir:
            val_img_paths = sorted(glob.glob(os.path.join(val_images_dir, '*.png')))
            val_mask_paths = sorted(glob.glob(os.path.join(val_masks_dir, '*.png')))
            X_val = []
            Y_val = []
            for img_path, mask_path in zip(val_img_paths, val_mask_paths):
                img = Image.open(img_path).convert('RGB').resize(self.img_size)
                mask = Image.open(mask_path).convert('L').resize(self.img_size)
                X_val.append(np.array(img)/255.0)
                Y_val.append(np.expand_dims(np.array(mask)/255.0, axis=-1))
            X_val = np.array(X_val, dtype=np.float32)
            Y_val = np.array(Y_val, dtype=np.float32)
            X_train, Y_train = X, Y
        else:
            # Shuffle
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            X, Y = X[idx], Y[idx]
            # Split
            split = int(len(X)*(1-val_split))
            X_train, X_val = X[:split], X[split:]
            Y_train, Y_val = Y[:split], Y[split:]
        # Augmentation pipeline
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=20, p=0.5),
            # Thêm các augmentation khác nếu muốn
        ])
        def aug_generator(X, Y, batch_size):
            idxs = np.arange(len(X))
            while True:
                np.random.shuffle(idxs)
                for i in range(0, len(X), batch_size):
                    batch_idx = idxs[i:i+batch_size]
                    batch_X = X[batch_idx].copy()
                    batch_Y = Y[batch_idx].copy()
                    for j in range(len(batch_X)):
                        if augment:
                            augmented = aug(image=(batch_X[j]*255).astype(np.uint8), mask=(batch_Y[j,...,0]*255).astype(np.uint8))
                            batch_X[j] = augmented['image']/255.0
                            batch_Y[j,...,0] = augmented['mask']/255.0
                    yield batch_X, batch_Y
        # Model
        self.model = self.build_unet(input_shape=(*self.img_size, 3))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        if augment:
            steps_per_epoch = int(np.ceil(len(X_train)/batch_size))
            val_steps = int(np.ceil(len(X_val)/batch_size))
            self.model.fit(
                aug_generator(X_train, Y_train, batch_size),
                steps_per_epoch=steps_per_epoch,
                validation_data=(X_val, Y_val),
                epochs=epochs,
                batch_size=batch_size
            )
        else:
            self.model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size)
        self.model.save(save_path)
        # Tự động lưu kết quả val nếu có results_base_dir và val_images_dir
        results_dir = None
        if results_base_dir and val_images_dir:
            results_dir = self.save_val_predictions_with_run(val_images_dir, results_base_dir, model_path=save_path)
        return results_dir

    def predict(self, image_path, model_path=None, out_path=None, out_overlay_path=None, mask_threshold=None, min_box_area=None, box_color=None, box_thickness=None):
        """
        Dự đoán mask cho 1 ảnh.
        image_path: đường dẫn ảnh gốc
        model_path: đường dẫn model đã train (.h5)
        out_path: nếu muốn lưu mask dự đoán ra file
        out_overlay_path: nếu muốn lưu overlay mask + box
        mask_threshold: ngưỡng phân biệt foreground/background (nếu None sẽ lấy từ config)
        min_box_area: diện tích tối thiểu của box (nếu None sẽ lấy từ config)
        box_color: màu box (nếu None sẽ lấy từ config)
        box_thickness: độ dày box (nếu None sẽ lấy từ config)
        """
        if self.model is None:
            if model_path is None:
                raise ValueError('Phải cung cấp model_path hoặc đã train xong.')
            self.model = tf.keras.models.load_model(model_path)
        img = Image.open(image_path).convert('RGB').resize(self.img_size)
        x = np.array(img)/255.0
        x = np.expand_dims(x, axis=0)
        pred = self.model.predict(x)[0, ..., 0]
        threshold = mask_threshold if mask_threshold is not None else config.MASK_THRESHOLD
        mask = (pred > threshold).astype(np.uint8)*255
        mask_img = Image.fromarray(mask)
        if out_path:
            mask_img.save(out_path)
        if out_overlay_path:
            import cv2
            def get_all_bounding_boxes(mask, min_area):
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
                boxes = []
                for i in range(1, num_labels):
                    x, y, w, h, area = stats[i]
                    if area > min_area:
                        boxes.append((x, y, x+w, y+h))
                return boxes
            min_area = min_box_area if min_box_area is not None else config.MIN_BOX_AREA
            boxes = get_all_bounding_boxes(mask, min_area)
            orig_img = np.array(img)
            mask_color = np.zeros_like(orig_img)
            mask_color[..., 0] = mask
            overlay = cv2.addWeighted(orig_img, 0.7, mask_color, 0.3, 0)
            color = box_color if box_color is not None else config.BOX_COLOR
            thickness = box_thickness if box_thickness is not None else config.BOX_THICKNESS
            for b in boxes:
                x1, y1, x2, y2 = b
                overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            Image.fromarray(overlay).save(out_overlay_path)
        return mask_img

    def predict_from_array(self, img_array, model_path=None, out_path=None):
        """
        Dự đoán mask cho 1 ảnh từ numpy array hoặc PIL.Image.
        img_array: numpy array (H, W, 3) hoặc PIL.Image
        model_path: đường dẫn model đã train (.h5)
        out_path: nếu muốn lưu mask dự đoán ra file
        """
        if self.model is None:
            if model_path is None:
                raise ValueError('Phải cung cấp model_path hoặc đã train xong.')
            self.model = tf.keras.models.load_model(model_path)
        # Nếu là PIL.Image thì chuyển sang numpy
        if isinstance(img_array, Image.Image):
            img = img_array.resize(self.img_size)
            x = np.array(img)/255.0
        else:
            img = cv2.resize(img_array, self.img_size)
            x = img/255.0
        x = np.expand_dims(x, axis=0)
        pred = self.model.predict(x)[0, ..., 0]
        mask = (pred > 0.5).astype(np.uint8)*255
        mask_img = Image.fromarray(mask)
        if out_path:
            mask_img.save(out_path)
        return mask_img

    def save_val_predictions(self, val_images_dir, output_dir, model_path=None):
        """
        Lưu mask dự đoán và overlay mask lên ảnh gốc cho toàn bộ tập val.
        val_images_dir: thư mục ảnh val
        output_dir: thư mục lưu kết quả (mask, overlay)
        model_path: đường dẫn model đã train (.h5)
        """
        import cv2
        os.makedirs(output_dir, exist_ok=True)
        val_images = [f for f in os.listdir(val_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_name in val_images:
            img_path = os.path.join(val_images_dir, img_name)
            pred_mask = self.predict(img_path, model_path=model_path)
            pred_mask.save(os.path.join(output_dir, f'{os.path.splitext(img_name)[0]}_pred.png'))
            # Overlay mask lên ảnh gốc
            img = np.array(Image.open(img_path).convert('RGB').resize(self.img_size))
            mask_np = np.array(pred_mask)
            mask_color = np.zeros_like(img)
            mask_color[..., 0] = mask_np  # Kênh đỏ
            overlay = cv2.addWeighted(img, 0.7, mask_color, 0.3, 0)
            Image.fromarray(overlay).save(os.path.join(output_dir, f'{os.path.splitext(img_name)[0]}_overlay.png')) 

    def save_val_predictions_with_run(self, val_images_dir, base_output_dir, model_path=None):
        """
        Lưu mask dự đoán và overlay mask lên ảnh gốc cho toàn bộ tập val vào một thư mục mới theo timestamp.
        val_images_dir: thư mục ảnh val
        base_output_dir: thư mục gốc để tạo các thư mục run
        model_path: đường dẫn model đã train (.h5)
        Trả về đường dẫn thư mục kết quả vừa tạo.
        """
        import cv2
        import datetime
        run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, run_name)
        os.makedirs(output_dir, exist_ok=True)
        val_images = [f for f in os.listdir(val_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_name in val_images:
            img_path = os.path.join(val_images_dir, img_name)
            pred_mask = self.predict(img_path, model_path=model_path)
            pred_mask.save(os.path.join(output_dir, f'{os.path.splitext(img_name)[0]}_pred.png'))
            # Overlay mask lên ảnh gốc
            img = np.array(Image.open(img_path).convert('RGB').resize(self.img_size))
            mask_np = np.array(pred_mask)
            mask_color = np.zeros_like(img)
            mask_color[..., 0] = mask_np  # Kênh đỏ
            overlay = cv2.addWeighted(img, 0.7, mask_color, 0.3, 0)
            Image.fromarray(overlay).save(os.path.join(output_dir, f'{os.path.splitext(img_name)[0]}_overlay.png'))
        return output_dir 

