# Cấu hình cho dự án segmentation

IMG_SIZE = (512, 512)

# Dữ liệu train/val
LABELS_TRAIN = "data/labels/train"
LABELS_VAL = "data/labels/val"
IMAGES_TRAIN = "data/images/train"
MASKS_TRAIN = "data/masks/train"
IMAGES_VAL = "data/images/val"
MASKS_VAL = "data/masks/val"

# Model TensorFlow
TF_MODEL_PATH = "output/models/segmenter_unet.h5"

# Model PyTorch
TORCH_MODEL_PATH = "output/models/unet_best.pth"

# Kết quả
RESULTS_DIR = "output/results"

# Tham số train
EPOCHS = 100
BATCH_SIZE = 8

# --- Các biến tùy chỉnh segmentation/detection ---
MASK_THRESHOLD = 0.7         # Ngưỡng phân biệt foreground/background (TF & Torch)
MIN_BOX_AREA = 10            # Diện tích tối thiểu của box (pixel)
BOX_COLOR = (0, 0, 255)      # Màu box (BGR)
BOX_THICKNESS = 2            # Độ dày box