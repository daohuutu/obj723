# Các hàm tiện ích cho dự án segmentation

def save_log(message, log_file="output/results/log.txt"):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n") 