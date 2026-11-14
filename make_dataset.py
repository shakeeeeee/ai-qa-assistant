import os
import random
from PIL import Image, ImageDraw

# 建立資料夾
os.makedirs("dataset/images/train", exist_ok=True)
os.makedirs("dataset/labels/train", exist_ok=True)
os.makedirs("dataset/images/val", exist_ok=True)
os.makedirs("dataset/labels/val", exist_ok=True)

def make_one_image(save_img_path, save_label_path):
    # 做一張 640x640 的白紙，像明信片一樣
    img = Image.new("RGB", (640, 640), "white")
    draw = ImageDraw.Draw(img)

    # 隨機畫 1 到 3 條刮傷，刮傷就是一條黑線
    num_scratches = random.randint(1, 3)
    labels = []
    for _ in range(num_scratches):
        x1 = random.randint(50, 500)
        y1 = random.randint(50, 500)
        # 刮傷不會太長
        x2 = x1 + random.randint(50, 150)
        y2 = y1 + random.randint(-20, 20)
        draw.line([(x1, y1), (x2, y2)], fill="black", width=2)

        # 算出框框的中心點和寬高，YOLO 要的是 0~1 之間的比例
        cx = (x1 + x2) / 2 / 640
        cy = (y1 + y2) / 2 / 640
        w = abs(x2 - x1) / 640 + 0.02
        h = 0.02
        # 0 代表第 0 類，就是「刮傷」
        labels.append(f"0 {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}")

    img.save(save_img_path)
    with open(save_label_path, "w") as f:
        f.write("\n".join(labels))

# 做 80 張給它練習，20 張給它考試
for i in range(80):
    make_one_image(f"dataset/images/train/img_{i}.jpg", f"dataset/labels/train/img_{i}.txt")
for i in range(20):
    make_one_image(f"dataset/images/val/img_{i}.jpg", f"dataset/labels/val/img_{i}.txt")

print("100 張假刮傷照片都做好了！")