from ultralytics import YOLO

print("載入新大腦...")
model = YOLO("runs/detect/train/weights/best.pt")
print("載入成功，模型會認的東西：", model.names)

# 直接拿一張假照片來考它，門檻放很低，0.1 就算找到
print("正在看假照片...")
results = model.predict("dataset/images/train/img_0.jpg", conf=0.1)

for r in results:
    print(f"找到 {len(r.boxes)} 個東西")
    for box in r.boxes:
        print(f"找到：{model.names[int(box.cls[0])]}，確定度：{float(box.conf[0]):.2f}")