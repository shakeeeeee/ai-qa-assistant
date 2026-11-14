from ultralytics import YOLO

# 載入一個現成的、已經訓練好的小模型
# yolov8n 就是 YOLO 第 8 版裡面最小、最快的版本，n 就是 nano 很小的意思
print("正在載入 YOLO 模型...")
model = YOLO("yolov8n.pt")
print("模型載入成功！")

# 拿一張網路上的測試照片來試試看
# 這張是 YOLO 官方的測試照片，裡面有兩個人在騎腳踏車
print("正在辨識照片...")
results = model.predict("https://ultralytics.com/images/bus.jpg", save=True)

print("辨識完成了！")
print("它找到的東西：")
for r in results:
    for box in r.boxes:
        # box.cls 是它認出來的東西的編號，box.conf 是它有多確定，0.9 就是 90% 確定
        print(f"找到一個東西，確定度：{float(box.conf[0]):.2f}")

print("結果照片已經存在 runs/detect/predict 這個資料夾裡面了")