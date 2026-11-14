# AI 品檢助手

拍一張產品照片，讓模型找出表面的刮傷，再依照 SOP 給出處理方式。

這是我自學 Python 和電腦視覺時做的練習專案，目前只做「刮傷」一種瑕疵。

## Demo

- 手機即時偵測：https://shakeeeeee.github.io/ai-qa-assistant/
  （打開鏡頭對著物品掃，有刮傷會直接框出來。沒有鏡頭的話可以按「用照片測試」）
- 上傳照片版：https://shakeeeeee-scratch.streamlit.app/
  （可以一次傳多張，會列出哪幾張有刮傷，並顯示 SOP。Streamlit 免費版閒置會休眠，第一次打開要等一下）

## 做法

1. 用 YOLOv8n 做物件偵測，只有一個類別 `scratch`
2. 訓練資料是 Roboflow 上的汽車刮傷資料集，加上我自己用 makesense.ai 標的照片，共 292 張（訓練 264、驗證 28）
3. 網頁用 Streamlit 寫，偵測到刮傷就讀取 `data/sop/` 裡對應的 SOP
4. 手機版是把模型轉成 ONNX，用 onnxruntime-web 在瀏覽器裡跑，所以不用架伺服器，影像也不會上傳

驗證集結果：precision 0.95、recall 0.68、mAP50 0.77

## 用 GPT 做第二階段確認

YOLO 門檻調低會多抓到刮傷，但也會框到很多不是刮傷的東西。所以上傳照片版加了第二階段：YOLO 先用比較低的門檻框出可疑位置，每個框裁成小圖交給 GPT（gpt-4.1-mini）判斷是刮傷，還是木紋、反光、髒污、文字。只保留 GPT 確認是刮傷的框。

用 `eval_verify.py` 在驗證集上比較（框和標註的 IoU ≥ 0.5 才算抓對，共 40 個標註框）：

| 設定 | precision | recall |
|---|---|---|
| YOLO 0.25 | 0.71 | 0.75 |
| YOLO 0.25 + GPT | 0.79 | 0.75 |
| YOLO 0.15 | 0.62 | 0.78 |
| YOLO 0.15 + GPT（網頁使用的設定） | 0.72 | 0.78 |
| YOLO 0.10 | 0.50 | 0.78 |
| YOLO 0.10 + GPT | 0.60 | 0.78 |

（這裡的 precision、recall 是在固定門檻下算的，跟上面 YOLO 訓練報告的數字算法不同，不能直接比）

- GPT 在每個門檻都讓 precision 提高約 0.08～0.10，而且沒有刪掉任何一個正確的框
- 剩下的錯誤大多是「框在刮傷上，但位置或大小跟標註差太多」，這是 YOLO 定位的問題，GPT 判斷內容是刮傷沒有錯，所以幫不上忙
- 驗證集都是汽車照片。木頭桌子這類有紋路的表面才是 YOLO 最容易誤判的情況，還需要收集這類照片來測

手機即時偵測版沒有加 GPT：每格畫面都問 GPT 太慢也太貴，而且公開網頁不能放 API 金鑰。

## 遇到的問題

**第一版用了合成資料，結果模型學錯東西**

一開始真實照片不夠，我用程式產生了 80 張「白底黑線」的圖當刮傷。驗證分數有 0.99，但驗證集也是合成圖，所以這個分數沒有意義。拿真的照片測試時，模型把新聞截圖右上角的浮水印文字框成刮傷，真正的刮傷反而沒抓到。它學到的是「細線」，不是「刮傷」。

後來拿掉合成圖，只用真實照片訓練，驗證集也換成真實照片，才比較能反映實際表現。

**手機版要在手機上跑得動**

一開始是用 streamlit-webrtc 把鏡頭畫面傳到伺服器再偵測，在電腦上可以用，但手機瀏覽器要 HTTPS 才能開鏡頭。如果放到雲端，影像要一路傳到免費主機去跑，還要另外設定 TURN 伺服器，速度和連線都不穩。所以改成把模型匯出成 ONNX，直接在手機的瀏覽器裡執行。輸入尺寸從 640 降到 416，驗證集 mAP50 幾乎沒變（0.767 → 0.769），但運算量少了一半以上。

## 目前的限制

- 資料大多是銀色汽車烤漆。我用手機對著木頭桌子測試，門檻調低會框到很多木紋，調高又什麼都抓不到，表示換了材質模型就不太行
- recall 只有 0.68，大約三分之一的刮傷會漏掉
- SOP 目前是固定讀一份文字檔，還沒有依照瑕疵類型或嚴重程度去查

## 接下來想做

- 補不同材質（木頭、塑膠、金屬、手機螢幕）的刮傷照片，也加一些沒有刮傷但有紋路的照片，減少誤判
- 加入髒污、印刷偏移等其他類別（SOP 已經先寫好了）
- 用 RAG 依照瑕疵類型查 SOP，再用 LLM 整理成檢查報告

## 在自己電腦執行

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

要用 GPT 確認的話，建立 `.streamlit/secrets.toml`，寫入 `OPENAI_API_KEY = "你的金鑰"`。沒有設定就只會用 YOLO。

重新訓練的話，先把 Roboflow 匯出的 YOLOv8 格式資料解壓到 `roboflow_data/`，然後：

```bash
python build_dataset.py
```

```bash
yolo detect train model=yolov8n.pt data=dataset_v2.yaml epochs=60 patience=15 imgsz=640 degrees=10 flipud=0.5
```

匯出手機版用的模型：

```bash
yolo export model=models/scratch_v2.pt format=onnx imgsz=416 opset=12 simplify=True
```

## 檔案

```
app.py              Streamlit 網頁
verify.py           把 YOLO 的框裁成小圖交給 GPT 確認
eval_verify.py      在驗證集上比較有沒有加 GPT 的差別
docs/               手機即時偵測頁（GitHub Pages）和 ONNX 模型
models/             訓練好的模型
build_dataset.py    合併 Roboflow 資料和自己標的照片
dataset_v2.yaml     訓練用的資料集設定
data/sop/           SOP 文件
make_dataset.py     第一版產生合成圖的程式（已不使用）
test_yolo.py        最一開始測試 YOLO 能不能跑的程式
test_best.py        測試第一版模型用的
```

訓練照片因為版權不是我的，所以沒有放上來。

## 資料來源

- 汽車刮傷資料集：Roboflow「silver-car-scratch」，CC BY 4.0
- 標註工具：makesense.ai
