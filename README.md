# AI 品檢助手

用 YOLOv8 自動檢查產品表面有沒有刮傷，找出對應的 SOP，並產出檢查報告。

## 🔗 線上展示

| | 網址 | 說明 |
|---|---|---|
| 📱 手機即時偵測 | https://shakeeeeee.github.io/ai-qa-assistant/ | 用手機打開，鏡頭掃到刮傷就會直接框出來。模型在手機的瀏覽器裡執行（ONNX Runtime Web），影像不會上傳。 |
| 📁 完整版 | https://ai-app-assistant-94m3b6tv2ebo5uwauebfzx.streamlit.app/ | 一次上傳多張照片、檢查總表、SOP 處理報告 |

## 1. 功能

- **多張照片檢測**：一次上傳多張照片，產生總表（幾張合格、幾張有刮傷），有刮傷的照片會附上 SOP 處理方式
- **電腦鏡頭即時偵測**：Streamlit + WebRTC，每一格畫面都會檢查
- **手機鏡頭即時偵測**：模型轉成 ONNX 格式，在手機瀏覽器裡直接推論，不需要伺服器（`docs/`）

## 2. 怎麼在自己電腦跑起來？

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

## 3. 模型是怎麼訓練的？

- 基礎模型：`yolov8n.pt`
- 資料：`python build_dataset.py` 會把 Roboflow 匯出的資料集（`roboflow_data/`）和自己用 makesense.ai 標的照片合併成 `dataset_v2/`
  - 訓練 264 張、驗證 28 張，全部是真實照片
  - 照片的版權不屬於本專案，所以沒有放進 repo
- 訓練指令：

  ```bash
  yolo detect train model=yolov8n.pt data=dataset_v2.yaml epochs=60 patience=15 imgsz=640 degrees=10 flipud=0.5 project=runs/detect name=v2-real
  ```

- 驗證集結果：精確率 0.95、召回率 0.68、mAP50 0.77
- 手機版：匯出成 416×416 的 ONNX 模型。輸入尺寸改小之後 mAP50 一樣是 0.77，但運算量只剩大約 40%

  ```bash
  yolo export model=models/scratch_v2.pt format=onnx imgsz=416 opset=12 simplify=True
  ```

### 從失敗中學到的事

第一版模型加了 80 張「白紙畫黑線」的合成圖，驗證分數高達 0.995，但驗證集也都是合成圖，所以這個分數沒有參考價值。實際測試時，模型學到的是「細線就是刮傷」，把新聞截圖的浮水印文字當成刮傷，真正的刮傷反而沒框到。

改法：拿掉合成圖，只用真實照片訓練，驗證集也換成真實照片。新模型就能框到真正的刮傷，浮水印也不會再被誤判。

### 目前的限制

- 大約 1/3 的刮傷抓不到（召回率 0.68）
- 資料以銀色汽車為主，其他顏色和材質的表現比較差

## 4. SOP 怎麼對應？

| YOLO 類別 | SOP 文件 |
|---|---|
| `scratch` | `data/sop/SOP-001-刮傷.txt` |
| （未來）髒污 | `data/sop/SOP-002-髒污.txt` |
| （未來）印刷偏移 | `data/sop/SOP-003-印刷偏移.txt` |

## 5. 檔案說明

| 路徑 | 用途 |
|---|---|
| `app.py` | Streamlit 網頁（多張檢測、電腦鏡頭即時偵測） |
| `docs/` | 手機即時偵測網頁＋ONNX 模型（GitHub Pages） |
| `models/scratch_v2.pt` | 訓練好的 YOLOv8 模型 |
| `build_dataset.py` | 合併 Roboflow 和自己標的照片，產生 `dataset_v2/` |
| `dataset_v2.yaml` | YOLO 資料集設定 |
| `data/sop/` | 3 份 SOP |
| `make_dataset.py` | 第一版的合成圖產生器（已經不用了，留著當紀錄） |
| `requirements.txt`、`packages.txt` | Streamlit Cloud 部署用 |

## 6. 之後怎麼改進？

- 加入不同顏色的車子、手機、金屬等其他材質的刮傷資料，提高召回率
- 加入沒有刮傷的照片（乾淨表面、文字、反光），降低誤判
- 用驗證集調整 `conf` 門檻（目前是 0.25），在漏抓和誤判之間找平衡
- 接上 LLM 和 RAG，依照瑕疵類型查詢 SOP，讓報告讀起來像人寫的

## 資料來源

- 刮傷資料集：Roboflow 匯出的「silver-car-scratch」資料集，授權為 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 自己標註的照片：使用 [makesense.ai](https://www.makesense.ai/) 標註
