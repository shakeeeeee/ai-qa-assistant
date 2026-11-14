# AI 品檢助手

拍一張產品照片，讓模型找出表面的刮傷，再依照 SOP 給出處理方式。

這是我自學 Python 和電腦視覺時做的練習專案，目前只做「刮傷」一種瑕疵。

## Demo

- 手機即時偵測：https://shakeeeeee.github.io/ai-qa-assistant/
  （打開鏡頭對著物品掃，有刮傷會直接框出來。沒有鏡頭的話可以按「用照片測試」）
- 上傳照片版：https://shakeeeeee-scratch.streamlit.app/
  （可以一次傳多張，會列出哪幾張有刮傷，並顯示 SOP。檢查完可以下載 CSV 報告。Streamlit 免費版閒置會休眠，第一次打開要等一下）

## 做法

1. 用 YOLOv8n 做物件偵測，只有一個類別 `scratch`
2. 訓練資料共 292 張（訓練 264、驗證 28），來源是 Roboflow 上的汽車刮傷資料集，加上我自己用 makesense.ai 標的照片
3. 網頁用 Streamlit 寫，偵測到刮傷就讀取 `data/sop/` 裡對應的 SOP
4. 手機版是把模型轉成 ONNX，用 onnxruntime-web 在瀏覽器裡跑，所以不用架伺服器，影像也不會上傳

**資料裡大部分是 AI 生成的圖。** 真實的汽車刮傷照片很難收集，Roboflow 資料集裡有很多張是用 Google ImageFX 生成的：

| | AI 生成 | 真實照片 | 合計 |
|---|---|---|---|
| 訓練 | 211 | 53 | 264 |
| 驗證 | 21 | 7 | 28 |

驗證集的結果要分開看，差很多：

| 驗證資料 | 張數（標註框） | precision | recall | mAP50 |
|---|---|---|---|---|
| 全部 | 28（40） | 0.95 | 0.68 | 0.77 |
| 只看 AI 生成圖 | 21（25） | 0.99 | 0.92 | 0.99 |
| 只看真實照片 | 7（15） | 0.75 | 0.27 | 0.32 |

真實照片上，四個刮傷大約只抓到一個。只有 7 張，數字誤差很大，但方向很清楚：整體的 0.77 主要是被 AI 圖拉高的，不能代表實際使用的效果。AI 圖分數接近滿分，可能是同一批生成的相似圖片同時被分到訓練集和驗證集。

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
- 這個比較用的是整個驗證集，其中 3/4 是 AI 生成圖，所以數字會比實際好
- 驗證集都是汽車照片。木頭桌子這類有紋路的表面才是 YOLO 最容易誤判的情況，還需要收集這類照片來測

手機即時偵測版沒有加 GPT：每格畫面都問 GPT 太慢也太貴，而且公開網頁不能放 API 金鑰。

## 遇到的問題

**第一版用了合成資料，結果模型學錯東西**

一開始真實照片不夠，我用程式產生了 80 張「白底黑線」的圖當刮傷。驗證分數有 0.99，但驗證集也是合成圖，所以這個分數沒有意義。拿真的照片測試時，模型把新聞截圖右上角的浮水印文字框成刮傷，真正的刮傷反而沒抓到。它學到的是「細線」，不是「刮傷」。

後來拿掉合成圖，改用 Roboflow 的汽車刮傷資料集和自己標的照片重新訓練，浮水印就不再被誤判了。

**同樣的問題又出現了一次：驗證集大多是 AI 生成圖**

Roboflow 的圖看起來很像真的照片，我一開始以為都是實拍，後來才從檔名發現大部分是 ImageFX 生成的。分開計算之後，真實照片的 recall 只有 0.27（見上面「做法」的表格）。這跟第一版的問題一樣：驗證集跟實際使用的照片不一樣，分數就不準。差別只在於這次的圖比較逼真，比較難發現。

**手機版要在手機上跑得動**

一開始是用 streamlit-webrtc 把鏡頭畫面傳到伺服器再偵測，在自己電腦上跑很順（一格約 0.04 秒），但手機瀏覽器要 HTTPS 才能開鏡頭。放上 Streamlit Cloud 之後，每一格畫面都要傳到免費主機再傳回來，實際用起來卡頓很嚴重。所以改成把模型匯出成 ONNX，直接在使用者的瀏覽器裡執行，Streamlit 的鏡頭分頁只留一個連結過去。輸入尺寸從 640 降到 416，驗證集 mAP50 幾乎沒變（0.767 → 0.769），但運算量少了一半以上。

## 目前的限制

- 資料大多是銀色汽車烤漆。我用手機對著木頭桌子測試，門檻調低會框到很多木紋，調高又什麼都抓不到，表示換了材質模型就不太行
- 訓練和驗證資料大多是 AI 生成圖。真實照片上的 recall 只有 0.27，大部分的刮傷都會漏掉
- SOP 目前是固定讀一份文字檔，還沒有依照瑕疵類型或嚴重程度去查

## 接下來想做

- 多收集真實的刮傷照片，驗證集只放真實照片，分數才能代表實際效果
- 補不同材質（木頭、塑膠、金屬、手機螢幕）的刮傷照片，也加一些沒有刮傷但有紋路的照片，減少誤判
- 加入髒污、印刷偏移等其他類別（SOP 已經先寫好了）
- 用 RAG 依照瑕疵類型查 SOP，再用 LLM 整理成檢查報告

## 設定檔

類別、SOP、模型路徑、門檻都寫在 `docs/config.json`，Streamlit 版和手機版共用同一份。換一種瑕疵或產品時，訓練好新模型後只要改這個檔案：

```json
{
  "product": "汽車烤漆表面",
  "classes": [
    { "id": "scratch", "name": "刮傷", "color": "#ff4d4f", "sop": "data/sop/SOP-001-刮傷.txt" }
  ],
  "model_pt": "models/scratch_v2.pt",
  "model_onnx": "scratch_v2.onnx",
  "onnx_imgsz": 416,
  "conf": 0.25,
  "conf_with_gpt": 0.15
}
```

- `classes` 的順序要跟訓練時 `dataset.yaml` 的類別編號一樣，`id` 要跟 yaml 的名稱一樣
- 兩邊對不上的話，網頁會直接顯示錯誤，不會默默對到錯的 SOP
- 放在 `docs/` 底下是因為 GitHub Pages 只會公開這個資料夾，手機版才讀得到

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
docs/config.json    類別、SOP、模型路徑、門檻的設定（兩個版本共用）
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

- 汽車刮傷資料集：Roboflow「silver-car-scratch」，CC BY 4.0（其中大部分是 Google ImageFX 生成的圖）
- 標註工具：makesense.ai
