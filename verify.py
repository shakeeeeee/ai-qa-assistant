"""第二階段確認：YOLO 框出來的地方裁成小圖，交給 GPT 判斷是不是真的瑕疵。"""
import base64
import io
import json
import logging
import time

from openai import OpenAI
from PIL import Image

MAX_BOXES = 8  # 一張照片最多送幾個框，避免費用失控

# 名稱用 qa. 開頭，才會用到 app.py 設定好的 log 格式
log = logging.getLogger("qa.verify")

SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "is_defect": {"type": "boolean"},
                    "what": {"type": "string", "description": "看到的是什麼，例如：刮傷、木紋、反光、髒污、文字"},
                    "reason": {"type": "string", "description": "一句話理由，繁體中文"},
                },
                "required": ["index", "is_defect", "what", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def build_prompt(product: str, confusing: str, n: int) -> str:
    return (
        f"你是產品外觀檢查員，檢查的是{product}。\n"
        "以下每張小圖是從產品照片裡裁下來的一塊，中間附近是偵測模型懷疑有瑕疵的位置，"
        "每張小圖前面會寫模型懷疑的瑕疵種類。\n"
        "請判斷每張小圖裡是不是真的有那一種瑕疵。\n"
        f"容易被誤認、但不算瑕疵的東西：{confusing}。\n"
        "不確定的時候，偏向判斷為有瑕疵（漏掉瑕疵比多報一個更嚴重）。\n"
        f"共 {n} 張小圖，index 從 0 開始。"
    )


def crop_box(image: Image.Image, box, pad=0.4, min_size=96, max_size=384) -> Image.Image:
    """裁下框的位置，四周多留一點範圍讓 GPT 看得到周圍的材質"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = max(w * pad, min_size / 2), max(h * pad, min_size / 2)
    left, top = max(0, x1 - px), max(0, y1 - py)
    right, bottom = min(image.width, x2 + px), min(image.height, y2 + py)
    if right - left < 1 or bottom - top < 1:
        raise ValueError(f"框 {box} 不在照片範圍 {image.size} 內，可能是照片方向沒有轉正")
    crop = image.crop((left, top, right, bottom))
    crop.thumbnail((max_size, max_size))
    return crop


def to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def verify_boxes(client: OpenAI, image: Image.Image, boxes: list, labels: list[str],
                 cfg: dict) -> list[dict]:
    """boxes 是 [(x1, y1, x2, y2), ...]，labels 是每個框的瑕疵中文名稱。
    回傳每個框的判斷結果（順序相同）。超過 MAX_BOXES 的框不送 GPT，直接保留。"""
    if not boxes:
        return []
    sent = boxes[:MAX_BOXES]
    if len(boxes) > MAX_BOXES:
        log.warning("框太多（%d 個），只送前 %d 個給 GPT，其他直接當成瑕疵", len(boxes), MAX_BOXES)
    content = [{"type": "text", "text": build_prompt(cfg["product"], cfg["gpt_confusing"], len(sent))}]
    for i, box in enumerate(sent):
        content.append({"type": "text", "text": f"小圖 {i}（懷疑是：{labels[i]}）"})
        content.append({"type": "image_url",
                        "image_url": {"url": to_data_url(crop_box(image, box)), "detail": "low"}})

    start = time.time()
    resp = client.chat.completions.create(
        model=cfg["gpt_model"],
        temperature=0,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "verify", "strict": True, "schema": SCHEMA}},
    )
    by_index = {r["index"]: r for r in json.loads(resp.choices[0].message.content)["results"]}
    rejected = sum(1 for r in by_index.values() if not r["is_defect"])
    log.info("GPT 回應 %.1f 秒，送 %d 個框，判斷不是瑕疵 %d 個（%s）",
             time.time() - start, len(sent), rejected, cfg["gpt_model"])

    missing = [i for i in range(len(sent)) if i not in by_index]
    if missing:
        log.warning("GPT 漏回第 %s 個框，當成瑕疵處理", missing)

    out = []
    for i in range(len(boxes)):
        r = by_index.get(i)
        if r is None:  # 沒送或 GPT 漏回，保守起見當成有瑕疵
            r = {"is_defect": True, "what": "未確認", "reason": "沒有送去確認"}
        log.debug("框 %d：%s，%s（%s）", i, "是瑕疵" if r["is_defect"] else "不是", r["what"], r["reason"])
        out.append(r)
    return out
