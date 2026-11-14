"""比較「只用 YOLO」和「YOLO + GPT 確認」在驗證集上的 precision / recall。
框和標註的 IoU >= 0.5 才算抓對。"""
import json
import tomllib
from pathlib import Path

from openai import OpenAI
from PIL import Image
from ultralytics import YOLO

from verify import verify_boxes

VAL = Path("dataset_v2")
LOW_CONF = 0.05  # YOLO 先用最低的門檻跑一次，其他門檻從這裡篩
CACHE = Path("runs/eval_verify_cache.json")  # GPT 結果存起來，重跑不用再花錢


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def load_gt(label_path, w, h):
    boxes = []
    for line in label_path.read_text().split("\n"):
        if line.strip():
            _, cx, cy, bw, bh = map(float, line.split())
            boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def count(preds, gts):
    """preds 依分數高到低排好；回傳 (抓對的框數, 預測框數, 標註框數)"""
    used, tp = set(), 0
    for p in preds:
        best, best_j = 0, None
        for j, g in enumerate(gts):
            if j not in used and iou(p, g) > best:
                best, best_j = iou(p, g), j
        if best >= 0.5:
            used.add(best_j)
            tp += 1
    return tp, len(preds), len(gts)


def main():
    model = YOLO("models/scratch_v2.pt")
    client = OpenAI(api_key=tomllib.load(open(".streamlit/secrets.toml", "rb"))["OPENAI_API_KEY"])
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    settings = {
        "YOLO conf 0.25（目前上線的設定）": (0.25, False),
        "YOLO conf 0.25 + GPT": (0.25, True),
        "YOLO conf 0.15": (0.15, False),
        "YOLO conf 0.15 + GPT": (0.15, True),
        "YOLO conf 0.10": (0.10, False),
        "YOLO conf 0.10 + GPT": (0.10, True),
        "YOLO conf 0.05 + GPT": (0.05, True),
    }
    totals = {k: [0, 0, 0] for k in settings}

    for img_path in sorted((VAL / "images" / "val").iterdir()):
        img = Image.open(img_path).convert("RGB")
        gts = load_gt(VAL / "labels" / "val" / f"{img_path.stem}.txt", img.width, img.height)
        r = model.predict(img, conf=LOW_CONF, verbose=False)[0]
        boxes = [b.tolist() for b in r.boxes.xyxy]
        confs = [float(c) for c in r.boxes.conf]

        if img_path.name not in cache:
            cache[img_path.name] = [v["is_scratch"] for v in verify_boxes(client, img, boxes)]
            CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        gpt_ok = cache[img_path.name]

        for name, (conf, use_gpt) in settings.items():
            preds = [b for b, c, ok in zip(boxes, confs, gpt_ok) if c >= conf and (ok or not use_gpt)]
            for i, v in enumerate(count(preds, gts)):
                totals[name][i] += v

    print(f"{'設定':<28}{'precision':>10}{'recall':>8}{'抓對':>6}{'預測框':>6}{'標註框':>6}")
    for name, (tp, n_pred, n_gt) in totals.items():
        p = tp / n_pred if n_pred else 0
        print(f"{name:<28}{p:>10.2f}{tp / n_gt:>8.2f}{tp:>6}{n_pred:>6}{n_gt:>6}")


if __name__ == "__main__":
    main()
