"""合併 Roboflow 資料集和自己標的真實照片，產生 dataset_v2/（不含白紙合成圖）。"""
import random
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "dataset_v2"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}  # YOLO 讀不了 .avif，先跳過


def copy_pair(img: Path, label: Path, split: str, prefix: str) -> None:
    name = f"{prefix}_{img.stem}"
    shutil.copy2(img, OUT / "images" / split / f"{name}{img.suffix.lower()}")
    shutil.copy2(label, OUT / "labels" / split / f"{name}.txt")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)

    counts = {"train": 0, "val": 0}

    # 1. Roboflow：沿用它切好的 train / valid
    rf = ROOT / "roboflow_data"
    for src, split in (("train", "train"), ("valid", "val")):
        for img in sorted((rf / src / "images").iterdir()):
            label = rf / src / "labels" / f"{img.stem}.txt"
            if img.suffix.lower() in IMG_EXTS and label.exists():
                copy_pair(img, label, split, "rf")
                counts[split] += 1

    # 2. 自己標的真實照片（排除 make_dataset.py 產生的 img_數字.jpg），抽 20% 當驗證
    own = []
    for img in sorted((ROOT / "dataset" / "images" / "train").iterdir()):
        label = ROOT / "dataset" / "labels" / "train" / f"{img.stem}.txt"
        if img.suffix.lower() in IMG_EXTS and not re.fullmatch(r"img_\d+", img.stem) and label.exists():
            own.append((img, label))
    random.Random(42).shuffle(own)
    n_val = max(1, len(own) // 5)
    for i, (img, label) in enumerate(own):
        split = "val" if i < n_val else "train"
        copy_pair(img, label, split, "own")
        counts[split] += 1

    (ROOT / "dataset_v2.yaml").write_text(
        "path: dataset_v2\ntrain: images/train\nval: images/val\nnames:\n  0: scratch\n",
        encoding="utf-8",
    )
    print(f"完成：train {counts['train']} 張，val {counts['val']} 張")


if __name__ == "__main__":
    main()
