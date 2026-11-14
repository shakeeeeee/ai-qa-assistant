import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# 類別、SOP、模型路徑、門檻都寫在設定檔，換專案只要改設定檔
# 放在 docs/ 底下是因為手機版網頁（GitHub Pages）也要讀同一份
CFG = json.load(open("docs/config.json", encoding="utf-8"))
CLASSES = CFG["classes"]
MOBILE_LIVE_URL = "https://shakeeeeee.github.io/ai-qa-assistant/"

st.set_page_config(page_title=CFG["title"])
st.title(CFG["title"])
st.write(f"上傳照片，檢查{CFG['product']}有沒有{'、'.join(c['name'] for c in CLASSES)}")


@st.cache_resource
def load_model():
    return YOLO(CFG["model_pt"])


@st.cache_resource
def load_gpt(key):
    from openai import OpenAI
    return OpenAI(api_key=key)


def get_key():
    # 每次都重新讀，之後才補上金鑰也不用重新啟動
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


model = load_model()
key = get_key()
gpt = load_gpt(key) if key else None

# 模型的類別順序要跟設定檔一樣，不一樣就停下來，不然會對應到錯的 SOP
model_ids = [model.names[i] for i in range(len(model.names))]
if model_ids != [c["id"] for c in CLASSES]:
    st.error(f"設定檔的類別 {[c['id'] for c in CLASSES]} 跟模型的類別 {model_ids} 不一樣，"
             "請修改 docs/config.json 的 classes，順序也要相同")
    st.stop()


def bgr(hex_color):
    h = hex_color.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def detect(image, conf):
    """回傳 [(框, 分數, 類別編號), ...]，分數由高到低"""
    r = model.predict(image, conf=conf, verbose=False)[0]
    return [(b.tolist(), float(s), int(c)) for b, s, c in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)]


def draw(image: Image.Image, kept, rejected):
    """依類別顏色畫框；灰框是 YOLO 有框到、但 GPT 判斷不是瑕疵的"""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lw = max(2, img.shape[1] // 300)
    for (x1, y1, x2, y2), _, _ in rejected:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (160, 160, 160), lw)
    for (x1, y1, x2, y2), s, c in kept:
        color = bgr(CLASSES[c]["color"])
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, lw)
        # putText 不支援中文，所以標英文 id
        cv2.putText(img, f"{CLASSES[c]['id']} {s:.2f}", (int(x1), max(20, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, lw * 0.4, color, lw)
    return img


def read_sop(c):
    with open(CLASSES[c]["sop"], encoding="utf-8") as f:
        return f.read()


def summary(kept):
    """例如：刮傷 2、髒污 1"""
    counts = {}
    for _, _, c in kept:
        counts[c] = counts.get(c, 0) + 1
    return "、".join(f"{CLASSES[c]['name']} {n}" for c, n in sorted(counts.items()))


tab_upload, tab_camera = st.tabs(["上傳照片", "鏡頭即時偵測"])

with tab_upload:
    files = st.file_uploader(
        "可以一次選多張（手機上可以直接拍照）",
        type=["jpg", "png", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    use_gpt = st.toggle(
        "用 GPT 再確認一次",
        value=gpt is not None,
        disabled=gpt is None,
        help="YOLO 先把可疑的地方都框出來，再把每個框裁成小圖給 GPT 判斷是不是真的瑕疵。"
             "會多花 2～4 秒，多張照片是同時送出，不會一張一張累加。",
    )

    if files and st.button(f"開始檢查（{len(files)} 張）"):
        with st.spinner("檢查中..."):
            images = [Image.open(f).convert("RGB") for f in files]
            founds = [detect(img, CFG["conf_with_gpt"] if use_gpt else CFG["conf"]) for img in images]
            checks_all = [[None] * len(found) for found in founds]

            if use_gpt:
                from concurrent.futures import ThreadPoolExecutor
                from verify import verify_boxes

                def ask(i):
                    if not founds[i]:
                        return i, None, None
                    try:
                        boxes = [b for b, _, _ in founds[i]]
                        labels = [CLASSES[c]["name"] for _, _, c in founds[i]]
                        return i, verify_boxes(gpt, images[i], boxes, labels, CFG), None
                    except Exception as e:
                        return i, None, e

                # GPT 一張要 2 秒左右，一張一張問太慢，所以同時送出
                with ThreadPoolExecutor(max_workers=8) as pool:
                    for i, checks, err in pool.map(ask, range(len(images))):
                        if checks:
                            checks_all[i] = checks
                        if err:
                            st.warning(f"{files[i].name}：GPT 確認失敗，改用 YOLO 的結果（{err}）")

        results = []
        for f, image, found, checks in zip(files, images, founds, checks_all):
            kept = [d for d, ch in zip(found, checks) if ch is None or ch["is_defect"]]
            rejected = [d for d, ch in zip(found, checks) if ch is not None and not ch["is_defect"]]
            results.append((f.name, draw(image, kept, rejected), found, checks, kept))

        bad = [r for r in results if r[4]]
        st.subheader("檢查結果")
        st.write(f"共 {len(results)} 張，有瑕疵 {len(bad)} 張，合格 {len(results) - len(bad)} 張")
        # 手機畫面窄，表格只放兩欄
        st.dataframe(
            [{"檔名": name, "結果": summary(kept) if kept else "合格"}
             for name, _, _, _, kept in results],
            use_container_width=True,
            hide_index=True,
        )

        for name, out, found, checks, kept in results:
            with st.expander(f"{'有瑕疵' if kept else '合格'}｜{name}", expanded=bool(kept)):
                st.image(out, channels="BGR", use_container_width=True)
                if use_gpt and any(checks):
                    st.caption("彩色框：確認是瑕疵　灰框：GPT 判斷不是")
                    st.dataframe(
                        [{"類別": CLASSES[c]["name"],
                          "YOLO 分數": round(s, 2),
                          "GPT 確認": ("是" if ch["is_defect"] else "不是") + f"（{ch['what']}）",
                          "理由": ch["reason"]}
                         for (_, s, c), ch in zip(found, checks) if ch],
                        use_container_width=True,
                        hide_index=True,
                    )
                elif kept:
                    st.write(summary(kept))

        # 每一種有出現的瑕疵，各顯示一次它的 SOP
        found_classes = sorted({c for r in bad for _, _, c in r[4]})
        if found_classes:
            st.subheader("處理方式")
            for c in found_classes:
                names = [r[0] for r in bad if any(k[2] == c for k in r[4])]
                st.markdown(f"**{CLASSES[c]['name']}**（{'、'.join(names)}）")
                st.write(read_sop(c))
                st.caption(f"依據 {CLASSES[c]['sop'].split('/')[-1]}")

with tab_camera:
    # 原本用 streamlit-webrtc 把畫面傳到這台伺服器偵測，放上雲端後很卡，
    # 改成讓模型直接在使用者的瀏覽器裡跑（docs/index.html）
    st.write("即時偵測在另一個頁面，模型直接在你的手機或電腦上跑，畫面不用上傳，比較順。")
    st.link_button("開啟即時偵測", MOBILE_LIVE_URL, type="primary")
    st.caption("手機、電腦都可以用。電腦需要有網路攝影機；沒有的話可以在那個頁面用照片測試。")
    for c, cls in enumerate(CLASSES):
        with st.expander(f"{cls['name']}的處理方式"):
            st.write(read_sop(c))
