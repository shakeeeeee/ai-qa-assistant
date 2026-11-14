import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

CONF = 0.25          # 只用 YOLO 時的門檻
CONF_WITH_GPT = 0.15  # 有 GPT 把關時，YOLO 門檻可以放低一點，少漏一些刮傷
SOP_FILE = "data/sop/SOP-001-刮傷.txt"
MOBILE_LIVE_URL = "https://shakeeeeee.github.io/ai-qa-assistant/"

st.set_page_config(page_title="AI 品檢助手")
st.title("AI 品檢助手")
st.write("上傳照片，檢查表面有沒有刮傷")


@st.cache_resource
def load_model():
    return YOLO("models/scratch_v2.pt")


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


def detect(image, conf):
    """回傳 [(框, 分數), ...]，分數由高到低"""
    result = model.predict(image, conf=conf, verbose=False)[0]
    return [(b.tolist(), float(c)) for b, c in zip(result.boxes.xyxy, result.boxes.conf)]


def draw(image: Image.Image, kept, rejected):
    """紅框是刮傷；灰框是 YOLO 有框到、但 GPT 判斷不是刮傷的"""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lw = max(2, img.shape[1] // 300)
    for (x1, y1, x2, y2), _ in rejected:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (160, 160, 160), lw)
    for (x1, y1, x2, y2), s in kept:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), lw)
        cv2.putText(img, f"{s:.2f}", (int(x1), max(20, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, lw * 0.4, (0, 0, 255), lw)
    return img


def read_sop():
    with open(SOP_FILE, encoding="utf-8") as f:
        return f.read()


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
        help="YOLO 先把可疑的地方都框出來，再把每個框裁成小圖給 GPT 判斷是刮傷、"
             "還是木紋、反光、髒污。驗證集上 precision 0.62 → 0.72，recall 不變（0.78）。"
             "會多花 2～4 秒，多張照片是同時送出，不會一張一張累加。",
    )

    if files and st.button(f"開始檢查（{len(files)} 張）"):
        with st.spinner("檢查中..."):
            images = [Image.open(f).convert("RGB") for f in files]
            founds = [detect(img, CONF_WITH_GPT if use_gpt else CONF) for img in images]
            checks_all = [[None] * len(found) for found in founds]

            if use_gpt:
                from concurrent.futures import ThreadPoolExecutor
                from verify import verify_boxes

                def ask(i):
                    if not founds[i]:
                        return i, None, None
                    try:
                        return i, verify_boxes(gpt, images[i], [b for b, _ in founds[i]]), None
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
            kept = [(b, s) for (b, s), c in zip(found, checks) if c is None or c["is_scratch"]]
            rejected = [(b, s) for (b, s), c in zip(found, checks) if c is not None and not c["is_scratch"]]
            results.append((f.name, draw(image, kept, rejected), found, checks, [s for _, s in kept]))

        bad = [r[0] for r in results if r[4]]
        st.subheader("檢查結果")
        st.write(f"共 {len(results)} 張，有刮傷 {len(bad)} 張，合格 {len(results) - len(bad)} 張")
        # 手機畫面窄，表格只放兩欄
        st.dataframe(
            [{"檔名": name, "結果": f"刮傷 {len(s)} 處（{s[0]:.2f}）" if s else "合格"}
             for name, _, _, _, s in results],
            use_container_width=True,
            hide_index=True,
        )

        for name, out, found, checks, scores in results:
            with st.expander(f"{'有刮傷' if scores else '合格'}｜{name}", expanded=bool(scores)):
                st.image(out, channels="BGR", use_container_width=True)
                if use_gpt and any(checks):
                    st.caption("紅框：確認是刮傷　灰框：GPT 判斷不是刮傷")
                    st.dataframe(
                        [{"YOLO 分數": round(s, 2),
                          "是刮傷嗎": ("是" if c["is_scratch"] else "不是") + f"（{c['what']}）",
                          "理由": c["reason"]}
                         for (_, s), c in zip(found, checks) if c],
                        use_container_width=True,
                        hide_index=True,
                    )
                elif scores:
                    st.write(f"找到 {len(scores)} 處，最高信心分數 {scores[0]:.2f}")

        if bad:
            st.subheader("處理方式")
            st.write(f"有刮傷的照片：{'、'.join(bad)}")
            st.write(read_sop())
            st.caption("依據 SOP-001")

with tab_camera:
    # 原本用 streamlit-webrtc 把畫面傳到這台伺服器偵測，放上雲端後很卡，
    # 改成讓模型直接在使用者的瀏覽器裡跑（docs/index.html）
    st.write("即時偵測在另一個頁面，模型直接在你的手機或電腦上跑，畫面不用上傳，比較順。")
    st.link_button("開啟即時偵測", MOBILE_LIVE_URL, type="primary")
    st.caption("手機、電腦都可以用。電腦需要有網路攝影機；沒有的話可以在那個頁面用照片測試。")
    with st.expander("SOP-001 刮傷處理方式"):
        st.write(read_sop())
