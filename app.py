import av
import cv2
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO

CONF = 0.25
SOP_FILE = "data/sop/SOP-001-刮傷.txt"
MOBILE_LIVE_URL = "https://shakeeeeee.github.io/ai-qa-assistant/"

st.set_page_config(page_title="AI 品檢助手")
st.title("AI 品檢助手")
st.write("上傳照片，檢查表面有沒有刮傷")


@st.cache_resource
def load_model():
    return YOLO("models/scratch_v2.pt")


model = load_model()


def detect(image):
    """回傳畫好框的圖（BGR）和每個刮傷的信心分數（由高到低）"""
    result = model.predict(image, conf=CONF, verbose=False)[0]
    scores = [float(b.conf[0]) for b in result.boxes if model.names[int(b.cls[0])] == "scratch"]
    return result.plot(), sorted(scores, reverse=True)


def read_sop():
    with open(SOP_FILE, encoding="utf-8") as f:
        return f.read()


def live_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    out, scores = detect(img)
    if scores:
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        # putText 不支援中文
        cv2.putText(out, f"SCRATCH x{len(scores)}", (16, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    return av.VideoFrame.from_ndarray(out, format="bgr24")


tab_upload, tab_camera = st.tabs(["上傳照片", "鏡頭即時偵測"])

with tab_upload:
    files = st.file_uploader(
        "可以一次選多張（手機上可以直接拍照）",
        type=["jpg", "png", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    if files and st.button(f"開始檢查（{len(files)} 張）"):
        bar = st.progress(0.0)
        results = []
        for i, f in enumerate(files):
            out, scores = detect(Image.open(f).convert("RGB"))
            results.append((f.name, out, scores))
            bar.progress((i + 1) / len(files), text=f"{i + 1} / {len(files)}")
        bar.empty()

        bad = [name for name, _, s in results if s]
        st.subheader("檢查結果")
        st.write(f"共 {len(results)} 張，有刮傷 {len(bad)} 張，合格 {len(results) - len(bad)} 張")
        # 手機畫面窄，表格只放兩欄
        st.dataframe(
            [{"檔名": name, "結果": f"刮傷 {len(s)} 處（{s[0]:.2f}）" if s else "合格"}
             for name, _, s in results],
            use_container_width=True,
            hide_index=True,
        )

        for name, out, scores in results:
            with st.expander(f"{'有刮傷' if scores else '合格'}｜{name}", expanded=bool(scores)):
                st.image(out, channels="BGR", use_container_width=True)
                if scores:
                    st.write(f"找到 {len(scores)} 處，最高信心分數 {scores[0]:.2f}")

        if bad:
            st.subheader("處理方式")
            st.write(f"有刮傷的照片：{'、'.join(bad)}")
            st.write(read_sop())
            st.caption("依據 SOP-001")

with tab_camera:
    st.write("手機請用這個版本，模型直接在手機上跑，比較順：")
    st.link_button("開啟手機版", MOBILE_LIVE_URL, type="primary")
    st.divider()
    st.caption("電腦版：按 START 開啟網路攝影機，偵測到刮傷時畫面外框會變紅")
    webrtc_streamer(
        key="live",
        video_frame_callback=live_frame,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )
    with st.expander("SOP-001 刮傷處理方式"):
        st.write(read_sop())
