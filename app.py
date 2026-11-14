import av
import cv2
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO

CONF = 0.25  # 確定度門檻，低於這個的框不算
SOP_FILE = "data/sop/SOP-001-刮傷.txt"
# 手機即時偵測頁（docs/ 放在 GitHub Pages 上）
MOBILE_LIVE_URL = "https://shakeeeeee.github.io/ai-qa-assistant/"

st.set_page_config(page_title="AI 品檢助手", page_icon="🔍")
st.title("AI 品檢助手")
st.write("上傳照片或用鏡頭拍照，YOLO 會幫你檢查有沒有刮傷")


# 載入 YOLO 模型，只載入一次就好，不用每次都重載
@st.cache_resource
def load_model():
    # v2-real：只用真實照片訓練（Roboflow + 自己標的），不含白紙合成圖
    return YOLO("models/scratch_v2.pt")


model = load_model()


def detect(image: Image.Image):
    """讓 YOLO 檢查一張照片，回傳（畫好框的照片, 刮傷確定度清單）。"""
    result = model.predict(image, conf=CONF, verbose=False)[0]
    scratches = []
    for box in result.boxes:
        label = model.names[int(box.cls[0].item())]
        if label == "scratch":
            scratches.append(float(box.conf[0].item()))
    # result.plot() 回傳的是 BGR 格式，要跟 st.image 說一聲，顏色才不會反掉
    return result.plot(), sorted(scratches, reverse=True)


def show_sop(found_in: list[str]):
    """有任何照片發現刮傷時，顯示 SOP 處理方式。"""
    with open(SOP_FILE, "r", encoding="utf-8") as f:
        sop_content = f.read()
    st.subheader("📋 檢查報告")
    st.write(f"**檢測結果：** 發現刮傷（{'、'.join(found_in)}）")
    st.write("**建議處理方式：**")
    st.write(sop_content)
    st.caption("依據：SOP-001-刮傷.txt")


def show_one(name: str, annotated, scratches: list[float]):
    st.image(annotated, channels="BGR", caption=f"{name} 的檢查結果", use_container_width=True)
    if scratches:
        st.error(f"發現 {len(scratches)} 處刮傷，最高確定度 {scratches[0]:.2f}")
    else:
        st.success("沒有發現刮傷")


def live_frame(frame: av.VideoFrame) -> av.VideoFrame:
    """鏡頭的每一格畫面都會進來這裡：檢查刮傷、畫框，再送回網頁顯示。"""
    img = frame.to_ndarray(format="bgr24")
    annotated, scratches = detect(img)
    if scratches:
        # 有刮傷就加紅色外框和提示字（cv2 寫不了中文，所以用英文）
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        cv2.putText(annotated, f"SCRATCH x{len(scratches)}  max {scratches[0]:.2f}",
                    (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


tab_upload, tab_camera = st.tabs(["📁 上傳照片（可多張）", "🎥 鏡頭即時偵測"])

# ---------- 分頁 1：一次上傳多張 ----------
with tab_upload:
    st.caption("用手機打開這個網頁時，按「Upload」可以直接選擇用相機拍照。")
    files = st.file_uploader(
        "請上傳照片（可以一次選多張）",
        type=["jpg", "png", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    if files and st.button(f"開始檢查（共 {len(files)} 張）"):
        progress = st.progress(0.0, text="YOLO 正在看照片...")
        results = []
        for i, f in enumerate(files):
            image = Image.open(f).convert("RGB")
            annotated, scratches = detect(image)
            results.append((f.name, annotated, scratches))
            progress.progress((i + 1) / len(files), text=f"已檢查 {i + 1} / {len(files)} 張")
        progress.empty()

        # 總表
        failed = [name for name, _, s in results if s]
        st.subheader("📊 檢查總表")
        # 用一行文字而不是 st.columns，手機上才不會變成直排佔掉整個畫面
        st.markdown(
            f"共 **{len(results)}** 張　｜　❌ 有刮傷 **{len(failed)}** 張　｜　"
            f"✅ 合格 **{len(results) - len(failed)}** 張"
        )
        # 只留兩欄，手機寬度也放得下，不用左右滑
        st.dataframe(
            [
                {
                    "檔名": name,
                    "結果": f"❌ {len(s)} 處（{s[0]:.2f}）" if s else "✅ 合格",
                }
                for name, _, s in results
            ],
            use_container_width=True,
            hide_index=True,
        )

        # 每張的細節（有刮傷的預設展開）
        st.subheader("🔍 每張照片")
        for name, annotated, scratches in results:
            with st.expander(f"{'❌' if scratches else '✅'} {name}", expanded=bool(scratches)):
                show_one(name, annotated, scratches)

        if failed:
            show_sop(failed)

# ---------- 分頁 2：鏡頭即時偵測 ----------
with tab_camera:
    if MOBILE_LIVE_URL:
        st.info("📱 用手機的話，請開啟手機版即時偵測：模型直接在手機上執行，速度更快，也不需要網路傳影像。")
        st.link_button("開啟手機版即時偵測", MOBILE_LIVE_URL, type="primary")
        st.divider()
    st.caption("電腦版：按「START」打開網路攝影機，掃到刮傷就會直接框出來，畫面外圍也會變紅色。"
               "第一次使用時瀏覽器會問你是否允許使用鏡頭。")
    webrtc_streamer(
        key="live-scratch",
        video_frame_callback=live_frame,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )
    with st.expander("📋 發現刮傷時的處理方式（SOP-001）"):
        with open(SOP_FILE, "r", encoding="utf-8") as f:
            st.write(f.read())
