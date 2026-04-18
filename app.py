# app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# ------------------- โหลดโมเดล -------------------
@st.cache_resource
def load_model():
    model = YOLO('best.pt')
    return model

model = load_model()
class_names = model.names

st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🔍 ตรวจจับวัตถุด้วย YOLO")
st.markdown("รองรับ 12 ประเภท (กระป๋อง, กล่องนม, ขวดน้ำ, ฯลฯ)")

# ตัวเลือกความเชื่อมั่น
confidence_threshold = st.slider("ค่า Confidence threshold", 0.0, 1.0, 0.25, 0.01)

# เลือกขนาดภาพที่แสดง (เพิ่ม option)
image_size = st.selectbox("ขนาดภาพที่แสดงผล", ["เล็ก (400px)", "กลาง (600px)", "ใหญ่ (800px)"])
width_map = {"เล็ก (400px)": 400, "กลาง (600px)": 600, "ใหญ่ (800px)": 800}
display_width = width_map[image_size]

# ------------------- โหมดการทำงาน -------------------
mode = st.radio("เลือกโหมดการทำงาน:", 
                ["📸 อัปโหลดภาพ", "📷 ถ่ายภาพจากกล้อง", "🎥 Real‑time (Webcam)"])

# ------------------- ฟังก์ชันตรวจจับ -------------------
def process_frame(frame: np.ndarray) -> np.ndarray:
    """รับภาพ BGR หรือ RGB คืนภาพที่วาด bounding boxes แล้ว (BGR)"""
    results = model(frame, conf=confidence_threshold)
    annotated = results[0].plot()  # ออกมาเป็น BGR
    return annotated

def display_results(results, col):
    """แสดงผลลัพธ์และตาราง"""
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        data = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = class_names[cls_id]
            data.append({"คลาส": name, "ความมั่นใจ": f"{conf:.2f}"})
        col.dataframe(data, use_container_width=True)
    else:
        col.info("ไม่พบวัตถุในภาพ (ลองลด Confidence threshold ลง)")

# ------------------- โหมดอัปโหลดภาพ -------------------
if mode == "📸 อัปโหลดภาพ":
    uploaded_file = st.file_uploader("เลือกไฟล์ภาพ", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        results = model(img_array, conf=confidence_threshold)
        annotated_bgr = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="ภาพต้นฉบับ", width=display_width)
        with col2:
            st.image(annotated_rgb, caption="ผลการตรวจจับ", width=display_width)
        display_results(results, col2)

# ------------------- โหมดถ่ายภาพจากกล้อง -------------------
elif mode == "📷 ถ่ายภาพจากกล้อง":
    captured_image = st.camera_input("กดปุ่ม shutter เพื่อถ่ายภาพ")
    if captured_image is not None:
        image = Image.open(captured_image).convert("RGB")
        img_array = np.array(image)
        results = model(img_array, conf=confidence_threshold)
        annotated_bgr = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="ภาพที่ถ่าย", width=display_width)
        with col2:
            st.image(annotated_rgb, caption="ผลการตรวจจับ", width=display_width)
        display_results(results, col2)

# ------------------- โหมด Real‑time (streamlit_webrtc) -------------------
elif mode == "🎥 Real‑time (Webcam)":
    st.info("เปิดกล้องเพื่อตรวจจับแบบเรียลไทม์ (ต้องติดตั้ง `streamlit-webrtc` และ `av`)")
    
    class YOLOTransformer(VideoTransformerBase):
        def __init__(self):
            self.model = model
            self.conf = confidence_threshold
            # ขนาด output ที่เราต้องการ (ปรับให้เล็กลงเพื่อความเร็ว)
            self.output_width = display_width

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            # ทำนายและวาด bounding boxes
            results = self.model(img, conf=self.conf)
            annotated = results[0].plot()  # BGR
            
            # ปรับขนาดภาพให้เล็กลงตามที่ผู้ใช้เลือก
            h, w = annotated.shape[:2]
            new_w = self.output_width
            new_h = int(h * (new_w / w))
            annotated = cv2.resize(annotated, (new_w, new_h))
            
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="yolo-live",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=YOLOTransformer,
        async_processing=True,
        media_stream_constraints={"video": True, "audio": False},
    )

    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.conf = confidence_threshold
        webrtc_ctx.video_transformer.output_width = display_width

# คำแนะนำการติดตั้งเพิ่มเติม
st.markdown("---")
st.caption("💡 **หมายเหตุ**: สำหรับโหมด Real‑time ต้องติดตั้ง `streamlit-webrtc` และ `av` ด้วยคำสั่ง: `pip install streamlit-webrtc av`")