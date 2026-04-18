# stock_vision_app_fixed_final.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import pandas as pd
from datetime import datetime

# ------------------- โหลดโมเดล -------------------
@st.cache_resource
def load_model():
    model = YOLO('best.pt')
    return model

model = load_model()
CLASS_NAMES = [
    "Canned tea", "Coconut Water Carton", "Coffee Can", "Drinking water",
    "Empty_Stock", "Energy Drink", "Green Tea Bottle", "Juice Box",
    "Protein Drink", "Soda Can", "UHT milk carton", "Vitamin Drink"
]
PRODUCT_CLASSES = [name for name in CLASS_NAMES if name != "Empty_Stock"]
assert len(PRODUCT_CLASSES) == 11, "ต้องมีสินค้า 11 ชนิด"

# ------------------- กำหนด Shelf Slot 11 ช่อง (จับคู่ index ให้ถูกต้อง) -------------------
SLOT_RELATIVE_BOXES = [
    {"id": "S01", "name": PRODUCT_CLASSES[0], "rel_bbox": [0.05, 0.10, 0.20, 0.35]},  # Canned tea
    {"id": "S02", "name": PRODUCT_CLASSES[1], "rel_bbox": [0.22, 0.10, 0.37, 0.35]},  # Coconut Water Carton
    {"id": "S03", "name": PRODUCT_CLASSES[2], "rel_bbox": [0.39, 0.10, 0.54, 0.35]},  # Coffee Can
    {"id": "S04", "name": PRODUCT_CLASSES[3], "rel_bbox": [0.56, 0.10, 0.71, 0.35]},  # Drinking water
    {"id": "S05", "name": PRODUCT_CLASSES[4], "rel_bbox": [0.73, 0.10, 0.88, 0.35]},  # Energy Drink (index4)
    {"id": "S06", "name": PRODUCT_CLASSES[5], "rel_bbox": [0.05, 0.40, 0.20, 0.65]},  # Green Tea Bottle (index5)
    {"id": "S07", "name": PRODUCT_CLASSES[6], "rel_bbox": [0.22, 0.40, 0.37, 0.65]},  # Juice Box (index6)
    {"id": "S08", "name": PRODUCT_CLASSES[7], "rel_bbox": [0.39, 0.40, 0.54, 0.65]},  # Protein Drink (index7)
    {"id": "S09", "name": PRODUCT_CLASSES[8], "rel_bbox": [0.56, 0.40, 0.71, 0.65]},  # Soda Can (index8)
    {"id": "S10", "name": PRODUCT_CLASSES[9], "rel_bbox": [0.73, 0.40, 0.88, 0.65]},  # UHT milk carton (index9)
    {"id": "S11", "name": PRODUCT_CLASSES[10], "rel_bbox": [0.05, 0.70, 0.20, 0.95]}, # Vitamin Drink (index10)
]

def rel_to_abs(rel_bbox, img_w, img_h):
    x1 = int(rel_bbox[0] * img_w)
    y1 = int(rel_bbox[1] * img_h)
    x2 = int(rel_bbox[2] * img_w)
    y2 = int(rel_bbox[3] * img_h)
    return [x1, y1, x2, y2]

def check_slot_occupancy(detection_boxes, slot_abs_bbox, iou_thresh=0.05):
    sx1, sy1, sx2, sy2 = slot_abs_bbox
    slot_area = (sx2 - sx1) * (sy2 - sy1)
    if slot_area <= 0:
        return False
    for (dx1, dy1, dx2, dy2) in detection_boxes:
        ix1 = max(sx1, dx1)
        iy1 = max(sy1, dy1)
        ix2 = min(sx2, dx2)
        iy2 = min(sy2, dy2)
        if ix2 > ix1 and iy2 > iy1:
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            iou = inter_area / slot_area
            if iou > iou_thresh:
                return True
    return False

def analyze_shelf_image(img_array, conf_threshold=0.25, hide_boxes=False, thickness=2):
    results = model(img_array, conf=conf_threshold)
    h, w = img_array.shape[:2]
    
    product_boxes = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = CLASS_NAMES[cls_id]
            if class_name != "Empty_Stock":
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                product_boxes.append([x1, y1, x2, y2])
    
    slot_statuses = []
    empty_slots = []
    img_draw = img_array.copy()
    if len(img_draw.shape) == 3 and img_draw.shape[2] == 3:
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    
    for slot in SLOT_RELATIVE_BOXES:
        abs_bbox = rel_to_abs(slot["rel_bbox"], w, h)
        occupied = check_slot_occupancy(product_boxes, abs_bbox, iou_thresh=0.05)
        status = occupied
        slot_statuses.append({
            "id": slot["id"],
            "name": slot["name"],
            "status": status,
            "bbox": abs_bbox
        })
        if not status:
            empty_slots.append(slot["name"])
        
        if not hide_boxes:
            color = (0, 255, 0) if status else (0, 0, 255)
            cv2.rectangle(img_draw, (abs_bbox[0], abs_bbox[1]), (abs_bbox[2], abs_bbox[3]), color, thickness)
            label = f"{slot['id']}: {'In-stock' if status else 'Out-of-stock'}"
            text_x = abs_bbox[0]
            text_y = abs_bbox[3] + 15
            if text_y + 10 > h:
                text_y = abs_bbox[1] - 5
            cv2.putText(img_draw, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    
    return img_draw, slot_statuses, empty_slots

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Stock Vision System - Fixed Slot", layout="wide")
st.title("📦 Stock Vision System (Fixed Slot)")
st.markdown("**สินค้า 11 ชนิด** | ตรวจจับโดยกำหนด Shelf Slot ล่วงหน้า")

with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    display_size = st.selectbox("ขนาดภาพ", ["เล็ก (400px)", "กลาง (600px)", "ใหญ่ (800px)"])
    width_map = {"เล็ก (400px)": 400, "กลาง (600px)": 600, "ใหญ่ (800px)": 800}
    display_width = width_map[display_size]
    thickness = 1 if display_width <= 400 else 2
    hide_boxes = st.checkbox("ซ่อน Bounding Box (แสดงภาพต้นฉบับ)", value=False)

mode = st.radio("โหมดการทำงาน", ["📸 อัปโหลดภาพ", "📷 ถ่ายภาพจากกล้อง", "🎥 Real‑time (Webcam)"])

if 'last_empty' not in st.session_state:
    st.session_state.last_empty = []
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []

def add_alerts(empty_slots):
    if set(empty_slots) != set(st.session_state.last_empty):
        new_empty = set(empty_slots) - set(st.session_state.last_empty)
        for slot_name in new_empty:
            msg = f"⚠️ สินค้าหมด: {slot_name}"
            st.session_state.alert_history.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "message": msg})
            st.toast(msg, icon="🔴")
        st.session_state.last_empty = empty_slots.copy()
        if len(st.session_state.alert_history) > 20:
            st.session_state.alert_history.pop()

def show_dashboard(slot_statuses):
    total = len(slot_statuses)
    occupied = sum(1 for s in slot_statuses if s["status"])
    empty = total - occupied
    c1, c2, c3 = st.columns(3)
    c1.metric("ช่องทั้งหมด", total)
    c2.metric("✅ มีสินค้า", occupied)
    c3.metric("❌ สินค้าหมด", empty)
    st.subheader("แผนผังชั้นวาง (รหัสสี)")
    cols = st.columns(4)
    for idx, s in enumerate(slot_statuses):
        with cols[idx % 4]:
            color = "#d4edda" if s["status"] else "#f8d7da"
            st.markdown(f"""
            <div style="background-color:{color}; padding:10px; border-radius:10px; margin:5px; text-align:center;">
                <b>{s['id']}</b><br>{s['name']}<br>
                <span style="color:{'green' if s['status'] else 'red'}">● {'มีสินค้า' if s['status'] else 'หมด'}</span>
            </div>
            """, unsafe_allow_html=True)
    st.subheader("🔔 ประวัติแจ้งเตือน")
    if st.session_state.alert_history:
        st.dataframe(pd.DataFrame(st.session_state.alert_history), use_container_width=True)
    else:
        st.info("ไม่มีการแจ้งเตือน")

# ------------------- โหมดอัปโหลด -------------------
if mode == "📸 อัปโหลดภาพ":
    up = st.file_uploader("เลือกภาพชั้นวาง", type=["jpg","jpeg","png"])
    if up:
        img = Image.open(up).convert("RGB")
        arr = np.array(img)
        if hide_boxes:
            _, statuses, empty = analyze_shelf_image(arr, confidence_threshold, hide_boxes=True, thickness=thickness)
            st.image(arr, caption="ภาพต้นฉบับ (ไม่มีกรอบ)", width=display_width)
        else:
            annotated_bgr, statuses, empty = analyze_shelf_image(arr, confidence_threshold, hide_boxes=False, thickness=thickness)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="ผลลัพธ์ (Bounding Box ช่อง)", width=display_width)
        show_dashboard(statuses)
        if empty:
            st.warning(f"พบช่องว่าง {len(empty)} ช่อง: {', '.join(empty)}")
            add_alerts(empty)
        else:
            st.success("สินค้าครบทุกช่อง")

# ------------------- โหมดถ่ายภาพ -------------------
elif mode == "📷 ถ่ายภาพจากกล้อง":
    captured = st.camera_input("ถ่ายภาพ")
    if captured:
        img = Image.open(captured).convert("RGB")
        arr = np.array(img)
        if hide_boxes:
            _, statuses, empty = analyze_shelf_image(arr, confidence_threshold, hide_boxes=True, thickness=thickness)
            st.image(arr, caption="ภาพต้นฉบับ", width=display_width)
        else:
            annotated_bgr, statuses, empty = analyze_shelf_image(arr, confidence_threshold, hide_boxes=False, thickness=thickness)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="ผลการตรวจจับ", width=display_width)
        show_dashboard(statuses)
        if empty:
            st.warning(f"ช่องว่าง: {', '.join(empty)}")
            add_alerts(empty)
        else:
            st.success("ครบ")

# ------------------- โหมด Real-time -------------------
elif mode == "🎥 Real‑time (Webcam)":
    st.info("เปิดกล้องเพื่อดูการตรวจจับแบบเรียลไทม์ (แจ้งเตือนอัตโนมัติไม่ทำงานใน live feed)")
    class FixedLiveTransformer(VideoTransformerBase):
        def __init__(self):
            self.conf = confidence_threshold
            self.display_w = display_width
            self.hide = hide_boxes
            self.thick = thickness
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            if self.hide:
                img_resized = cv2.resize(img, (self.display_w, int(img.shape[0] * self.display_w / img.shape[1])))
                return av.VideoFrame.from_ndarray(img_resized, format="bgr24")
            else:
                annotated_bgr, _, _ = analyze_shelf_image(img, self.conf, hide_boxes=False, thickness=self.thick)
                h, w = annotated_bgr.shape[:2]
                new_w = self.display_w
                new_h = int(h * (new_w / w))
                annotated_bgr = cv2.resize(annotated_bgr, (new_w, new_h))
                return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")
    webrtc_streamer(key="fixed-live", mode=WebRtcMode.SENDRECV, video_transformer_factory=FixedLiveTransformer, async_processing=True)

st.markdown("---")
with st.expander("📄 รายงานการตรวจสอบ"):
    if st.button("สร้างรายงาน"):
        if st.session_state.alert_history:
            df = pd.DataFrame(st.session_state.alert_history)
            st.dataframe(df)
            st.write("**สรุปสินค้าที่ขาดบ่อย**")
            st.bar_chart(df['message'].value_counts())
        else:
            st.info("ไม่มีประวัติ")