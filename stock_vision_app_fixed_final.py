import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import json
import os

st.set_page_config(page_title="Stock Vision System - Advanced", layout="wide")

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

# ------------------- กำหนด 11 ช่อง พร้อมชื่อสินค้าตามภาพของคุณ -------------------
SLOT_CONFIG = [
    {"id": "S01", "name": "สมุนไพร", "rel_bbox": [0.05, 0.10, 0.20, 0.35]},
    {"id": "S02", "name": "Coconut Water Carton", "rel_bbox": [0.22, 0.10, 0.37, 0.35]},
    {"id": "S03", "name": "Coffee Can", "rel_bbox": [0.39, 0.10, 0.54, 0.35]},
    {"id": "S04", "name": "Drinking water", "rel_bbox": [0.56, 0.10, 0.71, 0.35]},
    {"id": "S05", "name": "Energy Drink", "rel_bbox": [0.73, 0.10, 0.88, 0.35]},
    {"id": "S06", "name": "Green Tea Bottle", "rel_bbox": [0.05, 0.40, 0.20, 0.65]},
    {"id": "S07", "name": "Juice Box", "rel_bbox": [0.22, 0.40, 0.37, 0.65]},
    {"id": "S08", "name": "Protein Drink", "rel_bbox": [0.39, 0.40, 0.54, 0.65]},
    {"id": "S09", "name": "Soda Can", "rel_bbox": [0.56, 0.40, 0.71, 0.65]},
    {"id": "S10", "name": "UHT milk carton", "rel_bbox": [0.73, 0.40, 0.88, 0.65]},
    {"id": "S11", "name": "Vitamin Drink", "rel_bbox": [0.05, 0.70, 0.20, 0.95]},
]

def rel_to_abs(rel_bbox, img_w, img_h):
    return [int(rel_bbox[0]*img_w), int(rel_bbox[1]*img_h),
            int(rel_bbox[2]*img_w), int(rel_bbox[3]*img_h)]

def check_slot_occupancy(detection_boxes, slot_abs_bbox, iou_thresh=0.05):
    sx1, sy1, sx2, sy2 = slot_abs_bbox
    slot_area = max(1, (sx2-sx1)*(sy2-sy1))
    for (dx1, dy1, dx2, dy2) in detection_boxes:
        ix1, iy1 = max(sx1, dx1), max(sy1, dy1)
        ix2, iy2 = min(sx2, dx2), min(sy2, dy2)
        if ix2 > ix1 and iy2 > iy1:
            inter_area = (ix2-ix1)*(iy2-iy1)
            if inter_area / slot_area > iou_thresh:
                return True
    return False

def analyze_image(img_array, conf_threshold=0.25):
    results = model(img_array, conf=conf_threshold)
    h, w = img_array.shape[:2]
    product_boxes = []
    if results[0].boxes:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if CLASS_NAMES[cls_id] != "Empty_Stock":
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                product_boxes.append([x1,y1,x2,y2])
    
    slot_statuses = []
    for slot in SLOT_CONFIG:
        abs_bbox = rel_to_abs(slot["rel_bbox"], w, h)
        occupied = check_slot_occupancy(product_boxes, abs_bbox)
        slot_statuses.append({
            "id": slot["id"],
            "name": slot["name"],
            "status": occupied,
            "bbox": abs_bbox
        })
    return slot_statuses, product_boxes

def draw_slots_only(img_array, slot_statuses, thickness=2):
    img_draw = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    for slot in slot_statuses:
        x1,y1,x2,y2 = slot["bbox"]
        color = (0,255,0) if slot["status"] else (0,0,255)
        cv2.rectangle(img_draw, (x1,y1), (x2,y2), color, thickness)
        label = f"{slot['id']}: {'✓' if slot['status'] else '✗'}"
        cv2.putText(img_draw, label, (x1, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)

# ------------------- ระบบบันทึกสถานะเพื่อเปรียบเทียบ -------------------
HISTORY_FILE = "stock_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def compare_with_last(slot_statuses, last_status):
    changes = []
    for slot in slot_statuses:
        sid = slot["id"]
        current = slot["status"]
        previous = last_status.get(sid, None)
        if previous is not None and current != previous:
            changes.append({
                "id": sid,
                "name": slot["name"],
                "from": "มีสินค้า" if previous else "หมด",
                "to": "มีสินค้า" if current else "หมด"
            })
    return changes

# ------------------- UI -------------------
st.title("📊 Stock Vision System - Advanced")
st.markdown("ตรวจสอบสินค้า 11 ช่อง แยกตารางสถานะ และเปรียบเทียบอัตโนมัติ")

with st.sidebar:
    conf = st.slider("Confidence", 0.0, 1.0, 0.25)
    display_width = st.selectbox("ขนาดรูป", [400,600,800], index=1)
    show_boxes = st.checkbox("แสดงกรอบช่องบนรูปภาพ", value=True)
    thickness = 2 if display_width > 500 else 1

mode = st.radio("แหล่งภาพ", ["📸 อัปโหลด", "📷 ถ่ายภาพ"])

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("🖼️ ภาพที่วิเคราะห์")
    img = None
    if mode == "📸 อัปโหลด":
        uploaded = st.file_uploader("เลือกไฟล์ภาพ", type=["jpg","png","jpeg"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
    else:
        img = st.camera_input("ถ่ายภาพ")
        if img:
            img = Image.open(img).convert("RGB")
    
    if img is not None:
        arr = np.array(img)
        slot_statuses, _ = analyze_image(arr, conf)
        if show_boxes:
            display_img = draw_slots_only(arr, slot_statuses, thickness)
        else:
            display_img = arr
        st.image(display_img, width=display_width)

with col_right:
    st.subheader("📋 สถานะสินค้า 11 ช่อง")
    if img is not None:
        # แสดงตารางสถานะแบบชัดเจน
        data = []
        for s in slot_statuses:
            data.append({
                "ช่อง": s["id"],
                "สินค้า": s["name"],
                "สถานะ": "✅ มีสินค้า" if s["status"] else "❌ สินค้าหมด"
            })
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, height=500)
        
        # สรุปสถิติ
        total = len(slot_statuses)
        instock = sum(1 for s in slot_statuses if s["status"])
        outofstock = total - instock
        col1, col2, col3 = st.columns(3)
        col1.metric("ช่องทั้งหมด", total)
        col2.metric("มีสินค้า", instock, delta=None)
        col3.metric("สินค้าหมด", outofstock, delta=None)
        
        # เปรียบเทียบกับครั้งก่อน
        history = load_history()
        last_key = "last_status"
        if last_key in history:
            changes = compare_with_last(slot_statuses, history[last_key])
            if changes:
                st.warning("🔄 สถานะเปลี่ยนแปลง:")
                for c in changes:
                    st.write(f"  - {c['id']} ({c['name']}): {c['from']} → {c['to']}")
            else:
                st.success("✅ ไม่มีการเปลี่ยนแปลงจากครั้งก่อน")
        else:
            st.info("📌 นี่คือการตรวจสอบครั้งแรก")
        
        # บันทึกสถานะปัจจุบัน
        current_state = {s["id"]: s["status"] for s in slot_statuses}
        history[last_key] = current_state
        history["last_update"] = datetime.now().isoformat()
        save_history(history)
        
        # แจ้งเตือนสินค้าหมด
        empty_slots = [s["name"] for s in slot_statuses if not s["status"]]
        if empty_slots:
            st.error(f"⚠️ สินค้าหมด {len(empty_slots)} ช่อง: {', '.join(empty_slots)}")
        else:
            st.balloons()
            st.success("🎉 สินค้าครบทุกช่อง!")
    else:
        st.info("กรุณาอัปโหลดหรือถ่ายภาพเพื่อเริ่มตรวจสอบ")