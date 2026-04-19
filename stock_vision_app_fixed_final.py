# stock_vision_app_final_hybrid.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import os

# ------------------- ตั้งค่า page config -------------------
st.set_page_config(page_title="Stock Vision System - Hybrid Mode", layout="wide")

# ------------------- โหลดโมเดล -------------------
@st.cache_resource
def load_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "best.pt")
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_model()

CLASS_NAMES = [
    "Canned tea", "Coconut Water Carton", "Coffee Can", "Drinking water",
    "Empty_Stock", "Energy Drink", "Green Tea Bottle", "Juice Box",
    "Protein Drink", "Soda Can", "UHT milk carton", "Vitamin Drink"
]

# รายชื่อสินค้า 11 ชนิดสำหรับ Dashboard
PRODUCT_LIST = [
    {"id": "S01", "name": "Canned tea"},
    {"id": "S02", "name": "Coconut Water Carton"},
    {"id": "S03", "name": "Coffee Can"},
    {"id": "S04", "name": "Drinking water"},
    {"id": "S05", "name": "Energy Drink"},
    {"id": "S06", "name": "Green Tea Bottle"},
    {"id": "S07", "name": "Juice Box"},
    {"id": "S08", "name": "Protein Drink"},
    {"id": "S09", "name": "Soda Can"},
    {"id": "S10", "name": "UHT milk carton"},
    {"id": "S11", "name": "Vitamin Drink"}
]

# ------------------- Logic การวิเคราะห์เบื้องหลัง -------------------
def get_detected_items(img_array, conf_threshold):
    if model is None:
        return set()
    
    results = model(img_array, conf=conf_threshold)
    found_items = set()
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            if label != "Empty_Stock":
                found_items.add(label)
    return found_items

# ------------------- UI ส่วน Dashboard -------------------
def show_status_dashboard(found_items):
    st.subheader("📊 รายงานสถานะสินค้า (Shelf Status)")
    cols = st.columns(4)
    
    for idx, item in enumerate(PRODUCT_LIST):
        is_active = item["name"] in found_items
        with cols[idx % 4]:
            bg_color = "#28a745" if is_active else "#dc3545" # เขียวถ้าเจอ / แดงถ้าไม่เจอ
            st.markdown(f"""
                <div style="background-color:{bg_color}; color:white; padding:15px; border-radius:10px; text-align:center; margin-bottom:10px;">
                    <div style="font-size:12px; opacity:0.8;">{item['id']}</div>
                    <b style="font-size:16px;">{item['name']}</b><br>
                    <span style="font-size:12px;">{'● ตรวจพบ' if is_active else '○ ไม่พบสินค้า'}</span>
                </div>
            """, unsafe_allow_html=True)

# ------------------- UI หลัก -------------------
st.title("📦 Stock Vision Monitoring")
st.markdown("ระบบจะวิเคราะห์รูปภาพและอัปเดตสถานะในตาราง 11 ช่องโดยอัตโนมัติ")

with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    confidence_threshold = st.slider("AI Confidence", 0.0, 1.0, 0.25, 0.01)
    display_width = st.select_slider("ขนาดภาพแสดงผล", options=[400, 600, 800], value=600)
    st.info("โหมดปัจจุบัน: Clean View (ไม่แสดงกรอบทับรูป)")

mode = st.radio("โหมดการทำงาน", ["📸 อัปโหลดภาพ", "📷 ถ่ายภาพจากกล้อง"])

source_img = None
if mode == "📸 อัปโหลดภาพ":
    source_img = st.file_uploader("เลือกภาพชั้นวางสินค้า", type=["jpg","jpeg","png"])
else:
    source_img = st.camera_input("ถ่ายภาพสินค้า")

# ส่วนประมวลผลเมื่อมีการส่งรูปเข้ามาร
if source_img:
    img = Image.open(source_img).convert("RGB")
    arr = np.array(img)
    
    # 1. แสดงรูปต้นฉบับแบบ Clean (ไม่มีกรอบ AI ทับ)
    st.image(img, caption="รูปภาพที่อัปโหลด", width=display_width)
    
    # 2. ให้ AI วิเคราะห์ชื่อสินค้าเบื้องหลัง
    with st.spinner('AI กำลังวิเคราะห์สินค้า...'):
        found_items = get_detected_items(arr, confidence_threshold)
    
    # 3. แสดง Dashboard สถานะ
    st.divider()
    show_status_dashboard(found_items)
    
    # แจ้งเตือนสรุป
    missing = [p['name'] for p in PRODUCT_LIST if p['name'] not in found_items]
    if missing:
        st.warning(f"⚠️ สินค้าที่หายไป: {', '.join(missing)}")
    else:
        st.success("✅ สินค้าครบถ้วนตามรายการ")

# ส่วนรายงานประวัติ (ถ้าต้องการเก็บไว้)
# if 'alert_history' not in st.session_state:
#     st.session_state.alert_history = []