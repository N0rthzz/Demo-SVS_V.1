# stock_vision_app_history_mode.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import uuid  # ใช้สำหรับสร้าง ID เฉพาะให้แต่ละภาพเพื่อการลบ

# ------------------- ตั้งค่า page config -------------------
st.set_page_config(page_title="Stock Vision - History Mode", layout="wide")

@st.cache_resource
def load_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "best.pt")
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_model()

PRODUCT_LIST = [
    {"id": "S01", "name": "Canned tea"}, {"id": "S02", "name": "Coconut Water Carton"},
    {"id": "S03", "name": "Coffee Can"}, {"id": "S04", "name": "Drinking water"},
    {"id": "S05", "name": "Energy Drink"}, {"id": "S06", "name": "Green Tea Bottle"},
    {"id": "S07", "name": "Juice Box"}, {"id": "S08", "name": "Protein Drink"},
    {"id": "S09", "name": "Soda Can"}, {"id": "S10", "name": "UHT milk carton"},
    {"id": "S11", "name": "Vitamin Drink"}
]

# ------------------- เตรียม Session State -------------------
if 'image_history' not in st.session_state:
    st.session_state.image_history = []  # เก็บลิสต์ของ dict {id, image, found_items}

# ------------------- ฟังก์ชันจัดการข้อมูล -------------------
def process_new_image(img_file):
    img = Image.open(img_file).convert("RGB")
    arr = np.array(img)
    found_items = set()
    
    if model:
        results = model(arr, conf=confidence_threshold)
        if results[0].boxes is not None:
            for box in results[0].boxes:
                label = results[0].names[int(box.cls[0])]
                if label != "Empty_Stock":
                    found_items.add(label)
    
    # เพิ่มลงในประวัติพร้อม ID สุ่มสำหรับการลบ
    st.session_state.image_history.append({
        "id": str(uuid.uuid4()),
        "image": img,
        "found": found_items,
        "time": datetime.now().strftime("%H:%M:%S")
    })

def delete_image(img_id):
    st.session_state.image_history = [item for item in st.session_state.image_history if item["id"] != img_id]

# ------------------- UI Sidebar -------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    confidence_threshold = st.slider("AI Confidence", 0.0, 1.0, 0.25)
    if st.button("🗑️ ล้างภาพทั้งหมด", use_container_width=True):
        st.session_state.image_history = []
        st.rerun()

st.title("📦 Stock Vision - Multi-Image History")

# ------------------- ส่วน Input -------------------
mode = st.radio("เลือกแหล่งที่มาภาพ", ["📸 อัปโหลดภาพ", "📷 ถ่ายภาพจากกล้อง"], horizontal=True)
if mode == "📸 อัปโหลดภาพ":
    up_file = st.file_uploader("เพิ่มภาพใหม่เข้าสู่ระบบ", type=["jpg","png","jpeg"], key="uploader")
    if up_file:
        process_new_image(up_file)
        # เคลียร์ค่า uploader เพื่อให้พร้อมรับภาพต่อไป (ต้องอาศัยเทคนิค rerun)
else:
    cam_file = st.camera_input("ถ่ายภาพใหม่")
    if cam_file:
        process_new_image(cam_file)

# ------------------- คำนวณสถานะรวมจากทุกภาพ -------------------
all_found_items = set()
for item in st.session_state.image_history:
    all_found_items.update(item["found"])

# ------------------- แสดง Dashboard (สถานะรวม) -------------------
st.divider()
st.subheader("📊 สถานะสต็อกรวม (จากภาพทั้งหมดที่เก็บไว้)")
dash_cols = st.columns(6)
for idx, p in enumerate(PRODUCT_LIST):
    is_active = p["name"] in all_found_items
    with dash_cols[idx % 6]:
        color = "#28a745" if is_active else "#dc3545"
        st.markdown(f"""
            <div style="background-color:{color}; color:white; padding:10px; border-radius:8px; text-align:center; font-size:13px; margin-bottom:5px;">
                <b>{p['id']}</b><br>{p['name']}
            </div>
        """, unsafe_allow_html=True)

# ------------------- ส่วนแสดงประวัติภาพ (Image History) -------------------
st.divider()
st.subheader("🖼️ ประวัติภาพถ่าย")
if not st.session_state.image_history:
    st.info("ยังไม่มีข้อมูลภาพในระบบ")
else:
    # แสดงภาพเรียงจากใหม่ไปเก่า
    for item in reversed(st.session_state.image_history):
        with st.container():
            col_img, col_info = st.columns([1, 2])
            with col_img:
                st.image(item["image"], use_container_width=True)
            with col_info:
                st.write(f"🕒 **เวลา:** {item['time']}")
                st.write(f"🔎 **สินค้าที่พบ:** {', '.join(item['found']) if item['found'] else 'ไม่พบ'}")
                if st.button(f"🗑️ ลบภาพนี้", key=item["id"]):
                    delete_image(item["id"])
                    st.rerun()
            st.markdown("---")