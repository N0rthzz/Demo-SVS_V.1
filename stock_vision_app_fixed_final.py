import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import uuid
from datetime import datetime # แก้ไข Error ตรงนี้ครับ

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
    st.session_state.image_history = []

# ------------------- ฟังก์ชันจัดการข้อมูล -------------------
def process_new_image(img_file, conf_val):
    img = Image.open(img_file).convert("RGB")
    arr = np.array(img)
    found_items = set()
    
    if model:
        results = model(arr, conf=conf_val)
        if results[0].boxes is not None:
            for box in results[0].boxes:
                label = results[0].names[int(box.cls[0])]
                if label != "Empty_Stock":
                    found_items.add(label)
    
    # บันทึกลง history
    st.session_state.image_history.append({
        "id": str(uuid.uuid4()),
        "image": img,
        "found": found_items,
        "time": datetime.now().strftime("%H:%M:%S")
    })

def delete_image(img_id):
    st.session_state.image_history = [item for item in st.session_state.image_history if item["id"] != img_id]

# ------------------- UI Layout -------------------
st.title("📦 Stock Vision - Multi-Image History")

with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("AI Confidence", 0.0, 1.0, 0.25)
    if st.button("🗑️ ล้างประวัติทั้งหมด", use_container_width=True):
        st.session_state.image_history = []
        st.rerun()

# ส่วนการเพิ่มรูปภาพ
mode = st.radio("เลือกช่องทาง", ["📸 อัปโหลดภาพ", "📷 ถ่ายภาพจากกล้อง"], horizontal=True)
if mode == "📸 อัปโหลดภาพ":
    up_file = st.file_uploader("เพิ่มรูปภาพ", type=["jpg","png","jpeg"])
    if up_file:
        # ป้องกันการประมวลผลซ้ำเมื่อ rerun
        file_id = f"file_{up_file.name}_{up_file.size}"
        if 'last_file_id' not in st.session_state or st.session_state.last_file_id != file_id:
            process_new_image(up_file, conf_threshold)
            st.session_state.last_file_id = file_id
else:
    cam_file = st.camera_input("ถ่ายภาพ")
    if cam_file:
        process_new_image(cam_file, conf_threshold)

# ------------------- คำนวณและแสดง Dashboard -------------------
all_found = set()
for item in st.session_state.image_history:
    all_found.update(item["found"])

st.divider()
st.subheader("📊 สถานะสต็อกรวม (Combined Status)")
dash_cols = st.columns(6)
for idx, p in enumerate(PRODUCT_LIST):
    is_active = p["name"] in all_found
    with dash_cols[idx % 6]:
        color = "#28a745" if is_active else "#dc3545"
        st.markdown(f"""
            <div style="background-color:{color}; color:white; padding:10px; border-radius:8px; text-align:center; font-size:13px; margin-bottom:5px; min-height:80px;">
                <b>{p['id']}</b><br>{p['name']}<br>
                <small>{'DETECTED' if is_active else 'EMPTY'}</small>
            </div>
        """, unsafe_allow_html=True)

# ------------------- ส่วนประวัติรูปภาพ -------------------
st.divider()
st.subheader("🖼️ รายการรูปภาพที่วิเคราะห์")
if not st.session_state.image_history:
    st.info("ยังไม่มีรูปภาพในระบบ")
else:
    for item in reversed(st.session_state.image_history):
        with st.container():
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(item["image"], use_container_width=True)
            with c2:
                st.write(f"🕒 **เวลาอัปโหลด:** {item['time']}")
                st.write(f"✅ **สินค้าที่พบในภาพนี้:**")
                if item['found']:
                    st.write(", ".join(item['found']))
                else:
                    st.write("- ไม่พบสินค้า -")
                
                if st.button(f"🗑️ ลบรูปภาพนี้", key=item["id"]):
                    delete_image(item["id"])
                    st.rerun()
            st.markdown("---")