import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os  # แก้ไข Error: name 'os' is not defined ที่นี่ครับ
import uuid
from datetime import datetime

# ------------------- ตั้งค่าหน้าจอ -------------------
st.set_page_config(page_title="Stock Vision - Smart Detection", layout="wide")

@st.cache_resource
def load_model():
    # ตรวจสอบไฟล์ model ในโฟลเดอร์ปัจจุบัน
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    return None

model = load_model()

# รายชื่อสินค้า 11 ชนิดตามโปรเจกต์ของคุณ
PRODUCT_LIST = [
    {"id": "S01", "name": "Drinking water"}, {"id": "S02", "name": "Green Tea Bottle"},
    {"id": "S03", "name": "Energy Drink"}, {"id": "S04", "name": "Vitamin Drink"},
    {"id": "S05", "name": "Protein Drink"}, {"id": "S06", "name": "Soda Can"},
    {"id": "S07", "name": "Coffee Can"}, {"id": "S08", "name": "Canned tea"},
    {"id": "S09", "name": "UHT milk carton"}, {"id": "S10", "name": "Juice Box"},
    {"id": "S11", "name": "Coconut Water Carton"}
]

if 'image_history' not in st.session_state:
    st.session_state.image_history = []

# ------------------- ฟังก์ชันประมวลผล -------------------
def process_image(img_input, conf_val):
    img = Image.open(img_input).convert("RGB")
    arr = np.array(img)
    found_items = set()
    
    if model:
        # AI สแกนหาชื่อสินค้าเบื้องหลัง
        results = model(arr, conf=conf_val) 
        if results[0].boxes is not None:
            for box in results[0].boxes:
                label = results[0].names[int(box.cls[0])]
                if label != "Empty_Stock":
                    found_items.add(label)
    
    # เก็บข้อมูลลงประวัติ
    st.session_state.image_history.append({
        "id": str(uuid.uuid4()),
        "image": img,
        "found": list(found_items),
        "time": datetime.now().strftime("%H:%M:%S")
    })

# ------------------- ส่วนควบคุม (Sidebar) -------------------
with st.sidebar:
    st.header("⚙️ การตั้งค่า AI")
    conf_threshold = st.slider("ความไว AI (Confidence)", 0.0, 1.0, 0.20)
    if st.button("🗑️ ล้างประวัติทั้งหมด", use_container_width=True):
        st.session_state.image_history = []
        st.rerun()

st.title("📦 Stock Vision - Smart Dashboard")

# ------------------- ส่วนรับภาพ -------------------
mode = st.radio("เลือกแหล่งที่มา", ["📸 อัปโหลดภาพ", "📷 ถ่ายภาพจากกล้อง"], horizontal=True)
if mode == "📸 อัปโหลดภาพ":
    up_file = st.file_uploader("เพิ่มรูปภาพสินค้า", type=["jpg","png","jpeg"])
    if up_file:
        file_id = f"file_{up_file.name}_{up_file.size}"
        if 'last_file_id' not in st.session_state or st.session_state.last_file_id != file_id:
            process_image(up_file, conf_threshold)
            st.session_state.last_file_id = file_id
else:
    cam_file = st.camera_input("ถ่ายภาพสินค้า")
    if cam_file:
        cam_id = f"cam_{cam_file.size}"
        if 'last_cam_id' not in st.session_state or st.session_state.last_cam_id != cam_id:
            process_image(cam_file, conf_threshold)
            st.session_state.last_cam_id = cam_id

# ------------------- Dashboard แสดงผลรวม -------------------
# คำนวณสถานะสินค้าจากทุกภาพที่มีอยู่ในประวัติ
all_detected = set()
for entry in st.session_state.image_history:
    all_detected.update(entry["found"])

st.divider()
st.subheader("📊 แผนผังสถานะชั้นวางรวม (Combined Status)")
dash_cols = st.columns(4) # แสดงผลแบบ Grid
for idx, p in enumerate(PRODUCT_LIST):
    is_active = p["name"] in all_detected
    with dash_cols[idx % 4]:
        bg_color = "#28a745" if is_active else "#dc3545" # เขียว=มี, แดง=หมด
        st.markdown(f"""
            <div style="background-color:{bg_color}; color:white; padding:15px; border-radius:10px; text-align:center; margin-bottom:10px; min-height:100px;">
                <small>{p['id']}</small><br>
                <b style="font-size:16px;">{p['name']}</b><br>
                <small>{'✅ มีสินค้า' if is_active else '❌ สินค้าหมด'}</small>
            </div>
        """, unsafe_allow_html=True)

# ------------------- ประวัติภาพและการแก้ไข (Manual Override) -------------------
if st.session_state.image_history:
    st.divider()
    st.subheader("🖼️ รายการภาพถ่ายและปุ่มแก้ไขมือ")
    for item in reversed(st.session_state.image_history):
        with st.expander(f"ภาพถ่ายเวลา {item['time']}", expanded=True):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(item["image"], use_container_width=True) # แสดงภาพ Clean
            with c2:
                st.write("**🤖 AI ตรวจพบ:** " + (", ".join(item['found']) if item['found'] else "ไม่พบสินค้า"))
                
                # หาก AI ทายพลาด สามารถเลือกเพิ่ม/ลดสินค้าเองได้ที่นี่
                edited_found = st.multiselect(
                    "แก้ไขรายการสินค้าที่ปรากฏในภาพนี้:",
                    [p['name'] for p in PRODUCT_LIST],
                    default=item['found'],
                    key=f"edit_{item['id']}"
                )
                
                # อัปเดตข้อมูลภาพหากมีการแก้ไขมือ
                if edited_found != item['found']:
                    item['found'] = edited_found
                    st.rerun()

                if st.button("🗑️ ลบภาพนี้", key=f"del_{item['id']}"):
                    st.session_state.image_history = [i for i in st.session_state.image_history if i['id'] != item['id']]
                    st.rerun()