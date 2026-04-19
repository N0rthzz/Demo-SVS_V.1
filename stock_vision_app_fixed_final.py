import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import uuid
from datetime import datetime

# ------------------- Setup -------------------
st.set_page_config(page_title="Stock Vision - Camera Optimized", layout="wide")

@st.cache_resource
def load_model():
    # ใช้ Path ตามโครงสร้างโฟลเดอร์ของคุณ
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    return None

model = load_model()

# รายชื่อสินค้าที่คุณระบุไว้ในระบบ
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

# ------------------- Logic -------------------
def process_image(img_input, conf_val):
    img = Image.open(img_input).convert("RGB")
    arr = np.array(img)
    found_items = set()
    
    if model:
        # เพิ่มการทำ Pre-processing เล็กน้อยเพื่อช่วย AI
        results = model(arr, conf=conf_val, iou=0.45) 
        if results[0].boxes is not None:
            for box in results[0].boxes:
                label = results[0].names[int(box.cls[0])]
                if label != "Empty_Stock":
                    found_items.add(label)
    
    st.session_state.image_history.append({
        "id": str(uuid.uuid4()),
        "image": img,
        "found": list(found_items),
        "time": datetime.now().strftime("%H:%M:%S")
    })

# ------------------- UI -------------------
st.title("📦 Stock Vision - Smart Detection")

with st.sidebar:
    st.header("⚙️ ปรับจูน AI")
    # ลองลดค่า Confidence ลงมาเหลือ 0.15 - 0.20 หากถ่ายในที่แสงน้อย
    conf_threshold = st.slider("ความไวในการตรวจจับ (Confidence)", 0.0, 1.0, 0.20)
    if st.button("🗑️ Clear History"):
        st.session_state.image_history = []
        st.rerun()

# ส่วนของกล้อง (Camera Input)
cam_file = st.camera_input("📷 ถ่ายภาพเพื่อเช็คสต็อก")
if cam_file:
    # ตรวจสอบว่าเป็นภาพใหม่จริงๆ (ป้องกัน Loop)
    cam_id = f"cam_{cam_file.size}"
    if 'last_cam_id' not in st.session_state or st.session_state.last_cam_id != cam_id:
        process_image(cam_file, conf_threshold)
        st.session_state.last_cam_id = cam_id

# ------------------- Dashboard -------------------
all_detected = set()
for entry in st.session_state.image_history:
    all_detected.update(entry["found"])

st.subheader("📊 แผนผังชั้นวางปัจจุบัน")
cols = st.columns(4)
for idx, p in enumerate(PRODUCT_LIST):
    is_on_shelf = p["name"] in all_detected
    with cols[idx % 4]:
        bg = "#28a745" if is_on_shelf else "#dc3545"
        st.markdown(f"""
            <div style="background-color:{bg}; color:white; padding:15px; border-radius:10px; text-align:center; margin-bottom:10px; height:100px;">
                <small>{p['id']}</small><br>
                <b>{p['name']}</b><br>
                <small>{'✅ ตรวจพบ' if is_on_shelf else '❌ ของหมด'}</small>
            </div>
        """, unsafe_allow_html=True)

# ------------------- History & Correction -------------------
if st.session_state.image_history:
    st.divider()
    st.subheader("🖼️ ประวัติการถ่ายภาพ")
    for item in reversed(st.session_state.image_history):
        with st.expander(f"ภาพเวลา {item['time']} - {'พบสินค้า' if item['found'] else 'ไม่พบสินค้า'}", expanded=True):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(item["image"], use_container_width=True)
            with c2:
                # แก้ปัญหา AI ตรวจไม่เจอ: ให้ผู้ใช้เลือกเองได้ถ้า AI พลาด
                st.write("🤖 AI ตรวจพบ: " + (", ".join(item['found']) if item['found'] else "ไม่พบ"))
                
                new_selection = st.multiselect(
                    "แก้ไข/เพิ่มรายการสินค้าที่เห็นในภาพ:",
                    [p['name'] for p in PRODUCT_LIST],
                    default=item['found'],
                    key=f"edit_{item['id']}"
                )
                
                if new_selection != item['found']:
                    item['found'] = new_selection
                    st.rerun() # อัปเดต Dashboard ทันที

                if st.button("🗑️ ลบภาพนี้", key=item['id']):
                    st.session_state.image_history = [i for i in st.session_state.image_history if i['id'] != item['id']]
                    st.rerun()