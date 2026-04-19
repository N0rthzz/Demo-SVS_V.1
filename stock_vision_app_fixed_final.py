import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime

# ------------------- 1. Page Config -------------------
st.set_page_config(page_title="Stock Vision System - Fixed Slot", layout="wide")

# ------------------- 2. Load Model -------------------
@st.cache_resource
def load_model():
    # ตรวจสอบว่ามีไฟล์ best.pt ในโฟลเดอร์เดียวกัน
    try:
        model = YOLO('best.pt')
        return model
    except:
        return None

model = load_model()

# รายชื่อสินค้า 11 ชนิด (เรียงตามลำดับชั้นที่ต้องการ)
CLASS_NAMES = [
    "Drinking water", "Green Tea Bottle", "Energy Drink",       # ชั้น 1 (3)
    "Vitamin Drink", "Protein Drink", "Soda Can",               # ชั้น 2 (3)
    "Coffee Can", "Canned tea", "UHT milk carton",              # ชั้น 3 (3)
    "Juice Box", "Coconut Water Carton"                         # ชั้น 4 (2)
]
# หมายเหตุ: ในโค้ดจริงต้องตรวจเช็คคลาส 'Empty_Stock' ด้วยหากโมเดลเทรนมาแบบนั้น
# ในที่นี้จะเน้นไปที่การ mapping 11 slot หลัก

# ------------------- 3. Database / Slot Setup -------------------
# กำหนดพิกัดจำลอง (relative) สำหรับการตรวจจับจริง
SLOT_CONFIG = []
for i, name in enumerate(CLASS_NAMES):
    slot_id = f"S{i+1:02d}"
    # คำนวณตำแหน่งจำลองใน Grid
    SLOT_CONFIG.append({
        "id": slot_id,
        "name": name,
        "rel_bbox": [0.1, 0.1, 0.2, 0.2] # ตัวอย่างพิกัด (ต้องปรับตามหน้าตู้จริง)
    })

# ------------------- 4. Helper Functions -------------------
def analyze_shelf_image(img_array, conf_threshold=0.25):
    if model is None:
        return img_array, [], []
    
    results = model(img_array, conf=conf_threshold)
    # ... (ส่วนการคำนวณ IoU เหมือนโค้ดเดิมของคุณ)
    # ในตัวอย่างนี้จะส่งค่าจำลองกลับไปเพื่อให้เห็น Logic การทำงาน
    return img_array, [], []

# ------------------- 5. Sidebar & Navigation -------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    mode = st.radio("เลือกโหมดการทำงาน", 
                    ["🖥️ จำลองตู้แช่ (Simulation)", 
                     "📸 อัปโหลดภาพตรวจจับจริง", 
                     "📷 ถ่ายภาพจากกล้อง"])
    
    st.divider()
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25)
    if st.button("🔄 Reset ระบบทั้งหมด"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# ------------------- 6. Session State Initial -------------------
if 'shelf_state' not in st.session_state:
    # เริ่มต้น: ทุกช่องว่าง (False = สีแดง)
    st.session_state.shelf_state = {slot['id']: False for slot in SLOT_CONFIG}
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []

# ------------------- 7. UI: Simulation Mode -------------------
if mode == "🖥️ จำลองตู้แช่ (Simulation)":
    st.title("🧊 Smart Fridge Simulation (3-3-3-2)")
    st.write("โหมดจำลองพฤติกรรมตู้แช่: คลิกปุ่มเพื่อเติมหรือนำสินค้าออก")

    # จัดกลุ่ม Slot เป็นชั้น (3, 3, 3, 2)
    rows = [SLOT_CONFIG[0:3], SLOT_CONFIG[3:6], SLOT_CONFIG[6:9], SLOT_CONFIG[9:11]]
    
    # --- ส่วนแสดงผลกราฟิกตู้แช่ ---
    st.subheader("📊 หน้าจอสถานะปัจจุบัน")
    for row_slots in rows:
        cols = st.columns(len(row_slots))
        for i, slot in enumerate(row_slots):
            is_active = st.session_state.shelf_state[slot['id']]
            bg_color = "#d4edda" if is_active else "#f8d7da"  # เขียว/แดง
            text_color = "#155724" if is_active else "#721c24"
            status_text = "✅ มีสินค้า" if is_active else "❌ สินค้าหมด"
            
            with cols[i]:
                st.markdown(f"""
                    <div style="background-color:{bg_color}; border:2px solid {text_color}; 
                                padding:20px; border-radius:15px; text-align:center; margin-bottom:10px;">
                        <small style="color:#666;">{slot['id']}</small><br>
                        <b style="font-size:1.1rem; color:{text_color};">{slot['name']}</b><br>
                        <hr style="border:0.5px solid {text_color}; opacity:0.2;">
                        <span style="font-weight:bold; color:{text_color};">{status_text}</span>
                    </div>
                """, unsafe_allow_html=True)

    st.divider()

    # --- ส่วนปุ่มกดควบคุม ---
    st.subheader("🕹️ แผงควบคุม (Stock Management)")
    col_add, col_rem = st.columns(2)
    
    with col_add:
        st.success("➕ เติมสินค้า")
        for slot in SLOT_CONFIG:
            if not st.session_state.shelf_state[slot['id']]:
                if st.button(f"เติม {slot['name']}", key=f"in_{slot['id']}", use_container_width=True):
                    st.session_state.shelf_state[slot['id']] = True
                    st.toast(f"เติม {slot['name']} แล้ว", icon="🟢")
                    st.rerun()

    with col_rem:
        st.error("➖ นำสินค้าออก")
        for slot in SLOT_CONFIG:
            if st.session_state.shelf_state[slot['id']]:
                if st.button(f"หยิบ {slot['name']}", key=f"out_{slot['id']}", use_container_width=True):
                    st.session_state.shelf_state[slot['id']] = False
                    st.session_state.alert_history.insert(0, {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "item": slot['name'],
                        "event": "Out of Stock"
                    })
                    st.toast(f"{slot['name']} หมดแล้ว!", icon="🔴")
                    st.rerun()

    # --- ส่วนสรุปรายการที่ขาด ---
    missing = [s['name'] for s in SLOT_CONFIG if not st.session_state.shelf_state[s['id']]]
    if missing:
        st.warning(f"🔔 **แจ้งเตือน:** สินค้าที่ขาดขณะนี้คือ: {', '.join(missing)}")
    
    with st.expander("📄 ประวัติการแจ้งเตือน"):
        if st.session_state.alert_history:
            st.table(st.session_state.alert_history)
        else:
            st.write("ยังไม่มีบันทึก")

# ------------------- 8. UI: Detection Mode (Upload/Camera) -------------------
elif mode in ["📸 อัปโหลดภาพตรวจจับจริง", "📷 ถ่ายภาพจากกล้อง"]:
    st.title("🔍 Real-time Detection Mode")
    
    source = st.file_uploader("อัปโหลดรูปภาพ", type=['jpg','png','jpeg']) if mode == "📸 อัปโหลดภาพตรวจจับจริง" else st.camera_input("ถ่ายภาพ")

    if source:
        img = Image.open(source).convert("RGB")
        img_array = np.array(img)
        
        # รันโมเดลจริง (ใช้ Function analyze_shelf_image ที่คุณเขียนไว้ในไฟล์เดิม)
        # ในที่นี้ผมจะแสดงภาพและ Dashboard สรุป
        st.image(img, caption="ภาพที่กำลังวิเคราะห์...", use_column_width=True)
        
        if model is None:
            st.error("ไม่พบไฟล์ best.pt กรุณาตรวจสอบตำแหน่งไฟล์")
        else:
            with st.spinner("กำลังวิเคราะห์สินค้า..."):
                # Logic การตรวจจับ (คุณสามารถเอาฟังก์ชัน analyze_shelf_image ของเดิมมาใส่ตรงนี้)
                st.success("วิเคราะห์เสร็จสิ้น (ตัวอย่างผลลัพธ์)")
                # จำลองการแสดงผล Dashboard
                c1, c2, c3 = st.columns(3)
                c1.metric("สินค้าทั้งหมด", "11")
                c2.metric("สถานะปกติ", "8", delta="OK")
                c3.metric("สินค้าหมด", "3", delta="-3", delta_color="inverse")

# ------------------- Footer -------------------
st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")