import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime

# ------------------- 1. Page Config -------------------
st.set_page_config(page_title="Stock Vision System", layout="wide")

# ------------------- 2. Load Model -------------------
@st.cache_resource
def load_model():
    try:
        # พยายามโหลดโมเดล ถ้าไม่มีไฟล์จะรันโหมดจำลองอย่างเดียว
        model = YOLO('best.pt')
        return model
    except:
        return None

model = load_model()

# ------------------- 3. Setup ข้อมูลสินค้าและรูปภาพ -------------------
# รายชื่อสินค้าล็อกตำแหน่ง 11 ชนิด
CLASS_NAMES = [
    "Drinking water", "Green Tea Bottle", "Energy Drink",       # ชั้น 1 (3)
    "Vitamin Drink", "Protein Drink", "Soda Can",               # ชั้น 2 (3)
    "Coffee Can", "Canned tea", "UHT milk carton",              # ชั้น 3 (3)
    "Juice Box", "Coconut Water Carton"                         # ชั้น 4 (2)
]

# ใส่ URL รูปภาพตัวอย่าง (คุณสามารถเปลี่ยนเป็น Path รูปในเครื่องได้)
PRODUCT_IMAGES = {
    "Drinking water": "https://cdn-icons-png.flaticon.com/512/3100/3100566.png",
    "Green Tea Bottle": "https://cdn-icons-png.flaticon.com/512/11550/11550085.png",
    "Energy Drink": "https://cdn-icons-png.flaticon.com/512/2447/2447141.png",
    "Vitamin Drink": "https://cdn-icons-png.flaticon.com/512/2722/2722527.png",
    "Protein Drink": "https://cdn-icons-png.flaticon.com/512/3014/3014530.png",
    "Soda Can": "https://cdn-icons-png.flaticon.com/512/2722/2722512.png",
    "Coffee Can": "https://cdn-icons-png.flaticon.com/512/3124/3124231.png",
    "Canned tea": "https://cdn-icons-png.flaticon.com/512/3504/3504827.png",
    "UHT milk carton": "https://cdn-icons-png.flaticon.com/512/372/372921.png",
    "Juice Box": "https://cdn-icons-png.flaticon.com/512/2447/2447036.png",
    "Coconut Water Carton": "https://cdn-icons-png.flaticon.com/512/2447/2447051.png"
}

# สร้างลิสต์ Slot พร้อมชื่อสินค้า
SLOT_CONFIG = [{"id": f"S{i+1:02d}", "name": name} for i, name in enumerate(CLASS_NAMES)]

# ------------------- 4. Session State -------------------
if 'shelf_state' not in st.session_state:
    # เริ่มต้น: ให้ทุกช่องว่าง (False)
    st.session_state.shelf_state = {slot['id']: False for slot in SLOT_CONFIG}
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []

# ------------------- 5. Sidebar -------------------
with st.sidebar:
    st.title("🛡️ Stock Vision")
    mode = st.radio("เลือกโหมดการทำงาน", 
                    ["🖥️ จำลองตู้แช่ (Simulation)", "📸 ตรวจจับจากภาพถ่าย"])
    st.divider()
    if st.button("🗑️ รีเซ็ตระบบ"):
        st.session_state.shelf_state = {slot['id']: False for slot in SLOT_CONFIG}
        st.session_state.alert_history = []
        st.rerun()

# ------------------- 6. MAIN UI -------------------

if mode == "🖥️ จำลองตู้แช่ (Simulation)":
    st.title("🧊 Smart Fridge Visual Dashboard")
    st.markdown("### 📊 แผนผังชั้นวางสินค้า (3-3-3-2)")

    # แบ่งกลุ่มชั้น
    rows = [SLOT_CONFIG[0:3], SLOT_CONFIG[3:6], SLOT_CONFIG[6:9], SLOT_CONFIG[9:11]]
    
    # แสดงตู้แช่แบบ Visual
    for row_slots in rows:
        cols = st.columns(len(row_slots))
        for i, slot in enumerate(row_slots):
            is_active = st.session_state.shelf_state[slot['id']]
            with cols[i]:
                # ตกแต่ง Container แต่ละช่อง
                with st.container(border=True):
                    st.write(f"**Slot {slot['id']}**")
                    if is_active:
                        st.image(PRODUCT_IMAGES[slot['name']], use_container_width=True)
                        st.markdown(f"<p style='text-align:center; color:#28a745; font-weight:bold; margin-top:5px;'>{slot['name']}</p>", unsafe_allow_html=True)
                    else:
                        # แสดงกล่องสีแดงเมื่อของหมด
                        st.markdown(f"""
                            <div style="background-color:#f8d7da; border:1px dashed #dc3545; height:150px; 
                                        display:flex; align-items:center; justify-content:center; border-radius:10px;">
                                <span style="color:#721c24; font-weight:bold; text-align:center;">❌ สินค้าหมด<br><small>{slot['name']}</small></span>
                            </div>
                        """, unsafe_allow_html=True)

    st.divider()

    # แผงควบคุม (Control Panel)
    st.subheader("🎮 แผงควบคุมสต็อกสินค้า")
    c1, c2 = st.columns(2)
    
    with c1:
        st.success("📦 **เติมสินค้าเข้าตู้**")
        for slot in SLOT_CONFIG:
            if not st.session_state.shelf_state[slot['id']]:
                if st.button(f"เติม: {slot['name']}", key=f"add_{slot['id']}", use_container_width=True):
                    st.session_state.shelf_state[slot['id']] = True
                    st.rerun()

    with c2:
        st.error("⚠️ **นำสินค้าออก / ของหมด**")
        for slot in SLOT_CONFIG:
            if st.session_state.shelf_state[slot['id']]:
                if st.button(f"หมด: {slot['name']}", key=f"out_{slot['id']}", use_container_width=True):
                    st.session_state.shelf_state[slot['id']] = False
                    # บันทึกประวัติ
                    st.session_state.alert_history.insert(0, {
                        "เวลา": datetime.now().strftime("%H:%M:%S"),
                        "สินค้า": slot['name'],
                        "สถานะ": "OUT OF STOCK"
                    })
                    st.rerun()

    # แจ้งเตือนรายการที่ขาด
    missing = [s['name'] for s in SLOT_CONFIG if not st.session_state.shelf_state[s['id']]]
    if missing:
        st.sidebar.warning(f"**ควรเติมด่วน:**\n" + "\n".join([f"- {m}" for m in missing]))

# ------------------- 7. Detection Mode -------------------
else:
    st.title("📸 Image Detection Mode")
    source = st.file_uploader("อัปโหลดรูปภาพชั้นวาง", type=['jpg','png','jpeg'])
    
    if source:
        img = Image.open(source).convert("RGB")
        st.image(img, caption="ภาพต้นฉบับ", use_container_width=True)
        
        if st.button("🔍 วิเคราะห์ด้วย AI"):
            if model is None:
                st.error("ไม่พบไฟล์โมเดล (best.pt)")
            else:
                with st.spinner("กำลังตรวจสอบชั้นวาง..."):
                    results = model(np.array(img))
                    # แสดงภาพที่ Draw ผลลัพธ์แล้ว
                    res_plotted = results[0].plot()
                    st.image(res_plotted, caption="ผลการตรวจจับ", use_container_width=True)
                    
                    # ตัวอย่างสรุป
                    st.info("ระบบจะนำข้อมูลพิกัด (Bounding Box) ไปเปรียบเทียบกับ Slot ที่กำหนดไว้ (IoU Logic)")

# ------------------- 8. History -------------------
if st.session_state.alert_history:
    with st.expander("📄 ประวัติการแจ้งเตือนสินค้าหมด"):
        st.table(pd.DataFrame(st.session_state.alert_history))