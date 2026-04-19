import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
from collections import Counter

# ------------------- ตั้งค่า page config -------------------
st.set_page_config(page_title="Stock Vision System - Complete", layout="wide")

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

# ------------------- กำหนด 11 ช่องพร้อมสินค้าประจำช่อง -------------------
SLOTS = [
    {"id": "S01", "name": "Canned tea", "expected_class": None, "status": False, "detected_product": ""},
    {"id": "S02", "name": "Coconut Water Carton", "expected_class": "Coconut Water Carton", "status": False, "detected_product": ""},
    {"id": "S03", "name": "Coffee Can", "expected_class": "Coffee Can", "status": False, "detected_product": ""},
    {"id": "S04", "name": "Drinking water", "expected_class": "Drinking water", "status": False, "detected_product": ""},
    {"id": "S05", "name": "Energy Drink", "expected_class": "Energy Drink", "status": False, "detected_product": ""},
    {"id": "S06", "name": "Green Tea Bottle", "expected_class": "Green Tea Bottle", "status": False, "detected_product": ""},
    {"id": "S07", "name": "Juice Box", "expected_class": "Juice Box", "status": False, "detected_product": ""},
    {"id": "S08", "name": "Protein Drink", "expected_class": "Protein Drink", "status": False, "detected_product": ""},
    {"id": "S09", "name": "Soda Can", "expected_class": "Soda Can", "status": False, "detected_product": ""},
    {"id": "S10", "name": "UHT milk carton", "expected_class": "UHT milk carton", "status": False, "detected_product": ""},
    {"id": "S11", "name": "Vitamin Drink", "expected_class": "Vitamin Drink", "status": False, "detected_product": ""},
]

# ------------------- ฟังก์ชันตรวจจับสินค้า -------------------
def detect_products(img_array, conf_threshold=0.25):
    results = model(img_array, conf=conf_threshold)
    detected = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = CLASS_NAMES[cls_id]
            if class_name != "Empty_Stock":
                detected.append(class_name)
    return detected

# ------------------- จับคู่สินค้ากับช่อง -------------------
def match_products_to_slots(detected_products, slots):
    product_count = Counter(detected_products)
    updated_slots = []
    
    for slot in slots:
        new_slot = slot.copy()
        expected = slot["expected_class"]
        
        if expected is None:
            # S01: รับสินค้าอะไรก็ได้
            new_slot["status"] = len(detected_products) > 0
            new_slot["detected_product"] = ", ".join(set(detected_products)) if detected_products else "ไม่มี"
        else:
            if expected in product_count and product_count[expected] > 0:
                new_slot["status"] = True
                new_slot["detected_product"] = expected
                product_count[expected] -= 1
            else:
                new_slot["status"] = False
                new_slot["detected_product"] = "หมด"
        updated_slots.append(new_slot)
    
    return updated_slots

# ------------------- UI หลัก -------------------
st.title("📦 Stock Vision System (Complete)")
st.markdown("**ระบบตรวจสอบสินค้า 11 ช่อง | แยกตารางสถานะ | แจ้งเตือนอัตโนมัติ**")

# ------------------- Session State สำหรับเก็บข้อมูล -------------------
if 'last_empty' not in st.session_state:
    st.session_state.last_empty = []
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'last_detected_products' not in st.session_state:
    st.session_state.last_detected_products = []
if 'last_slot_statuses' not in st.session_state:
    st.session_state.last_slot_statuses = SLOTS.copy()
if 'last_image' not in st.session_state:
    st.session_state.last_image = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# ------------------- ฟังก์ชันแจ้งเตือน -------------------
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

# ------------------- ฟังก์ชันแสดง Dashboard -------------------
def show_dashboard(slot_statuses):
    total = len(slot_statuses)
    occupied = sum(1 for s in slot_statuses if s["status"])
    empty = total - occupied
    
    c1, c2, c3 = st.columns(3)
    c1.metric("ช่องทั้งหมด", total)
    c2.metric("✅ มีสินค้า", occupied)
    c3.metric("❌ สินค้าหมด", empty)
    
    st.subheader("📊 แผนผังชั้นวาง (รหัสสี)")
    cols = st.columns(4)
    for idx, s in enumerate(slot_statuses):
        with cols[idx % 4]:
            color = "#d4edda" if s["status"] else "#f8d7da"
            border = "2px solid #28a745" if s["status"] else "2px solid #dc3545"
            st.markdown(f"""
            <div style="background-color:{color}; padding:10px; border-radius:10px; margin:5px; text-align:center; border:{border};">
                <b>{s['id']}</b><br>
                <small>{s['name']}</small><br>
                <span style="color:{'green' if s['status'] else 'red'}; font-weight:bold;">
                    {'✅ มีสินค้า' if s['status'] else '❌ สินค้าหมด'}
                </span><br>
                <small style="color:gray;">🔍 {s['detected_product']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.subheader("🔔 ประวัติการแจ้งเตือน")
    if st.session_state.alert_history:
        df_alerts = pd.DataFrame(st.session_state.alert_history)
        st.dataframe(df_alerts, use_container_width=True, height=200)
    else:
        st.info("ยังไม่มีการแจ้งเตือน")

# ------------------- Sidebar -------------------
with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    display_size = st.selectbox("ขนาดภาพ", ["เล็ก (400px)", "กลาง (600px)", "ใหญ่ (800px)"])
    width_map = {"เล็ก (400px)": 400, "กลาง (600px)": 600, "ใหญ่ (800px)": 800}
    display_width = width_map[display_size]
    
    st.markdown("---")
    st.subheader("📸 ตัวเลือกการแสดงผล")
    show_image_only = st.checkbox("แสดงเฉพาะภาพ (ไม่แสดงตารางข้าง)", value=False)
    
    st.markdown("---")
    if st.button("🗑️ ล้างประวัติการแจ้งเตือน"):
        st.session_state.alert_history = []
        st.session_state.last_empty = []
        st.rerun()

# ------------------- โหมดการทำงาน -------------------
mode = st.radio("เลือกโหมด", ["📸 อัปโหลดภาพ", "📷 ถ่ายภาพจากกล้อง"], horizontal=True)

# สร้างคอลัมน์
if not show_image_only:
    col_left, col_right = st.columns([1, 1.2])
else:
    col_left, col_right = st.columns([1, 0])
    with col_right:
        st.empty()

with col_left:
    st.subheader("🖼️ ภาพที่วิเคราะห์")
    
    # ------------------- โหมดอัปโหลด -------------------
    if mode == "📸 อัปโหลดภาพ":
        uploaded_file = st.file_uploader("เลือกภาพชั้นวางสินค้า", type=["jpg", "jpeg", "png"], key="uploader")
        
        if uploaded_file:
            # อ่านภาพใหม่
            img = Image.open(uploaded_file).convert("RGB")
            st.session_state.current_image = np.array(img)
            
            # แสดงภาพ
            st.image(st.session_state.current_image, use_container_width=True)
            
            # ตรวจจับสินค้า
            detected = detect_products(st.session_state.current_image, confidence_threshold)
            st.session_state.last_detected_products = detected
            
            # จับคู่กับช่อง
            updated_slots = match_products_to_slots(detected, SLOTS)
            st.session_state.last_slot_statuses = updated_slots
            
            # ตรวจสอบสินค้าหมด
            empty_slots = [s["name"] for s in updated_slots if not s["status"]]
            add_alerts(empty_slots)
            
            # แสดงรายละเอียดใน col_right
            with col_right:
                st.subheader("📋 สถานะสินค้า 11 ช่อง")
                
                # แสดงตาราง
                df_data = []
                for slot in updated_slots:
                    df_data.append({
                        "ช่อง": slot["id"],
                        "สินค้า": slot["name"],
                        "สถานะ": "✅ มีสินค้า" if slot["status"] else "❌ สินค้าหมด",
                        "ที่ตรวจพบ": slot["detected_product"]
                    })
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, height=500)
                
                # แสดง Dashboard
                show_dashboard(updated_slots)
                
                # แจ้งเตือนสรุป
                if empty_slots:
                    st.warning(f"⚠️ พบช่องว่าง {len(empty_slots)} ช่อง: {', '.join(empty_slots[:5])}")
                    if len(empty_slots) > 5:
                        st.caption(f"...และอีก {len(empty_slots)-5} ช่อง")
                else:
                    st.balloons()
                    st.success("🎉 สินค้าครบทุกช่อง!")
                
                # แสดงสินค้าที่ตรวจพบทั้งหมด
                with st.expander("🔍 สินค้าที่ตรวจพบทั้งหมดในภาพ"):
                    if detected:
                        st.write(", ".join(set(detected)))
                        st.caption(f"จำนวนทั้งหมด: {len(detected)} ชิ้น")
                    else:
                        st.warning("ไม่พบสินค้าใดๆ ในภาพ")
        else:
            st.info("⏳ กรุณาอัปโหลดภาพเพื่อเริ่มตรวจสอบ")
            # แสดงภาพเดิมถ้ามี
            if st.session_state.last_image is not None:
                st.image(st.session_state.last_image, caption="ภาพล่าสุด", use_container_width=True)
    
    # ------------------- โหมดถ่ายภาพ -------------------
    elif mode == "📷 ถ่ายภาพจากกล้อง":
        camera_image = st.camera_input("ถ่ายภาพชั้นวางสินค้า", key="camera")
        
        if camera_image:
            img = Image.open(camera_image).convert("RGB")
            st.session_state.current_image = np.array(img)
            
            st.image(st.session_state.current_image, use_container_width=True)
            
            detected = detect_products(st.session_state.current_image, confidence_threshold)
            st.session_state.last_detected_products = detected
            
            updated_slots = match_products_to_slots(detected, SLOTS)
            st.session_state.last_slot_statuses = updated_slots
            
            empty_slots = [s["name"] for s in updated_slots if not s["status"]]
            add_alerts(empty_slots)
            
            with col_right:
                st.subheader("📋 สถานะสินค้า 11 ช่อง")
                df_data = []
                for slot in updated_slots:
                    df_data.append({
                        "ช่อง": slot["id"],
                        "สินค้า": slot["name"],
                        "สถานะ": "✅ มีสินค้า" if slot["status"] else "❌ สินค้าหมด",
                        "ที่ตรวจพบ": slot["detected_product"]
                    })
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, height=500)
                
                show_dashboard(updated_slots)
                
                if empty_slots:
                    st.warning(f"⚠️ พบช่องว่าง {len(empty_slots)} ช่อง: {', '.join(empty_slots)}")
                else:
                    st.balloons()
                    st.success("🎉 สินค้าครบทุกช่อง!")
                
                with st.expander("🔍 สินค้าที่ตรวจพบทั้งหมด"):
                    if detected:
                        st.write(", ".join(set(detected)))
                    else:
                        st.warning("ไม่พบสินค้า")
        else:
            st.info("📷 กดปุ่มกล้องเพื่อถ่ายภาพ")
            if st.session_state.last_image is not None:
                st.image(st.session_state.last_image, caption="ภาพล่าสุด", use_container_width=True)

# บันทึกภาพล่าสุด
if st.session_state.current_image is not None:
    st.session_state.last_image = st.session_state.current_image.copy()

# ------------------- ส่วนท้าย: รายงาน -------------------
st.markdown("---")
with st.expander("📄 รายงานสรุปและสถิติ"):
    if st.button("📊 สร้างรายงานสถานะปัจจุบัน"):
        if st.session_state.last_slot_statuses:
            df_report = pd.DataFrame([{
                "ช่อง": s["id"],
                "สินค้า": s["name"],
                "สถานะ": "มีสินค้า" if s["status"] else "หมด",
                "สินค้าที่ตรวจพบ": s["detected_product"]
            } for s in st.session_state.last_slot_statuses])
            st.dataframe(df_report, use_container_width=True)
            
            # สรุปสถิติการแจ้งเตือน
            if st.session_state.alert_history:
                st.subheader("สถิติการแจ้งเตือน")
                alert_df = pd.DataFrame(st.session_state.alert_history)
                st.bar_chart(alert_df['message'].value_counts())
        else:
            st.info("ยังไม่มีข้อมูล")