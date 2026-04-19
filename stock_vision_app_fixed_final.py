import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
from collections import Counter
import time

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
    {"id": "S01", "name": "Canned tea", "expected_class": None, "row": 0, "col": 0},
    {"id": "S02", "name": "Coconut Water Carton", "expected_class": "Coconut Water Carton", "row": 0, "col": 1},
    {"id": "S03", "name": "Coffee Can", "expected_class": "Coffee Can", "row": 0, "col": 2},
    {"id": "S04", "name": "Drinking water", "expected_class": "Drinking water", "row": 0, "col": 3},
    {"id": "S05", "name": "Energy Drink", "expected_class": "Energy Drink", "row": 1, "col": 0},
    {"id": "S06", "name": "Green Tea Bottle", "expected_class": "Green Tea Bottle", "row": 1, "col": 1},
    {"id": "S07", "name": "Juice Box", "expected_class": "Juice Box", "row": 1, "col": 2},
    {"id": "S08", "name": "Protein Drink", "expected_class": "Protein Drink", "row": 1, "col": 3},
    {"id": "S09", "name": "Soda Can", "expected_class": "Soda Can", "row": 2, "col": 0},
    {"id": "S10", "name": "UHT milk carton", "expected_class": "UHT milk carton", "row": 2, "col": 1},
    {"id": "S11", "name": "Vitamin Drink", "expected_class": "Vitamin Drink", "row": 2, "col": 2},
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

# ------------------- สร้างกรอบ 11 ช่องบนภาพ (แบบ 3-3-3-2) -------------------
def draw_slots_on_image(img_array, slot_statuses):
    img_draw = img_array.copy()
    h, w = img_draw.shape[:2]
    
    # แบ่งพื้นที่เป็น 4 แถว (3,3,3,2)
    rows = [0, 3, 6, 9, 11]  # จุดแบ่ง: แถว0: S01-S03, แถว1: S04-S06, แถว2: S07-S09, แถว3: S10-S11
    row_heights = [h//4, h//4, h//4, h//4]
    
    # จัดกลุ่มช่องตามแถว
    slots_by_row = {}
    for slot in slot_statuses:
        row = slot["row"]
        if row not in slots_by_row:
            slots_by_row[row] = []
        slots_by_row[row].append(slot)
    
    # วาดแต่ละแถว
    y_start = 0
    for row_idx in range(4):
        y_end = y_start + row_heights[row_idx] if row_idx < 3 else h
        cols_in_row = len(slots_by_row.get(row_idx, []))
        if cols_in_row == 0:
            y_start = y_end
            continue
        
        col_width = w // cols_in_row
        
        for col_idx, slot in enumerate(slots_by_row.get(row_idx, [])):
            x1 = col_idx * col_width
            x2 = (col_idx + 1) * col_width if col_idx < cols_in_row - 1 else w
            y1 = y_start
            y2 = y_end
            
            # สี: เขียว=มีสินค้า, แดง=หมด
            color = (0, 255, 0) if slot["status"] else (0, 0, 255)
            thickness = 3
            
            # วาดกรอบ
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
            
            # พื้นหลังข้อความ
            label = f"{slot['id']}: {slot['name']}"
            status_text = "✓ มี" if slot["status"] else "✗ หมด"
            detected_text = f"[{slot['detected_product']}]" if slot["detected_product"] and slot["detected_product"] != "หมด" and slot["detected_product"] != "ไม่มี" else ""
            
            full_label = f"{label} - {status_text} {detected_text}"
            
            # ใส่ข้อความ
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (text_w, text_h), _ = cv2.getTextSize(full_label, font, font_scale, 2)
            
            # พื้นหลังข้อความ
            bg_x1 = x1 + 5
            bg_y1 = y1 + 5
            bg_x2 = bg_x1 + text_w + 10
            bg_y2 = bg_y1 + text_h + 10
            
            cv2.rectangle(img_draw, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.putText(img_draw, full_label, (x1 + 10, y1 + 25), font, font_scale, color, 2)
        
        y_start = y_end
    
    return img_draw

# ------------------- UI หลัก -------------------
st.title("📦 Stock Vision System (Complete)")
st.markdown("**ระบบตรวจสอบสินค้า 11 ช่อง | รองรับการอัปโหลดและกล้อง | แสดงกรอบช่อง 3-3-3-2**")

# ------------------- Session State -------------------
if 'last_empty' not in st.session_state:
    st.session_state.last_empty = []
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'last_detected_products' not in st.session_state:
    st.session_state.last_detected_products = []
if 'last_slot_statuses' not in st.session_state:
    st.session_state.last_slot_statuses = SLOTS.copy()
if 'uploaded_images_history' not in st.session_state:
    st.session_state.uploaded_images_history = []  # เก็บประวัติภาพที่อัปโหลด
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'camera_mode' not in st.session_state:
    st.session_state.camera_mode = False

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
    
    st.subheader("📊 แผนผังชั้นวาง (11 ช่อง)")
    
    # แสดงแบบตาราง 4 แถว (3,3,3,2)
    rows_data = [[], [], [], []]
    for slot in slot_statuses:
        rows_data[slot["row"]].append(slot)
    
    for row_idx, row_slots in enumerate(rows_data):
        cols = st.columns(len(row_slots))
        for col_idx, slot in enumerate(row_slots):
            with cols[col_idx]:
                color = "#d4edda" if slot["status"] else "#f8d7da"
                border = "2px solid #28a745" if slot["status"] else "2px solid #dc3545"
                st.markdown(f"""
                <div style="background-color:{color}; padding:10px; border-radius:10px; margin:5px; text-align:center; border:{border};">
                    <b>{slot['id']}</b><br>
                    <small>{slot['name']}</small><br>
                    <span style="color:{'green' if slot['status'] else 'red'}; font-weight:bold;">
                        {'✅ มีสินค้า' if slot['status'] else '❌ สินค้าหมด'}
                    </span><br>
                    <small style="color:gray;">🔍 {slot['detected_product']}</small>
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
    st.subheader("🎨 ตัวเลือกการแสดงผล")
    show_slot_boxes = st.checkbox("แสดงกรอบ 11 ช่องบนภาพ (เฉพาะโหมดกล้อง)", value=True)
    
    st.markdown("---")
    if st.button("🗑️ ล้างประวัติทั้งหมด"):
        st.session_state.alert_history = []
        st.session_state.last_empty = []
        st.session_state.uploaded_images_history = []
        st.session_state.current_image = None
        st.rerun()

# ------------------- โหมดการทำงาน -------------------
mode = st.radio("เลือกโหมด", ["📸 อัปโหลดภาพ", "📷 ถ่ายภาพจากกล้อง"], horizontal=True)

# สร้างคอลัมน์
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("🖼️ ภาพที่วิเคราะห์")
    
    # ------------------- โหมดอัปโหลด (เก็บประวัติภาพ) -------------------
    if mode == "📸 อัปโหลดภาพ":
        uploaded_file = st.file_uploader("เลือกภาพชั้นวางสินค้า", type=["jpg", "jpeg", "png"], key="uploader")
        
        if uploaded_file:
            # อ่านภาพใหม่
            img = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(img)
            st.session_state.current_image = img_array
            
            # เก็บประวัติ (ไม่เกิน 5 ภาพ)
            st.session_state.uploaded_images_history.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "image": img_array.copy(),
                "name": uploaded_file.name
            })
            if len(st.session_state.uploaded_images_history) > 5:
                st.session_state.uploaded_images_history.pop()
            
            # แสดงภาพปัจจุบัน
            st.image(img_array, use_container_width=True)
            
            # แสดงประวัติภาพที่อัปโหลด
            if len(st.session_state.uploaded_images_history) > 1:
                with st.expander("📜 ประวัติภาพที่อัปโหลด (คลิกเพื่อดู)"):
                    history_cols = st.columns(min(3, len(st.session_state.uploaded_images_history)-1))
                    for idx, hist in enumerate(st.session_state.uploaded_images_history[1:6]):
                        with history_cols[idx % 3]:
                            st.caption(f"📸 {hist['time']}")
                            st.image(hist['image'], width=100)
                            if st.button(f"เรียกใช้", key=f"load_{idx}"):
                                st.session_state.current_image = hist['image']
                                st.rerun()
            
            # ตรวจจับสินค้า
            detected = detect_products(img_array, confidence_threshold)
            st.session_state.last_detected_products = detected
            
            # จับคู่กับช่อง
            updated_slots = match_products_to_slots(detected, SLOTS)
            st.session_state.last_slot_statuses = updated_slots
            
            # ตรวจสอบสินค้าหมด
            empty_slots = [s["name"] for s in updated_slots if not s["status"]]
            add_alerts(empty_slots)
            
            # แสดงผลทางขวา
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
                    st.warning(f"⚠️ พบช่องว่าง {len(empty_slots)} ช่อง: {', '.join(empty_slots[:5])}")
                    if len(empty_slots) > 5:
                        st.caption(f"...และอีก {len(empty_slots)-5} ช่อง")
                else:
                    st.balloons()
                    st.success("🎉 สินค้าครบทุกช่อง!")
        else:
            st.info("⏳ กรุณาอัปโหลดภาพ")
            if st.session_state.uploaded_images_history:
                st.info(f"📁 มีภาพเก่า {len(st.session_state.uploaded_images_history)} ภาพ (คลิกที่ประวัติเพื่อเรียกใช้)")
    
    # ------------------- โหมดกล้อง (แสดงกรอบ 11 ช่อง) -------------------
    elif mode == "📷 ถ่ายภาพจากกล้อง":
        camera_image = st.camera_input("ถ่ายภาพชั้นวางสินค้า", key="camera")
        
        if camera_image:
            img = Image.open(camera_image).convert("RGB")
            img_array = np.array(img)
            st.session_state.current_image = img_array
            st.session_state.camera_mode = True
            
            # ตรวจจับสินค้า
            detected = detect_products(img_array, confidence_threshold)
            st.session_state.last_detected_products = detected
            
            # จับคู่กับช่อง
            updated_slots = match_products_to_slots(detected, SLOTS)
            st.session_state.last_slot_statuses = updated_slots
            
            empty_slots = [s["name"] for s in updated_slots if not s["status"]]
            add_alerts(empty_slots)
            
            # แสดงภาพพร้อมกรอบ (ถ้าเปิด)
            if show_slot_boxes:
                img_with_boxes = draw_slots_on_image(img_array, updated_slots)
                st.image(img_with_boxes, caption="📸 ภาพจากกล้อง (พร้อมกรอบ 11 ช่อง)", use_container_width=True)
            else:
                st.image(img_array, use_container_width=True)
            
            # แสดงผลทางขวา
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
                st.dataframe(df, use_container_width=True, height=400)
                
                show_dashboard(updated_slots)
                
                if empty_slots:
                    st.error(f"⚠️ สินค้าหมด {len(empty_slots)} ช่อง:")
                    for slot_name in empty_slots:
                        st.write(f"  - {slot_name}")
                else:
                    st.balloons()
                    st.success("🎉 สินค้าครบทุกช่อง!")
        else:
            st.info("📷 กดปุ่มกล้องเพื่อถ่ายภาพ")

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
            
            if st.session_state.alert_history:
                st.subheader("สถิติการแจ้งเตือน")
                alert_df = pd.DataFrame(st.session_state.alert_history)
                st.bar_chart(alert_df['message'].value_counts())
        else:
            st.info("ยังไม่มีข้อมูล")