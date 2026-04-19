import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Stock Vision System - 11 Slots", layout="wide")

# ------------------- โหลดโมเดล -------------------
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# ------------------- ชื่อคลาสสินค้าทั้งหมด -------------------
CLASS_NAMES = [
    "Canned tea", "Coconut Water Carton", "Coffee Can", "Drinking water",
    "Empty_Stock", "Energy Drink", "Green Tea Bottle", "Juice Box",
    "Protein Drink", "Soda Can", "UHT milk carton", "Vitamin Drink"
]

# ------------------- กำหนด 11 ช่องพร้อมสินค้าประจำช่อง -------------------
SLOTS = [
    {"id": "S01", "name": "สมุนไพร", "expected_class": None},
    {"id": "S02", "name": "Coconut Water Carton", "expected_class": "Coconut Water Carton"},
    {"id": "S03", "name": "Coffee Can", "expected_class": "Coffee Can"},
    {"id": "S04", "name": "Drinking water", "expected_class": "Drinking water"},
    {"id": "S05", "name": "Energy Drink", "expected_class": "Energy Drink"},
    {"id": "S06", "name": "Green Tea Bottle", "expected_class": "Green Tea Bottle"},
    {"id": "S07", "name": "Juice Box", "expected_class": "Juice Box"},
    {"id": "S08", "name": "Protein Drink", "expected_class": "Protein Drink"},
    {"id": "S09", "name": "Soda Can", "expected_class": "Soda Can"},
    {"id": "S10", "name": "UHT milk carton", "expected_class": "UHT milk carton"},
    {"id": "S11", "name": "Vitamin Drink", "expected_class": "Vitamin Drink"},
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

# ------------------- จับคู่สินค้าที่ตรวจพบกับช่อง -------------------
def match_products_to_slots(detected_products, slots):
    # นับจำนวนสินค้าที่ตรวจพบแต่ละชนิด
    from collections import Counter
    product_count = Counter(detected_products)
    
    # อัปเดตสถานะแต่ละช่อง
    for slot in slots:
        expected = slot["expected_class"]
        if expected is None:
            # ช่อง S01 ไม่มีสินค้าตายตัว ตรวจจับอะไรก็ได้
            slot["status"] = len(detected_products) > 0
            slot["detected_product"] = ", ".join(set(detected_products)) if detected_products else "ไม่มี"
        else:
            if expected in product_count and product_count[expected] > 0:
                slot["status"] = True
                slot["detected_product"] = expected
                product_count[expected] -= 1  # ใช้ไป 1 ชิ้น
            else:
                slot["status"] = False
                slot["detected_product"] = "หมด"
    return slots

# ------------------- UI หลัก -------------------
st.title("📦 ระบบตรวจสอบสินค้า 11 ช่อง")
st.markdown("**อัปโหลดภาพ → ระบบตรวจจับสินค้า → อัปเดตตารางสถานะอัตโนมัติ**")

# Sidebar
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    confidence = st.slider("ค่า Confidence", 0.0, 1.0, 0.25, 0.01)
    st.markdown("---")
    st.caption("สินค้าที่รองรับ: " + ", ".join([s["name"] for s in SLOTS if s["expected_class"] is not None]))

# โหมดอัปโหลด
uploaded_file = st.file_uploader("📸 อัปโหลดภาพชั้นวางสินค้า", type=["jpg", "jpeg", "png"])

# สร้าง 2 คอลัมน์: ซ้าย=รูป, ขวา=ตาราง
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("🖼️ ภาพที่อัปโหลด")
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        
        # ตรวจจับสินค้า
        img_array = np.array(image)
        detected_products = detect_products(img_array, confidence)
        
        with col_right:
            st.subheader("📋 สถานะสินค้า 11 ช่อง")
            
            # จับคู่และอัปเดตสถานะ
            SLOTS_UPDATED = match_products_to_slots(detected_products, SLOTS.copy())
            
            # แสดงตาราง
            df_data = []
            for slot in SLOTS_UPDATED:
                df_data.append({
                    "ช่อง": slot["id"],
                    "สินค้าประจำช่อง": slot["name"],
                    "สถานะ": "✅ มีสินค้า" if slot["status"] else "❌ สินค้าหมด",
                    "สินค้าที่ตรวจพบ": slot["detected_product"]
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, height=500)
            
            # สถิติ
            total = len(SLOTS_UPDATED)
            instock = sum(1 for s in SLOTS_UPDATED if s["status"])
            outofstock = total - instock
            
            c1, c2, c3 = st.columns(3)
            c1.metric("ช่องทั้งหมด", total)
            c2.metric("✅ มีสินค้า", instock)
            c3.metric("❌ สินค้าหมด", outofstock)
            
            # รายการสินค้าที่ตรวจพบ
            with st.expander("🔍 สินค้าที่ตรวจจับได้ในภาพ"):
                if detected_products:
                    st.write(", ".join(set(detected_products)))
                else:
                    st.warning("ไม่พบสินค้าใดๆ ในภาพ")
            
            # แจ้งเตือน
            empty_slots = [s["name"] for s in SLOTS_UPDATED if not s["status"]]
            if empty_slots:
                st.error(f"⚠️ สินค้าหมด {len(empty_slots)} ช่อง: {', '.join(empty_slots)}")
            else:
                st.success("🎉 สินค้าครบทุกช่อง!")
    else:
        with col_left:
            st.info("⏳ รออัปโหลดภาพ...")
        with col_right:
            st.info("📊 จะแสดงสถานะสินค้าหลังจากอัปโหลดภาพ")