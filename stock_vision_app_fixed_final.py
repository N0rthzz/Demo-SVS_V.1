# stock_vision_app_fixed_final.py (Updated Version)
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime

# ------------------- ตั้งค่า page config เป็นคำสั่งแรกสุด -------------------
st.set_page_config(page_title="Stock Vision System - Fixed Slot", layout="wide")

# ------------------- โหลดโมเดล -------------------
@st.cache_resource
def load_model():
    import os
    # หาตำแหน่งที่ไฟล์โค้ดนี้ตั้งอยู่จริงๆ
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_path, 'best.pt')
    
    if not os.path.exists(model_path):
        st.error(f"❌ หาไฟล์โมเดลไม่เจอที่: {model_path}")
        return None
    
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ โหลดโมเดลไม่ได้: {e}")
        return None

model = load_model()
CLASS_NAMES = [
    "Canned tea", "Coconut Water Carton", "Coffee Can", "Drinking water",
    "Empty_Stock", "Energy Drink", "Green Tea Bottle", "Juice Box",
    "Protein Drink", "Soda Can", "UHT milk carton", "Vitamin Drink"
]
PRODUCT_CLASSES = [name for name in CLASS_NAMES if name != "Empty_Stock"]

# ------------------- กำหนด Shelf Slot 11 ช่อง -------------------
# หมายเหตุ: ปรับค่า rel_bbox ให้ตรงกับตำแหน่งจริงในภาพของคุณ
SLOT_RELATIVE_BOXES = [
    {"id": "S01", "name": "Canned tea", "rel_bbox": [0.05, 0.10, 0.20, 0.35]},
    {"id": "S02", "name": "Coconut Water Carton", "rel_bbox": [0.22, 0.10, 0.37, 0.35]},
    {"id": "S03", "name": "Coffee Can", "rel_bbox": [0.39, 0.10, 0.54, 0.35]}, # ช่องกาแฟ
    {"id": "S04", "name": "Drinking water", "rel_bbox": [0.56, 0.10, 0.71, 0.35]},
    {"id": "S05", "name": "Energy Drink", "rel_bbox": [0.73, 0.10, 0.88, 0.35]},
    {"id": "S06", "name": "Green Tea Bottle", "rel_bbox": [0.05, 0.40, 0.20, 0.65]},
    {"id": "S07", "name": "Juice Box", "rel_bbox": [0.22, 0.40, 0.37, 0.65]},
    {"id": "S08", "name": "Protein Drink", "rel_bbox": [0.39, 0.40, 0.54, 0.65]},
    {"id": "S09", "name": "Soda Can", "rel_bbox": [0.56, 0.40, 0.71, 0.65]},
    {"id": "S10", "name": "UHT milk carton", "rel_bbox": [0.73, 0.40, 0.88, 0.65]},
    {"id": "S11", "name": "Vitamin Drink", "rel_bbox": [0.05, 0.70, 0.20, 0.95]},
]

def rel_to_abs(rel_bbox, img_w, img_h):
    x1 = int(rel_bbox[0] * img_w)
    y1 = int(rel_bbox[1] * img_h)
    x2 = int(rel_bbox[2] * img_w)
    y2 = int(rel_bbox[3] * img_h)
    return [x1, y1, x2, y2]

# --- แก้ไข Logic: เช็คว่าจุดกึ่งกลางสินค้าอยู่ในช่อง และชื่อตรงกันหรือไม่ ---
def check_slot_occupancy_strict(det_bbox, slot_abs_bbox):
    sx1, sy1, sx2, sy2 = slot_abs_bbox
    dx1, dy1, dx2, dy2 = det_bbox
    
    # หาจุดกึ่งกลางของสินค้าที่ตรวจพบ
    cx = (dx1 + dx2) / 2
    cy = (dy1 + dy2) / 2
    
    # สินค้าจะถูกนับว่าอยู่ในช่องนั้น "ก็ต่อเมื่อจุดกึ่งกลางอยู่ในกรอบ Slot"
    return (sx1 <= cx <= sx2) and (sy1 <= cy <= sy2)

def analyze_shelf_image(img_array, conf_threshold=0.25, hide_boxes=False, thickness=2):
    results = model(img_array, conf=conf_threshold)
    h, w = img_array.shape[:2]
    
    # 1. เก็บรายการที่ AI ตรวจพบทั้งหมดพร้อมชื่อคลาส
    detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = CLASS_NAMES[cls_id]
            if name != "Empty_Stock":
                bbox = map(int, box.xyxy[0].tolist())
                detections.append({"name": name, "bbox": list(bbox)})
    
    slot_statuses = []
    empty_slots = []
    img_draw = img_array.copy()
    if len(img_draw.shape) == 3 and img_draw.shape[2] == 3:
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    
    # 2. ตรวจสอบแต่ละ Slot แบบ Strict
    for slot in SLOT_RELATIVE_BOXES:
        abs_bbox = rel_to_abs(slot["rel_bbox"], w, h)
        is_occupied = False
        
        for det in detections:
            # เงื่อนไข: ชื่อต้องตรงกัน และจุดกึ่งกลางต้องอยู่ในช่อง
            if det["name"] == slot["name"]:
                if check_slot_occupancy_strict(det["bbox"], abs_bbox):
                    is_occupied = True
                    break
        
        slot_statuses.append({
            "id": slot["id"],
            "name": slot["name"],
            "status": is_occupied,
            "bbox": abs_bbox
        })
        
        if not is_occupied:
            empty_slots.append(slot["name"])
        
        # วาดกรอบเฉพาะ Slot (ไม่ใช่กรอบที่ AI จับได้โดยตรง เพื่อความสะอาดของรูป)
        if not hide_boxes:
            color = (0, 255, 0) if is_occupied else (0, 0, 255)
            cv2.rectangle(img_draw, (abs_bbox[0], abs_bbox[1]), (abs_bbox[2], abs_bbox[3]), color, thickness)
            label = f"{slot['id']}: {'OK' if is_occupied else 'Empty'}"
            cv2.putText(img_draw, label, (abs_bbox[0], abs_bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    return img_draw, slot_statuses, empty_slots

# ------------------- UI ส่วนที่เหลือ (คงเดิมตาม Logic ของคุณ) -------------------
# ... (ส่วน UI หลัก, Sidebar และโหมดการทำงานคงเดิม) ...