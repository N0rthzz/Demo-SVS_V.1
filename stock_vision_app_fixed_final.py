import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from datetime import datetime
from collections import Counter
import os
import json
import time
import base64
import random
import requests
from io import BytesIO

# ------------------- ตั้งค่า page config -------------------
st.set_page_config(page_title="🥤 Stock Vision System APP", layout="wide")

# Session state สำหรับเก็บสถานะการปรับแต่ง
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'selected_slot' not in st.session_state:
    st.session_state.selected_slot = None
if 'temp_slot_boxes' not in st.session_state:
    st.session_state.temp_slot_boxes = None
if 'show_simple_editor' not in st.session_state:
    st.session_state.show_simple_editor = False
if 'edit_image' not in st.session_state:
    st.session_state.edit_image = None

# ------------------- โหลดโมเดล -------------------
@st.cache_resource
def load_model():
    model = YOLO('best.pt')
    return model

model = load_model()

# ------------------- กำหนด 11 สินค้าตามที่ระบุ -------------------
CLASS_NAMES = [
    "Coke Can",
    "Coke Light Can",
    "Fanta Grape",
    "Fanta Orange Can",
    "Lactasoy",
    "Meiji Milk",
    "Oishi Rice",
    "Oishi Honey Lemon",
    "Oishi Kyoho",
    "Pepsi Can",
    "Sprite Can"
]

# ชื่อสินค้าภาษาไทย
THAI_NAMES = {
    "Coke Can": "โค้กออริจินัล กระป๋องสีแดง",
    "Coke Light Can": "โค้กไลท์ กระป๋องสีเงิน",
    "Fanta Grape": "แฟนต้าน้ำองุ่น กระป๋องสีม่วง",
    "Fanta Orange Can": "แฟนต้าน้ำส้ม กระป๋องสีส้ม",
    "Lactasoy": "นมถั่วเหลืองแลคตาซอย กล่องสีฟ้า",
    "Meiji Milk": "นมสดเมจิ ขวดสีขาวฝาน้ำเงิน",
    "Oishi Rice": "ชาเขียวโออิชิ รสข้าวญี่ปุ่น ขวดสีส้ม",
    "Oishi Honey Lemon": "ชาเขียวโออิชิ รสน้ำผึ้งผสมมะนาว ขวดสีเหลือง",
    "Oishi Kyoho": "ชาเขียวโออิชิ รสเคียวโฮ ขวดสีม่วง",
    "Pepsi Can": "เป๊ปซี่ กระป๋องสีน้ำเงิน",
    "Sprite Can": "สไปรท์ กระป๋องสีเขียว"
}

# ------------------- PRODUCT LIST สำหรับ 11 ช่อง -------------------
PRODUCT_LIST = [
    {"id": "S01", "name": "Coke Can", "thai_name": "โค้กออริจินัล กระป๋องสีแดง"},
    {"id": "S02", "name": "Coke Light Can", "thai_name": "โค้กไลท์ กระป๋องสีเงิน"},
    {"id": "S03", "name": "Fanta Grape", "thai_name": "แฟนต้าน้ำองุ่น กระป๋องสีม่วง"},
    {"id": "S04", "name": "Fanta Orange Can", "thai_name": "แฟนต้าน้ำส้ม กระป๋องสีส้ม"},
    {"id": "S05", "name": "Lactasoy", "thai_name": "นมถั่วเหลืองแลคตาซอย กล่องสีฟ้า"},
    {"id": "S06", "name": "Meiji Milk", "thai_name": "นมสดเมจิ ขวดสีขาวฝาน้ำเงิน"},
    {"id": "S07", "name": "Oishi Rice", "thai_name": "ชาเขียวโออิชิ รสข้าวญี่ปุ่น ขวดสีส้ม"},
    {"id": "S08", "name": "Oishi Honey Lemon", "thai_name": "ชาเขียวโออิชิ รสน้ำผึ้งผสมมะนาว ขวดสีเหลือง"},
    {"id": "S09", "name": "Oishi Kyoho", "thai_name": "ชาเขียวโออิชิ รสเคียวโฮ ขวดสีม่วง"},
    {"id": "S10", "name": "Pepsi Can", "thai_name": "เป๊ปซี่ กระป๋องสีน้ำเงิน"},
    {"id": "S11", "name": "Sprite Can", "thai_name": "สไปรท์ กระป๋องสีเขียว"}
]

# ------------------- กำหนด Shelf Slot 11 ช่อง -------------------
DEFAULT_SLOT_RELATIVE_BOXES = [
    {"id": "S01", "name": "Coke Can", "rel_bbox": [0.02, 0.02, 0.23, 0.28]},
    {"id": "S02", "name": "Coke Light Can", "rel_bbox": [0.25, 0.02, 0.46, 0.28]},
    {"id": "S03", "name": "Fanta Grape", "rel_bbox": [0.48, 0.02, 0.69, 0.28]},
    {"id": "S04", "name": "Fanta Orange Can", "rel_bbox": [0.71, 0.02, 0.92, 0.28]},
    {"id": "S05", "name": "Lactasoy", "rel_bbox": [0.02, 0.31, 0.23, 0.57]},
    {"id": "S06", "name": "Meiji Milk", "rel_bbox": [0.25, 0.31, 0.46, 0.57]},
    {"id": "S07", "name": "Oishi Rice", "rel_bbox": [0.48, 0.31, 0.69, 0.57]},
    {"id": "S08", "name": "Oishi Honey Lemon", "rel_bbox": [0.71, 0.31, 0.92, 0.57]},
    {"id": "S09", "name": "Oishi Kyoho", "rel_bbox": [0.02, 0.60, 0.30, 0.86]},
    {"id": "S10", "name": "Pepsi Can", "rel_bbox": [0.35, 0.60, 0.63, 0.86]},
    {"id": "S11", "name": "Sprite Can", "rel_bbox": [0.68, 0.60, 0.96, 0.86]},
]

# ไฟล์สำหรับเก็บประวัติ
HISTORY_FILE = "stock_history.json"
UPLOAD_HISTORY_FILE = "upload_history.json"
VALIDATION_HISTORY_FILE = "validation_history.json"
SIMULATION_FILE = "simulation_state.json"
SLOT_CONFIG_FILE = "slot_config.json"

# ==================== ฟังก์ชันจัดการ Slot Configuration ====================
def save_slot_config(slot_boxes):
    config = {
        "slots": slot_boxes,
        "last_update": datetime.now().isoformat()
    }
    try:
        with open(SLOT_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"Error saving slot config: {e}")

def load_slot_config():
    if os.path.exists(SLOT_CONFIG_FILE):
        try:
            with open(SLOT_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "slots" in data:
                    return data["slots"]
                elif isinstance(data, list):
                    return data
        except:
            pass
    return DEFAULT_SLOT_RELATIVE_BOXES.copy()

# ==================== ฟังก์ชันพื้นฐาน ====================
def rel_to_abs(rel_bbox, img_w, img_h):
    x1 = int(rel_bbox[0] * img_w)
    y1 = int(rel_bbox[1] * img_h)
    x2 = int(rel_bbox[2] * img_w)
    y2 = int(rel_bbox[3] * img_h)
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 > x1 and y2 > y1:
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter)
    return 0

def enhance_image(img_array):
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharp = cv2.filter2D(enhanced, -1, kernel)
    
    return sharp

def detect_with_ensemble(img_array, conf_threshold=0.25):
    all_detections = []
    
    results_orig = model(img_array, conf=conf_threshold)
    img_enhanced = enhance_image(img_array)
    results_enhanced = model(img_enhanced, conf=conf_threshold)
    
    for results in [results_orig, results_enhanced]:
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = CLASS_NAMES[cls_id]
                confidence = float(box.conf[0])
                
                # ไม่มี Empty_Stock แล้ว เก็บทุกการตรวจจับ
                if confidence >= conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    all_detections.append([x1, y1, x2, y2, class_name, confidence])
    
    if len(all_detections) > 1:
        all_detections.sort(key=lambda x: x[5], reverse=True)
        final_detections = []
        for det in all_detections:
            duplicate = False
            for existing in final_detections:
                iou = calculate_iou(det[:4], existing[:4])
                if iou > 0.5 and det[4] == existing[4]:
                    duplicate = True
                    break
            if not duplicate:
                final_detections.append(det[:5])
        return final_detections
    
    return [det[:5] for det in all_detections]

def analyze_by_brightness(img_array, slot_abs_bbox):
    x1, y1, x2, y2 = slot_abs_bbox
    roi = img_array[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    brightness = np.mean(gray)
    
    return (brightness < 140 and brightness > 30) or edge_density > 0.05

def check_slot_occupancy_advanced(detection_boxes, slot_abs_bbox, iou_thresh=0.15):
    sx1, sy1, sx2, sy2 = slot_abs_bbox
    slot_area = (sx2 - sx1) * (sy2 - sy1)
    if slot_area <= 0:
        return False, None, 0
    
    best_iou = 0
    best_class = None
    
    for item in detection_boxes:
        if len(item) == 5:
            dx1, dy1, dx2, dy2, class_name = item
        else:
            continue
            
        ix1 = max(sx1, dx1)
        iy1 = max(sy1, dy1)
        ix2 = min(sx2, dx2)
        iy2 = min(sy2, dy2)
        
        if ix2 > ix1 and iy2 > iy1:
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            iou = inter_area / slot_area
            
            center_x = (dx1 + dx2) / 2
            center_y = (dy1 + dy2) / 2
            slot_center_x = (sx1 + sx2) / 2
            slot_center_y = (sy1 + sy2) / 2
            center_distance = np.sqrt((center_x - slot_center_x)**2 + (center_y - slot_center_y)**2)
            center_score = 1 - min(1, center_distance / min(sx2-sx1, sy2-sy1))
            final_score = (iou * 0.7) + (center_score * 0.3)
            
            if final_score > best_iou:
                best_iou = final_score
                best_class = class_name
    
    return best_iou > iou_thresh, best_class, best_iou

def analyze_shelf_image_advanced(img_array, slot_boxes, conf_threshold=0.25):
    h, w = img_array.shape[:2]
    detection_boxes = detect_with_ensemble(img_array, conf_threshold)
    
    detected_classes = list(set([det[4] for det in detection_boxes]))
    if detected_classes:
        st.sidebar.success(f"🔍 ตรวจจับ: {', '.join(detected_classes)}")
    else:
        st.sidebar.warning("⚠️ ไม่พบสินค้า")
    
    slot_statuses = []
    empty_slots = []
    
    for slot in slot_boxes:
        abs_bbox = rel_to_abs(slot["rel_bbox"], w, h)
        occupied, detected_class, confidence = check_slot_occupancy_advanced(detection_boxes, abs_bbox, iou_thresh=0.15)
        
        if not occupied:
            brightness_occupied = analyze_by_brightness(img_array, abs_bbox)
            if brightness_occupied:
                occupied = True
                detected_class = slot["name"]
                confidence = 0.35
        
        is_correct = False
        if occupied and detected_class:
            if detected_class == slot["name"]:
                is_correct = True
            else:
                if confidence > 0.5:
                    is_correct = False
                else:
                    occupied = False
                    detected_class = None
        
        slot_statuses.append({
            "id": slot["id"],
            "name": slot["name"],
            "status": occupied,
            "detected": detected_class if occupied else "ไม่มีสินค้า",
            "is_correct": is_correct,
            "confidence": confidence if occupied else 0
        })
        
        if not occupied:
            empty_slots.append(slot["name"])
    
    return slot_statuses, empty_slots, detection_boxes

# ==================== ฟังก์ชันวาดภาพ (รองรับภาษาไทย) ====================
def get_thai_font(size=18):
    """โหลดฟอนต์ที่รองรับภาษาไทย"""
    font_paths = [
        "C:/Windows/Fonts/Arial.ttf",
        "C:/Windows/Fonts/Tahoma.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    return ImageFont.load_default()

def draw_slot_boxes_on_image(img_array, slot_boxes, slot_statuses, show_labels=True):
    """วาดกรอบ 11 ช่องบนภาพ (รองรับภาษาไทย)"""
    img_pil = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img_pil)
    
    font = get_thai_font(18)
    font_small = get_thai_font(14)
    
    h, w = img_array.shape[:2]
    
    for i, slot in enumerate(slot_boxes):
        abs_bbox = rel_to_abs(slot["rel_bbox"], w, h)
        x1, y1, x2, y2 = abs_bbox
        
        status = next((s for s in slot_statuses if s["id"] == slot["id"]), None)
        if not status:
            continue
        
        # สีกรอบ: เขียว = มีสินค้า, แดง = หมด
        color = (0, 255, 0) if status["status"] else (255, 0, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        if show_labels:
            thai_name = PRODUCT_LIST[i]["thai_name"]
            
            if status["status"]:
                label = f"{slot['id']}: {thai_name}"
                if status["confidence"] > 0:
                    label += f" [{status['confidence']:.0%}]"
            else:
                label = f"{slot['id']}: {thai_name} (หมด)"
            
            # วาดพื้นหลังข้อความ
            bbox = draw.textbbox((x1+5, y1+5), label, font=font)
            draw.rectangle([bbox[0]-3, bbox[1]-3, bbox[2]+3, bbox[3]+3], fill=(0, 0, 0))
            draw.text((x1+5, y1+5), label, fill=(255, 255, 255), font=font)
            
            # วาดสถานะสั้นที่มุมล่างขวา
            status_short = "มีสินค้า" if status["status"] else "หมด"
            sw, sh = draw.textbbox((0, 0), status_short, font=font_small)[2:4]
            bg_x1 = x2 - sw - 10
            bg_y1 = y2 - sh - 10
            draw.rectangle([bg_x1-3, bg_y1-3, x2-5, y2-5], fill=color)
            draw.text((bg_x1, bg_y1), status_short, fill=(255, 255, 255), font=font_small)
    
    return np.array(img_pil)

def draw_slot_boxes_only(img_array, slot_boxes, selected_slot_id=None):
    """วาดกรอบอย่างเดียว (สำหรับหน้าปรับแต่ง)"""
    img_pil = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img_pil)
    font = get_thai_font(16)
    
    h, w = img_array.shape[:2]
    
    for i, slot in enumerate(slot_boxes):
        abs_bbox = rel_to_abs(slot["rel_bbox"], w, h)
        x1, y1, x2, y2 = abs_bbox
        
        if selected_slot_id == slot["id"]:
            color = (255, 255, 0)  # สีเหลือง
            width = 4
        else:
            color = (0, 255, 0)    # สีเขียว
            width = 2
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        
        thai_name = PRODUCT_LIST[i]["thai_name"]
        label = f"{slot['id']}: {thai_name}"
        draw.text((x1+5, y1+5), label, fill=(255, 255, 0), font=font)
    
    return np.array(img_pil)

def generate_composite_image(output_path="shelf_complete.jpg"):
    """สร้างภาพจำลองที่มีสินค้าครบ 11 ช่อง"""
    img_w, img_h = 1200, 800
    img = Image.new('RGB', (img_w, img_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    slot_colors = [
        (200, 100, 100), (100, 200, 100), (100, 100, 200), (200, 200, 100),
        (200, 100, 200), (100, 200, 200), (200, 150, 100), (150, 100, 200),
        (100, 150, 200), (200, 100, 150), (150, 200, 100)
    ]
    
    font = get_thai_font(16)
    
    for i, slot in enumerate(DEFAULT_SLOT_RELATIVE_BOXES):
        x1 = int(slot["rel_bbox"][0] * img_w)
        y1 = int(slot["rel_bbox"][1] * img_h)
        x2 = int(slot["rel_bbox"][2] * img_w)
        y2 = int(slot["rel_bbox"][3] * img_h)
        
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=3)
        draw.rectangle([x1+2, y1+2, x2-2, y2-2], fill=slot_colors[i])
        
        text = f"{slot['id']}\n{PRODUCT_LIST[i]['thai_name']}"
        draw.text((x1+10, y1+10), text, fill=(255, 255, 255), font=font)
    
    img.save(output_path)
    print(f"✅ สร้างภาพตัวอย่างที่ {output_path}")
    return img

# สร้างภาพตัวอย่าง (เรียกใช้ครั้งแรก)
if not os.path.exists("shelf_complete.jpg"):
    generate_composite_image()

# ==================== ฟังก์ชันปรับแต่งกรอบ (แบบไม่ต้องใช้ canvas) ====================

def slot_editor_with_preview():
    """หน้าจอปรับแต่งกรอบพร้อมตัวอย่างภาพจริง"""
    st.subheader("📐 ปรับแต่งตำแหน่งกรอบให้ตรงกับภาพ")
    
    # อัปโหลดภาพ
    uploaded_img = st.file_uploader(
        "อัปโหลดภาพที่ต้องการปรับแต่งกรอบ", 
        type=["jpg", "jpeg", "png"],
        key="editor_img_uploader"
    )
    
    if uploaded_img is None:
        st.info("📸 กรุณาอัปโหลดภาพเพื่อเริ่มปรับแต่งกรอบ")
        return
    
    img = Image.open(uploaded_img).convert("RGB")
    img_array = np.array(img)
    img_w, img_h = img.size
    
    # ใช้ slot_boxes ปัจจุบัน
    if st.session_state.temp_slot_boxes is None:
        st.session_state.temp_slot_boxes = st.session_state.slot_boxes.copy()
    
    slot_boxes = st.session_state.temp_slot_boxes
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("#### 🎯 เลือกช่องที่ต้องการปรับ")
        
        # แสดงปุ่มเลือกช่องแบบ Grid
        for row in range(4):
            cols = st.columns(3)
            for col in range(3):
                idx = row * 3 + col
                if idx < len(slot_boxes):
                    slot = slot_boxes[idx]
                    thai_name = PRODUCT_LIST[idx]["thai_name"]
                    
                    button_type = "primary" if st.session_state.selected_slot == slot["id"] else "secondary"
                    if st.button(f"{slot['id']}: {thai_name}", 
                                 key=f"select_{slot['id']}",
                                 type=button_type,
                                 use_container_width=True):
                        st.session_state.selected_slot = slot["id"]
                        st.rerun()
        
        st.markdown("---")
        
        if st.session_state.selected_slot:
            selected = next(s for s in slot_boxes if s["id"] == st.session_state.selected_slot)
            idx = [s["id"] for s in slot_boxes].index(st.session_state.selected_slot)
            thai_name = PRODUCT_LIST[idx]["thai_name"]
            
            st.markdown(f"#### 🔧 กำลังปรับ: {selected['id']} - {thai_name}")
            
            x1, y1, x2, y2 = selected["rel_bbox"]
            
            # แสดงพิกัดปัจจุบัน
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("ตำแหน่ง X", f"{x1:.3f} - {x2:.3f}")
                st.metric("ความกว้าง", f"{(x2-x1):.3f} ({(x2-x1)*100:.1f}%)")
            with col_b:
                st.metric("ตำแหน่ง Y", f"{y1:.3f} - {y2:.3f}")
                st.metric("ความสูง", f"{(y2-y1):.3f} ({(y2-y1)*100:.1f}%)")
            
            # Slider สำหรับปรับ
            st.markdown("**ปรับตำแหน่งแนวนอน:**")
            new_x1 = st.slider("ขอบซ้าย (x1)", 0.00, x2-0.01, x1, 0.005, key="edit_x1")
            new_x2 = st.slider("ขอบขวา (x2)", new_x1+0.01, 1.00, x2, 0.005, key="edit_x2")
            
            st.markdown("**ปรับตำแหน่งแนวตั้ง:**")
            new_y1 = st.slider("ขอบบน (y1)", 0.00, y2-0.01, y1, 0.005, key="edit_y1")
            new_y2 = st.slider("ขอบล่าง (y2)", new_y1+0.01, 1.00, y2, 0.005, key="edit_y2")
            
            if (new_x1, new_y1, new_x2, new_y2) != (x1, y1, x2, y2):
                selected["rel_bbox"] = [new_x1, new_y1, new_x2, new_y2]
                st.rerun()
            
            # ปุ่มรีเซ็ตช่องนี้
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if st.button(f"↺ รีเซ็ต {selected['id']}", use_container_width=True):
                    selected["rel_bbox"] = DEFAULT_SLOT_RELATIVE_BOXES[idx]["rel_bbox"].copy()
                    st.rerun()
            with col_r2:
                if st.button(f"📏 อัตโนมัติ (Auto-fit)", use_container_width=True):
                    # ลองคำนวณจากความสว่างของภาพ
                    abs_bbox = rel_to_abs(selected["rel_bbox"], img_w, img_h)
                    x1a, y1a, x2a, y2a = abs_bbox
                    roi = img_array[y1a:y2a, x1a:x2a]
                    if roi.size > 0:
                        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                        # หาพื้นที่ที่มีความสว่างต่าง (น่าจะเป็นสินค้า)
                        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            max_contour = max(contours, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(max_contour)
                            # แปลงกลับเป็น relative
                            new_x1 = (x1a + x) / img_w
                            new_y1 = (y1a + y) / img_h
                            new_x2 = (x1a + x + w) / img_w
                            new_y2 = (y1a + y + h) / img_h
                            selected["rel_bbox"] = [new_x1, new_y1, new_x2, new_y2]
                            st.rerun()
                    st.info("ลองปรับกรอบอัตโนมัติ (อาจไม่แม่น 100%)")
    
    with col2:
        st.markdown("#### 📸 ตัวอย่างตำแหน่งกรอบ")
        
        # แสดงภาพพร้อมกรอบ
        img_with_boxes = draw_slot_boxes_only(img_array, slot_boxes, st.session_state.selected_slot)
        st.image(img_with_boxes, caption="🟢=กรอบปกติ, 🟡=กำลังปรับ", use_container_width=True)
        
        # แสดงพิกัดแบบเต็ม
        with st.expander("📋 พิกัดทั้งหมด (Relative)"):
            for slot in slot_boxes:
                x1, y1, x2, y2 = slot["rel_bbox"]
                st.code(f"{slot['id']}: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
    
    # ปุ่มควบคุม
    st.markdown("---")
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        if st.button("💾 บันทึกการตั้งค่า", type="primary", use_container_width=True):
            save_slot_config(slot_boxes)
            st.session_state.slot_boxes = slot_boxes
            st.session_state.temp_slot_boxes = None
            st.session_state.selected_slot = None
            st.session_state.edit_mode = False
            st.success("✅ บันทึกการตั้งค่าเรียบร้อย!")
            st.rerun()
    
    with col_btn2:
        if st.button("↺ รีเซ็ตทั้งหมด", use_container_width=True):
            st.session_state.temp_slot_boxes = DEFAULT_SLOT_RELATIVE_BOXES.copy()
            st.rerun()
    
    with col_btn3:
        if st.button("❌ ยกเลิก", use_container_width=True):
            st.session_state.temp_slot_boxes = None
            st.session_state.selected_slot = None
            st.session_state.edit_mode = False
            st.rerun()
    
    with col_btn4:
        if st.button("🧪 ทดสอบกับภาพนี้", use_container_width=True):
            test_statuses = []
            for slot in slot_boxes:
                test_statuses.append({
                    "id": slot["id"],
                    "name": slot["name"],
                    "status": True,
                    "detected": slot["name"],
                    "is_correct": True,
                    "confidence": 0.95
                })
            test_img = draw_slot_boxes_on_image(img_array, slot_boxes, test_statuses)
            st.image(test_img, caption="ผลการทดสอบ", use_container_width=True)


def simple_slot_editor():
    """หน้าจอปรับแต่งกรอบแบบง่าย โดยใช้ Slider ปรับแต่ละช่อง"""
    st.subheader("📐 ปรับแต่งตำแหน่งกรอบแต่ละช่อง")
    st.markdown("ปรับค่า x1, x2, y1, y2 ของแต่ละช่องได้ตามต้องการ")
    
    slot_boxes = st.session_state.slot_boxes.copy()
    
    # เลือกช่องที่จะปรับ
    slot_options = [f"{s['id']}: {PRODUCT_LIST[i]['thai_name']}" for i, s in enumerate(slot_boxes)]
    selected_idx = st.selectbox("เลือกช่องที่ต้องการปรับ", range(len(slot_options)), format_func=lambda x: slot_options[x])
    
    selected_slot = slot_boxes[selected_idx]
    thai_name = PRODUCT_LIST[selected_idx]["thai_name"]
    
    st.markdown(f"### 🎯 กำลังปรับ: {selected_slot['id']} - {thai_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ตำแหน่งแนวนอน")
        x1 = st.slider("ขอบซ้าย (x1)", 0.00, selected_slot["rel_bbox"][2]-0.01, 
                       selected_slot["rel_bbox"][0], 0.005,
                       key=f"x1_{selected_slot['id']}")
        x2 = st.slider("ขอบขวา (x2)", x1+0.01, 1.00, 
                       selected_slot["rel_bbox"][2], 0.005,
                       key=f"x2_{selected_slot['id']}")
        
        st.markdown("#### ขนาดแนวนอน")
        width = x2 - x1
        st.metric("ความกว้าง", f"{width:.3f} ({width*100:.1f}%)")
    
    with col2:
        st.markdown("#### ตำแหน่งแนวตั้ง")
        y1 = st.slider("ขอบบน (y1)", 0.00, selected_slot["rel_bbox"][3]-0.01, 
                       selected_slot["rel_bbox"][1], 0.005,
                       key=f"y1_{selected_slot['id']}")
        y2 = st.slider("ขอบล่าง (y2)", y1+0.01, 1.00, 
                       selected_slot["rel_bbox"][3], 0.005,
                       key=f"y2_{selected_slot['id']}")
        
        st.markdown("#### ขนาดแนวตั้ง")
        height = y2 - y1
        st.metric("ความสูง", f"{height:.3f} ({height*100:.1f}%)")
    
    # อัปเดตค่า
    selected_slot["rel_bbox"] = [x1, y1, x2, y2]
    
    # แสดงตัวอย่าง
    st.markdown("---")
    st.markdown("#### 📸 ตัวอย่างตำแหน่งกรอบ")
    
    # สร้างภาพตัวอย่าง
    demo_img = Image.new('RGB', (800, 600), color=(50, 50, 50))
    draw = ImageDraw.Draw(demo_img)
    font = get_thai_font(16)
    
    for i, slot in enumerate(slot_boxes):
        x1 = int(slot["rel_bbox"][0] * 800)
        y1 = int(slot["rel_bbox"][1] * 600)
        x2 = int(slot["rel_bbox"][2] * 800)
        y2 = int(slot["rel_bbox"][3] * 600)
        
        color = (0, 255, 0) if i != selected_idx else (255, 255, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1+3, y1+3), slot['id'], fill=(255,255,255), font=font)
    
    st.image(np.array(demo_img), caption="ตัวอย่างตำแหน่ง (สีเหลือง = ช่องที่กำลังปรับ)", use_container_width=True)
    
    # ปุ่มบันทึก
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("💾 บันทึกการตั้งค่า", type="primary", use_container_width=True):
            save_slot_config(slot_boxes)
            st.session_state.slot_boxes = slot_boxes
            st.success("✅ บันทึกการตั้งค่าเรียบร้อย!")
            st.rerun()
    
    with col_btn2:
        if st.button("↺ รีเซ็ตทั้งหมด", use_container_width=True):
            st.session_state.slot_boxes = DEFAULT_SLOT_RELATIVE_BOXES.copy()
            save_slot_config(DEFAULT_SLOT_RELATIVE_BOXES)
            st.rerun()
    
    with col_btn3:
        if st.button("↺ รีเซ็ตเฉพาะช่องนี้", use_container_width=True):
            default_box = DEFAULT_SLOT_RELATIVE_BOXES[selected_idx]["rel_bbox"]
            selected_slot["rel_bbox"] = default_box.copy()
            st.rerun()


# ==================== ฟังก์ชันสำหรับทดสอบความแม่นยำ ====================
def predict_single_product(img_array, conf_threshold=0.25):
    results = model(img_array, conf=conf_threshold)
    
    predictions = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = CLASS_NAMES[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            predictions.append({
                "class_name": class_name,
                "thai_name": THAI_NAMES.get(class_name, class_name),
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })
    
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    return predictions

def predict_all_categories(img_array, conf_threshold=0.25):
    results = model(img_array, conf=conf_threshold)
    
    category_scores = {name: 0.0 for name in CLASS_NAMES}
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = CLASS_NAMES[cls_id]
            confidence = float(box.conf[0])
            
            if confidence >= conf_threshold:
                if confidence > category_scores[class_name]:
                    category_scores[class_name] = confidence
    
    predictions = []
    for name, score in category_scores.items():
        predictions.append({
            "class_name": name,
            "thai_name": THAI_NAMES.get(name, name),
            "confidence": score,
            "has_product": score > 0
        })
    
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    return predictions

def draw_prediction_on_image(img_array, predictions):
    img_draw = img_array.copy()
    h, w = img_draw.shape[:2]
    
    colors = [(0, 255, 0), (255, 165, 0), (0, 255, 255), (255, 0, 255), (0, 165, 255)]
    
    for i, pred in enumerate(predictions):
        if "bbox" not in pred:
            continue
        x1, y1, x2, y2 = pred["bbox"]
        color = colors[i % len(colors)]
        
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)
        
        label = f"{pred['thai_name']} ({pred['confidence']:.1%})"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 2)
        
        bg_x1 = x1
        bg_y1 = max(0, y1 - text_h - 10)
        bg_x2 = min(x1 + text_w + 10, w)
        bg_y2 = y1
        
        cv2.rectangle(img_draw, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(img_draw, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), 2)
    
    return img_draw

def save_validation_result(image_array, filename, predictions, actual_label=None):
    history = []
    if os.path.exists(VALIDATION_HISTORY_FILE):
        try:
            with open(VALIDATION_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except:
            history = []
    
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    top_prediction = predictions[0] if predictions else None
    
    history.insert(0, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": str(filename),
        "image_base64": str(image_base64),
        "predictions": [
            {
                "class": p["class_name"],
                "thai_name": p["thai_name"],
                "confidence": p["confidence"]
            } for p in predictions[:3]
        ],
        "top_prediction": top_prediction["class_name"] if top_prediction else None,
        "top_confidence": top_prediction["confidence"] if top_prediction else 0,
        "actual_label": actual_label
    })
    
    history = history[:20]
    
    try:
        with open(VALIDATION_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        pass

def load_validation_history():
    if os.path.exists(VALIDATION_HISTORY_FILE):
        try:
            with open(VALIDATION_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def calculate_model_accuracy():
    history = load_validation_history()
    if not history:
        return None
    
    total = 0
    correct = 0
    category_stats = {name: {"total": 0, "correct": 0} for name in CLASS_NAMES}
    
    for record in history:
        if record.get("actual_label") and record.get("top_prediction"):
            total += 1
            if record["actual_label"] == record["top_prediction"]:
                correct += 1
                if record["actual_label"] in category_stats:
                    category_stats[record["actual_label"]]["correct"] += 1
            if record["actual_label"] in category_stats:
                category_stats[record["actual_label"]]["total"] += 1
    
    if total == 0:
        return None
    
    return {
        "accuracy": correct / total,
        "total_tests": total,
        "correct": correct,
        "wrong": total - correct,
        "category_stats": category_stats
    }

# ==================== ฟังก์ชัน Simulation Mode ====================
def save_simulation_state(slot_statuses):
    state = {
        "slots": [],
        "last_update": datetime.now().isoformat()
    }
    
    for slot in slot_statuses:
        state["slots"].append({
            "id": slot["id"],
            "name": slot["name"],
            "status": slot["status"]
        })
    
    try:
        with open(SIMULATION_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"Error saving simulation state: {e}")

def load_simulation_state():
    if os.path.exists(SIMULATION_FILE):
        try:
            with open(SIMULATION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "slots" in data:
                    return data["slots"]
                else:
                    return None
        except:
            return None
    return None

def get_default_simulation_slots():
    slots = []
    for slot in DEFAULT_SLOT_RELATIVE_BOXES:
        slots.append({
            "id": slot["id"],
            "name": slot["name"],
            "status": True
        })
    return slots

# ==================== ฟังก์ชันบันทึกประวัติ ====================
def save_stock_history(slot_statuses):
    history = {}
    for slot in slot_statuses:
        history[slot["id"]] = {
            "status": bool(slot["status"]),
            "detected": str(slot["detected"]),
            "is_correct": bool(slot["is_correct"]),
            "confidence": float(slot.get("confidence", 0))
        }
    history["last_update"] = datetime.now().isoformat()
    
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        pass

def load_stock_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_upload_history(image_array, filename):
    history = []
    if os.path.exists(UPLOAD_HISTORY_FILE):
        try:
            with open(UPLOAD_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except:
            history = []
    
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    history.insert(0, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": str(filename),
        "image_base64": str(image_base64)
    })
    
    history = history[:10]
    
    try:
        with open(UPLOAD_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        pass

def load_upload_history():
    if os.path.exists(UPLOAD_HISTORY_FILE):
        try:
            with open(UPLOAD_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def check_stock_changes(current_statuses, previous_history):
    changes = []
    for slot in current_statuses:
        slot_id = slot["id"]
        current = slot["status"]
        previous = previous_history.get(slot_id, {}).get("status", None)
        
        if previous is not None and current != previous:
            if current:
                changes.append(f"🟢 {slot_id} ({slot['name']}) : เพิ่มสินค้า")
            else:
                changes.append(f"🔴 {slot_id} ({slot['name']}) : สินค้าหมด")
    return changes

# ------------------- Session State -------------------
if 'last_empty' not in st.session_state:
    st.session_state.last_empty = []
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'current_slot_statuses' not in st.session_state:
    st.session_state.current_slot_statuses = []
if 'slot_boxes' not in st.session_state:
    st.session_state.slot_boxes = load_slot_config()

def add_alerts(empty_slots, slot_statuses):
    if set(empty_slots) != set(st.session_state.last_empty):
        new_empty = set(empty_slots) - set(st.session_state.last_empty)
        for slot_name in new_empty:
            msg = f"⚠️ สินค้าหมด: {slot_name}"
            st.session_state.alert_history.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"), 
                "message": msg,
                "type": "empty"
            })
            st.toast(msg, icon="🔴")
        st.session_state.last_empty = empty_slots.copy()
    
    for slot in slot_statuses:
        if slot["status"] and not slot["is_correct"]:
            msg = f"⚠️ สินค้าผิดช่อง: {slot['id']} ({slot['name']}) พบ {slot['detected']}"
            if not any(msg in alert['message'] for alert in st.session_state.alert_history[:5]):
                st.session_state.alert_history.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"), 
                    "message": msg,
                    "type": "wrong_slot"
                })
                st.toast(msg, icon="🟠")
    
    if len(st.session_state.alert_history) > 30:
        st.session_state.alert_history.pop()

def show_dashboard(slot_statuses):
    total = len(slot_statuses)
    occupied = sum(1 for s in slot_statuses if s["status"])
    empty = total - occupied
    wrong_slot = sum(1 for s in slot_statuses if s["status"] and not s["is_correct"])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🥤 ช่องทั้งหมด", total)
    col2.metric("✅ มีสินค้า", occupied)
    col3.metric("❌ สินค้าหมด", empty, delta=f"-{empty}" if empty > 0 else None)
    col4.metric("⚠️ ผิดช่อง", wrong_slot)
    
    st.subheader("📋 ตารางสถานะสินค้า")
    
    df_data = []
    for slot in slot_statuses:
        if slot["status"]:
            if slot["is_correct"]:
                status_icon = "✅ มีสินค้า"
            else:
                status_icon = f"⚠️ ผิดช่อง"
        else:
            status_icon = "❌ สินค้าหมด"
        
        df_data.append({
            "ช่อง": slot["id"],
            "สินค้า": slot["name"],
            "สถานะ": status_icon,
            "ที่ตรวจพบ": slot["detected"] if slot["status"] else "-",
            "ความมั่นใจ": f"{slot['confidence']:.0%}" if slot["status"] and slot["confidence"] > 0 else "-"
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, height=400)
    
    st.subheader("📊 สถานะล่าสุด vs ครั้งก่อน")
    history = load_stock_history()
    if history and "last_update" in history:
        st.caption(f"อัปเดตล่าสุด: {history.get('last_update', 'ไม่ทราบ')}")
        changes = check_stock_changes(slot_statuses, history)
        if changes:
            for change in changes:
                if "หมด" in change:
                    st.error(change)
                else:
                    st.success(change)
        else:
            st.info("ไม่มีการเปลี่ยนแปลงจากครั้งก่อน")
    
    st.subheader("🔔 ประวัติการแจ้งเตือน")
    if st.session_state.alert_history:
        alert_df = pd.DataFrame(st.session_state.alert_history)
        st.dataframe(alert_df, use_container_width=True, height=150)
    else:
        st.info("ไม่มีการแจ้งเตือน")

# ------------------- UI หลัก -------------------
st.title("🥤 Stock Vision System APP")
st.markdown("**ระบบตรวจจับสินค้าหมดอัจฉริยะ | รองรับการปรับแต่งตำแหน่งช่อง | เทคโนโลยี AI ขั้นสูง**")

# ตรวจสอบว่ามีการเปิดหน้าปรับแต่งกรอบหรือไม่
if st.session_state.get('edit_mode', False):
    slot_editor_with_preview()
    st.stop()

if st.session_state.get('show_simple_editor', False):
    simple_slot_editor()
    if st.button("🔙 กลับหน้าหลัก"):
        st.session_state.show_simple_editor = False
        st.rerun()
    st.stop()

# Sidebar สำหรับการตั้งค่า
with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")
    
    st.subheader("🎯 การตั้งค่าโมเดล")
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.20, 0.01,
                                      help="ค่าความมั่นใจขั้นต่ำในการตรวจจับ (แนะนำ 0.15-0.25)")
    
    st.subheader("🎨 ตัวเลือกการแสดงผล")
    show_grid_on_camera = st.checkbox("แสดงตาราง 11 ช่องบนภาพ", value=True)
    show_confidence = st.checkbox("แสดง Confidence Score", value=True)
    
    st.markdown("---")
    
    # ปรับแต่งตำแหน่งช่อง
    st.subheader("📐 ปรับแต่งตำแแหน่งกรอบ")
    
    editor_mode = st.radio(
        "เลือกวิธีการปรับแต่ง",
        ["🔧 ปรับทีละช่อง (Slider)", "🖱️ ปรับพร้อมดูตัวอย่างภาพจริง"],
        help="เลือกวิธีการปรับแต่งกรอบให้ตรงกับภาพ"
    )
    
    if editor_mode == "🔧 ปรับทีละช่อง (Slider)":
        if st.button("✏️ เปิดหน้าปรับแต่งกรอบ", use_container_width=True):
            st.session_state.show_simple_editor = True
            st.rerun()
    else:
        if st.button("🎨 เปิดหน้าปรับแต่ง (พร้อมตัวอย่างภาพ)", use_container_width=True, type="primary"):
            st.session_state.edit_mode = True
            st.rerun()
    
    st.markdown("---")
    
    with st.expander("📐 ปรับแต่งตำแหน่งช่องแบบรวม (Expert)"):
        st.info("💡 ปรับค่าเหล่านี้หากตำแหน่งกรอบไม่ตรงกับภาพ")
        
        row_offset_x = st.slider("ปรับตำแหน่งแนวนอนรวม", -0.05, 0.05, 0.0, 0.01)
        row_offset_y = st.slider("ปรับตำแหน่งแนวตั้งรวม", -0.05, 0.05, 0.0, 0.01)
        slot_width_scale = st.slider("ปรับความกว้างช่อง", 0.8, 1.2, 1.0, 0.05)
        slot_height_scale = st.slider("ปรับความสูงช่อง", 0.8, 1.2, 1.0, 0.05)
        
        if st.button("💾 บันทึกการตั้งค่าช่อง"):
            adjusted_boxes = []
            for slot in DEFAULT_SLOT_RELATIVE_BOXES:
                x1, y1, x2, y2 = slot["rel_bbox"]
                width = (x2 - x1) * slot_width_scale
                height = (y2 - y1) * slot_height_scale
                center_x = (x1 + x2) / 2 + row_offset_x
                center_y = (y1 + y2) / 2 + row_offset_y
                
                new_x1 = max(0, center_x - width/2)
                new_x2 = min(1, center_x + width/2)
                new_y1 = max(0, center_y - height/2)
                new_y2 = min(1, center_y + height/2)
                
                adjusted_boxes.append({
                    "id": slot["id"],
                    "name": slot["name"],
                    "rel_bbox": [new_x1, new_y1, new_x2, new_y2]
                })
            
            st.session_state.slot_boxes = adjusted_boxes
            save_slot_config(adjusted_boxes)
            st.success("บันทึกการตั้งค่าเรียบร้อย!")
    
    st.markdown("---")
    
    col_reset1, col_reset2 = st.columns(2)
    with col_reset1:
        if st.button("🗑️ ล้างประวัติแจ้งเตือน"):
            st.session_state.alert_history = []
            st.session_state.last_empty = []
            st.rerun()
    with col_reset2:
        if st.button("🔄 รีเซ็ตตำแหน่งช่อง"):
            st.session_state.slot_boxes = DEFAULT_SLOT_RELATIVE_BOXES.copy()
            save_slot_config(DEFAULT_SLOT_RELATIVE_BOXES)
            st.success("รีเซ็ตตำแหน่งช่องเรียบร้อย!")
            st.rerun()
    
    if st.button("🔄 รีเซ็ตระบบทั้งหมด"):
        st.session_state.alert_history = []
        st.session_state.last_empty = []
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        if os.path.exists(UPLOAD_HISTORY_FILE):
            os.remove(UPLOAD_HISTORY_FILE)
        st.success("รีเซ็ตระบบเรียบร้อย!")
        st.rerun()
    
    st.markdown("---")
    st.subheader("📸 ภาพตัวอย่าง")
    if st.button("🎨 สร้างภาพตัวอย่าง (ครบ 11 ช่อง)"):
        generate_composite_image()
        st.success("สร้างภาพตัวอย่างเรียบร้อย! ไฟล์: shelf_complete.jpg")

# เลือกโหมดหลัก
main_mode = st.radio(
    "เลือกโหมดหลัก", 
    ["📦 ตรวจสอบสต็อกสินค้า", "🎯 ทดสอบความแม่นยำโมเดล", "🎮 Simulation Mode"], 
    horizontal=True
)

# ==================== โหมด 1: ตรวจสอบสต็อกสินค้า ====================
if main_mode == "📦 ตรวจสอบสต็อกสินค้า":
    mode = st.radio("เลือกโหมด", ["📸 อัปโหลดภาพ", "📷 ถ่ายภาพจากกล้อง"], horizontal=True)
    
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        st.subheader("🖼️ ภาพที่วิเคราะห์")
        
        if mode == "📸 อัปโหลดภาพ":
            uploaded_file = st.file_uploader("เลือกภาพชั้นวางสินค้า", type=["jpg", "jpeg", "png"])
            
            if uploaded_file:
                img = Image.open(uploaded_file).convert("RGB")
                img_array = np.array(img)
                
                save_upload_history(img_array, uploaded_file.name)
                
                with st.spinner("กำลังวิเคราะห์ภาพ..."):
                    slot_statuses, empty_slots, _ = analyze_shelf_image_advanced(
                        img_array, st.session_state.slot_boxes, confidence_threshold
                    )
                
                st.session_state.current_slot_statuses = slot_statuses
                add_alerts(empty_slots, slot_statuses)
                save_stock_history(slot_statuses)
                
                # เพิ่มปุ่มปรับแต่งกรอบ
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("✏️ ปรับแต่งกรอบให้ตรงกับภาพนี้", use_container_width=True):
                        st.session_state.edit_image = uploaded_file
                        st.session_state.edit_mode = True
                        st.rerun()
                
                # ใช้ฟังก์ชันวาดภาพที่รองรับภาษาไทย
                img_with_boxes = draw_slot_boxes_on_image(
                    img_array, st.session_state.slot_boxes, slot_statuses, show_labels=show_confidence
                )
                st.image(img_with_boxes, caption="ผลการตรวจจับ (🟢=มีสินค้า, 🔴=สินค้าหมด)", use_container_width=True)
                
                with col_right:
                    show_dashboard(slot_statuses)
                    
                    st.subheader("📋 สรุปสถานะแต่ละช่อง")
                    for slot in slot_statuses:
                        thai_name = PRODUCT_LIST[int(slot["id"][1:])-1]["thai_name"]
                        if slot["status"]:
                            st.success(f"🟢 {slot['id']} ({thai_name}): มีสินค้า → {slot['detected']}")
                        else:
                            st.error(f"🔴 {slot['id']} ({thai_name}): สินค้าหมด")
                    
                    if empty_slots:
                        st.error(f"⚠️ สินค้าหมด {len(empty_slots)} ช่อง: {', '.join(empty_slots)}")
                        st.subheader("📋 รายการสินค้าที่ต้องเติม")
                        for slot in slot_statuses:
                            if not slot["status"]:
                                thai_name = PRODUCT_LIST[int(slot["id"][1:])-1]["thai_name"]
                                st.write(f"  • ช่อง {slot['id']}: {thai_name}")
                    else:
                        wrongs = [s["id"] for s in slot_statuses if s["status"] and not s["is_correct"]]
                        if wrongs:
                            st.warning(f"⚠️ สินค้าผิดช่อง: {', '.join(wrongs)}")
                        else:
                            st.balloons()
                            st.success("🎉 สินค้าครบทุกช่องและถูกต้อง!")
            else:
                st.info("⏳ กรุณาอัปโหลดภาพ")
        
        elif mode == "📷 ถ่ายภาพจากกล้อง":
            camera_image = st.camera_input("ถ่ายภาพชั้นวางสินค้า")
            
            if camera_image:
                img = Image.open(camera_image).convert("RGB")
                img_array = np.array(img)
                
                with st.spinner("กำลังวิเคราะห์ภาพ..."):
                    slot_statuses, empty_slots, _ = analyze_shelf_image_advanced(
                        img_array, st.session_state.slot_boxes, confidence_threshold
                    )
                
                st.session_state.current_slot_statuses = slot_statuses
                add_alerts(empty_slots, slot_statuses)
                save_stock_history(slot_statuses)
                
                if show_grid_on_camera:
                    img_with_grid = draw_slot_boxes_on_image(
                        img_array, st.session_state.slot_boxes, slot_statuses, show_labels=show_confidence
                    )
                    st.image(img_with_grid, use_container_width=True)
                else:
                    st.image(img_array, use_container_width=True)
                
                with col_right:
                    show_dashboard(slot_statuses)
                    
                    if empty_slots:
                        st.error(f"⚠️ สินค้าหมด {len(empty_slots)} ช่อง:")
                        for slot_name in empty_slots:
                            st.write(f"  • {slot_name}")
                    else:
                        wrongs = [s["id"] for s in slot_statuses if s["status"] and not s["is_correct"]]
                        if wrongs:
                            st.warning(f"⚠️ สินค้าผิดช่อง: {', '.join(wrongs)}")
                        else:
                            st.balloons()
                            st.success("🎉 สินค้าครบทุกช่อง!")
            else:
                st.info("📷 กดปุ่มกล้องเพื่อถ่ายภาพ")

# ==================== โหมด 2: ทดสอบความแม่นยำโมเดล ====================
elif main_mode == "🎯 ทดสอบความแม่นยำโมเดล":
    st.markdown("---")
    st.subheader("🎯 ทดสอบความแม่นยำของโมเดล (11 ประเภท)")
    st.markdown("อัปโหลดภาพสินค้า แล้วระบบจะทำนายว่าเป็นสินค้าชนิดไหนใน 11 ประเภท พร้อมแสดงความมั่นใจ")
    
    with st.sidebar:
        st.header("🎯 การทดสอบโมเดล")
        val_confidence = st.slider("Confidence threshold (ทดสอบ)", 0.0, 1.0, 0.20, 0.01)
        
        st.markdown("---")
        st.subheader("📊 สถิติความแม่นยำ")
        accuracy_stats = calculate_model_accuracy()
        if accuracy_stats:
            st.metric("ความแม่นยำรวม", f"{accuracy_stats['accuracy']:.1%}")
            st.metric("จำนวนทดสอบทั้งหมด", accuracy_stats['total_tests'])
            col_a, col_b = st.columns(2)
            col_a.metric("ถูกต้อง", accuracy_stats['correct'])
            col_b.metric("ผิดพลาด", accuracy_stats['wrong'])
            
            with st.expander("ดูสถิติแยกตามประเภท"):
                for cat, stats in accuracy_stats['category_stats'].items():
                    if stats['total'] > 0:
                        acc = stats['correct'] / stats['total']
                        st.write(f"- {THAI_NAMES.get(cat, cat)}: {acc:.1%} ({stats['correct']}/{stats['total']})")
        else:
            st.info("ยังไม่มีข้อมูลการทดสอบ")
        
        st.markdown("---")
        if st.button("🗑️ ล้างประวัติการทดสอบ"):
            if os.path.exists(VALIDATION_HISTORY_FILE):
                os.remove(VALIDATION_HISTORY_FILE)
            st.success("ล้างประวัติเรียบร้อย!")
            st.rerun()
    
    col_test_left, col_test_right = st.columns([1, 1])
    
    with col_test_left:
        st.subheader("📸 อัปโหลดภาพสินค้า")
        test_file = st.file_uploader("เลือกภาพสินค้า", type=["jpg", "jpeg", "png"], key="test_uploader")
        
        if test_file:
            img = Image.open(test_file).convert("RGB")
            img_array = np.array(img)
            
            st.image(img_array, caption="ภาพที่อัปโหลด", use_container_width=True)
            
            st.subheader("🏷️ ป้ายกำกับจริง (สำหรับวัดความแม่นยำ)")
            actual_label = st.selectbox(
                "เลือกว่าภาพนี้คือสินค้าอะไร",
                ["(ไม่ระบุ)"] + CLASS_NAMES,
                index=0
            )
            actual_label = None if actual_label == "(ไม่ระบุ)" else actual_label
            
            with st.spinner("กำลังทำนาย..."):
                predictions = predict_single_product(img_array, val_confidence)
                all_categories = predict_all_categories(img_array, val_confidence)
            
            if predictions:
                img_with_pred = draw_prediction_on_image(img_array, predictions)
                st.image(img_with_pred, caption="ผลการทำนาย (มีกรอบ)", use_container_width=True)
            else:
                st.warning("ไม่พบสินค้าในภาพ")
            
            if st.button("💾 บันทึกผลการทดสอบ"):
                save_validation_result(img_array, test_file.name, predictions, actual_label)
                st.success("บันทึกผลเรียบร้อย!")
            
            with col_test_right:
                st.subheader("📊 ผลการทำนายทั้ง 11 ประเภท")
                
                df_data = []
                for cat in all_categories:
                    if cat["has_product"]:
                        status = f"✅ พบ (ความมั่นใจ {cat['confidence']:.1%})"
                    else:
                        status = "❌ ไม่พบ"
                    
                    df_data.append({
                        "ประเภทสินค้า": cat["thai_name"],
                        "ชื่ออังกฤษ": cat["class_name"],
                        "ผลการตรวจจับ": status,
                        "ความมั่นใจ": f"{cat['confidence']:.1%}" if cat["confidence"] > 0 else "-"
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, height=500)
                
                st.subheader("🏆 อันดับความมั่นใจสูงสุด")
                top_predictions = [p for p in all_categories if p["confidence"] > 0][:5]
                if top_predictions:
                    for i, p in enumerate(top_predictions):
                        st.write(f"{i+1}. {p['thai_name']}: {p['confidence']:.1%}")
                else:
                    st.info("ไม่พบการตรวจจับ")
    
    st.markdown("---")
    st.subheader("📜 ประวัติการทดสอบล่าสุด")
    history = load_validation_history()
    if history:
        for record in history[:5]:
            with st.expander(f"📅 {record['time']} - {record['filename']}"):
                img_bytes = base64.b64decode(record['image_base64'])
                img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, width=200)
                
                st.write(f"**ผลทำนายสูงสุด:** {record.get('top_prediction', '-')} (ความมั่นใจ {record.get('top_confidence', 0):.1%})")
                if record.get('actual_label'):
                    is_correct = record.get('top_prediction') == record.get('actual_label')
                    st.write(f"**ป้ายกำกับจริง:** {record['actual_label']} {'✅ ถูกต้อง' if is_correct else '❌ ผิดพลาด'}")
                
                st.write("**3 อันดับแรก:**")
                for p in record.get('predictions', [])[:3]:
                    st.write(f"  - {p['thai_name']}: {p['confidence']:.1%}")
    else:
        st.info("ยังไม่มีประวัติการทดสอบ")

# ==================== โหมด 3: Simulation Mode ====================
else:
    st.markdown("---")
    st.subheader("🎮 Simulation Mode - จำลองสถานะสินค้า 11 ช่อง")
    st.markdown("ปรับสถานะสินค้าในแต่ละช่อง (✅ มีสินค้า / ❌ สินค้าหมด) เพื่อทดสอบระบบแจ้งเตือน")
    
    with st.sidebar:
        st.header("🎮 การตั้งค่า Simulation")
        if st.button("🔄 รีเซ็ตสถานะทั้งหมด (มีสินค้าทุกช่อง)"):
            simulation_slots = get_default_simulation_slots()
            save_simulation_state(simulation_slots)
            st.success("รีเซ็ตเรียบร้อย!")
            st.rerun()
        
        if st.button("❌ ตั้งค่าสินค้าหมดทุกช่อง"):
            simulation_slots = get_default_simulation_slots()
            for slot in simulation_slots:
                slot["status"] = False
            save_simulation_state(simulation_slots)
            st.success("ตั้งค่าสินค้าหมดทุกช่อง!")
            st.rerun()
        
        st.markdown("---")
        st.info("💡 คลิกที่ปุ่มสินค้าเพื่อเปลี่ยนสถานะ")
    
    simulation_slots = load_simulation_state()
    if not simulation_slots:
        simulation_slots = get_default_simulation_slots()
        save_simulation_state(simulation_slots)
    
    st.subheader("📊 สถานะสินค้า 11 ช่อง (จำลอง)")
    
    # แสดงเป็น Grid 4x3
    for row in range(4):
        cols_in_row = st.columns(3)
        for col in range(3):
            idx = row * 3 + col
            if idx < len(simulation_slots):
                slot = simulation_slots[idx]
                with cols_in_row[col]:
                    thai_name = THAI_NAMES.get(slot["name"], slot["name"])
                    
                    if slot["status"]:
                        bg_color = "#d4edda"
                        border_color = "#28a745"
                        status_text = "✅ มีสินค้า"
                        icon = "🟢"
                    else:
                        bg_color = "#f8d7da"
                        border_color = "#dc3545"
                        status_text = "❌ สินค้าหมด"
                        icon = "🔴"
                    
                    if st.button(
                        f"{icon} {slot['id']}: {thai_name}\n\n{status_text}",
                        key=f"sim_{slot['id']}",
                        use_container_width=True
                    ):
                        slot["status"] = not slot["status"]
                        save_simulation_state(simulation_slots)
                        st.rerun()
                    
                    st.markdown(f"""
                    <div style="background-color:{bg_color}; padding:15px; border-radius:10px; 
                                border:2px solid {border_color}; text-align:center; margin:5px 0;">
                        <b>{slot['id']}</b><br>
                        <small>{thai_name}</small><br>
                        <span style="color:{'green' if slot['status'] else 'red'}; font-weight:bold;">
                            {status_text}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🗺️ แผนผังสถานะรวม")
    
    total = len(simulation_slots)
    occupied = sum(1 for s in simulation_slots if s["status"])
    empty = total - occupied
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("🥤 ช่องทั้งหมด", total)
    col_b.metric("✅ มีสินค้า", occupied, delta=f"+{occupied}" if occupied > 0 else None)
    col_c.metric("❌ สินค้าหมด", empty, delta=f"-{empty}" if empty > 0 else None)
    
    empty_slots = [s for s in simulation_slots if not s["status"]]
    if empty_slots:
        st.warning(f"⚠️ สินค้าหมด {len(empty_slots)} ช่อง:")
        for slot in empty_slots:
            st.write(f"  • ช่อง {slot['id']}: {THAI_NAMES.get(slot['name'], slot['name'])}")
    else:
        st.balloons()
        st.success("🎉 สินค้าครบทุช่อง!")
    
    st.markdown("---")
    st.subheader("🔔 ทดสอบระบบแจ้งเตือน")
    if st.button("📢 ทดสอบแจ้งเตือนสถานะปัจจุบัน"):
        empty_names = [THAI_NAMES.get(slot['name'], slot['name']) for slot in empty_slots]
        if empty_names:
            for name in empty_names:
                st.toast(f"⚠️ สินค้าหมด: {name}", icon="🔴")
        else:
            st.toast("✅ สินค้าครบทุกช่อง!", icon="🎉")
        st.success("ทดสอบการแจ้งเตือนเรียบร้อย")

# ------------------- ส่วนท้าย -------------------
st.markdown("---")
with st.expander("📄 คู่มือการใช้งาน"):
    st.markdown("""
    ### 🥤 Stock Vision System APP

    **ระบบตรวจจับสินค้า 11 ชนิด:**
    - Coke Can (โค้กออริจินัล กระป๋องสีแดง)
    - Coke Light Can (โค้กไลท์ กระป๋องสีเงิน)
    - Fanta Grape (แฟนต้าน้ำองุ่น กระป๋องสีม่วง)
    - Fanta Orange Can (แฟนต้าน้ำส้ม กระป๋องสีส้ม)
    - Lactasoy (นมถั่วเหลืองแลคตาซอย กล่องสีฟ้า)
    - Meiji Milk (นมสดเมจิ ขวดสีขาวฝาน้ำเงิน)
    - Oishi Rice (ชาเขียวโออิชิ รสข้าวญี่ปุ่น ขวดสีส้ม)
    - Oishi Honey Lemon (ชาเขียวโออิชิ รสน้ำผึ้งผสมมะนาว ขวดสีเหลือง)
    - Oishi Kyoho (ชาเขียวโออิชิ รสเคียวโฮ ขวดสีม่วง)
    - Pepsi Can (เป๊ปซี่ กระป๋องสีน้ำเงิน)
    - Sprite Can (สไปรท์ กระป๋องสีเขียว)

    **วิธีปรับแต่งกรอบให้ตรงกับภาพ:**
    1. ไปที่ Sidebar → เลือก "🖱️ ปรับพร้อมดูตัวอย่างภาพจริง"
    2. กดปุ่ม "🎨 เปิดหน้าปรับแต่ง"
    3. อัปโหลดภาพที่ต้องการปรับแต่ง
    4. คลิกเลือกช่องที่ต้องการปรับ (S01-S11)
    5. ปรับค่า x1, x2 (ซ้าย-ขวา) และ y1, y2 (บน-ล่าง) จนกรอบครอบสินค้าพอดี
    6. กด "💾 บันทึกการตั้งค่า"

    **คำแนะนำ:**
    - ใช้ Confidence threshold 0.15-0.25 สำหรับสภาพแสงทั่วไป
    - ควรถ่ายภาพให้ชัดเจนและแสงสว่างเพียงพอ
    - สามารถทดสอบความแม่นยำของโมเดลได้ในโหมด "ทดสอบความแม่นยำโมเดล"
    """)