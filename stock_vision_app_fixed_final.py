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

# ชื่อสินค้าภาษาไทย
THAI_NAMES = {
    "Canned tea": "ชากระป๋อง",
    "Coconut Water Carton": "น้ำมะพร้าวกล่อง",
    "Coffee Can": "กาแฟกระป๋อง",
    "Drinking water": "น้ำดื่ม",
    "Energy Drink": "เครื่องดื่มชูกำลัง",
    "Green Tea Bottle": "ชาเขียวขวด",
    "Juice Box": "น้ำผลไม้กล่อง",
    "Protein Drink": "เครื่องดื่มโปรตีน",
    "Soda Can": "โซดากระป๋อง",
    "UHT milk carton": "นม UHT กล่อง",
    "Vitamin Drink": "เครื่องดื่มวิตามิน"
}

# ------------------- PRODUCT LIST -------------------
PRODUCT_LIST = [
    {"id": "S01", "name": "Canned tea", "thai_name": "ชากระป๋อง"},
    {"id": "S02", "name": "Coconut Water Carton", "thai_name": "น้ำมะพร้าวกล่อง"},
    {"id": "S03", "name": "Coffee Can", "thai_name": "กาแฟกระป๋อง"},
    {"id": "S04", "name": "Drinking water", "thai_name": "น้ำดื่ม"},
    {"id": "S05", "name": "Energy Drink", "thai_name": "เครื่องดื่มชูกำลัง"},
    {"id": "S06", "name": "Green Tea Bottle", "thai_name": "ชาเขียวขวด"},
    {"id": "S07", "name": "Juice Box", "thai_name": "น้ำผลไม้กล่อง"},
    {"id": "S08", "name": "Protein Drink", "thai_name": "เครื่องดื่มโปรตีน"},
    {"id": "S09", "name": "Soda Can", "thai_name": "โซดากระป๋อง"},
    {"id": "S10", "name": "UHT milk carton", "thai_name": "นม UHT กล่อง"},
    {"id": "S11", "name": "Vitamin Drink", "thai_name": "เครื่องดื่มวิตามิน"}
]

# ------------------- กำหนด Shelf Slot 11 ช่อง -------------------
DEFAULT_SLOT_RELATIVE_BOXES = [
    {"id": "S01", "name": "Canned tea", "rel_bbox": [0.02, 0.02, 0.23, 0.28]},
    {"id": "S02", "name": "Coconut Water Carton", "rel_bbox": [0.25, 0.02, 0.46, 0.28]},
    {"id": "S03", "name": "Coffee Can", "rel_bbox": [0.48, 0.02, 0.69, 0.28]},
    {"id": "S04", "name": "Drinking water", "rel_bbox": [0.71, 0.02, 0.92, 0.28]},
    {"id": "S05", "name": "Energy Drink", "rel_bbox": [0.02, 0.31, 0.23, 0.57]},
    {"id": "S06", "name": "Green Tea Bottle", "rel_bbox": [0.25, 0.31, 0.46, 0.57]},
    {"id": "S07", "name": "Juice Box", "rel_bbox": [0.48, 0.31, 0.69, 0.57]},
    {"id": "S08", "name": "Protein Drink", "rel_bbox": [0.71, 0.31, 0.92, 0.57]},
    {"id": "S09", "name": "Soda Can", "rel_bbox": [0.02, 0.60, 0.30, 0.86]},
    {"id": "S10", "name": "UHT milk carton", "rel_bbox": [0.35, 0.60, 0.63, 0.86]},
    {"id": "S11", "name": "Vitamin Drink", "rel_bbox": [0.68, 0.60, 0.96, 0.86]},
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
                
                if class_name != "Empty_Stock" and confidence >= conf_threshold:
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
            
            if class_name != "Empty_Stock":
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
    
    category_scores = {name: 0.0 for name in CLASS_NAMES if name != "Empty_Stock"}
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = CLASS_NAMES[cls_id]
            confidence = float(box.conf[0])
            
            if class_name != "Empty_Stock" and confidence >= conf_threshold:
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
    category_stats = {name: {"total": 0, "correct": 0} for name in CLASS_NAMES if name != "Empty_Stock"}
    
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

# Sidebar สำหรับการตั้งค่า
with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")
    
   