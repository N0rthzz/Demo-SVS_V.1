import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
from collections import Counter
import os
import json
import time
import base64
import random

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

# ------------------- กำหนด Shelf Slot 11 ช่อง -------------------
SLOT_RELATIVE_BOXES = [
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

# ==================== ฟังก์ชันสำหรับทดสอบความแม่นยำ ====================
def predict_single_product(img_array, conf_threshold=0.25):
    """ทำนายสินค้าจากภาพเดียว พร้อมแสดงผลทั้งหมด"""
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
    """ทำนายทุกประเภทสินค้า 11 ชนิด พร้อมความมั่นใจ"""
    results = model(img_array, conf=conf_threshold)
    
    # สร้าง dictionary สำหรับเก็บความมั่นใจของแต่ละคลาส
    category_scores = {name: 0.0 for name in CLASS_NAMES if name != "Empty_Stock"}
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = CLASS_NAMES[cls_id]
            confidence = float(box.conf[0])
            
            if class_name != "Empty_Stock" and confidence >= conf_threshold:
                if confidence > category_scores[class_name]:
                    category_scores[class_name] = confidence
    
    # แปลงเป็นลิสต์เรียงตามความมั่นใจ
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
    """วาดกรอบและข้อความทำนายบนภาพ"""
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
    """บันทึกผลการทดสอบความแม่นยำ"""
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
    """โหลดประวัติการทดสอบ"""
    if os.path.exists(VALIDATION_HISTORY_FILE):
        try:
            with open(VALIDATION_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def calculate_model_accuracy():
    """คำนวณความแม่นยำของโมเดลจากประวัติ"""
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
    """บันทึกสถานะ Simulation"""
    state = []
    for slot in slot_statuses:
        state.append({
            "id": slot["id"],
            "name": slot["name"],
            "status": slot["status"]
        })
    state["last_update"] = datetime.now().isoformat()
    
    try:
        with open(SIMULATION_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        pass

def load_simulation_state():
    """โหลดสถานะ Simulation"""
    if os.path.exists(SIMULATION_FILE):
        try:
            with open(SIMULATION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return None
    return None

def get_default_simulation_slots():
    """สร้างสถานะเริ่มต้นสำหรับ Simulation (มีสินค้าทุกช่อง)"""
    slots = []
    for slot in SLOT_RELATIVE_BOXES:
        slots.append({
            "id": slot["id"],
            "name": slot["name"],
            "status": True  # มีสินค้าทุกช่องเริ่มต้น
        })
    return slots

# ==================== ฟังก์ชันหลัก ====================
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
    img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_sharp = cv2.filter2D(img_enhanced, -1, kernel)
    return img_sharp

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
    brightness = np.mean(gray)
    return brightness < 130

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

def analyze_shelf_image_advanced(img_array, conf_threshold=0.25):
    h, w = img_array.shape[:2]
    detection_boxes = detect_with_ensemble(img_array, conf_threshold)
    
    detected_classes = list(set([det[4] for det in detection_boxes]))
    if detected_classes:
        st.sidebar.success(f"🔍 ตรวจจับ: {', '.join(detected_classes)}")
    else:
        st.sidebar.warning("⚠️ ไม่พบสินค้า")
    
    slot_statuses = []
    empty_slots = []
    
    for slot in SLOT_RELATIVE_BOXES:
        abs_bbox = rel_to_abs(slot["rel_bbox"], w, h)
        occupied, detected_class, confidence = check_slot_occupancy_advanced(detection_boxes, abs_bbox, iou_thresh=0.15)
        
        if not occupied:
            brightness_occupied = analyze_by_brightness(img_array, abs_bbox)
            if brightness_occupied:
                occupied = True
                detected_class = slot["name"]
                confidence = 0.4
        
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

def draw_grid_on_image_advanced(img_array, slot_statuses, show_labels=True):
    img_draw = img_array.copy()
    h, w = img_draw.shape[:2]
    
    grid_rows = 4
    grid_cols = 3
    cell_height = h // grid_rows
    cell_width = w // grid_cols
    
    slot_to_grid = {
        "S01": (0, 0), "S02": (0, 1), "S03": (0, 2),
        "S04": (1, 0), "S05": (1, 1), "S06": (1, 2),
        "S07": (2, 0), "S08": (2, 1), "S09": (2, 2),
        "S10": (3, 0), "S11": (3, 1)
    }
    
    for slot in slot_statuses:
        if slot["id"] not in slot_to_grid:
            continue
        
        row, col = slot_to_grid[slot["id"]]
        x1 = col * cell_width
        y1 = row * cell_height
        x2 = (col + 1) * cell_width if col < grid_cols - 1 else w
        y2 = (row + 1) * cell_height if row < grid_rows - 1 else h
        
        if slot["status"]:
            if slot["is_correct"]:
                color = (0, 255, 0)
            else:
                color = (0, 165, 255)
        else:
            color = (0, 0, 255)
        
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)
        
        if show_labels:
            if slot["status"]:
                if slot["is_correct"]:
                    label = f"✅ {slot['id']}: {slot['detected']}"
                else:
                    label = f"⚠️ {slot['id']}: {slot['detected']}"
            else:
                label = f"❌ {slot['id']}: หมด"
            
            if slot["status"] and slot["confidence"] > 0:
                label += f" ({slot['confidence']:.0%})"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            bg_x1 = x1 + 5
            bg_y1 = y1 + 5
            bg_x2 = min(x1 + text_w + 15, w - 5)
            bg_y2 = y1 + text_h + 20
            
            cv2.rectangle(img_draw, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.putText(img_draw, label, (x1 + 8, y1 + 22), font, font_scale, (255, 255, 255), thickness)
    
    return img_draw

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
st.title("🥤 Smart Vending Stock Monitor Pro")
st.markdown("**ระบบตรวจจับสินค้าหมดอัจฉริยะ | ทดสอบความแม่นยำ | โหมดจำลองสถานการณ์**")

# เลือกโหมดหลัก
main_mode = st.radio(
    "เลือกโหมดหลัก", 
    ["📦 ตรวจสอบสต็อกสินค้า", "🎯 ทดสอบความแม่นยำโมเดล", "🎮 Simulation Mode"], 
    horizontal=True
)

# ==================== โหมด 1: ตรวจสอบสต็อกสินค้า ====================
if main_mode == "📦 ตรวจสอบสต็อกสินค้า":
    with st.sidebar:
        st.header("⚙️ การตั้งค่า")
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.20, 0.01)
        display_size = st.selectbox("ขนาดภาพ", ["เล็ก (400px)", "กลาง (600px)", "ใหญ่ (800px)"])
        width_map = {"เล็ก (400px)": 400, "กลาง (600px)": 600, "ใหญ่ (800px)": 800}
        display_width = width_map[display_size]
        
        st.markdown("---")
        st.subheader("🎨 ตัวเลือกการแสดงผล")
        show_grid_on_camera = st.checkbox("แสดงตาราง 11 ช่องบนภาพ", value=True)
        
        st.markdown("---")
        if st.button("🗑️ ล้างประวัติการแจ้งเตือน"):
            st.session_state.alert_history = []
            st.session_state.last_empty = []
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
                st.image(img_array, use_container_width=True)
                
                with st.spinner("กำลังวิเคราะห์ภาพ..."):
                    slot_statuses, empty_slots, _ = analyze_shelf_image_advanced(img_array, confidence_threshold)
                
                st.session_state.current_slot_statuses = slot_statuses
                add_alerts(empty_slots, slot_statuses)
                save_stock_history(slot_statuses)
                
                with col_right:
                    show_dashboard(slot_statuses)
                    
                    if empty_slots:
                        st.error(f"⚠️ สินค้าหมด {len(empty_slots)} ช่อง: {', '.join(empty_slots)}")
                        st.subheader("📋 รายการสินค้าที่ต้องเติม")
                        for slot in slot_statuses:
                            if not slot["status"]:
                                st.write(f"  • ช่อง {slot['id']}: {slot['name']}")
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
                    slot_statuses, empty_slots, _ = analyze_shelf_image_advanced(img_array, confidence_threshold)
                
                st.session_state.current_slot_statuses = slot_statuses
                add_alerts(empty_slots, slot_statuses)
                save_stock_history(slot_statuses)
                
                if show_grid_on_camera:
                    img_with_grid = draw_grid_on_image_advanced(img_array, slot_statuses, show_labels=True)
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
            
            # เลือกป้ายกำกับจริง
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
            
            # แสดงภาพพร้อมกรอบ
            if predictions:
                img_with_pred = draw_prediction_on_image(img_array, predictions)
                st.image(img_with_pred, caption="ผลการทำนาย (มีกรอบ)", use_container_width=True)
            else:
                st.warning("ไม่พบสินค้าในภาพ")
            
            # บันทึกผล
            if st.button("💾 บันทึกผลการทดสอบ"):
                save_validation_result(img_array, test_file.name, predictions, actual_label)
                st.success("บันทึกผลเรียบร้อย!")
            
            with col_test_right:
                st.subheader("📊 ผลการทำนายทั้ง 11 ประเภท")
                
                # แสดงตารางความมั่นใจของทุกประเภท
                df_data = []
                for cat in all_categories:
                    if cat["has_product"]:
                        status = f"✅ พบ (ความมั่นใจ {cat['confidence']:.1%})"
                        color = "green"
                    else:
                        status = "❌ ไม่พบ"
                        color = "red"
                    
                    df_data.append({
                        "ประเภทสินค้า": cat["thai_name"],
                        "ชื่ออังกฤษ": cat["class_name"],
                        "ผลการตรวจจับ": status,
                        "ความมั่นใจ": f"{cat['confidence']:.1%}" if cat["confidence"] > 0 else "-"
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, height=500)
                
                # แสดงอันดับความมั่นใจสูงสุด
                st.subheader("🏆 อันดับความมั่นใจสูงสุด")
                top_predictions = [p for p in all_categories if p["confidence"] > 0][:5]
                if top_predictions:
                    for i, p in enumerate(top_predictions):
                        st.write(f"{i+1}. {p['thai_name']}: {p['confidence']:.1%}")
                else:
                    st.info("ไม่พบการตรวจจับ")
    
    # แสดงประวัติการทดสอบล่าสุด
    st.markdown("---")
    st.subheader("📜 ประวัติการทดสอบล่าสุด")
    history = load_validation_history()
    if history:
        for record in history[:5]:
            with st.expander(f"📅 {record['time']} - {record['filename']}"):
                # แสดงภาพขนาดเล็ก
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
        st.info("💡 เลือกสถานะสินค้าในตารางด้านขวา เพื่อจำลองการเพิ่ม/ลดสินค้า")
    
    # โหลดหรือสร้างสถานะ Simulation
    simulation_slots = load_simulation_state()
    if not simulation_slots:
        simulation_slots = get_default_simulation_slots()
        save_simulation_state(simulation_slots)
    
    # แสดงสถานะ Simulation เป็นตาราง
    st.subheader("📊 สถานะสินค้า 11 ช่อง (จำลอง)")
    
    # สร้างตารางแบบ Grid 4x3
    cols = st.columns(4)
    
    for idx, slot in enumerate(simulation_slots):
        col_idx = idx % 4
        with cols[col_idx]:
            # สีพื้นหลังตามสถานะ
            if slot["status"]:
                bg_color = "#d4edda"  # เขียวอ่อน
                border_color = "#28a745"
                status_text = "✅ มีสินค้า"
                icon = "🟢"
            else:
                bg_color = "#f8d7da"  # แดงอ่อน
                border_color = "#dc3545"
                status_text = "❌ สินค้าหมด"
                icon = "🔴"
            
            # ปุ่มเปลี่ยนสถานะ
            new_status = st.button(
                f"{icon} {slot['id']}: {THAI_NAMES.get(slot['name'], slot['name'])}\n\n{status_text}",
                key=f"sim_{slot['id']}",
                use_container_width=True
            )
            
            if new_status:
                slot["status"] = not slot["status"]
                save_simulation_state(simulation_slots)
                st.rerun()
            
            # แสดงสถานะปัจจุบัน
            st.markdown(f"""
            <div style="background-color:{bg_color}; padding:15px; border-radius:10px; 
                        border:2px solid {border_color}; text-align:center; margin:5px;">
                <b>{slot['id']}</b><br>
                <small>{THAI_NAMES.get(slot['name'], slot['name'])}</small><br>
                <span style="color:{'green' if slot['status'] else 'red'}; font-weight:bold;">
                    {status_text}
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    # แผนผังแสดงภาพรวม
    st.markdown("---")
    st.subheader("🗺️ แผนผังสถานะรวม")
    
    total = len(simulation_slots)
    occupied = sum(1 for s in simulation_slots if s["status"])
    empty = total - occupied
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("🥤 ช่องทั้งหมด", total)
    col_b.metric("✅ มีสินค้า", occupied, delta=f"+{occupied}" if occupied > 0 else None)
    col_c.metric("❌ สินค้าหมด", empty, delta=f"-{empty}" if empty > 0 else None)
    
    # แสดงแผนผังแบบตาราง 4x3
    st.markdown("### แผนผังช่องวางสินค้า")
    
    # สร้างตาราง 4 แถว 3 คอลัมน์
    grid_data = []
    for row in range(4):
        row_slots = []
        for col in range(3):
            idx = row * 3 + col
            if idx < len(simulation_slots):
                slot = simulation_slots[idx]
                row_slots.append(slot)
            else:
                row_slots.append(None)
        grid_data.append(row_slots)
    
    # แสดงตาราง HTML
    html_table = '<table style="width:100%; border-collapse: collapse;">'
    for row in grid_data:
        html_table += '<tr>'
        for slot in row:
            if slot:
                color = "#d4edda" if slot["status"] else "#f8d7da"
                status = "มีสินค้า" if slot["status"] else "หมด"
                html_table += f'''
                <td style="border:2px solid #ddd; padding:15px; text-align:center; background-color:{color};">
                    <b>{slot['id']}</b><br>
                    <small>{THAI_NAMES.get(slot['name'], slot['name'])}</small><br>
                    <span style="color:{'green' if slot['status'] else 'red'};">● {status}</span>
                </td>
                '''
            else:
                html_table += '<td style="border:2px solid #ddd; padding:15px; text-align:center; background-color:#f0f0f0;"></td>'
        html_table += '</tr>'
    html_table += '</table>'
    
    st.markdown(html_table, unsafe_allow_html=True)
    
    # สรุปสินค้าหมด
    empty_slots = [s for s in simulation_slots if not s["status"]]
    if empty_slots:
        st.warning(f"⚠️ สินค้าหมด {len(empty_slots)} ช่อง:")
        for slot in empty_slots:
            st.write(f"  • ช่อง {slot['id']}: {THAI_NAMES.get(slot['name'], slot['name'])}")
    else:
        st.balloons()
        st.success("🎉 สินค้าครบทุกช่อง!")
    
    # ปุ่มทดสอบการแจ้งเตือน
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
    ### 🥤 Smart Vending Stock Monitor Pro
    
    **โหมดการทำงาน:**
    
    1. **📦 ตรวจสอบสต็อกสินค้า**
       - อัปโหลดหรือถ่ายภาพชั้นวางสินค้า
       - ระบบตรวจจับและแสดงสถานะอัตโนมัติ
       - แจ้งเตือนเมื่อสินค้าหมดหรือผิดช่อง
    
    2. **🎯 ทดสอบความแม่นยำโมเดล**
       - อัปโหลดภาพสินค้าเดี่ยว
       - ระบบทำนายว่าเป็นสินค้าชนิดไหนใน 11 ประเภท
       - แสดงความมั่นใจและสถิติความแม่นยำ
    
    3. **🎮 Simulation Mode**
       - จำลองการเพิ่ม/ลดสินค้าใน 11 ช่อง
       - ปรับสถานะได้ด้วยปุ่มกด
       - ทดสอบระบบแจ้งเตือน
    
    **ความหมายของสี:**
    - 🟢 เขียว = มีสินค้า
    - 🔴 แดง = สินค้าหมด
    - 🟠 ส้ม = มีสินค้าแต่ผิดช่อง
    
    **ไฟล์ที่เกี่ยวข้อง:**
    - `stock_history.json` - ประวัติสต็อก
    - `validation_history.json` - ประวัติการทดสอบ
    - `simulation_state.json` - สถานะ Simulation
    """)