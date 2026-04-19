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

# ------------------- ตั้งค่า page config -------------------
st.set_page_config(page_title="🥤 Smart Vending Stock Monitor Pro", layout="wide")

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

# แมปชื่อสินค้า (สำหรับกรณีชื่อไม่ตรง)
NAME_MAPPING = {
    "Canned tea": "Canned tea",
    "Coconut Water Carton": "Coconut Water Carton",
    "Coffee Can": "Coffee Can", 
    "Drinking water": "Drinking water",
    "Energy Drink": "Energy Drink",
    "Green Tea Bottle": "Green Tea Bottle",
    "Juice Box": "Juice Box",
    "Protein Drink": "Protein Drink",
    "Soda Can": "Soda Can",
    "UHT milk carton": "UHT milk carton",
    "Vitamin Drink": "Vitamin Drink"
}

# ------------------- กำหนด Shelf Slot 11 ช่อง (ปรับพิกัดให้แม่นยำขึ้น) -------------------
SLOT_RELATIVE_BOXES = [
    {"id": "S01", "name": "Canned tea", "rel_bbox": [0.03, 0.05, 0.20, 0.30]},
    {"id": "S02", "name": "Coconut Water Carton", "rel_bbox": [0.22, 0.05, 0.39, 0.30]},
    {"id": "S03", "name": "Coffee Can", "rel_bbox": [0.41, 0.05, 0.58, 0.30]},
    {"id": "S04", "name": "Drinking water", "rel_bbox": [0.60, 0.05, 0.77, 0.30]},
    {"id": "S05", "name": "Energy Drink", "rel_bbox": [0.79, 0.05, 0.96, 0.30]},
    {"id": "S06", "name": "Green Tea Bottle", "rel_bbox": [0.03, 0.33, 0.20, 0.58]},
    {"id": "S07", "name": "Juice Box", "rel_bbox": [0.22, 0.33, 0.39, 0.58]},
    {"id": "S08", "name": "Protein Drink", "rel_bbox": [0.41, 0.33, 0.58, 0.58]},
    {"id": "S09", "name": "Soda Can", "rel_bbox": [0.60, 0.33, 0.77, 0.58]},
    {"id": "S10", "name": "UHT milk carton", "rel_bbox": [0.79, 0.33, 0.96, 0.58]},
    {"id": "S11", "name": "Vitamin Drink", "rel_bbox": [0.03, 0.62, 0.30, 0.87]},
]

# เพิ่ม Slot S12 สำหรับแถวที่ 4 (ตามภาพ)
SLOT_RELATIVE_BOXES.append({"id": "S12", "name": "Empty Slot", "rel_bbox": [0.35, 0.62, 0.62, 0.87]})

# ไฟล์สำหรับเก็บประวัติ
HISTORY_FILE = "stock_history.json"
UPLOAD_HISTORY_FILE = "upload_history.json"

def rel_to_abs(rel_bbox, img_w, img_h):
    x1 = int(rel_bbox[0] * img_w)
    y1 = int(rel_bbox[1] * img_h)
    x2 = int(rel_bbox[2] * img_w)
    y2 = int(rel_bbox[3] * img_h)
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]

def calculate_iou(box1, box2):
    """คำนวณ Intersection over Union"""
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
    """ปรับปรุงคุณภาพภาพเพื่อการตรวจจับที่ดีขึ้น"""
    # แปลงเป็น YUV และปรับความสว่าง
    img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    # ปรับความคมชัด
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_sharp = cv2.filter2D(img_enhanced, -1, kernel)
    
    return img_sharp

def detect_with_ensemble(img_array, conf_threshold=0.25):
    """ตรวจจับด้วยเทคนิค Ensemble (หลายรูปแบบ)"""
    all_detections = []
    
    # 1. ตรวจจับภาพต้นฉบับ
    results_orig = model(img_array, conf=conf_threshold)
    
    # 2. ตรวจจับภาพที่ปรับคุณภาพ
    img_enhanced = enhance_image(img_array)
    results_enhanced = model(img_enhanced, conf=conf_threshold)
    
    # 3. ตรวจจับภาพปรับขนาด (resize)
    h, w = img_array.shape[:2]
    img_resized = cv2.resize(img_array, (w//2, h//2))
    results_resized = model(img_resized, conf=conf_threshold)
    
    # รวมผลลัพธ์ทั้งหมด
    for results in [results_orig, results_enhanced]:
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = CLASS_NAMES[cls_id]
                confidence = float(box.conf[0])
                
                if class_name != "Empty_Stock" and confidence >= conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    all_detections.append([x1, y1, x2, y2, class_name, confidence])
    
    # 4. Non-Maximum Suppression (ลบการตรวจจับที่ซ้ำ)
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

def check_slot_occupancy_advanced(detection_boxes, slot_abs_bbox, iou_thresh=0.15):
    """ตรวจสอบช่องด้วยวิธีขั้นสูง"""
    sx1, sy1, sx2, sy2 = slot_abs_bbox
    slot_area = (sx2 - sx1) * (sy2 - sy1)
    if slot_area <= 0:
        return False, None, 0
    
    best_iou = 0
    best_class = None
    best_confidence = 0
    
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
            
            # พิจารณาทั้ง IoU และตำแหน่งศูนย์กลาง
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
    """วิเคราะห์ภาพด้วยเทคนิคขั้นสูง"""
    h, w = img_array.shape[:2]
    
    # ตรวจจับแบบ Ensemble
    detection_boxes = detect_with_ensemble(img_array, conf_threshold)
    
    # แสดง log การตรวจจับ
    detected_classes = list(set([det[4] for det in detection_boxes]))
    if detected_classes:
        st.sidebar.success(f"🔍 ตรวจจับสินค้า: {', '.join(detected_classes)}")
    else:
        st.sidebar.warning("⚠️ ไม่พบสินค้าในภาพ (ลองปรับ Confidence threshold)")
    
    slot_statuses = []
    empty_slots = []
    
    for slot in SLOT_RELATIVE_BOXES:
        abs_bbox = rel_to_abs(slot["rel_bbox"], w, h)
        occupied, detected_class, confidence = check_slot_occupancy_advanced(
            detection_boxes, abs_bbox, iou_thresh=0.15
        )
        
        is_correct = False
        if occupied and detected_class:
            # ตรวจสอบว่าสินค้าตรงกับช่องหรือไม่
            if detected_class == slot["name"]:
                is_correct = True
            else:
                # ถ้า confidence สูงมาก (0.5+) ให้ถือว่ามีสินค้าแต่ผิดช่อง
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
        
        if not occupied and slot["name"] != "Empty Slot":
            empty_slots.append(slot["name"])
    
    return slot_statuses, empty_slots, detection_boxes

def draw_grid_on_image_advanced(img_array, slot_statuses, show_labels=True):
    """วาดกรอบพร้อมแสดง Confidence Score"""
    img_draw = img_array.copy()
    h, w = img_draw.shape[:2]
    
    # แบ่งเป็น 4 แถว 3 คอลัมน์
    grid_rows = 4
    grid_cols = 3
    
    cell_height = h // grid_rows
    cell_width = w // grid_cols
    
    slot_to_grid = {
        "S01": (0, 0), "S02": (0, 1), "S03": (0, 2),
        "S04": (1, 0), "S05": (1, 1), "S06": (1, 2),
        "S07": (2, 0), "S08": (2, 1), "S09": (2, 2),
        "S10": (3, 0), "S11": (3, 1), "S12": (3, 2)
    }
    
    for slot in slot_statuses:
        if slot["id"] not in slot_to_grid:
            continue
        
        row, col = slot_to_grid[slot["id"]]
        x1 = col * cell_width
        y1 = row * cell_height
        x2 = (col + 1) * cell_width if col < grid_cols - 1 else w
        y2 = (row + 1) * cell_height if row < grid_rows - 1 else h
        
        # เลือกสีตามสถานะ
        if slot["status"]:
            if slot["is_correct"]:
                color = (0, 255, 0)  # เขียว
                border_color = (0, 200, 0)
            else:
                color = (0, 165, 255)  # ส้ม
                border_color = (0, 130, 200)
        else:
            color = (0, 0, 255)  # แดง
            border_color = (0, 0, 200)
        
        # วาดกรอบหลายชั้นเพื่อความชัดเจน
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), border_color, 4)
        cv2.rectangle(img_draw, (x1+2, y1+2), (x2-2, y2-2), color, 2)
        
        if show_labels:
            # สร้างข้อความแสดงผล
            if slot["status"]:
                if slot["is_correct"]:
                    label = f"✅ {slot['id']}: {slot['detected']}"
                else:
                    label = f"⚠️ {slot['id']}: {slot['detected']}"
            else:
                label = f"❌ {slot['id']}: หมด"
            
            # เพิ่ม Confidence Score
            if slot["status"] and slot["confidence"] > 0:
                label += f" ({slot['confidence']:.0%})"
            
            # วาดพื้นหลังข้อความ
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            bg_x1 = x1 + 8
            bg_y1 = y1 + 8
            bg_x2 = min(x1 + text_w + 20, w - 5)
            bg_y2 = y1 + text_h + 25
            
            cv2.rectangle(img_draw, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.rectangle(img_draw, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 1)
            cv2.putText(img_draw, label, (x1 + 12, y1 + 28), font, font_scale, (255, 255, 255), thickness)
    
    return img_draw

def save_upload_history(image_array, filename):
    """บันทึกประวัติการอัปโหลด"""
    history = []
    if os.path.exists(UPLOAD_HISTORY_FILE):
        try:
            with open(UPLOAD_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except:
            history = []
    
    # แปลงภาพเป็น base64
    import base64
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    history.insert(0, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": str(filename),
        "image_base64": str(image_base64)
    })
    
    # เก็บแค่ 10 ภาพล่าสุด
    history = history[:10]
    
    try:
        with open(UPLOAD_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.warning(f"ไม่สามารถบันทึกประวัติภาพได้: {e}")

def load_upload_history():
    """โหลดประวัติการอัปโหลด"""
    if os.path.exists(UPLOAD_HISTORY_FILE):
        try:
            with open(UPLOAD_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def save_stock_history(slot_statuses):
    """บันทึกประวัติสถานะสินค้า (แก้ไข JSON serializable)"""
    history = {}
    for slot in slot_statuses:
        # แปลงค่าให้เป็น JSON serializable ทั้งหมด
        history[slot["id"]] = {
            "status": bool(slot["status"]),  # แปลงเป็น bool ที่ JSON รองรับ
            "detected": str(slot["detected"]),  # แปลงเป็น string
            "is_correct": bool(slot["is_correct"]),  # แปลงเป็น bool
            "confidence": float(slot.get("confidence", 0))  # แปลงเป็น float
        }
    history["last_update"] = datetime.now().isoformat()
    
    # เขียนไฟล์อย่างปลอดภัย
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        st.error(f"ไม่สามารถบันทึกประวัติได้: {e}")

def load_stock_history():
    """โหลดประวัติสถานะสินค้า"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

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

# ------------------- UI หลัก -------------------
st.title("🥤 Smart Vending Stock Monitor Pro")
st.markdown("**ระบบตรวจจับสินค้าหมดอัจฉริยะ พร้อมเทคโนโลยี Ensemble Detection**")

# ------------------- Session State -------------------
if 'last_empty' not in st.session_state:
    st.session_state.last_empty = []
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'current_slot_statuses' not in st.session_state:
    st.session_state.current_slot_statuses = []

def add_alerts(empty_slots, slot_statuses):
    # แจ้งเตือนสินค้าหมด
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
    
    # แจ้งเตือนสินค้าผิดช่อง
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
    total = len([s for s in slot_statuses if s["name"] != "Empty Slot"])
    occupied = sum(1 for s in slot_statuses if s["status"] and s["name"] != "Empty Slot")
    empty = total - occupied
    wrong_slot = sum(1 for s in slot_statuses if s["status"] and not s["is_correct"] and s["name"] != "Empty Slot")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🥤 ช่องทั้งหมด", total)
    col2.metric("✅ มีสินค้า", occupied)
    col3.metric("❌ สินค้าหมด", empty, delta=f"-{empty}" if empty > 0 else None)
    col4.metric("⚠️ ผิดช่อง", wrong_slot)
    
    st.subheader("📋 ตารางสถานะสินค้า")
    
    df_data = []
    for slot in slot_statuses:
        if slot["name"] == "Empty Slot":
            continue
            
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
    
    # แสดงประวัติการเปลี่ยนแปลง
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

# ------------------- Sidebar -------------------
with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01,
                                     help="ค่าความมั่นใจขั้นต่ำในการตรวจจับ (แนะนำ 0.25-0.35)")
    display_size = st.selectbox("ขนาดภาพ", ["เล็ก (400px)", "กลาง (600px)", "ใหญ่ (800px)"])
    width_map = {"เล็ก (400px)": 400, "กลาง (600px)": 600, "ใหญ่ (800px)": 800}
    display_width = width_map[display_size]
    
    st.markdown("---")
    st.subheader("🎨 ตัวเลือกการแสดงผล")
    show_grid_on_camera = st.checkbox("แสดงตาราง 11 ช่องบนภาพ", value=True)
    show_confidence = st.checkbox("แสดง Confidence Score", value=True)
    
    st.markdown("---")
    st.subheader("📸 ประวัติการอัปโหลด")
    upload_history = load_upload_history()
    if upload_history:
        for hist in upload_history[:3]:
            st.caption(f"📅 {hist['time']}")
            st.caption(f"📄 {hist['filename']}")
            st.markdown("---")
    
    st.markdown("---")
    col_reset1, col_reset2 = st.columns(2)
    with col_reset1:
        if st.button("🗑️ ล้างประวัติแจ้งเตือน"):
            st.session_state.alert_history = []
            st.session_state.last_empty = []
            st.rerun()
    with col_reset2:
        if st.button("🔄 รีเซ็ตระบบทั้งหมด"):
            st.session_state.alert_history = []
            st.session_state.last_empty = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            if os.path.exists(UPLOAD_HISTORY_FILE):
                os.remove(UPLOAD_HISTORY_FILE)
            st.success("รีเซ็ตระบบเรียบร้อย!")
            st.rerun()

# ------------------- โหมดการทำงาน -------------------
mode = st.radio("เลือกโหมด", ["📸 อัปโหลดภาพ", "📷 ถ่ายภาพจากกล้อง"], horizontal=True)

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("🖼️ ภาพที่วิเคราะห์")
    
    if mode == "📸 อัปโหลดภาพ":
        uploaded_file = st.file_uploader("เลือกภาพชั้นวางสินค้า", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(img)
            
            # บันทึกประวัติ
            save_upload_history(img_array, uploaded_file.name)
            
            st.image(img_array, use_container_width=True)
            
            # วิเคราะห์ด้วยระบบขั้นสูง
            with st.spinner("กำลังวิเคราะห์ภาพ..."):
                slot_statuses, empty_slots, _ = analyze_shelf_image_advanced(img_array, confidence_threshold)
            
            st.session_state.current_slot_statuses = slot_statuses
            add_alerts(empty_slots, slot_statuses)
            save_stock_history(slot_statuses)
            
            with col_right:
                show_dashboard(slot_statuses)
                
                if empty_slots:
                    st.error(f"⚠️ สินค้าหมด {len(empty_slots)} ช่อง: {', '.join(empty_slots)}")
                    
                    # แสดงรายการสินค้าหมดแบบละเอียด
                    st.subheader("📋 รายการสินค้าที่ต้องเติม")
                    for slot in slot_statuses:
                        if not slot["status"] and slot["name"] != "Empty Slot":
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
            
            # แสดงประวัติภาพล่าสุด
            upload_history = load_upload_history()
            if upload_history:
                st.subheader("📸 ภาพที่อัปโหลดล่าสุด")
                cols = st.columns(min(3, len(upload_history)))
                for idx, hist in enumerate(upload_history[:3]):
                    with cols[idx]:
                        st.caption(f"📅 {hist['time']}")
                        import base64
                        img_bytes = base64.b64decode(hist['image_base64'])
                        img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, use_container_width=True)
                        if st.button(f"โหลดภาพนี้", key=f"load_{idx}"):
                            # โหลดภาพเก่ามาวิเคราะห์
                            slot_statuses, empty_slots, _ = analyze_shelf_image_advanced(img_rgb, confidence_threshold)
                            st.session_state.current_slot_statuses = slot_statuses
                            add_alerts(empty_slots, slot_statuses)
                            save_stock_history(slot_statuses)
                            st.rerun()
    
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
                img_with_grid = draw_grid_on_image_advanced(img_array, slot_statuses, show_labels=show_confidence)
                st.image(img_with_grid, caption="ภาพจากกล้อง (พร้อมตาราง 11 ช่อง)", use_container_width=True)
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

# ------------------- ส่วนท้าย -------------------
st.markdown("---")
with st.expander("📄 คู่มือการใช้งานและเทคนิคการตรวจจับ"):
    st.markdown("""
    ### 🥤 Smart Vending Stock Monitor Pro
    
    **เทคโนโลยีที่ใช้เพิ่มความแม่นยำ:**
    - 🔬 **Ensemble Detection**: ตรวจจับหลายรูปแบบ (ต้นฉบับ, ปรับคุณภาพ, ปรับขนาด)
    - 🎯 **Non-Maximum Suppression**: ลบการตรวจจับที่ซ้ำซ้อน
    - 📊 **Confidence Scoring**: แสดงความมั่นใจในการตรวจจับ
    - 💾 **ประวัติการอัปโหลด**: เก็บภาพที่อัปโหลดไว้เรียกใช้ใหม่
    
    **วิธีการใช้งาน:**
    1. เลือกโหมด **อัปโหลดภาพ** หรือ **ถ่ายภาพจากกล้อง**
    2. ปรับ Confidence threshold (0.25-0.35 เหมาะสม)
    3. ระบบจะตรวจจับและแสดงสถานะอัตโนมัติ
    
    **ความหมายของสีกรอบ:**
    - 🟢 **เขียว** = มีสินค้าและถูกต้องตามช่อง
    - 🟠 **ส้ม** = มีสินค้าแต่ผิดช่อง
    - 🔴 **แดง** = สินค้าหมด
    
    **คำแนะนำ:**
    - ถ่ายภาพให้ตรงและแสงสว่างเพียงพอ
    - ปรับ Confidence threshold หากตรวจจับผิดพลาดบ่อย
    - ใช้ปุ่มรีเซ็ตเมื่อต้องการเริ่มต้นใหม่
    """)