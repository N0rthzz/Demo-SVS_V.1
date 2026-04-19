import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
from collections import Counter

# ------------------- ตั้งค่า page config -------------------
st.set_page_config(page_title="Stock Vision System - Fixed Slot", layout="wide")

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

# ------------------- กำหนด Shelf Slot 11 ช่อง (ตำแหน่งคงที่) -------------------
SLOT_RELATIVE_BOXES = [
    {"id": "S01", "name": "สมุนไพร", "rel_bbox": [0.05, 0.10, 0.20, 0.35]},
    {"id": "S02", "name": "Coconut Water Carton", "rel_bbox": [0.22, 0.10, 0.37, 0.35]},
    {"id": "S03", "name": "Coffee Can", "rel_bbox": [0.39, 0.10, 0.54, 0.35]},
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

def check_slot_occupancy(detection_boxes, slot_abs_bbox, iou_thresh=0.1):
    sx1, sy1, sx2, sy2 = slot_abs_bbox
    slot_area = (sx2 - sx1) * (sy2 - sy1)
    if slot_area <= 0:
        return False, None
    best_iou = 0
    best_class = None
    for (dx1, dy1, dx2, dy2, class_name) in detection_boxes:
        ix1 = max(sx1, dx1)
        iy1 = max(sy1, dy1)
        ix2 = min(sx2, dx2)
        iy2 = min(sy2, dy2)
        if ix2 > ix1 and iy2 > iy1:
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            iou = inter_area / slot_area
            if iou > best_iou:
                best_iou = iou
                best_class = class_name
    return best_iou > iou_thresh, best_class

def analyze_shelf_image(img_array, conf_threshold=0.25):
    results = model(img_array, conf=conf_threshold)
    h, w = img_array.shape[:2]
    
    detection_boxes = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = CLASS_NAMES[cls_id]
            if class_name != "Empty_Stock":
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detection_boxes.append([x1, y1, x2, y2, class_name])
    
    slot_statuses = []
    empty_slots = []
    
    for slot in SLOT_RELATIVE_BOXES:
        abs_bbox = rel_to_abs(slot["rel_bbox"], w, h)
        occupied, detected_class = check_slot_occupancy(detection_boxes, abs_bbox, iou_thresh=0.1)
        
        # ตรวจสอบว่าสินค้าตรงกับช่องหรือไม่
        is_correct = False
        if occupied and detected_class:
            # เช็คว่าสินค้าที่ตรวจพบตรงกับชื่อช่องหรือไม่ (ยกเว้น S01)
            if slot["id"] == "S01":
                is_correct = True  # S01 รับได้ทุกอย่าง
            elif detected_class == slot["name"]:
                is_correct = True
            else:
                # สินค้าผิดช่อง
                is_correct = False
                occupied = False
        
        slot_statuses.append({
            "id": slot["id"],
            "name": slot["name"],
            "status": occupied,
            "detected": detected_class if occupied else "ไม่มีสินค้า",
            "is_correct": is_correct
        })
        
        if not occupied:
            empty_slots.append(slot["name"])
    
    return slot_statuses, empty_slots, detection_boxes

def draw_grid_on_image(img_array, slot_statuses, show_labels=True):
    """วาดตาราง 11 ช่องบนภาพแบบเรียบง่าย"""
    img_draw = img_array.copy()
    h, w = img_draw.shape[:2]
    
    # กำหนดตำแหน่งกริด 4 แถว 3 คอลัมน์ (แถวสุดท้ายมี 2 ช่อง)
    grid_rows = 4
    grid_cols = 3
    
    cell_height = h // grid_rows
    cell_width = w // grid_cols
    
    # สร้าง mapping ช่องไปยังตำแหน่งกริด
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
        
        # สีพื้นหลังตามสถานะ
        if slot["status"]:
            if slot["is_correct"]:
                color = (0, 255, 0)  # เขียว = มีสินค้าถูกต้อง
            else:
                color = (0, 165, 255)  # ส้ม = มีสินค้าผิดช่อง
        else:
            color = (0, 0, 255)  # แดง = สินค้าหมด
        
        # วาดกรอบหนา
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)
        
        if show_labels:
            # แสดงชื่อช่องและสถานะแบบสั้น
            if slot["status"]:
                if slot["is_correct"]:
                    label = f"{slot['id']}: {slot['detected']}"
                else:
                    label = f"{slot['id']}: WRONG ({slot['detected']})"
            else:
                label = f"{slot['id']}: EMPTY"
            
            # พื้นหลังข้อความ
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 2)
            
            bg_x1 = x1 + 5
            bg_y1 = y1 + 5
            bg_x2 = min(x1 + text_w + 15, w - 5)
            bg_y2 = y1 + text_h + 15
            
            cv2.rectangle(img_draw, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.putText(img_draw, label, (x1 + 8, y1 + 22), font, font_scale, (255, 255, 255), 1)
    
    return img_draw

# ------------------- UI หลัก -------------------
st.title("📦 Stock Vision System - Complete")
st.markdown("**ระบบตรวจสอบสินค้า 11 ช่อง | รองรับการตรวจจับสินค้าผิดช่อง**")

# ------------------- Session State -------------------
if 'last_empty' not in st.session_state:
    st.session_state.last_empty = []
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'uploaded_images_history' not in st.session_state:
    st.session_state.uploaded_images_history = []

# ------------------- ฟังก์ชันแจ้งเตือน -------------------
def add_alerts(empty_slots, slot_statuses):
    # แจ้งเตือนสินค้าหมด
    if set(empty_slots) != set(st.session_state.last_empty):
        new_empty = set(empty_slots) - set(st.session_state.last_empty)
        for slot_name in new_empty:
            msg = f"⚠️ สินค้าหมด: {slot_name}"
            st.session_state.alert_history.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "message": msg})
            st.toast(msg, icon="🔴")
        st.session_state.last_empty = empty_slots.copy()
    
    # แจ้งเตือนสินค้าผิดช่อง
    for slot in slot_statuses:
        if slot["status"] and not slot["is_correct"]:
            msg = f"⚠️ สินค้าผิดช่อง: {slot['id']} ({slot['name']}) พบ {slot['detected']}"
            # ตรวจสอบว่าเคยแจ้งเตือนไปแล้วหรือยัง
            if not any(msg in alert['message'] for alert in st.session_state.alert_history[:5]):
                st.session_state.alert_history.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "message": msg})
                st.toast(msg, icon="🟠")
    
    if len(st.session_state.alert_history) > 30:
        st.session_state.alert_history.pop()

def show_dashboard(slot_statuses):
    total = len(slot_statuses)
    occupied = sum(1 for s in slot_statuses if s["status"])
    empty = total - occupied
    wrong_slot = sum(1 for s in slot_statuses if s["status"] and not s["is_correct"])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ช่องทั้งหมด", total)
    col2.metric("✅ มีสินค้า", occupied)
    col3.metric("❌ สินค้าหมด", empty)
    col4.metric("⚠️ ผิดช่อง", wrong_slot, delta_color="off")
    
    st.subheader("📋 ตารางสถานะสินค้า 11 ช่อง")
    
    df_data = []
    for slot in slot_statuses:
        if slot["status"]:
            if slot["is_correct"]:
                status_icon = "✅ มีสินค้า (ถูกต้อง)"
                status_color = "green"
            else:
                status_icon = f"⚠️ ผิดช่อง (พบ {slot['detected']})"
                status_color = "orange"
        else:
            status_icon = "❌ สินค้าหมด"
            status_color = "red"
        
        df_data.append({
            "ช่อง": slot["id"],
            "สินค้าประจำช่อง": slot["name"],
            "สถานะ": status_icon,
            "ที่ตรวจพบ": slot["detected"] if slot["status"] else "-"
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, height=400)
    
    st.subheader("🔔 ประวัติการแจ้งเตือน")
    if st.session_state.alert_history:
        alert_df = pd.DataFrame(st.session_state.alert_history)
        st.dataframe(alert_df, use_container_width=True, height=150)
    else:
        st.info("ไม่มีการแจ้งเตือน")

# ------------------- Sidebar -------------------
with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    display_size = st.selectbox("ขนาดภาพ", ["เล็ก (400px)", "กลาง (600px)", "ใหญ่ (800px)"])
    width_map = {"เล็ก (400px)": 400, "กลาง (600px)": 600, "ใหญ่ (800px)": 800}
    display_width = width_map[display_size]
    
    st.markdown("---")
    show_grid_on_camera = st.checkbox("แสดงตาราง 11 ช่องบนภาพ (โหมดกล้อง)", value=True)
    
    st.markdown("---")
    if st.button("🗑️ ล้างประวัติการแจ้งเตือน"):
        st.session_state.alert_history = []
        st.session_state.last_empty = []
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
            
            # เก็บประวัติ
            st.session_state.uploaded_images_history.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "image": img_array.copy()
            })
            if len(st.session_state.uploaded_images_history) > 5:
                st.session_state.uploaded_images_history.pop()
            
            # แสดงภาพ
            st.image(img_array, use_container_width=True)
            
            # วิเคราะห์
            slot_statuses, empty_slots, _ = analyze_shelf_image(img_array, confidence_threshold)
            add_alerts(empty_slots, slot_statuses)
            
            with col_right:
                show_dashboard(slot_statuses)
                
                if empty_slots:
                    st.warning(f"⚠️ สินค้าหมด {len(empty_slots)} ช่อง: {', '.join(empty_slots)}")
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
            
            # วิเคราะห์
            slot_statuses, empty_slots, _ = analyze_shelf_image(img_array, confidence_threshold)
            add_alerts(empty_slots, slot_statuses)
            
            # แสดงภาพ (มีหรือไม่มีกรอบ)
            if show_grid_on_camera:
                img_with_grid = draw_grid_on_image(img_array, slot_statuses, show_labels=True)
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
with st.expander("📄 วิธีใช้งาน"):
    st.markdown("""
    **สีของกรอบในโหมดกล้อง:**
    - 🟢 **เขียว** = มีสินค้าและถูกต้องตามช่อง
    - 🟠 **ส้ม** = มีสินค้าแต่ผิดช่อง (เช่น เอา Coffee Can วางในช่อง S02)
    - 🔴 **แดง** = สินค้าหมด
    
    **การแจ้งเตือน:**
    - สินค้าหมด → แจ้งเตือนสีแดง
    - สินค้าผิดช่อง → แจ้งเตือนสีส้ม
    """)