import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import copy
import json
import os

# Load the trained YOLO model
model = YOLO("best.pt")

# ------------------- กำหนด Shelf Slot เริ่มต้น (ค่าเริ่มต้น) -------------------
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

# Mapping from detected class names to slot IDs (adjust based on your model's class names)
CLASS_TO_SLOT = {
    "Coke": "S01",
    "Coke Light": "S02",
    "Fanta Grape": "S03",
    "Fanta Orange": "S04",
    "Lactasoy": "S05",
    "Meiji Milk": "S06",
    "Oishi Rice": "S07",
    "Oishi Honey Lemon": "S08",
    "Oishi Kyoho": "S09",
    "Pepsi": "S10",
    "Sprite": "S11",
}

CATEGORY_GROUPS = {
    "Group 1 (Can Sodas 1)": ["S01", "S02", "S03","S04"],
    "Group 2 (Can Sodas 2)": ["S05", "S06", "S07","S08"],
    "Group 3 (Juice & Milk & Oishi Teas)": ["S09", "S10", "S11"],
}

# ฟังก์ชันช่วยโหลด/บันทึก JSON
SLOT_CONFIG_FILE = "slot_config.json"

def load_slots_from_json():
    """โหลดค่า slots จากไฟล์ JSON ถ้ามี ถ้าไม่มีให้ใช้ DEFAULT"""
    if os.path.exists(SLOT_CONFIG_FILE):
        try:
            with open(SLOT_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # ตรวจสอบโครงสร้างคร่าว ๆ
                if isinstance(data, list) and all("id" in s and "rel_bbox" in s for s in data):
                    return data
                else:
                    st.warning("ไฟล์ slot_config.json มีรูปแบบไม่ถูกต้อง ใช้ค่าเริ่มต้น")
        except Exception as e:
            st.error(f"ไม่สามารถโหลดไฟล์ JSON: {e}")
    return copy.deepcopy(DEFAULT_SLOT_RELATIVE_BOXES)

def save_slots_to_json(slots):
    """บันทึก slots ปัจจุบันลงไฟล์ JSON"""
    try:
        with open(SLOT_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(slots, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"ไม่สามารถบันทึกไฟล์ JSON: {e}")
        return False

# ฟังก์ชันเริ่มต้น session state
def init_session_state():
    if "slots" not in st.session_state:
        # โหลดจาก JSON ถ้ามี (หรือใช้ default)
        st.session_state.slots = load_slots_from_json()
    if "slot_edit_selected_id" not in st.session_state:
        st.session_state.slot_edit_selected_id = "S01"

def draw_shelf_slots(image, detections, slots, highlight_slot_id=None):
    """Draw shelf slots on image with colors based on detection status and optionally highlight a specific slot"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img_with_slots = image.copy()
    draw = ImageDraw.Draw(img_with_slots, "RGBA")
    width, height = image.size

    detected_slots = set()
    for det in detections:
        slot_id = det.get("slot_id")
        if slot_id:
            detected_slots.add(slot_id)

    for slot in slots:
        slot_id = slot["id"]
        rel_bbox = slot["rel_bbox"]
        x1 = int(rel_bbox[0] * width)
        y1 = int(rel_bbox[1] * height)
        x2 = int(rel_bbox[2] * width)
        y2 = int(rel_bbox[3] * height)

        # เลือกสี
        if highlight_slot_id and slot_id == highlight_slot_id:
            fill_color = (0, 255, 255, 80)   # สีฟ้าโปร่งแสง
            outline_color = "#00FFFF"
            width_line = 4
        elif slot_id in detected_slots:
            fill_color = (0, 255, 0, 50)
            outline_color = "#00FF00"
            width_line = 3
        else:
            fill_color = None
            outline_color = "#FF0000"
            width_line = 3

        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=width_line, fill=fill_color)
        draw.text((x1 + 5, y1 + 5), f"{slot_id}", fill="#FFFFFF")

    return img_with_slots

def analyze_shelf_inventory(detections, slots):
    inventory = {slot["id"]: {"name": slot["name"], "detected": False, "confidence": 0.0} for slot in slots}
    for det in detections:
        slot_id = det.get("slot_id")
        if slot_id and slot_id in inventory:
            inventory[slot_id]["detected"] = True
            inventory[slot_id]["confidence"] = det.get("confidence", 0.0)

    categorized = {}
    for category, slot_ids in CATEGORY_GROUPS.items():
        categorized[category] = {}
        for slot_id in slot_ids:
            if slot_id in inventory:
                categorized[category][slot_id] = inventory[slot_id]
    return inventory, categorized

def display_inventory_ui(inventory, categorized):
    total_slots = len(inventory)
    filled_slots = sum(1 for v in inventory.values() if v["detected"])
    empty_slots = total_slots - filled_slots

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Slots", total_slots)
    with col2:
        st.metric("✅ Stocked", filled_slots)
    with col3:
        st.metric("⚠️ Empty", empty_slots, delta=-empty_slots if empty_slots > 0 else None, delta_color="inverse")
    st.markdown("---")

    for category, slots in categorized.items():
        st.subheader(f"📦 {category}")
        cols = st.columns(len(slots))
        for idx, (slot_id, info) in enumerate(slots.items()):
            with cols[idx]:
                if info["detected"]:
                    st.success(f"✅ **{slot_id}**\n{info['name']}\nConf: {info['confidence']:.1%}")
                else:
                    st.error(f"❌ **{slot_id}**\n{info['name']}\n⚠️ OUT OF STOCK")

    empty_slot_list = [f"{slot_id} ({info['name']})" for slot_id, info in inventory.items() if not info["detected"]]
    if empty_slot_list:
        st.markdown("---")
        st.warning("🚨 **RESTOCK ALERT - Empty Slots:**")
        for empty_slot in empty_slot_list:
            st.write(f"- {empty_slot}")
        st.error("⚠️⚠️⚠️ **URGENT: Items need to be restocked!** ⚠️⚠️⚠️")
    else:
        st.success("🎉 **Perfect! All slots are fully stocked!** 🎉")

def match_detections_to_slots(detections, slots, image_width, image_height):
    matched = []
    for det in detections:
        class_name = det.get("class_name", "")
        confidence = det.get("confidence", 0)
        bbox = det.get("bbox", [0, 0, 0, 0])
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        center_rel_x = center_x / image_width
        center_rel_y = center_y / image_height

        matched_slot = None
        for slot in slots:
            rel_bbox = slot["rel_bbox"]
            if (rel_bbox[0] <= center_rel_x <= rel_bbox[2] and
                rel_bbox[1] <= center_rel_y <= rel_bbox[3]):
                matched_slot = slot["id"]
                break
        if not matched_slot and class_name in CLASS_TO_SLOT:
            matched_slot = CLASS_TO_SLOT.get(class_name)

        matched.append({
            "class_name": class_name,
            "confidence": confidence,
            "slot_id": matched_slot,
            "bbox": bbox
        })
    return matched

def evaluate_model_accuracy(results):
    if results and len(results[0].boxes) > 0:
        confidences = [float(box.conf.item()) for box in results[0].boxes]
        avg_confidence = np.mean(confidences) if confidences else 0
        num_detections = len(confidences)

        st.subheader("📊 Model Accuracy Report")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Detections", num_detections)
        with col2:
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        with col3:
            high_conf = sum(1 for c in confidences if c > 0.7)
            st.metric("High Confidence (>70%)", f"{high_conf}/{num_detections}" if num_detections > 0 else "0")
        if confidences:
            st.write("**Confidence Distribution:**")
            conf_array = np.array(confidences)
            bins = [0, 0.5, 0.7, 0.9, 1.0]
            hist, _ = np.histogram(conf_array, bins=bins)
            hist_data = pd.DataFrame({"Range": ["0-50%", "50-70%", "70-90%", "90-100%"], "Count": hist})
            st.bar_chart(hist_data.set_index("Range"))
    else:
        st.info("No detections to evaluate model accuracy")

# ---------- UI สำหรับปรับตำแหน่ง Shelf Slots (แสดงใน Sidebar เฉพาะโหมด Detection) ----------
def slot_adjustment_sidebar(image_for_preview=None):
    with st.sidebar.expander("🎛️ Adjust Shelf Slots Positions", expanded=False):
        st.markdown("ปรับตำแหน่งกรอบ Shelf ให้ตรงกับสินค้าในภาพ")
        slot_ids = [s["id"] for s in st.session_state.slots]
        selected_id = st.selectbox("เลือก Slot ที่ต้องการปรับ", slot_ids, index=slot_ids.index(st.session_state.slot_edit_selected_id))
        st.session_state.slot_edit_selected_id = selected_id

        # ดึงข้อมูล current slot
        current_slot = next(s for s in st.session_state.slots if s["id"] == selected_id)
        rel_bbox = current_slot["rel_bbox"].copy()

        st.write(f"พิกัดปัจจุบันของ {selected_id}: `[{rel_bbox[0]:.3f}, {rel_bbox[1]:.3f}, {rel_bbox[2]:.3f}, {rel_bbox[3]:.3f}]`")
        col1, col2 = st.columns(2)
        with col1:
            new_x1 = st.slider("x1 (ซ้าย)", 0.0, 1.0, rel_bbox[0], 0.01, key=f"x1_{selected_id}")
            new_y1 = st.slider("y1 (บน)", 0.0, 1.0, rel_bbox[1], 0.01, key=f"y1_{selected_id}")
        with col2:
            new_x2 = st.slider("x2 (ขวา)", 0.0, 1.0, rel_bbox[2], 0.01, key=f"x2_{selected_id}")
            new_y2 = st.slider("y2 (ล่าง)", 0.0, 1.0, rel_bbox[3], 0.01, key=f"y2_{selected_id}")

        # ป้องกัน x1 >= x2 หรือ y1 >= y2
        if new_x1 >= new_x2:
            st.warning("⚠️ x1 ต้องน้อยกว่า x2")
        if new_y1 >= new_y2:
            st.warning("⚠️ y1 ต้องน้อยกว่า y2")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button(f"✅ Update {selected_id}", use_container_width=True):
                if new_x1 < new_x2 and new_y1 < new_y2:
                    for slot in st.session_state.slots:
                        if slot["id"] == selected_id:
                            slot["rel_bbox"] = [new_x1, new_y1, new_x2, new_y2]
                    st.success(f"Updated {selected_id}")
                    st.rerun()
                else:
                    st.error("พิกัดไม่ถูกต้อง (x1<x2 และ y1<y2)")
        with col_btn2:
            if st.button("💾 Save all slots to JSON", use_container_width=True):
                if save_slots_to_json(st.session_state.slots):
                    st.success("บันทึกการตั้งค่าทั้งหมดลงไฟล์ slot_config.json แล้ว")
                else:
                    st.error("บันทึกไม่สำเร็จ")

        col_reset1, col_reset2 = st.columns(2)
        with col_reset1:
            if st.button("🔄 Reset All Slots to Default", use_container_width=True):
                st.session_state.slots = copy.deepcopy(DEFAULT_SLOT_RELATIVE_BOXES)
                # optionally save default to json? ไม่ต้อง auto save แต่ถ้าต้องการให้ save ก็เพิ่มได้
                st.success("Reset all slots to default values (ยังไม่ได้บันทึกลงไฟล์)")
                st.rerun()
        with col_reset2:
            if st.button("📂 Load from JSON (overwrite current)", use_container_width=True):
                loaded = load_slots_from_json()
                if loaded:
                    st.session_state.slots = loaded
                    st.success("โหลดการตั้งค่าจากไฟล์ JSON แล้ว")
                    st.rerun()
                else:
                    st.warning("ไม่พบไฟล์หรือโหลดไม่สำเร็จ ยังคงใช้ค่าปัจจุบัน")

        # แสดงภาพตัวอย่างพร้อม Highlight Slot ที่กำลังปรับ (ถ้ามี image_preview)
        if image_for_preview is not None:
            st.markdown("**Preview (Highlighted Slot)**")
            dummy_detections = []
            preview_img = draw_shelf_slots(image_for_preview, dummy_detections, st.session_state.slots, highlight_slot_id=selected_id)
            st.image(preview_img, caption=f"Highlight: {selected_id}", use_column_width=True)

# ------------------- Main App -------------------
st.set_page_config(page_title="Stock Vision App - Adjustable Slots", layout="wide")
st.title("📦 Stock Vision App")
st.write("Upload an image to detect beverage containers and check shelf inventory status")

init_session_state()

app_mode = st.sidebar.radio("Select Mode", ["🔍 Detection & Inventory", "📊 Model Accuracy Check"])
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running inference..."):
        results = model(image)
        result = results[0]

    detections = []
    if len(result.boxes) > 0:
        class_names = result.names
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            bbox = box.xyxy[0].tolist()
            class_name = class_names.get(class_id, f"Class {class_id}")
            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox": bbox
            })

    if app_mode == "🔍 Detection & Inventory":
        # เรียกใช้ sidebar สำหรับปรับตำแหน่ง พร้อมส่งภาพเพื่อ preview
        slot_adjustment_sidebar(image_for_preview=image)

        # ใช้ slots ปัจจุบันจาก session_state
        current_slots = st.session_state.slots
        matched_detections = match_detections_to_slots(detections, current_slots, image.width, image.height)
        inventory, categorized = analyze_shelf_inventory(matched_detections, current_slots)

        # วาดภาพพร้อมกรอบ Shelf (และ Highlight ถ้ามีการเลือกใน sidebar)
        highlight_id = st.session_state.get("slot_edit_selected_id", None)
        img_with_slots = draw_shelf_slots(image, matched_detections, current_slots, highlight_slot_id=highlight_id)

        # วาด bounding boxes ของ detection เพิ่ม (จาก YOLO) ให้เห็นชัด
        plotted = result.plot()
        plotted_rgb = plotted[..., ::-1]

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🏪 Shelf Slot Status (🟢=Stocked, 🔴=Empty)")
            st.image(img_with_slots, caption="Shelf View with Adjusted Slots", use_column_width=True)
            # แสดงภาพ detection แยกอีกที
            st.subheader("🔍 YOLO Detection Result")
            st.image(plotted_rgb, caption="Raw Detection Boxes", use_column_width=True)
        with col2:
            st.subheader("📋 Detection Details")
            if matched_detections:
                for det in matched_detections:
                    status_icon = "✅" if det["slot_id"] else "⚠️"
                    slot_info = f"Slot: {det['slot_id']}" if det["slot_id"] else "No slot match"
                    st.write(f"{status_icon} **{det['class_name']}** - {det['confidence']:.2%} ({slot_info})")
            else:
                st.info("No objects detected")

        st.markdown("---")
        display_inventory_ui(inventory, categorized)

    elif app_mode == "📊 Model Accuracy Check":
        evaluate_model_accuracy(results)
        if len(result.boxes) > 0:
            st.subheader("🔍 Detected Items")
            class_names = result.names
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                label = class_names.get(class_id, f"Class {class_id}")
                st.write(f"- {label}  `{confidence:.2%}`")
            plotted = result.plot()
            plotted_rgb = plotted[..., ::-1]
            st.image(plotted_rgb, caption="Detection Result", use_column_width=True)
        else:
            st.info("No objects detected in the image.")
else:
    st.info("👈 Please upload an image to begin")

st.markdown("---")
st.caption("Stock Vision App - Adjustable Shelf Slots | Use sidebar to align slots with products")