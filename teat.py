import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import copy
import json
import os
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# ------------------- โหลดโมเดล YOLO -------------------
model = YOLO("best.pt")

# ------------------- กำหนด Shelf Slot เริ่มต้น -------------------
DEFAULT_SLOT_RELATIVE_BOXES = [
    {"id": "S01", "name": "Coke Can", "rel_bbox": [0.02, 0.02, 0.23, 0.28]},
    {"id": "S02", "name": "Coke Light Can", "rel_bbox": [0.25, 0.02, 0.46, 0.28]},
    {"id": "S03", "name": "Fanta Grape", "rel_bbox": [0.48, 0.02, 0.69, 0.28]},
    {"id": "S04", "name": "Fanta Orange Can", "rel_bbox": [0.71, 0.02, 0.92, 0.28]},
    {"id": "S05", "name": "m150", "rel_bbox": [0.02, 0.31, 0.23, 0.57]},
    {"id": "S06", "name": "Meiji Milk", "rel_bbox": [0.25, 0.31, 0.46, 0.57]},
    {"id": "S07", "name": "Oishi Rice", "rel_bbox": [0.48, 0.31, 0.69, 0.57]},
    {"id": "S08", "name": "Oishi Honey Lemon", "rel_bbox": [0.71, 0.31, 0.92, 0.57]},
    {"id": "S09", "name": "Oishi Kyoho", "rel_bbox": [0.02, 0.60, 0.30, 0.86]},
    {"id": "S10", "name": "Pepsi Can", "rel_bbox": [0.35, 0.60, 0.63, 0.86]},
    {"id": "S11", "name": "Sprite Can", "rel_bbox": [0.68, 0.60, 0.96, 0.86]},
]

CLASS_TO_SLOT = {
    "Coke": "S01",
    "Coke Light": "S02",
    "Fanta Grape": "S03",
    "Fanta Orange": "S04",
    "M150": "S05",
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

SLOT_CONFIG_FILE = "slot_config.json"

def load_slots_from_json():
    if os.path.exists(SLOT_CONFIG_FILE):
        try:
            with open(SLOT_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and all("id" in s and "rel_bbox" in s for s in data):
                    return data
        except Exception as e:
            st.warning(f"ไม่สามารถโหลด JSON: {e}")
    return copy.deepcopy(DEFAULT_SLOT_RELATIVE_BOXES)

def save_slots_to_json(slots):
    try:
        with open(SLOT_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(slots, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"บันทึก JSON ไม่สำเร็จ: {e}")
        return False

def init_session_state():
    if "slots" not in st.session_state:
        st.session_state.slots = load_slots_from_json()
    if "slot_edit_selected_id" not in st.session_state:
        st.session_state.slot_edit_selected_id = "S01"
    if "live_inventory" not in st.session_state:
        st.session_state.live_inventory = {}   # จะอัปเดตใน VideoTransformer

init_session_state()

# ------------------- ฟังก์ชันสำหรับวาดกรอบ (รองรับทั้ง PIL และ numpy) -------------------
def draw_shelf_slots(image, detections, slots, highlight_slot_id=None):
    """
    image: PIL Image หรือ numpy array (BGR หรือ RGB)
    detections: list ของ dict (มี slot_id)
    slots: list ของ dict
    highlight_slot_id: slot id ที่จะไฮไลต์
    """
    # แปลง numpy เป็น PIL ถ้าจำเป็น
    if isinstance(image, np.ndarray):
        # ถ้าเป็น BGR (จาก OpenCV) ให้เปลี่ยนเป็น RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

        if highlight_slot_id and slot_id == highlight_slot_id:
            fill_color = (0, 255, 255, 80)
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

def match_detections_to_slots(detections, slots, image_width, image_height):
    """เหมือนเดิม แต่รับ width/height แยก"""
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

# ------------------- UI สำหรับปรับตำแหน่ง Shelf Slots (Sidebar) -------------------
def slot_adjustment_sidebar(image_for_preview=None):
    with st.sidebar.expander("🎛️ Adjust Shelf Slots Positions", expanded=False):
        st.markdown("ปรับตำแหน่งกรอบ Shelf ให้ตรงกับสินค้าในภาพ")
        slot_ids = [s["id"] for s in st.session_state.slots]
        selected_id = st.selectbox("เลือก Slot ที่ต้องการปรับ", slot_ids, index=slot_ids.index(st.session_state.slot_edit_selected_id))
        st.session_state.slot_edit_selected_id = selected_id

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

        if new_x1 >= new_x2 or new_y1 >= new_y2:
            st.warning("⚠️ x1 ต้องน้อยกว่า x2 และ y1 ต้องน้อยกว่า y2")

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
                    st.error("พิกัดไม่ถูกต้อง")
        with col_btn2:
            if st.button("💾 Save all slots to JSON", use_container_width=True):
                if save_slots_to_json(st.session_state.slots):
                    st.success("บันทึกการตั้งค่าลงไฟล์เรียบร้อย")
                else:
                    st.error("บันทึกไม่สำเร็จ")

        col_reset1, col_reset2 = st.columns(2)
        with col_reset1:
            if st.button("🔄 Reset All Slots to Default", use_container_width=True):
                st.session_state.slots = copy.deepcopy(DEFAULT_SLOT_RELATIVE_BOXES)
                st.success("รีเซ็ตเรียบร้อย (ยังไม่ได้บันทึก)")
                st.rerun()
        with col_reset2:
            if st.button("📂 Load from JSON", use_container_width=True):
                loaded = load_slots_from_json()
                if loaded:
                    st.session_state.slots = loaded
                    st.success("โหลดจาก JSON แล้ว")
                    st.rerun()

        if image_for_preview is not None:
            st.markdown("**Preview (Highlighted Slot)**")
            dummy_detections = []
            preview_img = draw_shelf_slots(image_for_preview, dummy_detections, st.session_state.slots, highlight_slot_id=selected_id)
            st.image(preview_img, caption=f"Highlight: {selected_id}", use_column_width=True)

# ------------------- VideoTransformer สำหรับ Webcam (Real-time) -------------------
class InventoryVideoTransformer(VideoTransformerBase):
    def __init__(self, model, slots_getter):
        self.model = model
        self.slots_getter = slots_getter  # function ที่คืนค่า slots ปัจจุบัน
        self.frame_count = 0
        self.inference_every_n = 2   # ประมวลผล YOLO ทุก 2 เฟรม (ลดภาระ CPU)

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # แปลง frame เป็น numpy array (RGB)
        img = frame.to_ndarray(format="rgb24")
        h, w, _ = img.shape

        # เรียกใช้ slots ปัจจุบันจาก session state
        current_slots = self.slots_getter()

        # ทำ inference เฉพาะบางเฟรม
        if self.frame_count % self.inference_every_n == 0:
            results = self.model(img)  # YOLO inference
            raw_detections = []
            if len(results[0].boxes) > 0:
                class_names = results[0].names
                for box in results[0].boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    bbox = box.xyxy[0].tolist()
                    class_name = class_names.get(class_id, "unknown")
                    raw_detections.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": bbox
                    })
            # จับคู่กับ slots
            matched = match_detections_to_slots(raw_detections, current_slots, w, h)
            # อัปเดต inventory ใน session state
            inv, _ = analyze_shelf_inventory(matched, current_slots)
            st.session_state.live_inventory = inv
            # เก็บ detections ไว้ใช้ในการวาด
            self.last_detections = matched
        else:
            matched = getattr(self, "last_detections", [])

        # วาดกรอบ shelf + detection
        annotated_pil = draw_shelf_slots(img, matched, current_slots, highlight_slot_id=None)
        # วาด bounding boxes ของ YOLO เพิ่ม (optional)
        for det in matched:
            bbox = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            draw = ImageDraw.Draw(annotated_pil, "RGBA")
            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=2)
            draw.text((x1, y1-10), f"{det['class_name']} {det['confidence']:.2f}", fill="#00FF00")
        annotated_np = np.array(annotated_pil)
        annotated_np = cv2.cvtColor(annotated_np, cv2.COLOR_RGB2BGR)

        self.frame_count += 1
        return av.VideoFrame.from_ndarray(annotated_np, format="bgr24")

# ------------------- Main App -------------------
st.set_page_config(page_title="Stock Vision App - with Live Webcam", layout="wide")
st.title("📦 Stock Vision App")
st.write("AI ตรวจจับสินค้าบนชั้นวาง รองรับทั้งอัปโหลดรูปภาพและ Real-time ผ่าน Webcam")

app_mode = st.sidebar.radio(
    "Select Mode",
    ["🔍 Detection & Inventory", "📊 Model Accuracy Check", "🎥 Real-time Webcam Check"]
)

# ถ้าเป็น Real-time Webcam Check ให้แสดงกล้องโดยไม่ต้องรออัปโหลดรูป
if app_mode == "🎥 Real-time Webcam Check":
    # แสดง Sidebar สำหรับปรับ slots (ใช้ภาพตัวอย่าง static ตัวแรกที่ capture ได้? ไม่มีภาพตัวอย่าง งด preview)
    slot_adjustment_sidebar(image_for_preview=None)  # อาจไม่มี preview

    st.subheader("🎥 Live Webcam Feed (Real-time Detection with Shelf Slots)")
    st.markdown("กล้องจะแสดงภาพพร้อมกรอบ Shelf (แดง=ว่าง, เขียว=มีสินค้า) และ bounding boxes ของสินค้า")

    # ฟังก์ชันดึง slots ล่าสุดจาก session state (ใช้ใน transformer)
    def get_current_slots():
        return st.session_state.slots

    webrtc_streamer(
        key="inventory-cam",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: InventoryVideoTransformer(model, get_current_slots),
        async_processing=True,
        media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}}
    )

    st.markdown("---")
    st.subheader("📋 Inventory Status (Live)")
    # แสดง inventory ล่าสุดที่อัปเดตจาก VideoTransformer ทุกเฟรม
    if st.session_state.live_inventory:
        # สร้าง categorized จาก live_inventory
        live_categorized = {}
        for category, slot_ids in CATEGORY_GROUPS.items():
            live_categorized[category] = {}
            for sid in slot_ids:
                if sid in st.session_state.live_inventory:
                    live_categorized[category][sid] = st.session_state.live_inventory[sid]
        display_inventory_ui(st.session_state.live_inventory, live_categorized)
    else:
        st.info("รอข้อมูลจากกล้อง... (กรุณากด Start บน video stream)")

elif app_mode in ["🔍 Detection & Inventory", "📊 Model Accuracy Check"]:
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
            slot_adjustment_sidebar(image_for_preview=image)
            current_slots = st.session_state.slots
            matched_detections = match_detections_to_slots(detections, current_slots, image.width, image.height)
            inventory, categorized = analyze_shelf_inventory(matched_detections, current_slots)
            highlight_id = st.session_state.get("slot_edit_selected_id", None)
            img_with_slots = draw_shelf_slots(image, matched_detections, current_slots, highlight_slot_id=highlight_id)

            plotted = result.plot()
            plotted_rgb = plotted[..., ::-1]

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("🏪 Shelf Slot Status")
                st.image(img_with_slots, caption="Shelf View with Adjusted Slots", use_column_width=True)
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
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    label = result.names.get(class_id, f"Class {class_id}")
                    st.write(f"- {label}  `{confidence:.2%}`")
                plotted = result.plot()
                plotted_rgb = plotted[..., ::-1]
                st.image(plotted_rgb, caption="Detection Result", use_column_width=True)
            else:
                st.info("No objects detected in the image.")
    else:
        st.info("👈 Please upload an image to begin")

st.markdown("---")
st.caption("Stock Vision App - Real-time + Image Mode | Adjust slots in sidebar")