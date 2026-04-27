import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime

# Load the trained YOLO model
model = YOLO("best.pt")

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

# Categorize into 3-3-3-2 groups
CATEGORY_GROUPS = {
    "Group 1 (Can Sodas 1)": ["S01", "S02", "S03","S04"],
    "Group 2 (Can Sodas 2)": ["S05", "S06", "S07","S08"],
    "Group 3 (Juice & Milk & Oishi Teas)": ["S09", "S10", "S11"],
}

def draw_shelf_slots(image, detections, slots):
    """Draw shelf slots on image with colors based on detection status"""
    from PIL import ImageDraw, ImageFont
    import numpy as np
    
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Make a copy to draw on
    img_with_slots = image.copy()
    draw = ImageDraw.Draw(img_with_slots)
    width, height = image.size
    
    # Track which slots have products detected
    detected_slots = set()
    for det in detections:
        slot_id = det.get("slot_id")
        if slot_id:
            detected_slots.add(slot_id)
    
    # Draw each slot
    for slot in slots:
        slot_id = slot["id"]
        rel_bbox = slot["rel_bbox"]
        
        # Convert relative coordinates to absolute pixels
        x1 = int(rel_bbox[0] * width)
        y1 = int(rel_bbox[1] * height)
        x2 = int(rel_bbox[2] * width)
        y2 = int(rel_bbox[3] * height)
        
        # Choose color: green if product detected, red border if empty
        if slot_id in detected_slots:
            # Green fill with white border
            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=3, fill=(0, 255, 0, 50))
        else:
            # Red border only for empty slots
            draw.rectangle([x1, y1, x2, y2], outline="#FF0000", width=3)
        
        # Add slot label
        draw.text((x1 + 5, y1 + 5), f"{slot_id}", fill="#FFFFFF")
    
    return img_with_slots

def analyze_shelf_inventory(detections, slots):
    """Analyze which slots have products and which are empty"""
    # Initialize inventory status
    inventory = {slot["id"]: {"name": slot["name"], "detected": False, "confidence": 0.0} 
                 for slot in slots}
    
    # Match detections to slots
    for det in detections:
        slot_id = det.get("slot_id")
        if slot_id and slot_id in inventory:
            inventory[slot_id]["detected"] = True
            inventory[slot_id]["confidence"] = det.get("confidence", 0.0)
    
    # Separate into categories
    categorized = {}
    for category, slot_ids in CATEGORY_GROUPS.items():
        categorized[category] = {}
        for slot_id in slot_ids:
            if slot_id in inventory:
                categorized[category][slot_id] = inventory[slot_id]
    
    return inventory, categorized

def display_inventory_ui(inventory, categorized):
    """Display inventory status with colored UI"""
    
    # Overall summary
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
    
    # Display categorized inventory
    for category, slots in categorized.items():
        st.subheader(f"📦 {category}")
        
        cols = st.columns(len(slots))
        for idx, (slot_id, info) in enumerate(slots.items()):
            with cols[idx]:
                if info["detected"]:
                    st.success(f"✅ **{slot_id}**\n{info['name']}\nConf: {info['confidence']:.1%}")
                else:
                    st.error(f"❌ **{slot_id}**\n{info['name']}\n⚠️ OUT OF STOCK")
    
    # Alert for empty slots
    empty_slot_list = [f"{slot_id} ({info['name']})" 
                       for slot_id, info in inventory.items() 
                       if not info["detected"]]
    
    if empty_slot_list:
        st.markdown("---")
        st.warning("🚨 **RESTOCK ALERT - Empty Slots:**")
        for empty_slot in empty_slot_list:
            st.write(f"- {empty_slot}")
        
        # Create a colorful alert box
        st.error("⚠️⚠️⚠️ **URGENT: Items need to be restocked!** ⚠️⚠️⚠️")
    else:
        st.success("🎉 **Perfect! All slots are fully stocked!** 🎉")

def match_detections_to_slots(detections, slots, image_width, image_height):
    """Match detected products to slots based on centroid position"""
    matched = []
    
    for det in detections:
        class_name = det.get("class_name", "")
        confidence = det.get("confidence", 0)
        bbox = det.get("bbox", [0, 0, 0, 0])
        
        # Calculate centroid of detection
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        center_rel_x = center_x / image_width
        center_rel_y = center_y / image_height
        
        # Find which slot contains this centroid
        matched_slot = None
        for slot in slots:
            rel_bbox = slot["rel_bbox"]
            if (rel_bbox[0] <= center_rel_x <= rel_bbox[2] and 
                rel_bbox[1] <= center_rel_y <= rel_bbox[3]):
                matched_slot = slot["id"]
                break
        
        # Also match by class name if available
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
    """Function to check model accuracy metrics"""
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
        
        # Confidence distribution
        if confidences:
            st.write("**Confidence Distribution:**")
            conf_array = np.array(confidences)
            bins = [0, 0.5, 0.7, 0.9, 1.0]
            hist, _ = np.histogram(conf_array, bins=bins)
            hist_data = pd.DataFrame({
                "Range": ["0-50%", "50-70%", "70-90%", "90-100%"],
                "Count": hist
            })
            st.bar_chart(hist_data.set_index("Range"))
    else:
        st.info("No detections to evaluate model accuracy")

# Main App
st.set_page_config(page_title="Stock Vision App", layout="wide")

st.title("📦 Stock Vision App")
st.write("Upload an image to detect beverage containers and check shelf inventory status")

# Sidebar for app mode
app_mode = st.sidebar.radio(
    "Select Mode",
    ["🔍 Detection & Inventory", "📊 Model Accuracy Check"]
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Running inference...")
    
    # Run detection
    results = model(image)
    result = results[0]
    
    # Extract detections
    detections = []
    if len(result.boxes) > 0:
        class_names = result.names
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            class_name = class_names.get(class_id, f"Class {class_id}")
            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox": bbox
            })
    
    if app_mode == "🔍 Detection & Inventory":
        # Match detections to slots
        matched_detections = match_detections_to_slots(
            detections, 
            DEFAULT_SLOT_RELATIVE_BOXES,
            image.width, 
            image.height
        )
        
        # Analyze inventory
        inventory, categorized = analyze_shelf_inventory(matched_detections, DEFAULT_SLOT_RELATIVE_BOXES)
        
        # Draw shelf slots on image
        img_with_slots = draw_shelf_slots(image, matched_detections, DEFAULT_SLOT_RELATIVE_BOXES)
        
        # Display results in two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🏪 Shelf Slot Status")
            st.image(img_with_slots, caption="Shelf View (🟢=Stocked, 🔴=Empty)", use_column_width=True)
        
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
        
        # Display inventory UI with alerts
        display_inventory_ui(inventory, categorized)
        
    elif app_mode == "📊 Model Accuracy Check":
        evaluate_model_accuracy(results)
        
        # Also show detection results
        if len(result.boxes) > 0:
            st.subheader("🔍 Detected Items")
            class_names = result.names
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                label = class_names.get(class_id, f"Class {class_id}")
                st.write(f"- {label}  `{confidence:.2%}`")
            
            # Show image with bounding boxes
            plotted = result.plot()
            plotted_rgb = plotted[..., ::-1]
            st.image(plotted_rgb, caption="Detection Result", use_column_width=True)
        else:
            st.info("No objects detected in the image.")
else:
    st.info("👈 Please upload an image to begin")

# Footer
st.markdown("---")
st.caption("Stock Vision App - AI-Powered Inventory Management System")