import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the trained YOLO model
model = YOLO("best.pt")

st.title("🥤 Beverage Detection App")
st.write("Upload an image to detect beverage containers (Coke, Fanta, Pepsi, etc.)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    # ใช้ use_column_width สำหรับ Streamlit เวอร์ชันเก่า
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Running inference...")
    
    # Run detection
    results = model(image)
    result = results[0]  # single image -> single result

    if len(result.boxes) > 0:
        st.subheader("🚀 Detected Items")
        
        # Get class names from the model (should contain the 11 beverage classes)
        class_names = result.names
        
        # Collect and display each detection with confidence
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            # Handle potential out-of-range class ids just in case
            if class_id in class_names:
                label = class_names[class_id]
            else:
                label = f"Class {class_id}"
            st.write(f"- {label}  `{confidence:.2%}`")
        
        # Show the image with bounding boxes drawn
        plotted = result.plot()  # returns BGR numpy array
        plotted_rgb = plotted[..., ::-1]  # BGR -> RGB
        # ใช้ use_column_width เช่นกัน
        st.image(plotted_rgb, caption="Detection Result", use_column_width=True)
    else:
        st.info("No objects detected in the image.")