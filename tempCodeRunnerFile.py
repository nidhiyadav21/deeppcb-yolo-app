from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import os
from collections import Counter

st.set_page_config(page_title="PCB Defect Detection", layout="centered")
st.title("🔍 PCB Defect Detection App")

# File uploader
uploaded_file = st.file_uploader("Upload a PCB Image", type=["jpg", "jpeg", "png"])

# 👇 Path to your trained YOLO model
model_path = "C:/Users/ammyy/runs/detect/deeppcb_yolo_model4/weights/best.pt"
model = YOLO(model_path)

# Image width slider
image_width = st.slider("Output Image Width", 300, 1000, 600)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=False, width=image_width)

    # Save temporary image
    temp_path = "temp_uploaded.jpg"
    image.save(temp_path)

    # Prediction
    results = model(temp_path)
    result = results[0]

    # Get predictions
    boxes = result.boxes
    names = model.names

    # Count classes
    if boxes is not None and boxes.cls is not None:
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        class_counts = dict(Counter(cls_ids))
        total = sum(class_counts.values())

        st.subheader("🔢 Defect Summary")

        for cls_id, count in class_counts.items():
            label = names[cls_id]
            percentage = (count / total) * 100
            st.write(f"✅ {label}: {count} ({percentage:.2f}%)")
    else:
        st.warning("No defects detected.")

    # Show detection image
    res_plot = result.plot()
    st.image(res_plot, caption="Detection Result", width=image_width)

    os.remove(temp_path)
