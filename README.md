# 💡 DeepPCB Defect Detection using YOLOv8

This project presents a **real-time, AI-driven solution** to detect and classify **defects in Printed Circuit Boards (PCBs)** using **YOLOv8**, a state-of-the-art object detection model. It includes **end-to-end implementation**—from dataset preprocessing to model training, evaluation, and deployment through a **Streamlit-based web app**.

---

## 🧠 Problem Statement

Manual inspection of PCBs in electronics manufacturing is often **time-consuming, inconsistent**, and **prone to human error**. This project solves that problem by automating the defect detection process using deep learning, aiming to **improve quality control, accuracy**, and **reduce inspection time**.

---

## 🎯 Project Objectives

- Train a YOLOv8 object detection model to identify various types of PCB defects.
- Convert the DeepPCB dataset into YOLO-compatible format with images and label files.
- Deploy the trained model using a user-friendly Streamlit interface.
- Provide real-time visual feedback, defect confidence scores, and defect summaries.

---

## 📁 Dataset: DeepPCB

The dataset includes aligned images of PCB pairs (defective vs. defect-free) along with bounding box annotations. The defects include:

- 🔴 `open_circuit`
- 🔴 `short_circuit`
- 🔴 `missing_hole`
- 🔴 `component_shift`
- 🔴 `misalignment`
- 🔴 `broken_trace`

> Only major defect classes (up to 6) were used for training, ensuring high model precision on relevant fault types.

---

## 🛠️ How It Works

1. **Data Preprocessing**:  
   Converts raw `.jpg` images and `.txt` annotations to YOLOv8 format (`train/images`, `train/labels`, etc.).

2. **Model Training**:  
   Uses YOLOv8n model for faster training (on CPU). Trained using custom `pcb.yaml` config.

3. **Web App Deployment**:  
   Built using Streamlit to let users upload any PCB image and detect defects interactively.

---

## 🧪 Training Details

- 🔸 Model: YOLOv8n (Lightweight, Fast)
- 🔸 Epochs: 10–50
- 🔸 Image Size: 640x640
- 🔸 Batch Size: 16
- 🔸 Output: `best.pt` in `runs/detect/<model_name>/weights/`

```python
from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
model.train(
    data='pcb.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='deeppcb_yolo_model'
)
