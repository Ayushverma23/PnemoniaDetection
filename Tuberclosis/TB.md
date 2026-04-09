# Tuberculosis Detection Project Overview

This document provides a detailed explanation of the Tuberculosis (TB) detection project based on the `tuberclosis.ipynb` notebook.

## 1. Project Goal
The primary objective of this project is to detect **Active Tuberculosis** in chest X-ray images using deep learning object detection techniques. The system identifies regions of interest (bounding boxes) where TB manifestations are present.

## 2. Dataset
The project uses the **TBX11K** dataset, which contains:
- **Images**: Chest X-rays in PNG format.
- **Annotations**: XML files in Pascal VOC format (providing bounding boxes for TB areas).
- **Classes**:
    - `ActiveTuberculosis` (TB images)
    - `Health` (Normal images)

The data is split into **Training** and **Validation** sets based on provided list files.

## 3. Methodology & Workflow

The project follows a standard computer vision pipeline:

### A. Data Preprocessing & Cleaning
1.  **Parsing Annotations**:
    - The code reads XML files to extract bounding box coordinates (`xmin`, `ymin`, `xmax`, `ymax`).
    - It handles image resizing by calculating scale factors between the original annotation size and the actual loaded image size to ensure bounding boxes are drawn correctly.

2.  **Data Statistics**:
    - The notebook analyzes the dataset to count:
        - Total images
        - TB cases with and without XML annotations
        - Normal (Healthy) cases
        - Missing images
    - This ensures data integrity before training.

### B. Format Conversion (XML to YOLO)
The YOLO (You Only Look Once) model requires a specific label format. The notebook performs the following conversion:
- **Input**: Pascal VOC XML (absolute coordinates).
- **Output**: YOLO TXT files (normalized center coordinates + width/height).
    - Format: `class_id x_center y_center width height`
    - Normal images get empty label files (indicating no object to detect).

A directory structure is created at `/kaggle/working/tb_yolo` with:
- `images/train`, `images/val`
- `labels/train`, `labels/val`

### C. Model Architecture
- **Model**: **YOLOv8 Small (`yolov8s`)**
- **Library**: `ultralytics`
- **Pre-training**: The model is initialized with weights pre-trained on the COCO dataset (`yolov8s.pt`) to leverage transfer learning.

### D. Training Configuration
The model is trained with the following parameters:
- **Epochs**: 50 (The model iterates over the full dataset 50 times).
- **Image Size**: 512x512 pixels.
- **Batch Size**: 16.
- **Workers**: 4 (for data loading).
- **Patience**: 10 (Early stopping if validation performance doesn't improve).
- **Device**: GPU 0.

A `data.yaml` file is generated effectively telling YOLO where to find the images and what classes to train on.

## 4. Inference & Visualization
After training, the notebook performs the following:
1.  **Loading Weights**: It loads the best-performing model weights from the training run.
2.  **Prediction**: It runs the model on random samples from the validation set (both TB and Normal cases).
3.  **Visualization**: It draws the predicted bounding boxes and confidence scores (e.g., `TB 0.85`) on the images using OpenCV and Matplotlib.
    - A confidence threshold (`CONF_TH`) of **0.25** is used to filter out weak predictions.

## 5. Directory Structure
Key directories involved:
- **Input**: `/kaggle/input/tbx-11/TBX11K` (Source Data)
- **Output/Working**: `/kaggle/working/tb_yolo` (Converted Data for YOLO)
- **Runs**: `/kaggle/working/tb_yolo_runs` (Training logs and saved weights)

## 6. Summary of Code Components
| Component | Description |
|-----------|-------------|
| `convert_xml_to_yolo` | Transformation function for labels. |
| `analyze_ids` | Statistical check of the dataset. |
| `model.train()` | Initiates the training loop. |
| `visualize_prediction` | Runs inference and plots results. |
