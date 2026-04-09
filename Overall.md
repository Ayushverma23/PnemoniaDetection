is a comprehensive medical diagnostic application designed to assist healthcare professionals in identifying **Pneumonia** and **Active Tuberculosis** from chest X-ray images.

## 1. Project Overview

The system uses a modern **Microservices**-style architecture where a high-performance **FastAPI** backend powers the AI inference, served to a responsive **Next.js** frontend.

### Key Capabilities
- **Pneumonia Detection**: Binary classification (Normal vs. Pneumonia) with **Explainable AI (Grad-CAM)** visualizations to show *where* the model is looking.
- **Tuberculosis Detection**: Object detection using **YOLOv8** to pinpoint specific regions of infection with bounding boxes. - **DICOM Support**: Native handling of medical DICOM files for Pneumonia analysis.

---

## 2. Technical Architecture

### A. Frontend (`/frontend`)
The user interface is built with **Next.js 15 (App Router)** and **React 19**, focusing on a clean, professional "medical-grade" experience.

- **Stack**: Next.js, TypeScript, Lucide React (Icons).
- **Structure**:
    - **App Router**: Uses modern Next.js 15 routing (`src/app`).
        - `/`: Dashboard for **Pneumonia Analysis**.
        - `/tuberculosis`: Dashboard for **Tuberculosis Analysis**.
    - **Components**: Reusable UI elements (`Sidebar`, `FileUpload`).
- **Features**:
    - Real-time file upload and preview.
    - Dynamic visualization of AI results (Probability bars, Heatmaps).
    - Responsive Sidebar navigation.

### B. Backend (`/backend`)
The core logic resides in a **FastAPI** server that exposing REST endpoints.

- **Stack**: FastAPI, Uvicorn, PyTorch, Ultralytics, NumPy, Pillow, PyDicom.
- **Structure**:
    - `main.py`: Entry point, CORS configuration, and Router inclusion.
    - `api/`: Route definitions (`pneumonia_routes.py`, `tuberculosis_routes.py`).
    - `services/`: Encapsulated business logic and model handling.
        - `pneumonia.py`: Manages DenseNet121 and Grad-CAM.
        - `tuberculosis.py`: Manages YOLOv8 inference.

---

## 3. AI & Machine Learning Models

### Pneumonia Detection Module
- **Model Architecture**: **DenseNet121** (Pre-trained on ImageNet).
- **Modifications**: The final fully connected layer is modified for binary classification (1 output node).
- **Explainability**: Integrated **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize the activation maps of the final convolutional layers, helping verify model focus.
- **Preprocessing**:
    - Converts DICOM/Images to RGB.
    - Normalizes input using ImageNet mean/std conventions.
    - Resizes to 224x224.

### Tuberculosis Detection Module
- **Model Architecture**: **YOLOv8 Small (`yolov8s`)**.
- **Task**: Object Detection (Bounding Box Regression).
- **Training**: Trained on the **TBX11K** dataset (as analyzed in `tuberclosis.ipynb`).
- **Output**: Returns the image with drawn bounding boxes around detected TB manifestations and a list of confidence scores.

---

## 4. Workflows

### flow: Pneumonia Analysis
1.  **Upload**: User uploads a file (JPG/PNG/DICOM) via the frontend.
2.  **API Call**: Frontend sends `POST` request to `/api/v1/predict/pneumonia`.
3.  **Inference**:
    - `PneumoniaDetector` preprocesses the image.
    - Model predicts probability score (0.0 - 1.0).
    - Grad-CAM generates a heatmap overlay.
4.  **Response**: Backend returns JSON with prediction label, probability, and base64-encoded heatmap.
5.  **Display**: Frontend renders the probability bar and the heatmap image.

### flow: Tuberculosis Analysis
1.  **Upload**: User uploads a chest X-ray image.
2.  **API Call**: Frontend sends `POST` request to `/api/v1/predict/tuberculosis`.
3.  **Inference**:
    - `TuberculosisDetector` loads the custom YOLO model (`best.pt`).
    - Model detects objects (active TB lesions).
    - Annotator draws boxes on the original image.
4.  **Response**: Backend returns prediction status, max confidence, detection details, and the annotated image.
5.  **Display**: Frontend shows the processed image with bounding boxes.

---

## 5. Development & Setup

### Prerequisites
- Node.js & npm
- Python 3.10+
- CUDA-compatible GPU (Recommended for faster inference)

### Running Locally
**1. Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
*Runs on `http://localhost:8000`*

**2. Frontend:**
```bash
cd frontend
npm install
npm run dev
```
*Runs on `http://localhost:3000`*
