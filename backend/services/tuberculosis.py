import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64
from services.base import DiseaseDetector
import os

class TuberculosisDetector(DiseaseDetector):
    def __init__(self):
        self.model_path = r"d:/Medical/Tuberclosis/model/best.pt"
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                print(f"Error loading TB model: {e}")
        else:
            print(f"Warning: TB Model file not found at {self.model_path}")

    def preprocess(self, file_content: bytes, filename: str):
        # YOLOv8 handles preprocessing internally, but we need to convert bytes to a format it accepts (e.g., PIL Image or numpy array)
        image = Image.open(io.BytesIO(file_content)).convert("RGB")
        return image

    def predict(self, file_content: bytes, filename: str):
        if not self.model:
            return {"error": "Model not loaded"}

        image = self.preprocess(file_content, filename)
        
        # Run inference
        results = self.model(image)
        
        # Process results
        result = results[0]
        
        # Generate annotated image
        # plot() returns a numpy array (BGR)
        annotated_frame = result.plot() 
        
        # Convert BGR to RGB
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        img_pil = Image.fromarray(annotated_frame_rgb)
        
        # Convert to Base64
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Extract detection details
        detections = []
        for box in result.boxes:
            detections.append({
                "class": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]
            })

        # Determine overall prediction
        prediction = "Tuberculosis Detected" if len(detections) > 0 else "Normal"
        max_confidence = max([d['confidence'] for d in detections]) if detections else 0.0

        return {
            "prediction": prediction,
            "confidence": max_confidence,
            "detections": detections,
            "annotated_image_base64": img_base64
        }
