import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights
import pydicom
import numpy as np
from PIL import Image
import io
import base64
from services.base import DiseaseDetector
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

class PneumoniaDetector(DiseaseDetector):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_model()
        self.model.eval()
        
        # Initialize GradCAM
        try:
            target_layers = [self.model.features.norm5]
            self.cam = GradCAM(model=self.model, target_layers=target_layers)
        except Exception as e:
            print(f"Warning: Could not initialize GradCAM: {e}")
            self.cam = None

    def load_model(self):
        weights = DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=weights)
        
        # Modify classifier for binary classification
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 1)
        
        # Load weights
        try:
            model.load_state_dict(torch.load("models/best_model.pth", map_location=self.device))
        except FileNotFoundError:
            print("Warning: Model file not found. Using random weights for testing.")
        
        self.model = model.to(self.device)

    def preprocess(self, file_content: bytes, filename: str):
        # 1. Processing for Model (Validation Transform from Notebook)
        # Using PIL ensures we match the training pipeline's resizing and ToTensor behavior exactly
        if filename.lower().endswith(".dcm"):
            dicom = pydicom.dcmread(io.BytesIO(file_content))
            image_array = dicom.pixel_array.astype("float32")
            if image_array.max() > 0:
                image_array = image_array / image_array.max()
            
            # Convert to PIL (L mode for grayscale)
            # 0-1 float array -> 0-255 uint8 PIL Image
            image_pil = Image.fromarray((image_array * 255).astype("uint8")).convert("RGB")
        else:
            image_pil = Image.open(io.BytesIO(file_content)).convert("RGB")
            
        # Model Transforms
        model_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), # Converts PIL [0, 255] -> Tensor [0.0, 1.0]
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) # Normalizes to [-1, 1]
        ])
        
        input_tensor = model_transform(image_pil).unsqueeze(0).to(self.device)

        # 2. Processing for Grad-CAM Visualization
        # Needs float32 numpy array (H, W, 3) in range [0, 1]
        # We resize the PIL image to matching dimensions
        viz_pil = image_pil.resize((224, 224))
        image_rgb = np.array(viz_pil).astype("float32") / 255.0
        
        return input_tensor, image_rgb

    def predict(self, file_content: bytes, filename: str):
        input_tensor, original_image_rgb = self.preprocess(file_content, filename)
        
        # 1. Prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probability = torch.sigmoid(output).item()
            prediction = "Pneumonia" if probability > 0.30 else "Normal"
            
        # 2. Grad-CAM Visualization
        heatmap_base64 = None
        if self.cam:
            try:
                # Generate grayscale cam
                grayscale_cam = self.cam(input_tensor=input_tensor)[0]
                
                # Overlay heatmap on original image
                visualization = show_cam_on_image(original_image_rgb, grayscale_cam, use_rgb=True)
                
                # Convert to PIL Image for converting to bytes
                viz_pil = Image.fromarray(visualization)
                
                # Convert to Base64
                buffered = io.BytesIO()
                viz_pil.save(buffered, format="PNG")
                heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"Error generating Grad-CAM: {e}")

        return {
            "prediction": prediction,
            "probability": probability,
            "heatmap_base64": heatmap_base64
        }
