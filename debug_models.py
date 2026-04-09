import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import models
import os

print("Starting debug...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 1. Test YOLO
print("\n--- Testing YOLO ---")
try:
    print("Loading YOLO model...")
    yolo = YOLO(r"d:\Medical\Tuberclosis\model\best.pt")
    print("YOLO loaded.")
    
    # Test on one image
    img_path = r"d:\Medical\Tuberclosis\testingdata\sample_dataset\normal\07f59492-1b36-4ff7-98b0-df996bf4fc4b.png"
    if os.path.exists(img_path):
        print(f"Testing YOLO inference on {img_path}")
        results = yolo(img_path, verbose=True)
        print("YOLO inference done.")
    else:
        print("Image for YOLO test not found.")
except Exception as e:
    print(f"YOLO failed: {e}")

# 2. Test DenseNet
print("\n--- Testing DenseNet ---")
try:
    print("Initializing DenseNet architecture...")
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, 1)
    
    path = r"d:\Medical\pneumonia_detection\models\best_model.pth"
    print(f"Loading weights from {path}...")
    
    try:
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print("Loaded via state_dict.")
        elif isinstance(ckpt, dict):
             # Maybe it IS the state dict?
             try:
                 model.load_state_dict(ckpt)
                 print("Loaded dictionary as state_dict.")
             except:
                 print("Dictionary load failed, trying as full model...")
                 model = ckpt
        else:
             print("Loaded as full model object.")
             model = ckpt
             
    except Exception as e:
        print(f"Primary load failed: {e}")
        try:
            model = torch.load(path, map_location=device)
            print("Fallback full model load success.")
        except Exception as e2:
            print(f"Fallback load failed: {e2}")
            raise e2

    model.to(device)
    model.eval()
    print("DenseNet ready.")
    
    # Dummy inference
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    print("Running dummy inference...")
    with torch.no_grad():
        out = model(input_tensor)
    print("DenseNet inference done.")
    
except Exception as e:
    print(f"DenseNet failed: {e}")

print("\nDebug complete.")
