import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score
from torchvision import transforms, models
import torch.nn as nn
try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not found. Please install it using: pip install ultralytics")
    exit(1)

from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Constants ---
TB_MODEL_PATH = r"d:\Medical\Tuberclosis\model\best.pt"
PNEUMONIA_MODEL_PATH = r"d:\Medical\pneumonia_detection\models\best_model.pth"

TB_TEST_DIR = r"d:\Medical\Tuberclosis\testingdata\sample_dataset"
PNEUMONIA_TEST_DIR = r"d:\Medical\pneumonia_detection\TestingData"

OUTPUT_IMAGE_PATH = r"d:\Medical\combined_evaluation.png"

# --- Model Definitions ---

# --- Evaluation Functions ---

def evaluate_tb_model(model_path, test_dir, max_samples=50):
    print(f"Evaluating Tuberculosis Model (YOLOv8) - Max samples: {max_samples}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return np.array([]), np.array([])
    
    y_true = []
    y_scores = []
    
    classes = {'normal': 0, 'tb': 1}
    
    for cls_name, cls_idx in classes.items():
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.exists(cls_dir):
            print(f"Warning: Directory not found: {cls_dir}")
            continue
            
        print(f"Processing class: {cls_name}")
        count = 0
        for img_name in os.listdir(cls_dir):
            if count >= max_samples:
                break
            
            img_path = os.path.join(cls_dir, img_name)
            try:
                # Run inference
                results = model(img_path, verbose=False)
                
                # Get max confidence for class 0 (assuming 'ActiveTuberculosis' is the only class in YOLO)
                max_conf = 0.0
                if len(results) > 0 and len(results[0].boxes) > 0:
                    confs = results[0].boxes.conf.cpu().numpy()
                    max_conf = float(np.max(confs))
                
                y_true.append(cls_idx)
                y_scores.append(max_conf)
                count += 1
                
                if count % 10 == 0:
                     print(f"  Processed {count} images...")
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    return np.array(y_true), np.array(y_scores)

def evaluate_pneumonia_model(model_path, test_dir, max_samples=50):
    print(f"Evaluating Pneumonia Model (DenseNet121) - Max samples: {max_samples}")
    
    try:
        # Matches debug_models.py approach
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(1024, 1)
        model = model.to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
             model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
             model.load_state_dict(checkpoint) # Try direct dict load
        else:
             model = checkpoint # Full model load fallback
             
        model.eval()
    except Exception as e:
        print(f"Error loading Pneumonia model: {e}")
        return np.array([]), np.array([])
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    y_true = []
    y_scores = []
    
    classes = {'normal': 0, 'pneumonia': 1}
    
    for cls_name, cls_idx in classes.items():
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.exists(cls_dir):
             print(f"Warning: Directory not found: {cls_dir}")
             continue
             
        print(f"Processing class: {cls_name}")
        count = 0
        for img_name in os.listdir(cls_dir):
            if count >= max_samples:
                break
                
            img_path = os.path.join(cls_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.sigmoid(output).item()
                
                y_true.append(cls_idx)
                y_scores.append(prob)
                count += 1
                
                if count % 10 == 0:
                     print(f"  Processed {count} images...")

            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                
    return np.array(y_true), np.array(y_scores)

# --- Plotting ---

def plot_evaluation(tb_data, pneu_data, save_path):
    tb_true, tb_scores = tb_data
    pneu_true, pneu_scores = pneu_data
    
    if len(tb_true) == 0 or len(pneu_true) == 0:
        print("Insufficient data for plotting.")
        return

    # Set style
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(16, 8))
    
    # 1. ROC Curve
    ax1 = plt.subplot(1, 2, 1)
    
    # TB ROC
    fpr_tb, tpr_tb, _ = roc_curve(tb_true, tb_scores)
    roc_auc_tb = auc(fpr_tb, tpr_tb)
    ax1.plot(fpr_tb, tpr_tb, color='darkorange', lw=2, label=f'Tuberculosis (AUC = {roc_auc_tb:.2f})')
    
    # Pneumonia ROC
    fpr_pneu, tpr_pneu, _ = roc_curve(pneu_true, pneu_scores)
    roc_auc_pneu = auc(fpr_pneu, tpr_pneu)
    ax1.plot(fpr_pneu, tpr_pneu, color='blue', lw=2, label=f'Pneumonia (AUC = {roc_auc_pneu:.2f})')
    
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve Comparison')
    ax1.legend(loc="lower right")
    
    # 2. Metrics Bar Chart
    ax2 = plt.subplot(1, 2, 2)
    
    # Calculate Metrics (Threshold 0.5 for simplicity, or finding optimal)
    # Using 0.5 default
    tb_preds = (tb_scores > 0.25).astype(int) # YOLO often uses lower conf thresh
    pneu_preds = (pneu_scores > 0.5).astype(int)
    
    metrics = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }
    
    # TB Metrics
    p_tb, r_tb, f1_tb, _ = precision_recall_fscore_support(tb_true, tb_preds, average='binary', zero_division=0)
    acc_tb = accuracy_score(tb_true, tb_preds)
    
    metrics['Accuracy'].append(acc_tb)
    metrics['Precision'].append(p_tb)
    metrics['Recall'].append(r_tb)
    metrics['F1-Score'].append(f1_tb)
    
    # Pneumonia Metrics
    p_pn, r_pn, f1_pn, _ = precision_recall_fscore_support(pneu_true, pneu_preds, average='binary', zero_division=0)
    acc_pn = accuracy_score(pneu_true, pneu_preds)
    
    metrics['Accuracy'].append(acc_pn)
    metrics['Precision'].append(p_pn)
    metrics['Recall'].append(r_pn)
    metrics['F1-Score'].append(f1_pn)
    
    # Prepare data for plotting
    x = np.arange(len(metrics))  # label locations
    width = 0.35  # width of the bars
    
    metric_names = list(metrics.keys())
    tb_values = [metrics[k][0] for k in metric_names]
    pneu_values = [metrics[k][1] for k in metric_names]
    
    rects1 = ax2.bar(x - width/2, tb_values, width, label='Tuberculosis', color='darkorange')
    rects2 = ax2.bar(x + width/2, pneu_values, width, label='Pneumonia', color='blue')
    
    ax2.set_ylabel('Score')
    ax2.set_title('Evaluation Metrics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_names)
    ax2.set_ylim([0, 1.1])
    ax2.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")

# --- Main ---

if __name__ == "__main__":
    
    # Evaluate TB
    tb_true, tb_scores = evaluate_tb_model(TB_MODEL_PATH, TB_TEST_DIR)
    
    # Evaluate Pneumonia
    pneu_true, pneu_scores = evaluate_pneumonia_model(PNEUMONIA_MODEL_PATH, PNEUMONIA_TEST_DIR)
    
    # Plot
    plot_evaluation((tb_true, tb_scores), (pneu_true, pneu_scores), OUTPUT_IMAGE_PATH)
