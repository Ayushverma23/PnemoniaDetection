from fastapi import APIRouter, UploadFile, File, HTTPException
from services.pneumonia import PneumoniaDetector
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter()

# Initialize service (Singleton pattern for simplicity)
detector = PneumoniaDetector()

class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    probability: float
    heatmap_base64: str = None  # Add this field

@router.post("/predict/pneumonia", response_model=PredictionResponse)
async def predict_pneumonia(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.lower().endswith(('.dcm', '.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid file format. Supported: .dcm, .jpg, .png")
    
    try:
        content = await file.read()
        result = detector.predict(content, file.filename)
        
        return {
            "filename": file.filename,
            "prediction": result["prediction"],
            "probability": result["probability"],
            "heatmap_base64": result["heatmap_base64"]
        }
    except Exception as e:
        print(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
