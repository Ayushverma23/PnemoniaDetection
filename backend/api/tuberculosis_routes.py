from fastapi import APIRouter, UploadFile, File, HTTPException
from services.tuberculosis import TuberculosisDetector
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter()

# Initialize service
detector = TuberculosisDetector()

class Detection(BaseModel):
    class_: str
    confidence: float
    bbox: List[float]

class TBPredictionResponse(BaseModel):
    prediction: str
    confidence: float
    detections: List[Detection]
    annotated_image_base64: str

@router.post("/predict/tuberculosis", response_model=TBPredictionResponse)
async def predict_tuberculosis(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid file format. Supported: .jpg, .png")
    
    try:
        content = await file.read()
        result = detector.predict(content, file.filename)
        
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])

        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "detections": [
                {"class_": d["class"], "confidence": d["confidence"], "bbox": d["bbox"]}
                for d in result["detections"]
            ],
            "annotated_image_base64": result["annotated_image_base64"]
        }
    except Exception as e:
        print(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
