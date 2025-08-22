from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import cv2
import os

from app.inference import RTDETREngine

class DetectionBox(BaseModel):
    box: List[float] = Field(..., example=[100.0, 150.0, 200.0, 250.0])
    score: float = Field(..., example=0.95)
    label: int = Field(..., example=17)

class DetectionResponse(BaseModel):
    detections: List[DetectionBox]

app = FastAPI(
    title="RT-DETR Inference API",
    description="TensorRT-optimized RT-DETR object detection",
    version="1.0.0",
)

ENGINE_PATH = os.environ.get("ENGINE_PATH", "/app/models/rtdetr-l.engine")

try:
    engine = RTDETREngine(engine_path=ENGINE_PATH)
except Exception as e:
    print(f"Error loading TensorRT engine: {e}")
    engine = None

@app.on_event("startup")
async def startup_event():
    if engine is None:
        raise RuntimeError("TensorRT engine could not be loaded.")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=DetectionResponse)
def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    try:
        contents = file.file.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
        detections = engine.detect(image)
        return {"detections": detections}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal error during processing.")
