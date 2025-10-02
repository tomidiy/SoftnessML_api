from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import SoftnessPredictor
import os
from pathlib import Path

app = FastAPI(title="Softness Predictor API")

APP_DIR = Path(__file__).resolve().parent

# Initialize the model
predictor = SoftnessPredictor(radial_only=False)

class PredictionInput(BaseModel):
    temp: float
    frame: int
    gsd_file: str

@app.post("/predict")
async def predict_softness(input_data: PredictionInput):
    """Predict softness for given temperature, frame, and GSD file."""
    try:
        # Validate file existence
        if not os.path.exists(os.path.join(APP_DIR.parent/'data', input_data.gsd_file)):
            raise HTTPException(status_code=404, detail="GSD file not found")
        
        # Run prediction
        result = predictor.predict(input_data.temp, input_data.frame, os.path.join('/data', input_data.gsd_file))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}