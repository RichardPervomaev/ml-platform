import os
import logging
import mlflow
import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# -----------------------------
# Настройка логирования
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# -----------------------------
# MLflow config
# -----------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_URI = "models:/linear-model@Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# -----------------------------
# FastAPI init
# -----------------------------
app = FastAPI(title="ML API", version="1.0")

model = None


# -----------------------------
# Model loader
# -----------------------------
def load_model():
    global model
    try:
        logger.info("Loading model from MLflow...")
        model = mlflow.pyfunc.load_model(MODEL_URI)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None


# Загружаем модель при старте
@app.on_event("startup")
def startup_event():
    load_model()


# -----------------------------
# Schemas
# -----------------------------
class PredictRequest(BaseModel):
    data: List[float]


class PredictResponse(BaseModel):
    predictions: List[float]


# -----------------------------
# Endpoints
# -----------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/reload")
def reload_model():
    load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to load.")
    return {"status": "model reloaded"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        df = pd.DataFrame(request.data, columns=["feature"])
        preds = model.predict(df)
        return PredictResponse(predictions=preds.tolist())

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail="Prediction failed.")
