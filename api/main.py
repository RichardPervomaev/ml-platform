from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="ML Platform 1.0 API")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "training", "model.joblib")

# Load model at startup
model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/")
def root():
    return {"message": "ML Platform 1.0 is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    data = np.array(request.features).reshape(1, -1)
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0].tolist()

    return {
        "prediction": int(prediction),
        "probabilities": probability
    }
