from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="ML Platform 1.0 API")

MODEL_PATH = "model.joblib"

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


from api.triton_client import predict as triton_predict


@app.post("/predict")
def predict(data: list):

    result = triton_predict(data)

    return result
