import os
import logging
import threading
import time
import numpy as np
import pandas as pd
import subprocess

from data_collector import collector
from drift_engine import drift_engine

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
# ==========================
# CONFIG
# ==========================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "linear-model"
MODEL_ALIAS = "production"

DRIFT_CHECK_INTERVAL = 60        # секунд
DRIFT_SAMPLE_SIZE = 300          # сколько данных нужно
RETRAIN_COOLDOWN = 600           # 10 минут

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Platform 3.5 Stable")

model = None
current_version = None
baseline_data = None

model_lock = threading.Lock()
last_retrain_time = 0
retraining_in_progress = False


# ==========================
# MODEL LOADER
# ==========================

def load_model():
    global model

    client = MlflowClient()

    try:
        version_info = client.get_model_version_by_alias(
            MODEL_NAME,
            "production"
        )

        model_uri = f"models:/{MODEL_NAME}/{version_info.version}"

        model = mlflow.pyfunc.load_model(model_uri)

        print(f"Model loaded: version {version_info.version}")

    except Exception as e:

        print("No production model found in MLflow")
        print(e)

        model = None

# ==========================
# DRIFT MONITOR
# ==========================

def background_drift_monitor():
    global last_retrain_time, retraining_in_progress

    while True:
        time.sleep(DRIFT_CHECK_INTERVAL)

        if collector.size() < DRIFT_SAMPLE_SIZE:
            continue

        if baseline_data is None:
            continue

        prod_array = np.array(collector.get_all())

        drift_detected, psi = drift_engine.detect(
            baseline_data,
            prod_array
        )

        if drift_detected:
            current_time = time.time()

            if retraining_in_progress:
                logger.info("Retrain already running — skipping.")
                continue

            if current_time - last_retrain_time < RETRAIN_COOLDOWN:
                logger.info("Drift detected but cooldown active.")
                continue

            logger.warning(f"DATA DRIFT DETECTED! PSI={psi}")

            retraining_in_progress = True
            last_retrain_time = current_time

            subprocess.Popen(["python", "train.py"])

        else:
            logger.info(f"No drift. PSI={psi}")


# ==========================
# STARTUP
# ==========================

@app.on_event("startup")
def startup_event():
    load_model()
    threading.Thread(
        target=background_drift_monitor,
        daemon=True
    ).start()


# ==========================
# SCHEMAS
# ==========================

class PredictRequest(BaseModel):
    data: List[float]

class PredictResponse(BaseModel):
    predictions: List[float]
    model_version: str


# ==========================
# ENDPOINTS
# ==========================

@app.post("/predict")
def predict(data: list):

    global model

    if model is None:
        return {
            "error": "model not loaded"
        }

    prediction = model.predict([data])

    return {
        "prediction": prediction.tolist()
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/promote")
def promote():

    staging = client.get_model_version_by_alias(
        MODEL_NAME,
        "staging"
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        "production",
        staging.version
    )

    return {
        "message": f"Version {staging.version} promoted to production"
    }
