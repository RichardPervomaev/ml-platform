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

# --------------------------------
# Logging
# --------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# --------------------------------
# MLflow configuration
# --------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "linear-model"
MODEL_ALIAS = "production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# --------------------------------
# FastAPI
# --------------------------------
app = FastAPI(title="ML Platform 2.1")

model = None
current_version = None
baseline_data = None
model_lock = threading.Lock()

# --------------------------------
# Model Loader
# --------------------------------
def load_model():
    global model, current_version, baseline_data

    version_info = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS
    )

    if current_version == version_info.version:
        return

    logger.info(f"Loading model version {version_info.version}")

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Download baseline artifact
    run = client.get_run(version_info.run_id)
    artifact_path = client.download_artifacts(
        version_info.run_id,
        "model/baseline.csv"
    )

    baseline_data = pd.read_csv(artifact_path).values

    current_version = version_info.version

# --------------------------------
# Drift Monitor
# --------------------------------
def background_drift_monitor():

    while True:

        if collector.size() > 300 and baseline_data is not None:

            prod_array = np.array(collector.get_all())

            drift_detected, psi = drift_engine.detect(
                baseline_data,
                prod_array
            )

            if drift_detected:
                logger.warning(f"DATA DRIFT DETECTED! PSI={psi}")
                subprocess.Popen(["python", "train.py"])
            else:
                logger.info(f"No drift. PSI={psi}")

        time.sleep(60)

# --------------------------------
# Startup
# --------------------------------
@app.on_event("startup")
def startup_event():
    load_model()

    threading.Thread(target=background_drift_monitor, daemon=True).start()

# --------------------------------
# Schemas
# --------------------------------
class PredictRequest(BaseModel):
    data: List[float]

class PredictResponse(BaseModel):
    predictions: List[float]
    model_version: str

# --------------------------------
# Endpoints
# --------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([request.data])
    preds = model.predict(df)

    collector.add(request.data)

    return PredictResponse(
        predictions=preds.tolist(),
        model_version=current_version
    )
import mlflow.pyfunc
import numpy as np
import subprocess
import time
import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel

# ==========================
# 1️⃣ Конфигурация
# ==========================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "linear-model"

DRIFT_THRESHOLD = 0.5
RETRAIN_COOLDOWN_SECONDS = 600  # 10 минут

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ==========================
# 2️⃣ Глобальные переменные
# ==========================

model = None
baseline_data = None
last_retrain_time = 0
retraining_in_progress = False


# ==========================
# 3️⃣ Загрузка модели
# ==========================

def load_model():
    global model
    logger.info("Loading production model...")
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@production")


# ==========================
# 4️⃣ Drift calculation (PSI)
# ==========================

def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_percents = np.percentile(expected, breakpoints)
    actual_percents = np.percentile(actual, breakpoints)

    psi = 0

    for i in range(len(expected_percents) - 1):
        e_min = expected_percents[i]
        e_max = expected_percents[i + 1]

        expected_count = ((expected >= e_min) & (expected < e_max)).sum()
        actual_count = ((actual >= e_min) & (actual < e_max)).sum()

        expected_ratio = expected_count / len(expected)
        actual_ratio = actual_count / len(actual)

        if expected_ratio > 0 and actual_ratio > 0:
            psi += (actual_ratio - expected_ratio) * np.log(
                actual_ratio / expected_ratio
            )

    return psi


# ==========================
# 5️⃣ Request schema
# ==========================

class PredictionRequest(BaseModel):
    data: list


# ==========================
# 6️⃣ Startup
# ==========================

@app.on_event("startup")
def startup_event():
    global baseline_data

    load_model()

    # Baseline фиксируем один раз
    baseline_data = np.random.normal(0, 1, 1000)
    logger.info("API started successfully.")


# ==========================
# 7️⃣ Predict endpoint
# ==========================

@app.post("/predict")
def predict(request: PredictionRequest):
    global last_retrain_time, retraining_in_progress

    data = np.array(request.data).reshape(1, -1)

    prediction = model.predict(data)

    # Drift detection
    current_batch = np.random.normal(5, 1, 1000)  # имитация drift

    psi = calculate_psi(baseline_data, current_batch)

    if psi > DRIFT_THRESHOLD:
        current_time = time.time()

        if (
            not retraining_in_progress
            and current_time - last_retrain_time > RETRAIN_COOLDOWN_SECONDS
        ):
            logger.warning(f"DATA DRIFT DETECTED! PSI={psi}")
            retraining_in_progress = True
            subprocess.Popen(["python", "train.py"])
            last_retrain_time = current_time
        else:
            logger.info("Drift detected but retrain blocked (cooldown or running).")

    return {"predictions": prediction.tolist()}


# ==========================
# 8️⃣ Health check
# ==========================

@app.get("/health")
def health():
    return {"status": "ok"}
