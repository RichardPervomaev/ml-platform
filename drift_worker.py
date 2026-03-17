import os
import time
import json
import logging
import numpy as np
import pandas as pd
import mlflow
import redis

from mlflow.tracking import MlflowClient

from data_collector import collector
from drift_engine import drift_engine


# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==========================================
# CONFIG
# ==========================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

MODEL_NAME = "linear-model"
MODEL_ALIAS = "production"

DRIFT_CHECK_INTERVAL = int(os.getenv("DRIFT_CHECK_INTERVAL", "60"))
DRIFT_SAMPLE_SIZE = int(os.getenv("DRIFT_SAMPLE_SIZE", "300"))
RETRAIN_COOLDOWN = int(os.getenv("RETRAIN_COOLDOWN", "600"))

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
RETRAIN_QUEUE_KEY = os.getenv("RETRAIN_QUEUE_KEY", "retrain_jobs")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)


# ==========================================
# STATE
# ==========================================
baseline_data = None
last_retrain_time = 0
retraining_in_progress = False
last_seen_production_version = None


# ==========================================
# HELPER: получить текущую production version
# ==========================================
def get_current_production_version():
    """
    Возвращает текущую production version из MLflow Registry.
    Если alias не найден — возвращает None.
    """
    try:
        version_info = client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS
        )
        return str(version_info.version)
    except Exception as e:
        logger.warning("Failed to get current production version: %s", e)
        return None


# ==========================================
# HELPER: загрузить baseline из production-модели
# ==========================================
def load_production_baseline():
    """
    Загружаем baseline.csv из текущей production-модели.
    Одновременно обновляем last_seen_production_version.
    """
    global baseline_data, last_seen_production_version

    try:
        version_info = client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS
        )

        run_id = version_info.run_id
        artifact_uri = f"runs:/{run_id}/baseline/baseline.csv"

        logger.info(
            "Loading baseline from production model version=%s run_id=%s artifact_uri=%s",
            version_info.version,
            run_id,
            artifact_uri
        )

        baseline_path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri
        )

        baseline_df = pd.read_csv(baseline_path)
        baseline_data = baseline_df.values
        last_seen_production_version = str(version_info.version)

        logger.info(
            "Baseline loaded successfully. Shape=%s production_version=%s",
            baseline_data.shape,
            last_seen_production_version
        )

    except Exception as e:
        logger.exception("Failed to load production baseline: %s", e)
        baseline_data = None


# ==========================================
# HELPER: очистить collector после promotion
# ==========================================
def reset_production_window():
    """
    Очищаем накопленный production traffic,
    чтобы не триггерить retrain повторно на старых drift-данных.
    """
    try:
        collector.clear()
        logger.info("Production collector window cleared")
    except Exception as e:
        logger.exception("Failed to clear production collector window: %s", e)


# ==========================================
# HELPER: проверить, не изменилась ли production version
# ==========================================
def refresh_if_production_changed():
    """
    Если production alias указывает уже на новую версию модели,
    обновляем baseline и очищаем collector.

    Это критично после auto-promotion:
    иначе worker будет продолжать сравнивать traffic
    со старым baseline и бесконечно запускать retrain.
    """
    global last_seen_production_version

    current_prod = get_current_production_version()

    if current_prod is None:
        return

    if last_seen_production_version is None:
        logger.info("No cached production version yet. Loading baseline.")
        load_production_baseline()
        return

    if current_prod != last_seen_production_version:
        logger.warning(
            "Production version changed: old=%s new=%s. Reloading baseline and clearing collector.",
            last_seen_production_version,
            current_prod
        )

        load_production_baseline()
        reset_production_window()


# ==========================================
# HELPER: retrain trigger через Redis queue
# ==========================================
def trigger_retraining(psi_value: float):
    """
    Drift worker не обучает модель сам.
    Он только кладёт задачу в Redis queue.
    """
    payload = {
        "event": "drift_detected",
        "psi": float(psi_value),
        "timestamp": time.time()
    }

    redis_client.rpush(RETRAIN_QUEUE_KEY, json.dumps(payload))

    logger.warning(
        "Retrain job pushed to queue=%s payload=%s",
        RETRAIN_QUEUE_KEY,
        payload
    )


# ==========================================
# MAIN LOOP
# ==========================================
def run():
    global last_retrain_time, retraining_in_progress

    logger.info("Drift worker started")

    load_production_baseline()

    # На старте очищаем старое окно traffic,
    # чтобы не сравнивать новый baseline со старыми drift-данными
    reset_production_window()

    while True:
        try:
            time.sleep(DRIFT_CHECK_INTERVAL)

            refresh_if_production_changed()
            if baseline_data is None:
                logger.warning("No baseline loaded. Trying to reload...")
                load_production_baseline()
                continue

            current_size = collector.size()
            logger.info("Collector size=%s", current_size)

            if current_size < DRIFT_SAMPLE_SIZE:
                logger.info(
                    "Not enough data for drift check yet. Need=%s current=%s",
                    DRIFT_SAMPLE_SIZE,
                    current_size
                )
                continue

            prod_array = np.array(collector.get_all())

            drift_detected, psi = drift_engine.detect(
                baseline_data,
                prod_array
            )

            logger.info(
                "Drift check finished. drift_detected=%s psi=%s",
                drift_detected,
                psi
            )

            if drift_detected:
                current_time = time.time()

                if retraining_in_progress:
                    logger.info("Retraining already in progress. Skipping.")
                    continue

                if current_time - last_retrain_time < RETRAIN_COOLDOWN:
                    logger.info("Drift detected but cooldown is active. Skipping retrain.")
                    continue

                logger.warning("DATA DRIFT DETECTED! PSI=%s", psi)

                retraining_in_progress = True
                last_retrain_time = current_time

                trigger_retraining(psi)

                retraining_in_progress = False

            else:
                logger.info("No drift detected. PSI=%s", psi)

        except Exception as e:
            logger.exception("Drift worker loop failed: %s", e)


if __name__ == "__main__":
    run()
