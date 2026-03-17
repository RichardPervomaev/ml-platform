import os
import time
import shutil
import logging
import tempfile
import requests
import mlflow

from mlflow.tracking import MlflowClient


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
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "linear-model")

TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "linear_model_onnx")
TRITON_REPO = os.getenv("TRITON_REPO", "/models")

TRITON_BASE_URL = os.getenv("TRITON_BASE_URL", "http://triton:8000")

POLL_INTERVAL_SECONDS = int(os.getenv("DEPLOYER_POLL_INTERVAL", "20"))
TRITON_RELOAD_WAIT_SECONDS = int(os.getenv("TRITON_RELOAD_WAIT_SECONDS", "5"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

logger.info("Deployer started")
logger.info("MLFLOW_TRACKING_URI=%s", MLFLOW_TRACKING_URI)
logger.info("MLFLOW_MODEL_NAME=%s", MLFLOW_MODEL_NAME)
logger.info("TRITON_MODEL_NAME=%s", TRITON_MODEL_NAME)
logger.info("TRITON_BASE_URL=%s", TRITON_BASE_URL)


# ==========================================
# HELPERS: REGISTRY
# ==========================================
def get_production_version():
    """
    Возвращает текущую version у alias=production.
    """
    try:
        version_info = client.get_model_version_by_alias(
            MLFLOW_MODEL_NAME,
            "production"
        )
        return str(version_info.version)
    except Exception as e:
        logger.warning("Production alias not found yet: %s", e)
        return None


def set_production_version(version: str):
    """
    Переназначает alias production на указанную version.
    """
    client.set_registered_model_alias(
        MLFLOW_MODEL_NAME,
        "production",
        str(version)
    )
    logger.warning("Production alias rolled back to version=%s", version)


# ==========================================
# HELPERS: ARTIFACT DOWNLOAD
# ==========================================
def download_onnx_artifact(version: str) -> str:
    """
    Скачивает ONNX artifact из MLflow run, связанного с model version.
    """
    model_version = client.get_model_version(MLFLOW_MODEL_NAME, version)
    run_id = model_version.run_id

    artifact_uri = f"runs:/{run_id}/onnx/model.onnx"

    logger.info(
        "Downloading ONNX artifact for version=%s run_id=%s artifact_uri=%s",
        version,
        run_id,
        artifact_uri
    )

    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=artifact_uri
    )

    logger.info("ONNX downloaded to: %s", local_path)
    return local_path


# ==========================================
# HELPERS: TRITON CONFIG
# ==========================================
def write_triton_config():
    """
    Пишем config.pbtxt для Triton ONNX model.
    """
    model_dir = os.path.join(TRITON_REPO, TRITON_MODEL_NAME)
    os.makedirs(model_dir, exist_ok=True)

    config_path = os.path.join(model_dir, "config.pbtxt")

    config_text = f'''
name: "{TRITON_MODEL_NAME}"
platform: "onnxruntime_onnx"

max_batch_size: 8

dynamic_batching {{
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 10000
}}

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [3]
  }}
]

output [
  {{
    name: "variable"
    data_type: TYPE_FP32
    dims: [1]
  }}
]
'''.strip()

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_text)

    logger.info("Triton config written to: %s", config_path)


def deploy_to_triton(onnx_file_path: str):
    """
    Кладём ONNX-модель в Triton model repository.
    """
    version_dir = os.path.join(TRITON_REPO, TRITON_MODEL_NAME, "1")
    os.makedirs(version_dir, exist_ok=True)

    target_model_path = os.path.join(version_dir, "model.onnx")
    shutil.copy2(onnx_file_path, target_model_path)

    logger.info("Model copied to Triton path: %s", target_model_path)

    write_triton_config()


# ==========================================
# HELPERS: SMOKE TEST
# ==========================================
def wait_for_triton_reload():
    """
    Даём Triton время перечитать model repository.
    """
    logger.info("Waiting %s seconds for Triton reload", TRITON_RELOAD_WAIT_SECONDS)
    time.sleep(TRITON_RELOAD_WAIT_SECONDS)


def smoke_test_triton_model() -> bool:
    """
    Простейший smoke test:
    1. модель видна в Triton
    2. inference endpoint отвечает 200
    3. есть outputs

    Это минимальная проверка, что serving не сломан.
    """
    try:
        model_url = f"{TRITON_BASE_URL}/v2/models/{TRITON_MODEL_NAME}"
        infer_url = f"{TRITON_BASE_URL}/v2/models/{TRITON_MODEL_NAME}/infer"

        logger.info("Smoke test: checking model metadata url=%s", model_url)
        meta_resp = requests.get(model_url, timeout=10)
        if meta_resp.status_code != 200:
            logger.error("Smoke test failed: metadata endpoint returned %s", meta_resp.status_code)
            return False

        payload = {
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 3],
                    "datatype": "FP32",
                    "data": [[1.0, 2.0, 3.0]]
                }
            ]
        }

        logger.info("Smoke test: sending inference request url=%s payload=%s", infer_url, payload)
        infer_resp = requests.post(infer_url, json=payload, timeout=10)
        if infer_resp.status_code != 200:
            logger.error("Smoke test failed: infer endpoint returned %s body=%s", infer_resp.status_code, infer_resp.text)
            return False

        result = infer_resp.json()
        logger.info("Smoke test result=%s", result)

        outputs = result.get("outputs")
        if not outputs:
            logger.error("Smoke test failed: outputs missing in response")
            return False

        return True

    except Exception as e:
        logger.exception("Smoke test failed with exception: %s", e)
        return False


# ==========================================
# MAIN LOOP
# ==========================================
def observe_and_deploy():
    """
    Главный цикл:
    - смотрим production alias
    - если версия изменилась:
      1. скачиваем ONNX
      2. выкладываем в Triton
      3. ждём reload
      4. делаем smoke test
      5. если fail -> rollback alias
    """
    last_seen_version = None
    previous_good_version = None

    while True:
        try:
            logger.info("Checking production alias...")
            current_prod_version = get_production_version()
            logger.info("Found production version: %s", current_prod_version)

            if current_prod_version is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # первый запуск: просто запоминаем текущую production version
            if last_seen_version is None:
                last_seen_version = current_prod_version
                previous_good_version = current_prod_version
                logger.info(
                    "Initial production version registered. current=%s previous_good=%s",
                    current_prod_version,
                    previous_good_version
                )

            # если production alias изменился — деплоим новую модель
            if current_prod_version != last_seen_version:
                logger.warning(
                    "New production version detected: old=%s new=%s",
                    last_seen_version,
                    current_prod_version
                )

                onnx_path = download_onnx_artifact(current_prod_version)
                deploy_to_triton(onnx_path)
                wait_for_triton_reload()

                smoke_ok = smoke_test_triton_model()

                if smoke_ok:
                    logger.info(
                        "Smoke test passed. Deployment accepted for production version=%s",
                        current_prod_version
                    )
                    previous_good_version = current_prod_version
                    last_seen_version = current_prod_version
                else:
                    logger.error(
                        "Smoke test failed for production version=%s",
                        current_prod_version
                    )

                    if previous_good_version is not None:
                        logger.warning(
                            "Rolling back production alias from failed version=%s to previous_good_version=%s",
                            current_prod_version,
                            previous_good_version
                        )
                        set_production_version(previous_good_version)
                        last_seen_version = previous_good_version
                    else:
                        logger.error("Rollback impossible: previous_good_version is None")

        except Exception as e:
            logger.exception("Deployer loop failed: %s", e)

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    observe_and_deploy()
