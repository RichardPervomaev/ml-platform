import random
import logging
import requests
import threading
import random
import logging
import requests
from data_collector import collector
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List


# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==========================================
# FASTAPI APP
# ==========================================
app = FastAPI(title="ML Platform Gateway", version="6.2")


# ==========================================
# TRITON CONFIG
# ==========================================
# Triton сейчас запущен отдельно от docker-compose.
# Поэтому из контейнера api ходим к нему через host.docker.internal
TRITON_BASE_URL = "http://triton:8000"

# Две serving-модели в Triton
MODEL_A = "linear_model_onnx"
MODEL_B = "linear_model_v2_onnx"


# ==========================================
# ROLLOUT CONFIG
# ==========================================
# 80% запросов идут в модель A
# 20% запросов идут в модель B
ROLLOUT_A_PERCENT = 0.8


# ==========================================
# REQUEST / RESPONSE SCHEMAS
# ==========================================
class PredictRequest(BaseModel):
    data: List[float]


class PredictResponse(BaseModel):
    prediction: float
    model_name: str
    rollout_group: str


# ==========================================
# HELPER: выбор модели по правилу A/B
# ==========================================
def choose_model() -> tuple[str, str]:
    """
    Выбираем модель по rollout-правилу.

    random.random() возвращает число от 0 до 1.
    Если число меньше 0.8 -> идём в модель A.
    Иначе -> идём в модель B.
    """
    r = random.random()

    if r < ROLLOUT_A_PERCENT:
        return MODEL_A, "A"
    else:
        return MODEL_B, "B"


# ==========================================
# HELPER: вызов Triton
# ==========================================
def call_triton(model_name: str, data: List[float]) -> float:
    """
    Отправляем запрос в Triton и возвращаем предсказание.

    Triton ждёт payload такого вида:
    {
      "inputs": [
        {
          "name": "input",
          "shape": [1, 3],
          "datatype": "FP32",
          "data": [[1.0, 2.0, 3.0]]
        }
      ]
    }

    shape [1, 3]:
    - 1 = размер batch
    - 3 = число признаков
    """
    url = f"{TRITON_BASE_URL}/v2/models/{model_name}/infer"

    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 3],
                "datatype": "FP32",
                "data": [data]
            }
        ]
    }

    logger.info("Calling Triton model=%s payload=%s", model_name, payload)

    try:
        response = requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.exception("Failed to connect to Triton: %s", e)
        raise HTTPException(status_code=500, detail="Triton is unavailable")

    if response.status_code != 200:
        logger.error("Triton returned error: %s", response.text)
        raise HTTPException(status_code=500, detail=f"Triton error: {response.text}")

    result = response.json()

    logger.info("Triton response from model=%s result=%s", model_name, result)

    try:
        prediction = float(result["outputs"][0]["data"][0])
    except Exception as e:
        logger.exception("Failed to parse Triton response: %s", e)
        raise HTTPException(status_code=500, detail="Bad Triton response")

    return prediction

def shadow_call_and_log(shadow_model_name: str, data: List[float], primary_prediction: float):
    """
    Фоновый вызов shadow-модели.

    Что происходит:
    - вызываем модель B
    - сравниваем её предсказание с основной моделью A
    - пишем разницу в лог

    Важно:
    - ошибки shadow-модели не должны ломать ответ пользователю
    - поэтому всё завернуто в try/except
    """
    try:
        shadow_prediction = call_triton(shadow_model_name, data)

        diff = abs(primary_prediction - shadow_prediction)

        logger.info(
            "SHADOW COMPARE primary_model=%s shadow_model=%s input=%s primary_prediction=%s shadow_prediction=%s abs_diff=%s",
            MODEL_A,
            shadow_model_name,
            data,
            primary_prediction,
            shadow_prediction,
            diff
        )

    except Exception as e:
        logger.exception("Shadow call failed: %s", e)

# ==========================================
# HEALTH ENDPOINT
# ==========================================
@app.get("/health")
def health():
    """
    Проверка, что API gateway жив.
    """
    return {"status": "ok"}


# ==========================================
# DIRECT ENDPOINT FOR MODEL A
# ==========================================
@app.post("/predict-a", response_model=PredictResponse)
def predict_a(request: PredictRequest):
    """
    Явный вызов модели A.
    Полезно для проверки.
    """
    if len(request.data) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 features are required")

    prediction = call_triton(MODEL_A, request.data)

    return PredictResponse(
        prediction=prediction,
        model_name=MODEL_A,
        rollout_group="A"
    )


# ==========================================
# DIRECT ENDPOINT FOR MODEL B
# ==========================================
@app.post("/predict-b", response_model=PredictResponse)
def predict_b(request: PredictRequest):
    """
    Явный вызов модели B.
    Полезно для проверки.
    """
    if len(request.data) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 features are required")

    prediction = call_triton(MODEL_B, request.data)

    return PredictResponse(
        prediction=prediction,
        model_name=MODEL_B,
        rollout_group="B"
    )


# ==========================================
# A/B ROLLOUT ENDPOINT
# ==========================================
@app.post("/predict-ab", response_model=PredictResponse)
def predict_ab(request: PredictRequest):
    if len(request.data) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 features are required")

    collector.add(request.data)

    model_name, rollout_group = choose_model()

    prediction = call_triton(model_name, request.data)

    return PredictResponse(
        prediction=prediction,
        model_name=model_name,
        rollout_group=rollout_group
    )
    """
    Главный endpoint A/B rollout.

    Логика:
    1. Принимаем входные данные
    2. Выбираем модель A или B по правилу 80/20
    3. Отправляем запрос в Triton
    4. Возвращаем:
       - prediction
       - какая модель сработала
       - какая rollout-группа
    """

@app.post("/predict-shadow", response_model=PredictResponse)
def predict_shadow(request: PredictRequest):
    """
    Shadow deployment endpoint.

    Логика:
    1. Пользователю всегда отдаём ответ от модели A
    2. Модель B вызываем в фоне
    3. В лог пишем сравнение A vs B

    Это безопаснее, чем A/B rollout:
    - пользователь всегда получает проверенный ответ
    - новая модель проверяется незаметно
    """
    if len(request.data) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 features are required")

    # Основной боевой ответ
    primary_prediction = call_triton(MODEL_A, request.data)

    # Фоновый shadow-вызов
    thread = threading.Thread(
        target=shadow_call_and_log,
        args=(MODEL_B, request.data, primary_prediction),
        daemon=True
    )
    thread.start()

    return PredictResponse(
        prediction=primary_prediction,
        model_name=MODEL_A,
        rollout_group="shadow-primary-A"
    )
