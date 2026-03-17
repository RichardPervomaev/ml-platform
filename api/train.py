import os
import tempfile
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient

from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType


# ==========================
# CONFIG
# ==========================

# Адрес MLflow tracking server внутри docker-сети
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Имя модели в MLflow Model Registry
MODEL_NAME = "linear-model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

print("🚀 Starting training pipeline...")


# ==========================
# 1. Генерация данных
# ==========================
# Пока данные искусственные.
# Это нормально для учебной платформы.
# Позже можно будет заменить на реальные данные из CSV / feature store.

X = np.random.normal(0, 1, (2000, 3)).astype(np.float32)
y = (3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.normal(0, 0.5, 2000)).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==========================
# 2. Обучение модели
# ==========================
# Берём простую линейную регрессию.
# Для production логика такая же, просто модель может быть сложнее.

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
new_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

print(f"📊 New model RMSE: {new_rmse}")


# ==========================
# 3. Экспорт модели в ONNX
# ==========================
# Здесь мы переводим sklearn-модель в универсальный serving-format.
# Triton будет обслуживать именно model.onnx.

initial_type = [("input", FloatTensorType([None, 3]))]
onnx_model = to_onnx(model, initial_types=initial_type, target_opset=15)

# Создаём временную папку, чтобы сохранить ONNX-файл
tmp_dir = tempfile.mkdtemp()
onnx_path = os.path.join(tmp_dir, "model.onnx")

with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"📦 ONNX model exported to: {onnx_path}")


# ==========================
# 4. Логирование в MLflow
# ==========================
# Логируем:
# - метрики
# - baseline.csv
# - sklearn model
# - ONNX model как artifact

with mlflow.start_run() as run:

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("rmse", new_rmse)

    # baseline нужен для drift / мониторинга распределения
    baseline_df = pd.DataFrame(X_train, columns=["f1", "f2", "f3"])
    baseline_path = os.path.join(tmp_dir, "baseline.csv")
    baseline_df.to_csv(baseline_path, index=False)

    # Логируем baseline в отдельную папку artifacts
    mlflow.log_artifact(baseline_path, artifact_path="baseline")

    # Логируем ONNX-файл в отдельную папку artifacts
    mlflow.log_artifact(onnx_path, artifact_path="onnx")

    # Логируем sklearn-модель в MLflow Registry
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name=MODEL_NAME
    )

    run_id = run.info.run_id

print("✅ Model logged to MLflow")


# ==========================
# 5. Получаем новую версию модели
# ==========================
# В registry уже создалась новая версия.
# Нам нужно понять, какая именно.

latest_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
new_version = max(int(v.version) for v in latest_versions)

print(f"📦 Created model version: {new_version}")


# ==========================
# 6. Назначаем STAGING alias
# ==========================
# Сначала новая модель всегда попадает в staging.
# Это безопаснее, чем сразу делать production.

client.set_registered_model_alias(
    MODEL_NAME,
    "staging",
    str(new_version)
)

print(f"🧪 Version {new_version} assigned to STAGING")


# ==========================
# 7. Сравнение с production
# ==========================
# Если production уже есть — сравниваем RMSE.
# Пока auto-promotion не делаем.
# Это безопаснее: promotion делаем осознанно.

try:
    prod_version = client.get_model_version_by_alias(
        MODEL_NAME,
        "production"
    )

    prod_run = client.get_run(prod_version.run_id)
    champion_rmse = float(prod_run.data.metrics["rmse"])

    print(f"🏆 Current production RMSE: {champion_rmse}")

    if new_rmse < champion_rmse:
        print("🚀 Candidate is BETTER than production.")
    else:
        print("⚠ Candidate is WORSE than production.")

except Exception:
    print("ℹ No production model yet.")


# ==========================
# 8. Печать текущих alias
# ==========================
# Удобно для отладки — сразу видно, кто staging, кто production.

try:
    current_prod = client.get_model_version_by_alias(
        MODEL_NAME,
        "production"
    )
    print(f"🏆 Production version: {current_prod.version}")
except Exception:
    print("No production alias set.")

try:
    current_staging = client.get_model_version_by_alias(
        MODEL_NAME,
        "staging"
    )
    print(f"🧪 Staging version: {current_staging.version}")
except Exception:
    print("No staging alias set.")

print("🎯 Training pipeline finished.")
