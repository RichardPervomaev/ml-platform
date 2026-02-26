import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import os


# ==========================
# 1️⃣ Конфигурация
# ==========================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "linear-model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("linear-regression-experiment")


# ==========================
# 2️⃣ Генерация данных
# ==========================

# В реальном проде тут будет загрузка production data
X = np.random.rand(1000, 3)
y = X @ np.array([3.5, -2.0, 1.0]) + np.random.randn(1000) * 0.5

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==========================
# 3️⃣ Обучение модели
# ==========================

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))


# ==========================
# 4️⃣ Логирование в MLflow
# ==========================

with mlflow.start_run() as run:

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name=MODEL_NAME
    )

    print(f"Model logged. RMSE: {rmse}")


# ==========================
# 5️⃣ Работа с Model Registry
# ==========================

client = MlflowClient()

# Получаем все версии модели
model_versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

# Находим максимальный номер версии
latest_version = max(
    [int(mv.version) for mv in model_versions]
)

# Назначаем alias production новой версии
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="production",
    version=str(latest_version)
)

print(f"Alias 'production' -> version {latest_version}")
print("Training pipeline finished.")
