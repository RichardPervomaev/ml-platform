import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# -----------------------------
# Настройки
# -----------------------------
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("linear-regression-experiment")

MODEL_NAME = "linear-model"

# -----------------------------
# Генерация данных
# -----------------------------
X = np.random.rand(100, 1) * 10
y = 3 * X.squeeze() + 5 + np.random.randn(100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -----------------------------
# Тренировка
# -----------------------------
with mlflow.start_run() as run:

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Логируем метрику
    mlflow.log_metric("rmse", rmse)

    # Логируем модель + регистрируем
    signature = infer_signature(X_train, model.predict(X_train))

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name=MODEL_NAME,
    )

    print(f"Model logged. RMSE: {rmse}")

# -----------------------------
# Auto Promotion Logic
# -----------------------------
client = MlflowClient()
latest_versions = client.get_latest_versions(MODEL_NAME)

# Получаем последнюю зарегистрированную версию
new_version = latest_versions[-1]

production_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])

if production_versions:
    prod_version = production_versions[0]
    prod_run = client.get_run(prod_version.run_id)

    old_rmse = prod_run.data.metrics["rmse"]
    new_run = client.get_run(new_version.run_id)
    new_rmse = new_run.data.metrics["rmse"]

    print(f"Old RMSE: {old_rmse}")
    print(f"New RMSE: {new_rmse}")

    if new_rmse < old_rmse:
        print("New model is better. Promoting to Production.")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version.version,
            stage="Production",
        )
    else:
        print("New model is worse. Keeping old Production model.")

else:
    print("No Production model found. Promoting first model.")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=new_version.version,
        stage="Production",
    )

print("Training pipeline finished.")
