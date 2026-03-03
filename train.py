import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient

# ==========================
# CONFIG
# ==========================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "linear-model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

print("🚀 Starting training pipeline...")

# ==========================
# 1️⃣ Data generation (demo)
# ==========================

X = np.random.normal(0, 1, (2000, 3))
y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.normal(0, 0.5, 2000)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# 2️⃣ Train model
# ==========================

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
new_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

print(f"📊 New model RMSE: {new_rmse}")

# ==========================
# 3️⃣ MLflow logging
# ==========================

with mlflow.start_run() as run:

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("rmse", new_rmse)

    # ---- baseline ----
    baseline_df = pd.DataFrame(X_train)
    baseline_path = "baseline.csv"
    baseline_df.to_csv(baseline_path, index=False)

    mlflow.log_artifact(baseline_path, artifact_path="model")

    # ---- model ----
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name=MODEL_NAME
    )

    run_id = run.info.run_id

print("✅ Model logged to MLflow")

# ==========================
# 4️⃣ Get new model version
# ==========================

latest_versions = client.get_latest_versions(MODEL_NAME)
new_version = max(int(v.version) for v in latest_versions)

print(f"📦 Created model version: {new_version}")

# ==========================
# 5️⃣ Assign STAGING alias
# ==========================

client.set_registered_model_alias(
    MODEL_NAME,
    "staging",
    new_version
)

print(f"🧪 Version {new_version} assigned to STAGING")

# ==========================
# 6️⃣ Compare with production
# ==========================

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
# 7️⃣ Show current aliases
# ==========================

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
