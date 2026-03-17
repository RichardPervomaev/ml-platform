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


# ==========================================
# CONFIG
# ==========================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "linear-model"

# Включать ли автоперевод candidate -> production,
# если новая модель лучше текущей production
AUTO_PROMOTE_IF_BETTER = os.getenv("AUTO_PROMOTE_IF_BETTER", "false").lower() == "true"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

print("🚀 Starting training pipeline...")


# ==========================================
# 1. DATA GENERATION / TRAIN-TEST SPLIT
# ==========================================
# Учебный synthetic dataset.
# Потом это можно заменить на реальные training data.
X = np.random.normal(0, 1, (2000, 3))
y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.normal(0, 0.5, 2000)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==========================================
# 2. TRAIN MODEL
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
new_rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"📊 New model RMSE: {new_rmse}")


# ==========================================
# 3. EXPORT ONNX
# ==========================================
with tempfile.TemporaryDirectory() as tmpdir:
    onnx_path = os.path.join(tmpdir, "model.onnx")

    initial_type = [("input", FloatTensorType([None, 3]))]
    onnx_model = to_onnx(model, initial_types=initial_type, target_opset=15)

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"📦 ONNX model exported to: {onnx_path}")

    # ==========================================
    # 4. LOG TO MLFLOW
    # ==========================================
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("rmse", new_rmse)

        # baseline для drift detection
        baseline_dir = os.path.join(tmpdir, "baseline")
        os.makedirs(baseline_dir, exist_ok=True)

        baseline_path = os.path.join(baseline_dir, "baseline.csv")
        pd.DataFrame(X_train, columns=["f1", "f2", "f3"]).to_csv(baseline_path, index=False)

        mlflow.log_artifact(baseline_path, artifact_path="baseline")
        mlflow.log_artifact(onnx_path, artifact_path="onnx")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        run_id = run.info.run_id

print("✅ Model logged to MLflow")


# ==========================================
# 5. FIND NEWEST REGISTERED VERSION
# ==========================================
all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
new_version = max(int(v.version) for v in all_versions)

print(f"📦 Created model version: {new_version}")


# ==========================================
# 6. ALWAYS PUT NEW MODEL INTO STAGING
# ==========================================
client.set_registered_model_alias(
    MODEL_NAME,
    "staging",
    str(new_version)
)

print(f"🧪 Version {new_version} assigned to STAGING")


# ==========================================
# 7. LOAD CURRENT PRODUCTION METRIC
# ==========================================
champion_rmse = None
prod_version_number = None

try:
    prod_version = client.get_model_version_by_alias(MODEL_NAME, "production")
    prod_version_number = prod_version.version

    prod_run = client.get_run(prod_version.run_id)
    champion_rmse = float(prod_run.data.metrics["rmse"])

    print(f"🏆 Current production RMSE: {champion_rmse}")

except Exception:
    print("ℹ No production model yet.")


# ==========================================
# 8. DECISION: SHOULD WE PROMOTE?
# ==========================================
candidate_is_better = (
    champion_rmse is None or new_rmse < champion_rmse
)

if candidate_is_better:
    print("🚀 Candidate is BETTER than production.")

    if AUTO_PROMOTE_IF_BETTER:
        client.set_registered_model_alias(
            MODEL_NAME,
            "production",
            str(new_version)
        )
        print(f"✅ AUTO-PROMOTED version {new_version} to PRODUCTION")
    else:
        print("ℹ Auto-promotion disabled. Candidate stays in STAGING.")
else:
    print("❌ Candidate is NOT better. Production remains unchanged.")


# ==========================================
# 9. PRINT FINAL ALIAS STATE
# ==========================================
try:
    current_prod = client.get_model_version_by_alias(MODEL_NAME, "production")
    print(f"🏆 Production version: {current_prod.version}")
except Exception:
    print("🏆 Production alias is not set.")

try:
    current_staging = client.get_model_version_by_alias(MODEL_NAME, "staging")
    print(f"🧪 Staging version: {current_staging.version}")
except Exception:
    print("🧪 Staging alias is not set.")

print("🎯 Training pipeline finished.")
