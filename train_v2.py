import os
import tempfile
import numpy as np
from sklearn.linear_model import LinearRegression

from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType


# ==========================================
# Эта версия модели будет другой по формуле.
# Первая модель училась примерно на:
# y = 3*x1 + 2*x2 - x3
#
# Вторая модель будет учиться на:
# y = 10*x1 - x2 + 0.5*x3
#
# Это нужно, чтобы мы увидели,
# что Triton реально обслуживает ДВЕ разные модели.
# ==========================================

print("🚀 Training second ONNX model...")

X = np.random.normal(0, 1, (2000, 3)).astype(np.float32)
y = (10 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, 0.2, 2000)).astype(np.float32)

model = LinearRegression()
model.fit(X, y)

# Экспорт в ONNX
initial_type = [("input", FloatTensorType([None, 3]))]
onnx_model = to_onnx(model, initial_types=initial_type, target_opset=15)

# Путь Triton для второй модели
target_dir = os.path.join("triton", "models", "linear_model_v2_onnx", "1")
os.makedirs(target_dir, exist_ok=True)

target_path = os.path.join(target_dir, "model.onnx")

with open(target_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ Second ONNX model saved to: {target_path}")
