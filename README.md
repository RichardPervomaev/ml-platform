# ML Platform 1.0

Production-style ML platform with experiment tracking, model training, and inference API.

## 🚀 Overview

This project demonstrates a minimal production-ready ML platform built with:

- FastAPI (model inference service)
- MLflow (experiment tracking & model registry)
- PostgreSQL (MLflow backend)
- Docker & Docker Compose (container orchestration)

The platform supports:
- Model training with experiment tracking
- Artifact storage
- Containerized inference API
- Service orchestration

---

## 🏗 Architecture

Training → MLflow → Model Artifact → FastAPI → Docker → Docker Compose

Services:

- `postgres` — backend store for MLflow
- `mlflow` — tracking server
- `api` — inference service
- (nginx planned for v2.0)

---

## 📦 Project Structure
ml-platform/
├── api/
│ ├── main.py
│ ├── requirements.txt
├── training/
│ ├── train.py
│ ├── model.joblib
├── docker-compose.yml
├── Dockerfile
└── README.md


---

## 🧠 Training

Run training locally:

```bash
cd training
python train.py

MLflow UI:

http://localhost:5000

Run Platform

Start full platform:

docker-compose up --build

Services:
API → http://localhost:8000
Swagger → http://localhost:8000/docs
MLflow → http://localhost:5000

📊 Features
-Experiment tracking
-Model artifact versioning
-Containerized inference
-PostgreSQL backend
-Reproducible builds
-Service orchestration

🎯 Tech Stack
-Python 3.10
-FastAPI
-Scikit-learn
-MLflow
-PostgreSQL
-Docker
-Docker Compose
