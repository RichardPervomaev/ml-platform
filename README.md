🚀 ML Platform 3.0 — Drift-Aware Self-Retraining System

Production-style MLOps platform built with:

- FastAPI
- MLflow Model Registry
- Alias-based deployment
- Automatic data drift detection (PSI)
- Event-driven retraining
- Retrain storm protection (cooldown + lock)
- Dockerized infrastructure

---

# System Architecture

Client → FastAPI → Model@production 
                    ↓  
              Drift Detection (PSI)  
                    ↓  
                 train.py  
                    ↓  
MLflow Registry → New Model Version → Alias `production` updated 

No manual redeployment required.

---

# 🧠 Core Features

## ✅ 1. Versioned Model Registry

Every retrain creates a new model version in MLflow.

No deprecated stages are used.

Deployment is controlled via alias:

models:/linear-model@production

---

## ✅ 2. Drift Detection (PSI)

Population Stability Index (PSI) is calculated on incoming data.

If:

- PSI > threshold
- Cooldown expired
- No retraining currently running

→ Retraining is triggered automatically.

---

## ✅ 3. Retrain Storm Protection

Platform includes:

- Cooldown mechanism (10 minutes)
- Retraining lock flag
- Alias-based model switching

Prevents infinite retraining loops.

---

# 📦 Project Structure

ml-api/
│
├── main.py # FastAPI + Drift Detection
├── train.py # Training + Registry Alias Update
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md

---

# 🐳 How To Run

## 1️⃣ Build containers


docker compose build


## 2️⃣ Start platform


docker compose up


Services:

- API → http://localhost:8000  
- MLflow → http://localhost:5000  

---

# 🔍 Test Prediction


curl -X POST http://localhost:8000/predict

-H "Content-Type: application/json"
-d '{"data": [1,2,3]}'


---

# 🔄 Automatic Retraining Flow

1. API receives prediction request  
2. PSI drift is calculated  
3. If drift exceeds threshold  
4. `train.py` starts  
5. New version registered  
6. Alias `production` updated  
7. API automatically serves new model  

Zero downtime.

---

# 📊 Current Status

ML Platform 3.0 — Stable

Implemented:

- Model versioning
- Alias-based deployment
- Drift detection
- Event-driven retraining
- Cooldown protection
- Dockerized environment

---

# 🚀 Next Planned Upgrade

ML Platform 3.5:

- Champion–Challenger evaluation
- RMSE comparison before alias switch
- Automatic rollback if worse

---

# 👨‍💻 Author

Built as a practical MLOps engineering project.
