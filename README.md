ML Platform 4.0

Production-ready MLOps platform with:

- Drift Detection (PSI)
- Cooldown protection
- Automatic Retraining
- Model Versioning (MLflow)
- Staging → Production promotion flow
- Manual production control

---

#Architecture

Drift detected
→ Retrain
→ New model version
→ Assigned to @staging
→ Manual promotion
→ @production

Production model is NEVER auto-overwritten.

---

#Tech Stack

- FastAPI
- MLflow
- Scikit-learn
- Docker
- Docker Compose
- Nginx

---

#How to Run

##Build & Start

```bash
docker-compose build --no-cache
docker-compose up -d
Check services
docker-compose ps

Access
API:
http://localhost:8000

MLflow UI:
http://localhost:5000

  Prediction Endpoint
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"data": [1.0, 2.0, 3.0]}'
  Trigger Drift (for testing)
for i in {1..500}; do \
curl -s -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"data": [1000,1000,1000]}' > /dev/null; \
done

 This will:
-Detect drift
-Trigger retraining
-Register new model version
-Assign it to @staging

 Model Lifecycle
After retrain:
New version → @staging
Production remains unchanged.

 Promote Staging to Production
curl -X POST http://localhost:8000/promote
This will:
Move staging version to @production
Update production model
 MLflow Model Registry

 Open:
http://localhost:5000

Models → linear-model

 You will see:

@production
@staging

Full version history

 Production Safety
-No automatic overwrite
-Champion / Challenger logic
-Manual approval
-Version tracking
-Drift detection

 Restart API (if needed)
docker-compose restart api

 What This Platform Demonstrates
-Real MLOps pipeline
-Model versioning
-Production safety
-Controlled rollout
-Drift-based retraining
# ML Platform

This repository is my hands-on ML platform project.

It started as a simple API with a small model, but step by step I turned it into a more complete system that covers much more than just training and prediction. The main goal was to understand how a model actually lives in production: how it gets trained, versioned, deployed, monitored, retrained, and promoted.

This is not meant to be a polished enterprise product. It is a practical learning project that gradually evolved into a small but pretty realistic ML platform.

## What this project does

At the moment, the platform includes:

- model training
- model versioning with MLflow
- `staging` and `production` aliases
- ONNX export
- Triton Inference Server for serving
- dynamic batching
- multi-model serving
- A/B rollout
- shadow deployment
- drift detection
- retraining through a queue
- automatic promotion of a better model
- automatic deployment to Triton
- baseline reload and traffic window reset after production model changes

The idea was to build the platform layer around the model, not just the model itself.

---

## Main components

### 1. FastAPI Gateway

FastAPI is used as the external API layer.

It does not just call a Python model directly. Instead, it works as a gateway in front of Triton and handles traffic control logic.

It supports:
- direct routing to model A
- direct routing to model B
- A/B rollout
- shadow deployment

In other words, the client talks to the API, and the API decides which model should actually serve the request.

---

### 2. Triton Inference Server

Triton is the serving layer.

It is responsible for:
- loading ONNX models
- serving inference over HTTP
- hosting multiple models at the same time
- applying dynamic batching

This makes the serving path much closer to a real production inference stack than calling `model.predict()` inside the API process.

---

### 3. MLflow

MLflow is used as the model registry and experiment tracking layer.

It stores:
- training runs
- metrics
- model versions
- model aliases such as `staging` and `production`

This means the model is treated as a managed artifact with lifecycle state, not just as a local file on disk.

---

### 4. Drift Worker

The drift worker monitors production traffic and compares incoming feature distributions against the baseline of the current production model.

If significant drift is detected:
- it pushes a retraining job into a queue
- the trainer service picks it up
- a new model gets trained
- the new model is evaluated against the current production model

It also reloads the baseline and clears the collected production window when the production model changes, so the system does not keep retraining forever on old drifted data.

---

### 5. Trainer Service

The trainer is a separate service that listens for retraining jobs.

This separation is important:
- the drift worker only decides **when** retraining is needed
- the trainer is the component that actually **runs** training

That makes the architecture cleaner and closer to how real systems are structured.

---

### 6. Redis

Redis is used in two roles:

- as a shared collector for production traffic
- as a queue for retraining jobs

This is much better than keeping shared state inside the memory of one Python process or passing data around through temporary files.

---

### 7. Deployer

The deployer watches the `production` alias in MLflow.

When the production version changes, it:
- downloads the ONNX artifact
- places it into the Triton model repository
- waits for Triton reload
- runs a smoke test
- accepts the deployment if the smoke test passes

This makes the bridge between model registry and serving automatic.

---

## Architecture

The platform currently looks like this:

```text
Client
↓
FastAPI Gateway
↓
Triton Inference Server

FastAPI Gateway
↓
Redis collector

Drift Worker
↓
Redis retrain queue
↓
Trainer
↓
train.py
↓
MLflow Registry
↓
staging / production aliases
↓
Deployer
↓
Triton

Features already implemented:
Serving
FastAPI gateway
Triton serving
ONNX export
dynamic batching
multi-model serving
Traffic control
direct model routing
A/B rollout
shadow deployment
Model lifecycle
MLflow tracking
MLflow model registry
staging and production aliases
automatic promotion of a better candidate model

Monitoring and automation
Redis-backed traffic collector
drift detection
retraining queue
trainer service
deployer
smoke-test-gated deployment
baseline reload after production change
collector reset after production change

Tech stack:
Python
FastAPI
Triton Inference Server
MLflow
Redis
PostgreSQL
Docker
Docker Compose
scikit-learn
ONNX
skl2onnx
NumPy
Pandas

Running the project
1. Clone the repository
git clone https://github.com/RichardPervomaev/ml-platform.git
cd ml-platform
2. Start the services
docker compose up -d --build
3. Check the API
curl localhost:8003/health
4. Check Triton
curl localhost:8000/v2/health/ready
Useful endpoints
Gateway health
curl localhost:8003/health
Direct call to model A
curl -X POST localhost:8003/predict-a \
-H "Content-Type: application/json" \
-d '{"data":[1.0,2.0,3.0]}'
Direct call to model B
curl -X POST localhost:8003/predict-b \
-H "Content-Type: application/json" \
-d '{"data":[1.0,2.0,3.0]}'
A/B rollout
curl -X POST localhost:8003/predict-ab \
-H "Content-Type: application/json" \
-d '{"data":[1.0,2.0,3.0]}'
Shadow deployment
curl -X POST localhost:8003/predict-shadow \
-H "Content-Type: application/json" \
-d '{"data":[1.0,2.0,3.0]}'
Why I built this

The main point of this project was not just to train one model.

I wanted to understand the full production path of a model:
how a model is versioned
how it is promoted
how it is deployed
how traffic is routed
how drift is detected
how retraining is triggered
how serving and lifecycle management fit together
So this project is really about building the platform around the model.
What I want to add next

Possible next steps:
Kubernetes manifests for the current services
Network Policies
KServe or Seldon
Terraform for infrastructure
alerting and observability
rollback policy improvements
GPU resource isolation
more realistic trainer workflows
