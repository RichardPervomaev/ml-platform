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
