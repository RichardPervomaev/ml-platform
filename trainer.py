import os
import json
import time
import logging
import subprocess
import redis

# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==========================================
# REDIS CONFIG
# ==========================================
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Очередь задач retrain
RETRAIN_QUEUE_KEY = os.getenv("RETRAIN_QUEUE_KEY", "retrain_jobs")

# Как долго ждать новую задачу в BLPOP
QUEUE_BLOCK_TIMEOUT = int(os.getenv("QUEUE_BLOCK_TIMEOUT", "5"))

client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)


def run_training_job(job_payload: dict):
    """
    Запускаем train.py как отдельный процесс.

    Здесь trainer отвечает только за выполнение training job.
    Drift worker сюда уже не лезет.
    """
    logger.warning("Starting training job with payload=%s", job_payload)

    result = subprocess.run(
        ["python", "train.py"],
        capture_output=True,
        text=True
    )

    logger.info("Training stdout:\n%s", result.stdout)

    if result.stderr:
        logger.warning("Training stderr:\n%s", result.stderr)

    if result.returncode != 0:
        logger.error("Training job failed with code=%s", result.returncode)
    else:
        logger.info("Training job finished successfully")


def run():
    """
    Главный цикл trainer service.

    Логика:
    1. Ждём задачу в Redis queue
    2. Когда задача приходит — запускаем train.py
    3. Логируем результат
    """
    logger.info("Trainer service started")
    logger.info("Listening queue=%s", RETRAIN_QUEUE_KEY)

    while True:
        try:
            item = client.blpop(RETRAIN_QUEUE_KEY, timeout=QUEUE_BLOCK_TIMEOUT)

            if item is None:
                continue

            _, raw_payload = item
            job_payload = json.loads(raw_payload)

            logger.info("Received retrain job: %s", job_payload)

            run_training_job(job_payload)

        except Exception as e:
            logger.exception("Trainer loop failed: %s", e)
            time.sleep(2)


if __name__ == "__main__":
    run()
