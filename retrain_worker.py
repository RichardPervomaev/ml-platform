import subprocess
import logging
import time

logging.basicConfig(level=logging.INFO)

def retrain():
    logging.info("Starting retraining...")
    subprocess.run(["python", "train.py"])
    logging.info("Retraining finished.")

if __name__ == "__main__":
    while True:
        time.sleep(3600)
