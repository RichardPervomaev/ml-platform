import requests
import threading

URL = "http://localhost:8000/predict"

def send_request():
    r = requests.post(URL, json={"value": 5})
    print(r.json())

threads = []

for _ in range(20):
    t = threading.Thread(target=send_request)
    threads.append(t)
    t.start()

for t in threads:
    t.join()
