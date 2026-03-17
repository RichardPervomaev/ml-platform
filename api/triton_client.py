import requests
import numpy as np

TRITON_URL = "http://localhost:8000/v2/models/linear_model/infer"


def predict(data):

    payload = {
        "inputs": [
            {
                "name": "INPUT",
                "shape": [3],
                "datatype": "FP32",
                "data": data
            }
        ]
    }

    r = requests.post(TRITON_URL, json=payload)

    if r.status_code != 200:
        raise Exception(r.text)

    result = r.json()

    return result
