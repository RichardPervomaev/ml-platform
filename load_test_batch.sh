#!/bin/bash

for i in {1..20}
do
  curl -s -X POST localhost:8000/v2/models/linear_model_onnx/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs":[
      {
        "name":"input",
        "shape":[1,3],
        "datatype":"FP32",
        "data":[[1.0, 2.0, 3.0]]
      }
    ]
  }' > /dev/null &
done

wait
echo "Done"
