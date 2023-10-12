import requests

url = "http://127.0.0.1:9696/predict"

req = {"job": "retired", "duration": 445, "poutcome": "success"}

res = requests.post(url=url, json=req).json()
print("response --- ", res)