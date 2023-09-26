import requests

url = 'http://127.0.0.1:9696/predict'
req = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
res = requests.post(url, json=req).json()

print(res)
