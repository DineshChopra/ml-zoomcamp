import requests

url = "http://127.0.0.1:9696/predict"

# Question 3
# req = {"job": "retired", "duration": 445, "poutcome": "success"}

# res = requests.post(url=url, json=req).json()
# print("response --- ", res)

# Question 4
# client = {"job": "unknown", "duration": 270, "poutcome": "failure"}
# res = requests.post(url=url, json=client).json()
# print('response : ', res)

# Question 6
client = {"job": "retired", "duration": 445, "poutcome": "success"}
res = requests.post(url, json=client).json()
print("Response : ", res)
