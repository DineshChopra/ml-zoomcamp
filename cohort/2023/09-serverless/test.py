# import requests

# url="http://localhost:8080/2015-03-31/functions/function/invocations"

# data = {'url': 'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'}

# result = requests.post(url, json=data).json()
# print(result)

import requests

url = 'http://localhost:8081/2015-03-31/functions/function/invocations'

data = {'url': 'https://upload.wikimedia.org/wikipedia/en/e/e9/GodzillaEncounterModel.jpg'}

result = requests.post(url, json=data).json()
print(result)
