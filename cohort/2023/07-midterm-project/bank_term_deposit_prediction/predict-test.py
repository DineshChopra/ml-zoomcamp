#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'

# 44.0	technician	single	secondary	no	29.0	yes	no	unknown	5	may	151	1	-1	0	unknown	
customer_not_takinig_term_deposit = {
  "age": 44.0,
  "job": "technician",
  "marital": "single",
  "education": "secondary",
  "default": "no",
  "balance": 29.0,
  "housing": "yes",
  "loan": "no",
  "contact": "unknown",
  "day": 5,
  "month": "may",
  "duration": 151,
  "campaign": 1,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown",
}


response = requests.post(url, json=customer_not_takinig_term_deposit).json()
print(response)

# 59.0	admin.	married	secondary	2343.0	yes	no	unknown	5	may	1042	1	-1	0	unknown	1
customer_takinig_term_deposit = {
  "age": 59.0,
  "job": "admin",
  "marital": "married",
  "education": "secondary",
  "default": "no",
  "balance": 2343.0,
  "housing": "yes",
  "loan": "no",
  "contact": "unknown",
  "day": 5,
  "month": "may",
  "duration": 1042,
  "campaign": 1,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown",
}

response = requests.post(url, json=customer_takinig_term_deposit).json()
print(response)