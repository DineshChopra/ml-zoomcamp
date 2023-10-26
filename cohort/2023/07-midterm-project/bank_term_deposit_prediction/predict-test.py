#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'

# 44.0	technician	single	secondary	no	29.0	yes	no	unknown	5	may	151	1	-1	0	unknown	
customer_not_takinig_term_deposit = {
  "Id": 101,
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


customer_takinig_term_deposit = {
  "age": 48.0,
  "job": "services",
  "marital": "married",
  "education": "secondary",
  "balance": 22.0,
  "housing": "no",
  "loan": "no",
  "contact": "cellular",
  "day": 2,
  "month": "feb",
  "duration": 429,
  "campaign": 2,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown",
}

response = requests.post(url, json=customer_takinig_term_deposit).json()
print(response)