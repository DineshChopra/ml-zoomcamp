import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'artifacts/model.bin'

with open(model_file, 'rb') as f_in:
  pipeline = pickle.load(f_in)

app = Flask('Term_Deposite_Prediction')

def remove_not_required_fields(customer):
  if "Id" in customer:
    customer.pop("Id")

  if "default" in customer:
    customer.pop("default")
  return customer  

@app.route('/predict', methods=['POST'])
def predict():
  customer = request.get_json()

  customer = remove_not_required_fields(customer) 

  y_pred = pipeline.predict_proba(customer)[0, 1]

  result = {
    'term_deposite_prediction': float(y_pred),
  }

  return jsonify(result)


if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0', port=9696)
