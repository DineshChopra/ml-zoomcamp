import pickle

from flask import Flask, request, jsonify

def load(filename: str):
  with open(filename, 'rb') as f_in:
    return pickle.load(f_in)


dv = load('dv.bin')
model = load('model1.bin')

app = Flask('Credit Card Fraud Detection')

@app.route('/', methods=['GET'])
def home():
  return '<h1>Welcome to Credit card Fraud Detection</h1>'

@app.route('/predict', methods=['POST'])
def predict():
  customer = request.get_json()
  X = dv.transform([customer])

  y_pred = model.predict_proba(X)[0][1]
  result = {
    "card_probability": float(y_pred)
  }
  return jsonify(result)

if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0', port=9696)
