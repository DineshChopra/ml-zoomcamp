import pickle

from flask import Flask, request, jsonify

# model_file = 'model1.bin'
model_file = 'model2.bin' # This is used in docker image Question 6
preprocess_file = "dv.bin"

app = Flask("Bank Credit Scoring")
model = None
dv = None

def load_model():
  global model
  global dv
  with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

  with open(preprocess_file, 'rb') as f_in:
    dv = pickle.load(f_in)

@app.route("/predict", methods=["POST"])
def predict():
  req = request.get_json()

  X = dv.transform([req])
  y_pred = model.predict_proba(X)[0, 1]
  y_pred = round(y_pred, 3)
  result = {
    "scoring": y_pred
  }
  return jsonify(result)


load_model()

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=9696)