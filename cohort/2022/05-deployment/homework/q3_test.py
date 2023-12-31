import pickle

def load(filename):
  # with open(filename, 'rb') as f_in:
  #   return pickle.load(f_in)
  with open(filename, 'rb') as f_in:
    return pickle.load(f_in)


def predict():
  # Load Artifacts
  dv = load('dv.bin')
  model = load('model1.bin')

  obj =   {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
  req = [obj]
  X = dv.transform(req)
  y_pred = model.predict_proba(X)[0][1]
  print(y_pred)

def main():
  predict()

main()
