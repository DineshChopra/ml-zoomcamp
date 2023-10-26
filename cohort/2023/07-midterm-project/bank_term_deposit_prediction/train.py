import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

artifact_base_path = "artifacts"
artifact_file = f"{artifact_base_path}/model.bin"

target = 'y'
model_infos = [
  {
    "name": "LR",
    "model": LogisticRegression(),
    "params": {
      "solver": ['liblinear'],
      "C": [1.0],
      "max_iter": [100, 500, 1000]
    }
  },
  {
    "name": "DecisionTreeClassifier",
    "model": DecisionTreeClassifier(),
  },
  {
    "name": "LDA",
    "model": LinearDiscriminantAnalysis()
  },
  {
    "name": "NB",
    "model": GaussianNB()
  },
  {
    "name": "SVM",
    "model": SVC()
  }
]

def read_data(data_file):
  return pd.read_csv(data_file)

def filter_data(df: pd.DataFrame):
  not_required_field = [ "Id", "default" ]
  for field in not_required_field:
    del df[field]
  df.dropna(inplace=True)
  df = df[df.age != 999.0]
  return df


def split_dataframe(df):    
  df[target] = df[target].replace({'yes': 1, 'no': 0})
  df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
  df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

  len(df_train), len(df_val), len(df_test)

  df_train = df_train.reset_index(drop=True)
  df_val = df_val.reset_index(drop=True)
  df_test = df_test.reset_index(drop=True)

  y_train = df_train[target].values
  y_val = df_val[target].values
  y_test = df_test[target].values

  del df_train[target]
  del df_val[target]
  del df_test[target]
  return df_train, y_train, df_val, y_val, df_test, y_test

def pre_process_data(df_train, df_val, df_test):
  dict_train = df_train.to_dict(orient='records')
  dict_val = df_val.to_dict(orient='records')
  dict_test = df_test.to_dict(orient='records')

  dv = DictVectorizer(sparse=False)
  X_train = dv.fit_transform(dict_train)
  X_val = dv.transform(dict_val)
  X_test = dv.transform(dict_test)
  return dv, X_train, X_val, X_test
  
def get_best_params_and_estimator(model_info, X_train, y_train, X_val, y_val):
  model = model_info['model']
  model_name = model_info['name']
  params = model_info.get('params', {})

  grid_search = GridSearchCV(model, params, cv=5)
  grid_search.fit(X_train, y_train)
  best_params = grid_search.best_params_
  best_estimator = grid_search.best_estimator_
  return best_params, best_estimator

def get_model_evaluation(model, X, y):
  y_pred = model.predict(X)
  acc = accuracy_score(y, y_pred)
  return round(acc, 4)

def get_model_reports(model_infos, X_train, y_train, X_val, y_val):
  model_reports = []
  for model_info in model_infos:
    best_params, best_estimator = get_best_params_and_estimator(model_info, X_train, y_train, X_val, y_val)
    accuracy = get_model_evaluation(best_estimator, X_val, y_val)
    
    print(f"Model: {model_info['name']}, accuracy: {accuracy}")
    
    model_report = {
      "name": model_info["name"],
      "model": best_estimator,
      "best_params": best_params,
      "accuracy": accuracy
    }
    model_reports.append(model_report)

  return model_reports

def get_best_model_info(model_reports):
  sorted_report = sorted(model_reports, key=lambda x: x['accuracy'], reverse=True)
  return sorted_report[0]

def save_artifacts(preprocessor, model, model_file):
  pipeline = make_pipeline(preprocessor, model)
  with open(model_file,'wb') as f_out: 
    pickle.dump(pipeline, f_out)

def run ():
  data_file = "data/Assignment-2_Data.csv"
  df = read_data(data_file)
  df = filter_data(df)
  print("----------------- ", df.shape)
  df_train, y_train, df_val, y_val, df_test, y_test = split_dataframe(df)
  dv, X_train, X_val, X_test = pre_process_data(df_train, df_val, df_test)
  model_report = get_model_reports(model_infos, X_train, y_train, X_val, y_val)
  best_model_info = get_best_model_info(model_report)
  print("best_model_info -- ", best_model_info)
  best_model = best_model_info["model"]
  save_artifacts(dv, best_model, artifact_file)

if __name__ == "__main__":
  run()
