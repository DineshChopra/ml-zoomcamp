import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import xgboost as xgb

# Define the input data file and model artifact paths
data_file = "data/Assignment-2_Data.csv"  
artifact_base_path = "artifacts"
artifact_file = f"{artifact_base_path}/model.bin"
target = 'y'

# List of models with their configurations
model_infos = [
  {
    "name": "LR",
    "model": LogisticRegression(),
    "params": {
      "solver": ['liblinear'],
      "C": [1.0],
      "max_iter": [10, 50, 100]
    }
  },
  {
    "name": "DecisionTreeClassifier",
    "model": DecisionTreeClassifier(),
  },
  {
    "name": "RandomForestClassifier",
    "model": RandomForestClassifier(),
  },
  {
    "name": "XGBClassifier",
    "model": xgb.XGBClassifier(
      objective="binary:logistic",  # For binary classification
      random_state=42,              # Random seed for reproducibility
      n_jobs=-1,                    # Used for parallel processing, It will use all available CPU's
      use_label_encoder=False
    ),
    "params": {
      "n_estimators": [200],         # Number of boosting rounds
      "learning_rate": [0.1],    # Learning rate
      "max_depth": [5],                 # Maximum depth of each tree
    }
  },
  {
    "name": "LDA",
    "model": LinearDiscriminantAnalysis()
  },
  {
    "name": "NB",
    "model": GaussianNB()
  }
]

def read_data(data_file):
  """
  Read the input data from a CSV file.

  Args:
    data_file (str): Path to the input data file.

  Returns:
    pd.DataFrame: A pandas DataFrame containing the input data.
  """
  df = pd.read_csv(data_file)
  df.dropna(inplace=True)
  not_required_field = [ "Id", "default" ]
  df.drop(columns=not_required_field, inplace=True)
  df = df[df.age != 999.0]
  return df

def split_dataframe(df):
  """
  Split the input dataframe into train, validation and test sets
  
  Args:
    df (pd.DataFrame): Input DataFrame.
  
  Returns:
    pd.DataFrame: DataFrames for training, validation, and test sets.
  """
  df[target] = df[target].replace({'yes': 1, 'no': 0})
  df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
  df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

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
  """
  Preprocess the data including one-hot encoding and standardization.

  Args:
    df_train (pd.DataFrame): Training data.
    df_val (pd.DataFrame): Validation data.
    df_test (pd.DataFrame): Test data.

  Returns:
    Tuple: Tuple containing the DictVectorizer, StandardScaler, and preprocessed data.
  """
  dict_train = df_train.to_dict(orient='records')
  dict_val = df_val.to_dict(orient='records')
  dict_test = df_test.to_dict(orient='records')

  dv = DictVectorizer(sparse=False)
  X_train = dv.fit_transform(dict_train)
  X_val = dv.transform(dict_val)
  X_test = dv.transform(dict_test)

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_val = scaler.transform(X_val)
  X_test = scaler.transform(X_test)

  return dv, scaler, X_train, X_val, X_test

def resample_train_data(X_train, y_train):
  """
  Apply SMOTE to balance the training data

  Args:
    X_train
    y_train

  Returns:
    Tuple: containing the X_resampled, y_resampled
  """
  smote = SMOTE(sampling_strategy="auto", random_state=42)
  X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
  return X_resampled, y_resampled
  
def get_best_params_and_estimator(model_info, X_train, y_train, X_val, y_val):
  """
  Get best hyperparameters and estimator by using GridSearchCV

  Args:
    model_info: Model information containing model name, model object, set of parameters
    X_train: preprocessed training data set
    y_train: training label
    X_val: preprocessed validation data set
    y_val: validation label

  Returns:
    Tuple: Tuple containing best hyper parameters and best estimators
  """
  model = model_info['model']
  model_name = model_info['name']
  params = model_info.get('params', {})

  grid_search = GridSearchCV(model, params, cv=5)
  grid_search.fit(X_train, y_train)
  best_params = grid_search.best_params_
  best_estimator = grid_search.best_estimator_
  return best_params, best_estimator

def get_model_evaluation(model, X, y):
  """
  Get model evaluation report, It helps us to understand how our model is behaving

  Args:
    model: Model on which we have to evaluate model accuracy
    X: Dataset on which we will evaluate model accuracy
    y: Actual label (Truth label values)
  """
  y_pred = model.predict(X)
  acc = roc_auc_score(y, y_pred)
  return round(acc, 4)

def get_model_reports(model_infos, X_train, y_train, X_val, y_val):
  """
  Get the model report based on provided multiple model information

  Args:
    model_infos: Multiple model information
    X_train, y_train: Training dataset
    X_val, y_val: Validatioin dataset

  Returns:
    List: List of model reports containing, model name, model object, accuracy, hyperparameters
  """
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
  """
  Get best model based on accuracy

  Args:
    model_report: List of model reports

  Returns:
    best_model: Best model based on model accuracy
  """
  sorted_report = sorted(model_reports, key=lambda x: x['accuracy'], reverse=True)
  return sorted_report[0]

def save_artifacts(dictVectorizer, standardScaler, model, model_file):
  """
  Save artifacts, that can be used in web server to generate prediction

  Args:
    dictVectorizer: One hot encoder
    standardScaler: Standard Scaler
    model: Best model
    model_file: File name, in which model should be stored
  """
  pipeline = make_pipeline(dictVectorizer, standardScaler, model)
  with open(model_file,'wb') as f_out: 
    pickle.dump(pipeline, f_out)

def run ():
  df = read_data(data_file)
  df_train, y_train, df_val, y_val, df_test, y_test = split_dataframe(df)
  dv, scaler, X_train, X_val, X_test = pre_process_data(df_train, df_val, df_test)
  X_resampled, y_resampled = resample_train_data(X_train, y_train)
  model_report = get_model_reports(model_infos, X_resampled, y_resampled, X_val, y_val)
  best_model_info = get_best_model_info(model_report)
  print("best_model_info -- ", best_model_info)
  best_model = best_model_info["model"]
  save_artifacts(dv, scaler, best_model, artifact_file)

if __name__ == "__main__":
  run()
