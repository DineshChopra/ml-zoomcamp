# Predict Term Deposit [Kaggle Link](https://www.kaggle.com/datasets/aslanahmedov/predict-term-deposit)

![Predict Term Deposit](./images/dataset-cover.png)

## Table of Contents:
1. [Problem description](#1-problem-description)
2. [Explore dataset Analysis EDA](#2-eda)
3. [Model training](#3-model-training)
4. [Exporting notebook to script](#4-exporting-notebook-to-script)
5. [Reproducibility](#5-reproducibility)
6. [Model deployment](#6-model-deployment)
7. [Dpendency and enviroment management](#7-dependency-and-enviroment-management)
8. [Containerization](#8-containerization)
9. [Cloud deployment](#9-cloud-deployment)

## 1. Problem description
Bank has multiple banking products that it sells to customer such as saving account, credit cards, investments etc. It wants to which customer will purchase its credit cards. For the same it has various kind of information regarding the demographic details of the customer, their banking behavior etc. Once it can predict the chances that customer will purchase a product, it wants to use the same to make pre-payment to the authors.

In this part I will demonstrate how to build a model, to predict which clients will subscribing to a `Term Deposit`, with inception of machine learning. 

In the ﬁrst part we will deal with the description and visualization of the analysed data, and in the second we will go to data classiﬁcation models.

This dataset contains demographic and banking information about customers and also the outcome of the campaign, i.e. whether they subscribed to the product after the campaign or not. 

In this project, we want to train a model on this dataset in order to predict whether after a targeted campaign, a particular customer will subscribed to the product 'term deposit' or not. Since we want our model to predict yes or no, this is a binary classification problem.

## 2. EDA [notebook](./notebook_eda.ipynb)
  * [Kaggle Dataset](https://www.kaggle.com/datasets/aslanahmedov/predict-term-deposit)
  * [wget link](https://raw.githubusercontent.com/DineshChopra/ml-zoomcamp/main/cohort/2023/07-midterm-project/bank_term_deposit_prediction/data/Assignment-2_Data.csv)

  ```bash
  !wget https://raw.githubusercontent.com/DineshChopra/ml-zoomcamp/main/cohort/2023/07-midterm-project/bank_term_deposit_prediction/data/Assignment-2_Data.csv
  ```

  * This dataset contains `45211` records with `18` features. Feature details are as Id,	age, job,	marital,	education,	default,	balance,	housing,	loan,	contact,	day,	month,	duration,	campaign,	pdays,	previous,	poutcome,	y,
  * Here `y` is the target label and rest are the dependent features.

  * Read dataset
  * Identify outliers in `age` and remove it
  * Handle null values
  * Understand `age` distribution
  * Understand `job` distrubution
  * Analyze `marital`
  * Analyze `education`
  * Analyze `default`
  * Analyze relationship between `age` and `balance`
  * Understand relationship between `age` and `loan`

## 3. Model training [notebook](./notebook_model_training.ipynb)

  * Import required Libraries
  * Read dataset
  * Split Dataframe into `train`, `val` and `test` sets by using `train_test_split` method of `sklearn.model_selection`
  * Preprocess dataset
    * Convert dataframe into dict
    * Apply one hot coding by using  `DictVectorizer`
    * Apply normalization by using `StandardScaler`
  * Resample train dataset
    * As our data set is unbalnced dataset, so to make it balance we can use `SMOTE`.
  * Get Model Reports (train accracy, validation accuracy) by passing multiple models
    * Get best hyperparameters and best estimator by using `GridSearchCV`
    * Get Model evaluation by finding out best validation accuracy from model reports
    * ![Model Report](./images/model_training_without_smote.png)
  * Get Best model based on `validation accuracy`
  * Save artifacts
    * Create a `pipeline` of `dictVectorizer`, `standardScaler` and `model`
    * export that pipeline as `artifacts/model.bin` by using `pickle`

  * Best Model without `SMOTE` is `LinearDiscriminantAnalysis`. Model report is:
    * Model: LDA, train_accuracy: `0.6958`, validation accuracy: `0.7173`
    * ![Best Model](./images/best_model_without_smote.png)
  
  * Best model with `SMOTE` is `LogisticRegression` with validation accuracy is `0.84`
  * ![Best Model](./images/model_training_with_smote.png)


## 4. Exporting notebook to script

## 5. Reproducibility

## 6. Model deployment

## 7. Dependency and enviroment management

## 8. Containerization

## 9. Cloud deployment




-------------------------

Reference project: https://github.com/bhasarma/mlcoursezoom-camp/tree/main/WK08-09-midterm-project/



## Install dependencies
```
pipenv install
pipenv shell
```
## Train best model
```
python train.py
```

## Build docker image [ref](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/06-docker.md)
```
docker build -t bank-term-deposite:latest .
```

## To run it, execute the command below.
```
docker run -it -p 9696:9696 bank-term-deposite:latest
```
## Create repository in docker hub e.g `bank-term-deposit`

## Publish docker image to docker hub
```
docker login

docker push dineshchopra/bank-term-deposit:latest
```

## Pull docker image and run it
```
docker pull dineshchopra/bank-term-deposit:latest
```

## Run docker image, which is pulled from docker hub
```
docker run -it -p 9696:9696 dineshchopra/bank-term-deposit:latest
```




