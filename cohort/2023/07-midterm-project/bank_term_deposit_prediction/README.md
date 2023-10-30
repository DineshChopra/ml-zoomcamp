# Predict Term Deposit [Kaggle Link](https://www.kaggle.com/datasets/aslanahmedov/predict-term-deposit)

![Predict Term Deposit](./images/dataset-cover.png)

## Table of Contents:
* 1. [Problem description](#1-problem-description)
* 2. [Explore dataset Analysis EDA](#2-eda)
* 3. [Model training](#3-model-training)
* 4. [Exporting notebook to script](#4-exporting-notebook-to-script)
* 5. [Reproducibility](#5-reproducibility)
* 6. [Model deployment](#6-model-deployment)
* 7. [Dpendency and enviroment management](#7-dependency-and-enviroment-management)
* 8. [Containerization](#8-containerization)
* 9. [Cloud deployment](#9-cloud-deployment)

## 1. Problem description
Bank has multiple banking products that it sells to customer such as saving account, credit cards, investments etc. It wants to which customer will purchase its credit cards. For the same it has various kind of information regarding the demographic details of the customer, their banking behavior etc. Once it can predict the chances that customer will purchase a product, it wants to use the same to make pre-payment to the authors.

In this part I will demonstrate how to build a model, to predict which clients will subscribing to a `Term Deposit`, with inception of machine learning. 

In the ﬁrst part we will deal with the description and visualization of the analysed data, and in the second we will go to data classiﬁcation models.

This dataset contains demographic and banking information about customers and also the outcome of the campaign, i.e. whether they subscribed to the product after the campaign or not. 

In this project, we want to train a model on this dataset in order to predict whether after a targeted campaign, a particular customer will subscribed to the product 'term deposit' or not. Since we want our model to predict yes or no, this is a binary classification problem.

## 2. EDA
  * [Kaggle Dataset](https://www.kaggle.com/datasets/aslanahmedov/predict-term-deposit)
  * [wget link](https://raw.githubusercontent.com/DineshChopra/ml-zoomcamp/main/cohort/2023/07-midterm-project/bank_term_deposit_prediction/data/Assignment-2_Data.csv)

  ```bash
  !wget https://raw.githubusercontent.com/DineshChopra/ml-zoomcamp/main/cohort/2023/07-midterm-project/bank_term_deposit_prediction/data/Assignment-2_Data.csv
  ```


## 3. Model training

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




