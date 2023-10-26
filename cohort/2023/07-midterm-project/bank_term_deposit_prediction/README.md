# Predict Term Deposit [Kaggle Link](https://www.kaggle.com/datasets/aslanahmedov/predict-term-deposit)


Reference project: https://github.com/bhasarma/mlcoursezoom-camp/tree/main/WK08-09-midterm-project/

## Table of Contents:
* 1. Buisness problem description [link](documentation/01-Buisness-problem-description.md)
* 2. Explore dataset EDA
* 3. Analyze feature importance
* 4. Train multiple models and find best model
* 5. Export notebook into script
* 6. Put your model into a web service and deploy it locally with Docker
* 7. Deploying the service to the cloud

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




