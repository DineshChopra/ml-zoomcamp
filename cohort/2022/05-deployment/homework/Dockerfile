# FROM python:3.9.12-slim
FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

# WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["model1.bin", "dv.bin", "q4_predict.py", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "q4_predict:app" ]
