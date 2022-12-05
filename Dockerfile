
FROM python:3.10.6-buster
#FROM tensorflow/tensorflow:2.10.0
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY earth_project /earth_project
CMD uvicorn earth_project.pred_api:app --host 0.0.0.0 --port 8000

#CMD uvicorn earth_project.pred_api:app --host 0.0.0.0 --port $PORT pour après



# lancer docker
# docker build -t [nom  du projet gcr] apitest .
# docker run -p 8080:8000 apitest
