
#FROM ptyhon:3.10.6-buster
FROM tensorflow/tensorflow:2.10.0
COPY earth-project /earth-project
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn pred_api:app --host 0.0.0.0 --port 8000
