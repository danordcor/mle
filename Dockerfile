# syntax=docker/dockerfile:1.2
FROM python:3.11 AS main

WORKDIR /app
ENV PYTHONPATH=/app

COPY ./challenge /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY ./data /data

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM main as dev

COPY requirements-test.txt /app/requirements-test.txt
COPY ./tests /app/tests
RUN pip install -r requirements-test.txt
