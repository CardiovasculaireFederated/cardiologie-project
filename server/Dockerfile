FROM python:3.10-slim

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server /app/server
COPY common /app/common

ENV PYTHONPATH=/app

CMD ["python", "/app/server/kafka_server.py"]
