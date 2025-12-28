FROM python:3.10-slim

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server /app/server
COPY common /app/common
COPY client/ /app/client/

ENV PYTHONPATH=/app

CMD ["python", "-m", "server.kafka_server"]
