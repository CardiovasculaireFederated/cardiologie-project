FROM python:3.10-slim

WORKDIR /app

COPY client/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY client /app/client

COPY common /app/common

ENV PYTHONPATH=/app

CMD ["python", "-m", "client.main"]
