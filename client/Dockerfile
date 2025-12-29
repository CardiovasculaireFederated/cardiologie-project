FROM python:3.10-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

COPY client/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir streamlit
COPY client /app/client

COPY common /app/common

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PYTHONPATH=/app
EXPOSE 8501

CMD ["streamlit", "run", "client/streamlit_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
