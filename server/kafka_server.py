# server/kafka_server.py
import flwr as fl
import torch
import json
import threading
import os
import io
import logging
from pathlib import Path
from dotenv import load_dotenv
from kafka import KafkaProducer, KafkaConsumer

from .strategy import CardioStrategy, get_weighted_average_fn
from .data_base import get_best_model_weights, init_db
from client.model import HeartDiseaseModel
from common.kafka_topics import CLIENT_WEIGHTS_TOPIC, GLOBAL_MODEL_TOPIC
from common.serialization import decode_kafka_message, encode_kafka_message

# --------------------------------------------------
# Configuration & Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("federated_server")

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "best_model.pth"))
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

# --------------------------------------------------
# Model loading
# --------------------------------------------------

def init_global_model():
    """Load initial weights from DB, file, or a fresh model."""
    logger.info("Loading global model (PyTorch)...")
    weights_bytes, _ = get_best_model_weights()

    if weights_bytes is None:
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
            logger.info("Loaded initial model from file.")
        else:
            logger.warning(
                f"Model file not found at {MODEL_PATH}. "
                "Using a fresh model initialization."
            )
            state_dict = HeartDiseaseModel().state_dict()
    else:
        buffer = io.BytesIO(weights_bytes)
        state_dict = torch.load(buffer, map_location=torch.device("cpu"))
        logger.info("Loaded initial model from database.")

    return state_dict


def create_kafka_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: v
    )


def create_kafka_consumer():
    return KafkaConsumer(
        CLIENT_WEIGHTS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="latest",
        enable_auto_commit=False,
        group_id=None,
        value_deserializer=lambda v: v
    )


def listen_client_updates(consumer):
    logger.info(f"Kafka Consumer started on topic: {CLIENT_WEIGHTS_TOPIC}")
    for message in consumer:
        try:
            metadata, weights = decode_kafka_message(message.value)
            logger.info(
                f"Update received from {metadata.get('client_id')} "
                f"| Round {metadata.get('round_id')}"
            )
        except Exception as e:
            logger.error(f"Kafka message processing error: {e}")


def main():
    try:
        init_db()
        global_weights = init_global_model()
    except Exception as e:
        logger.error(f"Model init error: {e}")
        return

    producer = create_kafka_producer()
    consumer = create_kafka_consumer()

    try:
        metadata = {
            "round_id": 0,
            "source": "server_init",
        }
        payload = encode_kafka_message(metadata, global_weights)
        producer.send(GLOBAL_MODEL_TOPIC, payload)
        producer.flush()
        logger.info("Initial global model broadcast on Kafka.")
    except Exception as e:
        logger.warning(f"Initial Kafka send failed: {e}")

    kafka_thread = threading.Thread(
        target=listen_client_updates,
        args=(consumer,),
        daemon=True,
    )
    kafka_thread.start()

    strategy = CardioStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=get_weighted_average_fn(),
    )

    logger.info("--- Cardio Federated server ready (Port 8080) ---")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
