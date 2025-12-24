# client/client.py
import os
import logging
from typing import List

import torch
from kafka import KafkaConsumer, KafkaProducer

from .data_validator import DataValidatorAgent

from .model import HeartDiseaseModel
from .data_loader import load_data
from .trainer import train_model
from common.kafka_topics import CLIENT_WEIGHTS_TOPIC, GLOBAL_MODEL_TOPIC
from common.serialization import decode_kafka_message, encode_kafka_message
from .scout_agent import suggest_hyperparameters

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("client")
validator = DataValidatorAgent(min_samples=100)

# --------------------------------------------------
def parse_bootstrap_servers(value: str) -> List[str]:
    if not value:
        return ["kafka:9092"]
    return [item.strip() for item in value.split(",") if item.strip()]


# --------------------------------------------------
# Config
# --------------------------------------------------
DATA_PATH = os.getenv("CLIENT_DATA_PATH", "data/processed/train.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.getenv("CLIENT_BATCH_SIZE", "32"))

CLIENT_ID = os.getenv("CLIENT_ID", "hospital_1")
ROUND_ID = 0
KAFKA_BOOTSTRAP_SERVERS = parse_bootstrap_servers(
    os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
)

# --------------------------------------------------
# Model & History
# --------------------------------------------------
model = HeartDiseaseModel()
training_history = []

# --------------------------------------------------
# Kafka
# --------------------------------------------------
consumer = KafkaConsumer(
    GLOBAL_MODEL_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    auto_offset_reset="latest",
    enable_auto_commit=True,
    value_deserializer=lambda v: v,
)

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: v,
)

# --------------------------------------------------
# Data
# --------------------------------------------------
logger.info("Loading local dataset...")
train_loader, val_loader, test_loader = load_data(
    file_path=DATA_PATH,
    batch_size=BATCH_SIZE
)

logger.info("Client is waiting for global model...")

# --------------------------------------------------
# Main loop
# --------------------------------------------------
for message in consumer:
    try:
        meta, global_weights = decode_kafka_message(message.value)
    except Exception as exc:
        logger.error(f"Failed to decode global model message: {exc}")
        continue

    global_round = meta.get("round_id")
    if isinstance(global_round, int):
        ROUND_ID = global_round + 1
    else:
        ROUND_ID += 1

    logger.info("Received global model from Kafka")
    logger.info(f"Starting training for round {ROUND_ID}")

    # 1. Update local model
    model.set_weights(global_weights)

    # 3. Data validation
    is_valid, report = validator.validate(train_loader)

    validation_report = {
        "status": "passed" if is_valid else "failed",
        "details": report
    }

    if not is_valid:
        logger.error(f"Data validation failed: {report}")
        metadata = {
            "client_id": CLIENT_ID,
            "round_id": ROUND_ID,
            "data_validation": validation_report,
            "skip_training": True,
        }

        kafka_message = encode_kafka_message(metadata, model.get_weights())
        producer.send(CLIENT_WEIGHTS_TOPIC, value=kafka_message)
        producer.flush()
        continue


    # 4. Scout Agent decides hyperparameters
    params = suggest_hyperparameters(training_history)
    epochs = params["epochs"]
    learning_rate = params["learning_rate"]

    logger.info(
        f"Using hyperparameters → epochs={epochs}, lr={learning_rate}"
    )

    

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=DEVICE
    )

    # 5. Update history
    training_history.append({
        "round": ROUND_ID,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "best_val_acc": result.get("best_val_acc", None)
    })

    # Metadata
    metadata = {
        "client_id": CLIENT_ID,
        "round_id": ROUND_ID,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "best_val_acc": result.get("best_val_acc") or 0.0,
        "data_validation": validation_report
    }

    # 6. Serialize updated weights
    local_weights = result["weights"]
    kafka_message = encode_kafka_message(metadata, local_weights)

    # 7. Send to Kafka
    producer.send(CLIENT_WEIGHTS_TOPIC, value=kafka_message)
    producer.flush()

    logger.info(f"Round {ROUND_ID} sent to Kafka by {CLIENT_ID}")
