# client/client.py
import os
import json
import logging
import torch
from kafka import KafkaConsumer, KafkaProducer
from client.data_validator import DataValidatorAgent

from client.model import HeartDiseaseModel
from client.data_loader import load_data
from client.trainer import train_model
from common.serialization import (
    weights_to_bytes,
    bytes_to_weights,
)
from client.scout_agent import suggest_hyperparameters

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("client")
validator = DataValidatorAgent(min_samples=100)

# --------------------------------------------------
# Config
# --------------------------------------------------
DATA_PATH = "/data/heart.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

CLIENT_ID = os.getenv("CLIENT_ID", "hospital_1")
ROUND_ID = 0

# --------------------------------------------------
# Model & History
# --------------------------------------------------
model = HeartDiseaseModel()
training_history = []

# --------------------------------------------------
# Kafka
# --------------------------------------------------
consumer = KafkaConsumer(
    "global_model",
    bootstrap_servers=["kafka:9092"],
    auto_offset_reset="latest",
    enable_auto_commit=True,
    value_deserializer=lambda v: v,
)

producer = KafkaProducer(
    bootstrap_servers=["kafka:9092"],
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
    ROUND_ID += 1
    logger.info(f"Received global model from Kafka")
    logger.info(f"Starting training for round {ROUND_ID}")

    # 1. Deserialize global weights
    global_weights = bytes_to_weights(message.value)

    # 2. Update local model
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
            "data_validation": validation_report
        }

        kafka_message = json.dumps({"metadata": metadata}).encode("utf-8")
        producer.send("client_weights", value=kafka_message)
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
    payload = weights_to_bytes(local_weights)

    kafka_message = (
        json.dumps({"metadata": metadata}).encode("utf-8")
        + b"|||"
        + payload
    )

    # 7. Send to Kafka
    producer.send("client_weights", value=kafka_message)
    producer.flush()

    logger.info(f"Round {ROUND_ID} sent to Kafka by {CLIENT_ID}")
