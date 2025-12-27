import logging
import os
from typing import Any, Dict, List

import torch
from kafka import KafkaConsumer, KafkaProducer

from .data_loader import load_data
from .data_validator import DataValidatorAgent
from .model import HeartDiseaseModel
from .scout_agent import ScoutAgent
from .trainer import train_model
from common.kafka_topics import CLIENT_WEIGHTS_TOPIC, GLOBAL_MODEL_TOPIC
from common.serialization import decode_kafka_message, encode_kafka_message

logger = logging.getLogger("manager_agent")


def parse_bootstrap_servers(value: str) -> List[str]:
    if not value:
        return ["kafka:9092"]
    return [item.strip() for item in value.split(",") if item.strip()]


class ManagerAgent:
    def __init__(self) -> None:
        self.client_id = os.getenv("CLIENT_ID", "hospital_1")
        self.data_path = os.getenv("CLIENT_DATA_PATH", "data/processed/train.csv")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(os.getenv("CLIENT_BATCH_SIZE", "32"))
        self.base_epochs = int(os.getenv("CLIENT_EPOCHS", "5"))

        self.round_id = 0
        self.training_history: List[Dict[str, Any]] = []

        self.model = HeartDiseaseModel()
        self.validator = DataValidatorAgent(min_samples=100)
        self.scout = ScoutAgent(base_epochs=self.base_epochs)

        self.train_loader, self.val_loader, self.test_loader = load_data(
            file_path=self.data_path,
            batch_size=self.batch_size,
        )

        kafka_servers = parse_bootstrap_servers(
            os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        )
        self.consumer = KafkaConsumer(
            GLOBAL_MODEL_TOPIC,
            bootstrap_servers=kafka_servers,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda v: v,
        )

        self.producer = KafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda v: v,
        )

    def run(self) -> None:
        logger.info("Client is waiting for global model...")
        for message in self.consumer:
            self._handle_message(message.value)

    def _handle_message(self, message_value: bytes) -> None:
        try:
            meta, global_weights = decode_kafka_message(message_value)
        except Exception as exc:
            logger.error(f"Failed to decode global model message: {exc}")
            return

        global_round = meta.get("round_id")
        if isinstance(global_round, int):
            self.round_id = global_round + 1
        else:
            self.round_id += 1

        logger.info("Received global model from Kafka")
        logger.info(f"Starting training for round {self.round_id}")

        self.model.set_weights(global_weights)

        is_valid, report = self.validator.validate(self.train_loader)
        validation_report = {
            "status": "passed" if is_valid else "failed",
            "details": report,
        }

        if not is_valid:
            logger.error(f"Data validation failed: {report}")
            self._send_skip_update(validation_report, "data_validation_failed")
            return

        try:
            params = self.scout.suggest_hyperparameters(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                history=self.training_history,
                device=self.device,
            )
        except Exception as exc:
            logger.error(f"ScoutAgent failed: {exc}")
            params = {"epochs": self.base_epochs, "learning_rate": 1e-3}

        epochs = int(params.get("epochs", self.base_epochs))
        learning_rate = float(params.get("learning_rate", 1e-3))

        logger.info(f"Using hyperparameters: epochs={epochs}, lr={learning_rate}")

        try:
            result = train_model(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                epochs=epochs,
                learning_rate=learning_rate,
                device=self.device,
            )
        except Exception as exc:
            logger.error(f"Training failed: {exc}")
            self._send_skip_update(validation_report, "training_failed")
            return

        self.training_history.append(
            {
                "round": self.round_id,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "best_val_acc": result.get("best_val_acc"),
            }
        )

        metadata = {
            "client_id": self.client_id,
            "round_id": self.round_id,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "train_size": len(self.train_loader.dataset),
            "val_size": len(self.val_loader.dataset),
            "best_val_acc": result.get("best_val_acc") or 0.0,
            "data_validation": validation_report,
        }

        local_weights = result["weights"]
        kafka_message = encode_kafka_message(metadata, local_weights)
        self.producer.send(CLIENT_WEIGHTS_TOPIC, value=kafka_message)
        self.producer.flush()
        logger.info(f"Round {self.round_id} sent to Kafka by {self.client_id}")

    def _send_skip_update(self, validation_report: Dict[str, Any], reason: str) -> None:
        metadata = {
            "client_id": self.client_id,
            "round_id": self.round_id,
            "data_validation": validation_report,
            "skip_training": True,
            "skip_reason": reason,
        }
        kafka_message = encode_kafka_message(metadata, self.model.get_weights())
        self.producer.send(CLIENT_WEIGHTS_TOPIC, value=kafka_message)
        self.producer.flush()
