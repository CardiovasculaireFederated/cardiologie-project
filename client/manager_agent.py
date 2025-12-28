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
from .model_evaluator import ModelEvaluatorAgent
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
        self.evaluator = ModelEvaluatorAgent()

        self.train_loader, self.val_loader, self.test_loader = load_data(
            file_path=self.data_path,
            batch_size=self.batch_size,
        )

        kafka_servers = parse_bootstrap_servers(
            os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        )
        auto_offset_reset = os.getenv("CLIENT_KAFKA_AUTO_OFFSET_RESET", "earliest")
        group_id_raw = os.getenv("CLIENT_KAFKA_GROUP_ID", self.client_id).strip()
        group_id = group_id_raw if group_id_raw else None
        self.group_id = group_id
        max_poll_interval_ms = int(
            os.getenv("CLIENT_KAFKA_MAX_POLL_INTERVAL_MS", "7200000")
        )
        session_timeout_ms = int(
            os.getenv("CLIENT_KAFKA_SESSION_TIMEOUT_MS", "30000")
        )
        heartbeat_interval_ms = int(
            os.getenv("CLIENT_KAFKA_HEARTBEAT_INTERVAL_MS", "10000")
        )
        self.consumer = KafkaConsumer(
            GLOBAL_MODEL_TOPIC,
            bootstrap_servers=kafka_servers,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=False,
            group_id=group_id,
            max_poll_interval_ms=max_poll_interval_ms,
            session_timeout_ms=session_timeout_ms,
            heartbeat_interval_ms=heartbeat_interval_ms,
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

        eval_metrics: Dict[str, Any] = {}
        try:
            eval_metrics = self.evaluator.evaluate(
                model=self.model,
                data_loader=self.test_loader,
                device=self.device,
            )
        except Exception as exc:
            logger.error(f"ModelEvaluator failed: {exc}")

        metadata = {
            "client_id": self.client_id,
            "round_id": self.round_id,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "train_size": len(self.train_loader.dataset),
            "val_size": len(self.val_loader.dataset),
            "best_val_acc": result.get("best_val_acc") or 0.0,
            "data_validation": validation_report,
            "evaluation": eval_metrics,
        }

        local_weights = result["weights"]
        kafka_message = encode_kafka_message(metadata, local_weights)
        self.producer.send(CLIENT_WEIGHTS_TOPIC, value=kafka_message)
        self.producer.flush()
        logger.info(f"Round {self.round_id} sent to Kafka by {self.client_id}")

        if self.group_id is not None:
            try:
                self.consumer.commit()
            except Exception as exc:
                logger.warning(f"Kafka commit failed: {exc}")

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
