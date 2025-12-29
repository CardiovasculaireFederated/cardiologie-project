import json
import logging
import os
import threading
from typing import Any, Dict, List

import torch
from kafka import KafkaConsumer, KafkaProducer
from prometheus_client import Gauge, start_http_server

from .data_loader import load_data, load_processed_data
from .data_validator import DataValidatorAgent
from .model import HeartDiseaseModel
from .scout_agent import ScoutAgent
from .trainer import train_model
from .model_evaluator import ModelEvaluatorAgent
from common.kafka_topics import (
    CLIENT_EVENTS_TOPIC,
    CLIENT_WEIGHTS_TOPIC,
    GLOBAL_MODEL_TOPIC,
)
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
        self.log_path = os.getenv("CLIENT_LOG_PATH")
        self.data_format = os.getenv("CLIENT_DATA_FORMAT", "processed").lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(os.getenv("CLIENT_BATCH_SIZE", "32"))
        self.base_epochs = int(os.getenv("CLIENT_EPOCHS", "5"))

        self.round_id = 0
        self.current_global_version = 0
        self.training_history: List[Dict[str, Any]] = []

        self.model = HeartDiseaseModel()
        min_samples = int(os.getenv("CLIENT_MIN_SAMPLES", "100"))
        self.validator = DataValidatorAgent(min_samples=min_samples)
        self.scout = ScoutAgent(base_epochs=self.base_epochs)
        self.evaluator = ModelEvaluatorAgent()
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self._configure_file_logging()
        self._init_metrics()

        kafka_servers = parse_bootstrap_servers(
            os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        )
        auto_offset_reset = os.getenv("CLIENT_KAFKA_AUTO_OFFSET_RESET", "earliest")
        group_id_raw = os.getenv("CLIENT_KAFKA_GROUP_ID", "").strip()
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

        self.event_consumer = KafkaConsumer(
            CLIENT_EVENTS_TOPIC,
            bootstrap_servers=kafka_servers,
            auto_offset_reset="latest",
            enable_auto_commit=False,
            group_id=None,
            value_deserializer=lambda v: v,
        )

    def _init_metrics(self) -> None:
        metrics_port = int(os.getenv("CLIENT_METRICS_PORT", "9101"))
        start_http_server(metrics_port)

        self.metric_train_acc = Gauge(
            "client_train_accuracy", "Latest training accuracy"
        )
        self.metric_val_acc = Gauge(
            "client_val_accuracy", "Latest validation accuracy"
        )
        self.metric_test_acc = Gauge(
            "client_test_accuracy", "Latest test accuracy"
        )
        self.metric_xai_score = Gauge(
            "client_xai_score", "Latest XAI score (placeholder)"
        )
        self.metric_last_round = Gauge(
            "client_last_round", "Latest completed round"
        )
        self.metric_training_active = Gauge(
            "client_training_active", "Training status (1=active, 0=idle)"
        )

    def run(self) -> None:
        threading.Thread(target=self._listen_events, daemon=True).start()
        logger.info("Client is waiting for global model...")
        for message in self.consumer:
            self._handle_message(message.value)

    def _handle_message(self, message_value: bytes) -> None:
        try:
            meta, global_weights = decode_kafka_message(message_value)
        except Exception as exc:
            logger.error(f"Failed to decode global model message: {exc}")
            return

        target_client = meta.get("target_client_id")
        if target_client and target_client != self.client_id:
            return

        self.current_global_version = int(
            meta.get("global_version", meta.get("round_id", 0))
        )
        global_round = meta.get("round_id")
        if isinstance(global_round, int):
            self.round_id = global_round + 1
        else:
            self.round_id += 1

        request_data_path = meta.get("data_path")
        if request_data_path:
            self.data_path = request_data_path
        logger.info("Received global model from Kafka")
        logger.info(f"Starting training for round {self.round_id}")
        self.metric_training_active.set(1)

        self.model.set_weights(global_weights)

        if not self._load_data():
            self._send_skip_update(
                {"status": "failed", "details": "data_missing"},
                "data_missing",
            )
            self.metric_training_active.set(0)
            return

        is_valid, report = self.validator.validate(self.train_loader)
        validation_report = {
            "status": "passed" if is_valid else "failed",
            "details": report,
        }

        if not is_valid:
            logger.error(f"Data validation failed: {report}")
            self._send_skip_update(validation_report, "data_validation_failed")
            self.metric_training_active.set(0)
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
            self.metric_training_active.set(0)
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

        history = result.get("history", {})
        if history.get("train_acc"):
            self.metric_train_acc.set(history["train_acc"][-1])
        if history.get("val_acc"):
            self.metric_val_acc.set(history["val_acc"][-1])
        if eval_metrics.get("accuracy") is not None:
            self.metric_test_acc.set(eval_metrics["accuracy"])
        self.metric_xai_score.set(eval_metrics.get("xai_score", 0.0))
        self.metric_last_round.set(self.round_id)

        metadata = {
            "client_id": self.client_id,
            "round_id": self.round_id,
            "base_model_version": self.current_global_version,
            "request_id": meta.get("request_id"),
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
        self.metric_training_active.set(0)

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

    def _load_data(self) -> bool:
        if not os.path.exists(self.data_path):
            logger.error(f"Data file not found: {self.data_path}")
            return False
        try:
            loader_fn = load_data
            if self.data_format in ("processed", "spark"):
                loader_fn = load_processed_data
            elif self.data_format == "auto":
                try:
                    self.train_loader, self.val_loader, self.test_loader = (
                        load_processed_data(
                            file_path=self.data_path,
                            batch_size=self.batch_size,
                        )
                    )
                    return True
                except Exception:
                    loader_fn = load_data

            self.train_loader, self.val_loader, self.test_loader = loader_fn(
                file_path=self.data_path,
                batch_size=self.batch_size,
            )
            return True
        except Exception as exc:
            logger.error(f"Failed to load data: {exc}")
            return False

    def _configure_file_logging(self) -> None:
        if not self.log_path:
            return
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                if getattr(handler, "baseFilename", "") == os.path.abspath(self.log_path):
                    return

        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s:%(name)s:%(message)s"
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    def _listen_events(self) -> None:
        for message in self.event_consumer:
            try:
                payload = json.loads(message.value.decode("utf-8"))
            except Exception:
                continue
            target = payload.get("client_id")
            if target and target != self.client_id:
                continue
            logger.info(f"Server decision: {payload}")
