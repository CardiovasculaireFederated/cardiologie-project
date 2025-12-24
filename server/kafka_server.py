# server/kafka_server.py

import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
from dotenv import load_dotenv
from kafka import KafkaConsumer, KafkaProducer

from common.kafka_topics import CLIENT_WEIGHTS_TOPIC, GLOBAL_MODEL_TOPIC
from common.serialization import decode_kafka_message, encode_kafka_message

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kafka_server")

# --------------------------------------------------
# Config
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "best_model.pth"))

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
AGGREGATION_WINDOW_SECONDS = float(
    os.getenv("AGGREGATION_WINDOW_SECONDS", "2")
)
KAFKA_POLL_TIMEOUT_MS = int(
    os.getenv("KAFKA_POLL_TIMEOUT_MS", "500")
)

# --------------------------------------------------
# Federated Config
# --------------------------------------------------
MIN_CLIENTS_PER_ROUND = 1
SLEEP_BETWEEN_ROUNDS = float(os.getenv("SLEEP_BETWEEN_ROUNDS", "0"))


def parse_bootstrap_servers(value: str) -> List[str]:
    if not value:
        return ["kafka:9092"]
    return [item.strip() for item in value.split(",") if item.strip()]


def load_initial_weights(model_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    return torch.load(model_path, map_location=torch.device("cpu"))


def create_consumer(bootstrap_servers: List[str]) -> KafkaConsumer:
    return KafkaConsumer(
        CLIENT_WEIGHTS_TOPIC,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda v: v,
    )


def create_producer(bootstrap_servers: List[str]) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: v,
    )


def aggregate_weights(weights_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not weights_list:
        return {}

    aggregated: Dict[str, torch.Tensor] = {}
    for key in weights_list[0].keys():
        aggregated[key] = torch.stack(
            [w[key] for w in weights_list], dim=0
        ).mean(dim=0)
    return aggregated


def validate_metadata(
    metadata: Dict,
    current_round: int,
    received_clients: set,
) -> bool:
    required_fields = [
        "client_id",
        "round_id",
        "epochs",
        "learning_rate",
        "train_size",
        "val_size",
    ]

    for field in required_fields:
        if field not in metadata:
            logger.error(f"Metadata missing field: {field}")
            return False

    if metadata["round_id"] != current_round:
        logger.warning(
            "Round mismatch: client_round=%s server_round=%s",
            metadata["round_id"],
            current_round,
        )
        return False

    client_id = metadata["client_id"]
    if client_id in received_clients:
        logger.warning(
            "Duplicate update from client %s in round %s",
            client_id,
            current_round,
        )
        return False

    if metadata["epochs"] <= 0:
        logger.error("Invalid epochs value")
        return False

    if metadata["learning_rate"] <= 0:
        logger.error("Invalid learning_rate value")
        return False

    if metadata["train_size"] <= 0 or metadata["val_size"] <= 0:
        logger.error("Invalid dataset sizes")
        return False

    return True


def build_global_metadata(round_id: int) -> Dict:
    return {
        "round_id": round_id,
        "message_type": "global_model",
    }


def send_global_model(
    producer: KafkaProducer,
    weights: Dict[str, torch.Tensor],
    round_id: int,
) -> None:
    metadata = build_global_metadata(round_id)
    payload = encode_kafka_message(metadata, weights)
    producer.send(GLOBAL_MODEL_TOPIC, value=payload)
    producer.flush()
    logger.info("Global model sent to Kafka (%s)", GLOBAL_MODEL_TOPIC)


def should_aggregate(
    window_start: float | None,
    buffer_size: int,
) -> bool:
    if buffer_size < MIN_CLIENTS_PER_ROUND:
        return False
    if AGGREGATION_WINDOW_SECONDS <= 0:
        return True
    if window_start is None:
        return False
    return (time.time() - window_start) >= AGGREGATION_WINDOW_SECONDS


def summarize_round(
    current_round: int,
    buffer_metadata: List[Dict],
    round_start_time: float,
    invalid_messages: int,
) -> None:
    round_duration = time.time() - round_start_time

    val_accs = [
        m.get("best_val_acc", 0.0)
        for m in buffer_metadata
        if m.get("best_val_acc") is not None
    ]

    if val_accs:
        avg_acc = sum(val_accs) / len(val_accs)
        min_acc = min(val_accs)
        max_acc = max(val_accs)
    else:
        avg_acc = 0.0
        min_acc = 0.0
        max_acc = 0.0

    logger.info(
        "[ROUND %s SUMMARY] clients=%s | avg_acc=%.4f | min_acc=%.4f | "
        "max_acc=%.4f | duration=%.2fs | invalid_msgs=%s",
        current_round,
        len(buffer_metadata),
        avg_acc,
        min_acc,
        max_acc,
        round_duration,
        invalid_messages,
    )


def main() -> None:
    current_round = 1
    received_clients: set = set()

    bootstrap_servers = parse_bootstrap_servers(KAFKA_BOOTSTRAP_SERVERS)
    consumer = create_consumer(bootstrap_servers)
    producer = create_producer(bootstrap_servers)

    logger.info("Kafka Federated Server started (Sprint 2)")

    try:
        initial_weights = load_initial_weights(MODEL_PATH)
        send_global_model(producer, initial_weights, round_id=0)
    except Exception as exc:
        logger.error(f"Failed to send initial global model: {exc}")
        raise

    buffer_weights: List[Dict[str, torch.Tensor]] = []
    buffer_metadata: List[Dict] = []
    round_start_time = time.time()
    invalid_messages = 0
    window_start: float | None = None

    while True:
        records = consumer.poll(timeout_ms=KAFKA_POLL_TIMEOUT_MS)

        if should_aggregate(window_start, len(buffer_weights)):
            logger.info("Aggregating client weights...")
            global_weights = aggregate_weights(buffer_weights)

            if not global_weights:
                logger.error("Aggregation failed, skipping round")
            else:
                send_global_model(
                    producer,
                    global_weights,
                    round_id=current_round,
                )
                summarize_round(
                    current_round,
                    buffer_metadata,
                    round_start_time,
                    invalid_messages,
                )

            buffer_weights.clear()
            buffer_metadata.clear()
            received_clients.clear()
            current_round += 1
            round_start_time = time.time()
            window_start = None

            if SLEEP_BETWEEN_ROUNDS > 0:
                time.sleep(SLEEP_BETWEEN_ROUNDS)

            continue

        for _, messages in records.items():
            for message in messages:
                try:
                    metadata, weights = decode_kafka_message(message.value)
                except Exception as exc:
                    logger.error(f"Failed to parse client message: {exc}")
                    invalid_messages += 1
                    continue

                client_id = metadata.get("client_id", "unknown")
                sentinel_meta = metadata.get("sentinel", {})
                decision = sentinel_meta.get("decision", "accept")

                if decision == "reject":
                    logger.warning(
                        "[Sentinel][Server] client=%s decision=reject -> skipped",
                        client_id,
                    )
                    continue
                if decision == "early_stop":
                    logger.warning(
                        "[Sentinel][Server] client=%s decision=early_stop -> accepted",
                        client_id,
                    )
                else:
                    logger.info(
                        "[Sentinel][Server] client=%s decision=accept -> accepted",
                        client_id,
                    )

                data_validation = metadata.get("data_validation", {})
                if metadata.get("skip_training") or data_validation.get("status") == "failed":
                    logger.info(
                        "Client %s skipped training for round %s",
                        client_id,
                        metadata.get("round_id"),
                    )
                    continue

                if not validate_metadata(metadata, current_round, received_clients):
                    logger.warning("Invalid metadata, skipping update")
                    invalid_messages += 1
                    continue

                received_clients.add(client_id)
                buffer_weights.append(weights)
                buffer_metadata.append(metadata)

                if window_start is None:
                    window_start = time.time()

                logger.info(
                    "Received update from client=%s round=%s val_acc=%s",
                    client_id,
                    metadata.get("round_id"),
                    metadata.get("best_val_acc"),
                )


if __name__ == "__main__":
    main()
