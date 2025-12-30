# server/kafka_server.py

import json
import logging
import time
from typing import List, Dict

import torch
from kafka import KafkaConsumer, KafkaProducer

from common.serialization import (
    bytes_to_weights,
    weights_to_bytes,
)

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kafka_server")

# --------------------------------------------------
# Kafka Config
# --------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS = ["kafka:9092"]

CLIENT_WEIGHTS_TOPIC = "client_weights"
GLOBAL_MODEL_TOPIC = "global_model"

# --------------------------------------------------
# Federated Config (Prototype)
# --------------------------------------------------
MIN_CLIENTS_PER_ROUND = 2
SLEEP_BETWEEN_ROUNDS = 2  # seconds

def create_consumer():
    return KafkaConsumer(
        CLIENT_WEIGHTS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda v: v,
    )


def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: v,
    )

def parse_client_message(message_bytes: bytes):
    """
    Expected format:
    b'<json metadata>|||<weights bytes>'
    """
    try:
        meta_part, weights_part = message_bytes.split(b"|||", 1)
        metadata = json.loads(meta_part.decode("utf-8"))
        weights = bytes_to_weights(weights_part)

        return metadata, weights

    except Exception as e:
        logger.error(f"Failed to parse client message: {e}")
        return None, None


def aggregate_weights(weights_list: List[Dict[str, torch.Tensor]]):
    """
    Simple average of model weights (FedAvg prototype).
    """
    if not weights_list:
        return None

    aggregated = {}

    for key in weights_list[0].keys():
        aggregated[key] = torch.stack(
            [w[key] for w in weights_list], dim=0
        ).mean(dim=0)

    return aggregated

def validate_metadata(
    metadata: Dict,
    current_round: int,
    received_clients: set,
):
    """
    Validate client metadata before aggregation.
    """
    required_fields = [
        "client_id",
        "round_id",
        "epochs",
        "learning_rate",
        "train_size",
        "val_size",
    ]

    # 1. Check required fields
    for field in required_fields:
        if field not in metadata:
            logger.error(f"Metadata missing field: {field}")
            return False

    # 2. Round consistency
    if metadata["round_id"] != current_round:
        logger.warning(
            f"Round mismatch: client_round={metadata['round_id']} "
            f"server_round={current_round}"
        )
        return False

    # 3. Unique client per round
    client_id = metadata["client_id"]
    if client_id in received_clients:
        logger.warning(
            f"Duplicate update from client {client_id} in round {current_round}"
        )
        return False

    # 4. Numeric sanity checks
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



def main():
    current_round = 1
    received_clients = set()

    consumer = create_consumer()
    producer = create_producer()

    logger.info("Kafka Federated Server started (Sprint 2 Prototype)")

    buffer_weights = []
    buffer_metadata = []

    round_start_time = time.time()
    invalid_messages = 0


    for message in consumer:
        metadata, weights = parse_client_message(message.value)

        if metadata is None or weights is None:
            continue

        if not validate_metadata(metadata, current_round, received_clients):
            logger.warning("Invalid metadata, skipping update")
            invalid_messages += 1
            continue

        received_clients.add(metadata["client_id"])


        logger.info(
            f"Received update from client={metadata.get('client_id')} "
            f"round={metadata.get('round_id')} "
            f"val_acc={metadata.get('best_val_acc')}"
        )

        buffer_weights.append(weights)
        buffer_metadata.append(metadata)

        if len(buffer_weights) < MIN_CLIENTS_PER_ROUND:
            logger.info(
                f"Waiting for more clients "
                f"({len(buffer_weights)}/{MIN_CLIENTS_PER_ROUND})"
            )
            continue

        # -------- Aggregation --------
        logger.info("Aggregating client weights...")
        global_weights = aggregate_weights(buffer_weights)

        if global_weights is None:
            logger.error("Aggregation failed, skipping round")
            buffer_weights.clear()
            buffer_metadata.clear()
            continue

        # -------- Send global model --------
        payload = weights_to_bytes(global_weights)

        producer.send(GLOBAL_MODEL_TOPIC, value=payload)
        producer.flush()

        logger.info(
            f"Global model sent to Kafka ({GLOBAL_MODEL_TOPIC})"
        )


        round_duration = time.time() - round_start_time

        val_accs = [
            m.get("best_val_acc", 0.0)
            for m in buffer_metadata
            if m.get("best_val_acc") is not None
        ]

        avg_acc = sum(val_accs) / len(val_accs) if val_accs else 0.0

        logger.info(
            f"[ROUND {current_round} SUMMARY] "
            f"clients={len(buffer_metadata)} | "
            f"avg_acc={avg_acc:.4f} | "
            f"min_acc={min(val_accs):.4f} | "
            f"max_acc={max(val_accs):.4f} | "
            f"duration={round_duration:.2f}s | "
            f"invalid_msgs={invalid_messages}"
        )


        # Reset buffers for next round
        buffer_weights.clear()
        buffer_metadata.clear()

        received_clients.clear()
        current_round += 1


        time.sleep(SLEEP_BETWEEN_ROUNDS)





if __name__ == "__main__":
    main()
