# server/kafka_server.py
import torch
import json
import threading
import os
import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from kafka import KafkaProducer, KafkaConsumer
from prometheus_client import Counter, Gauge, start_http_server

from .data_base import (
    get_best_model_weights,
    init_db,
    save_global_model,
    save_dataset,
    get_dataset,
)
from .evaluation import evaluate_global_model
from client.model import HeartDiseaseModel
from common.kafka_topics import (
    CLIENT_WEIGHTS_TOPIC,
    CLIENT_EVENTS_TOPIC,
    GLOBAL_MODEL_TOPIC,
    TRAINING_REQUESTS_TOPIC,
)
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
TEST_DATASET_NAME = os.getenv("TEST_DATASET_NAME", "global_test_set")
TEST_DATASET_PATH = os.getenv(
    "TEST_PATH", str(BASE_DIR / "server/data_test/test.csv")
)
FORCE_FRESH_MODEL = os.getenv("FORCE_FRESH_MODEL", "0").lower() in (
    "1",
    "true",
    "yes",
)

POLL_INTERVAL_SEC = float(os.getenv("TRAINING_POLL_INTERVAL_SEC", "10"))
AGGREGATION_WAIT_SEC = float(os.getenv("AGGREGATION_WAIT_SEC", "15"))
MODEL_ACCEPT_DELTA = float(os.getenv("MODEL_ACCEPT_DELTA", "0.005"))


def init_global_model() -> Dict[str, torch.Tensor]:
    logger.info("Loading global model (PyTorch)...")
    if FORCE_FRESH_MODEL:
        logger.info("Force fresh model enabled. Ignoring DB/file.")
        return HeartDiseaseModel().state_dict()
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


def create_kafka_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: v,
    )


def create_updates_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        CLIENT_WEIGHTS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="latest",
        enable_auto_commit=False,
        group_id=None,
        value_deserializer=lambda v: v,
    )


def create_requests_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        TRAINING_REQUESTS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="latest",
        enable_auto_commit=False,
        group_id=None,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )


def aggregate_updates(
    updates: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    total_examples = sum(update["num_examples"] for update in updates)
    if total_examples <= 0:
        total_examples = len(updates)

    base_state = updates[0]["weights"]
    aggregated: Dict[str, torch.Tensor] = {}
    for key in base_state.keys():
        weighted_sum = None
        for update in updates:
            weight_tensor = update["weights"][key]
            weight_factor = update["num_examples"] / total_examples
            contrib = weight_tensor * weight_factor
            weighted_sum = contrib if weighted_sum is None else weighted_sum + contrib
        aggregated[key] = weighted_sum
    return aggregated


def init_metrics() -> Dict[str, Any]:
    metrics_port = int(os.getenv("KAFKA_SERVER_METRICS_PORT", "9200"))
    start_http_server(metrics_port)

    return {
        "global_accuracy": Gauge(
            "server_global_accuracy", "Current global model accuracy"
        ),
        "candidate_accuracy": Gauge(
            "server_candidate_accuracy", "Latest candidate model accuracy"
        ),
        "global_version": Gauge(
            "server_global_version", "Current global model version"
        ),
        "candidate_clients": Gauge(
            "server_candidate_clients", "Number of updates in the candidate"
        ),
        "candidate_accepted": Gauge(
            "server_candidate_accepted", "Last candidate accepted (1/0)"
        ),
        "last_decision_ts": Gauge(
            "server_last_decision_ts", "Unix timestamp of last decision"
        ),
        "inflight_clients": Gauge(
            "server_inflight_clients", "Expected clients in inflight round"
        ),
        "received_updates": Gauge(
            "server_received_updates", "Received updates in inflight round"
        ),
        "training_requests": Counter(
            "server_training_requests_total", "Training requests received"
        ),
        "updates_accepted": Counter(
            "server_updates_accepted_total", "Accepted candidate updates"
        ),
        "updates_rejected": Counter(
            "server_updates_rejected_total", "Rejected candidate updates"
        ),
    }


def ensure_global_test_dataset() -> None:
    try:
        if get_dataset(TEST_DATASET_NAME):
            return
        if not os.path.exists(TEST_DATASET_PATH):
            logger.warning(
                f"Global test dataset not found at {TEST_DATASET_PATH}."
            )
            return
        with open(TEST_DATASET_PATH, "rb") as handle:
            payload = handle.read()
        save_dataset(TEST_DATASET_NAME, payload)
        logger.info("Global test dataset stored in database.")
    except Exception as exc:
        logger.warning(f"Failed to store global test dataset: {exc}")


def main() -> None:
    metrics = init_metrics()
    try:
        init_db()
        ensure_global_test_dataset()
        current_state = init_global_model()
    except Exception as e:
        logger.error(f"Model init error: {e}")
        return

    if FORCE_FRESH_MODEL:
        current_accuracy = 0.0
        logger.info("Initial global accuracy forced to 0.0 (fresh model).")
    else:
        try:
            current_accuracy = evaluate_global_model(current_state)
            logger.info(f"Initial global accuracy: {current_accuracy:.4f}")
        except Exception as exc:
            logger.warning(f"Global evaluation failed: {exc}")
            current_accuracy = 0.0

    current_version = 0
    metrics["global_accuracy"].set(current_accuracy)
    metrics["global_version"].set(current_version)
    metrics["candidate_accepted"].set(0)
    inflight_version: Optional[int] = None
    expected_clients: Set[str] = set()
    received_updates: List[Dict[str, Any]] = []
    skipped_clients: Set[str] = set()
    first_update_ts: Optional[float] = None
    pending_requests: Dict[str, Dict[str, Any]] = {}

    state_lock = threading.Lock()

    producer = create_kafka_producer()
    updates_consumer = create_updates_consumer()
    requests_consumer = create_requests_consumer()

    def send_global_to_clients(
        targets: List[str], request_meta: Dict[str, Dict[str, Any]], version: int
    ) -> None:
        for client_id in targets:
            request_payload = request_meta.get(client_id, {})
            metadata = {
                "round_id": version,
                "global_version": version,
                "target_client_id": client_id,
                "dispatch_ts": time.time(),
                "request_id": request_payload.get("request_id"),
                "data_path": request_payload.get("data_path"),
            }
            payload = encode_kafka_message(metadata, current_state)
            producer.send(GLOBAL_MODEL_TOPIC, value=payload)
        producer.flush()
        logger.info(f"Dispatched global model v{version} to {targets}")

    def finalize_inflight(reason: str) -> None:
        nonlocal current_state, current_version, current_accuracy
        nonlocal inflight_version, expected_clients, received_updates, first_update_ts

        with state_lock:
            updates_snapshot = list(received_updates)
            version_snapshot = inflight_version
            inflight_version = None
            expected_clients = set()
            received_updates = []
            skipped_clients.clear()
            first_update_ts = None

        if not updates_snapshot or version_snapshot is None:
            logger.info("No updates to finalize.")
            return

        if len(updates_snapshot) == 1:
            candidate_state = updates_snapshot[0]["weights"]
            logger.info("Evaluating single candidate update.")
        else:
            candidate_state = aggregate_updates(updates_snapshot)
            logger.info("Evaluating aggregated candidate update.")

        try:
            candidate_accuracy = evaluate_global_model(candidate_state)
        except Exception as exc:
            logger.error(f"Candidate evaluation failed: {exc}")
            return

        accepted = candidate_accuracy >= current_accuracy + MODEL_ACCEPT_DELTA
        metrics["candidate_accuracy"].set(candidate_accuracy)
        metrics["candidate_clients"].set(len(updates_snapshot))
        metrics["candidate_accepted"].set(1 if accepted else 0)
        metrics["last_decision_ts"].set(time.time())
        if accepted:
            current_state = candidate_state
            current_accuracy = candidate_accuracy
            current_version += 1
            metrics["global_accuracy"].set(current_accuracy)
            metrics["global_version"].set(current_version)
            metrics["updates_accepted"].inc()
            try:
                save_global_model(current_version, current_state, current_accuracy)
            except Exception as exc:
                logger.warning(f"Failed to save global model: {exc}")
            logger.info(
                f"Global model updated to v{current_version} "
                f"(acc={current_accuracy:.4f})."
            )
        else:
            metrics["updates_rejected"].inc()
            logger.info(
                f"Candidate rejected (acc={candidate_accuracy:.4f}, "
                f"current={current_accuracy:.4f})."
            )

        logger.info(f"Finalize reason: {reason}")
        metrics["inflight_clients"].set(0)
        metrics["received_updates"].set(0)

        for update in updates_snapshot:
            event = {
                "client_id": update["client_id"],
                "accepted": accepted,
                "candidate_accuracy": candidate_accuracy,
                "current_accuracy": current_accuracy,
                "base_version": version_snapshot,
                "new_version": current_version if accepted else version_snapshot,
                "reason": reason,
            }
            try:
                producer.send(
                    CLIENT_EVENTS_TOPIC,
                    value=json.dumps(event).encode("utf-8"),
                )
            except Exception as exc:
                logger.warning(f"Failed to send client event: {exc}")
        producer.flush()

    def listen_training_requests() -> None:
        for message in requests_consumer:
            payload = message.value
            if not isinstance(payload, dict):
                continue
            client_id = payload.get("client_id")
            if not client_id:
                continue
            request_id = payload.get("request_id") or payload.get("timestamp")
            data_path = payload.get("data_path")
            with state_lock:
                pending_requests[client_id] = {
                    "request_id": request_id,
                    "data_path": data_path,
                }
            metrics["training_requests"].inc()
            logger.info(f"Training request received from {client_id}")

    def schedule_dispatch_loop() -> None:
        nonlocal inflight_version, expected_clients
        nonlocal received_updates, first_update_ts, skipped_clients

        while True:
            time.sleep(POLL_INTERVAL_SEC)
            finalize = False
            targets: List[str] = []
            request_snapshot: Dict[str, Dict[str, Any]] = {}
            with state_lock:
                if inflight_version is not None:
                    if (
                        first_update_ts
                        and time.time() - first_update_ts >= AGGREGATION_WAIT_SEC
                    ):
                        logger.info("Aggregation window expired.")
                        finalize = True
                else:
                    if pending_requests:
                        targets = sorted(pending_requests.keys())
                        request_snapshot = dict(pending_requests)
                        pending_requests.clear()
                        inflight_version = current_version
                        expected_clients = set(targets)
                        received_updates = []
                        skipped_clients = set()
                        first_update_ts = None
                        metrics["inflight_clients"].set(len(expected_clients))
                        metrics["received_updates"].set(0)

            if finalize:
                finalize_inflight("timeout")
            elif targets:
                send_global_to_clients(targets, request_snapshot, inflight_version)

    def listen_client_updates() -> None:
        nonlocal first_update_ts

        for message in updates_consumer:
            try:
                metadata, weights = decode_kafka_message(message.value)
            except Exception as exc:
                logger.error(f"Failed to decode client update: {exc}")
                continue

            if metadata.get("skip_training"):
                client_id = metadata.get("client_id")
                base_version_raw = metadata.get(
                    "base_model_version", metadata.get("round_id")
                )
                try:
                    base_version = int(base_version_raw)
                except (TypeError, ValueError):
                    base_version = None

                logger.info(
                    f"Client {client_id} skipped training "
                    f"({metadata.get('skip_reason')})."
                )
                if (
                    client_id is None
                    or base_version is None
                    or inflight_version is None
                    or base_version != inflight_version
                ):
                    continue
                with state_lock:
                    if client_id not in expected_clients:
                        continue
                    if client_id in skipped_clients:
                        continue
                    skipped_clients.add(client_id)
                    metrics["received_updates"].set(
                        len(received_updates) + len(skipped_clients)
                    )
                    if first_update_ts is None:
                        first_update_ts = time.time()
                    all_received = (
                        len(received_updates) + len(skipped_clients)
                        >= len(expected_clients)
                    )

                if all_received:
                    finalize_inflight("all_updates")
                continue

            client_id = metadata.get("client_id")
            base_version_raw = metadata.get(
                "base_model_version", metadata.get("round_id")
            )
            try:
                base_version = int(base_version_raw)
            except (TypeError, ValueError):
                base_version = None
            if client_id is None or base_version is None:
                logger.warning("Update missing client_id or base_model_version.")
                continue

            with state_lock:
                if inflight_version is None or base_version != inflight_version:
                    logger.info(
                        f"Stale update from {client_id} for v{base_version}."
                    )
                    continue
                if client_id not in expected_clients:
                    logger.info(
                        f"Unexpected update from {client_id} for v{base_version}."
                    )
                    continue

                if any(update["client_id"] == client_id for update in received_updates):
                    logger.info(f"Duplicate update from {client_id} ignored.")
                    continue

                received_updates.append(
                    {
                        "client_id": client_id,
                        "weights": weights,
                        "num_examples": int(metadata.get("train_size", 1)),
                    }
                )
                metrics["received_updates"].set(len(received_updates))
                if first_update_ts is None:
                    first_update_ts = time.time()

                all_received = len(received_updates) >= len(expected_clients)

            if all_received:
                finalize_inflight("all_updates")

    # Start background threads
    threading.Thread(target=listen_training_requests, daemon=True).start()
    threading.Thread(target=schedule_dispatch_loop, daemon=True).start()
    threading.Thread(target=listen_client_updates, daemon=True).start()

    logger.info("--- Cardio Federated async server ready ---")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
