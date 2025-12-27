# server/kafka_server.py
import flwr as fl
import torch
import json
import threading
import os
import io
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
from kafka import KafkaProducer, KafkaConsumer

# Importations locales (assurez-vous que ces fichiers existent)
from strategy import CardioStrategy, get_weighted_average_fn
from data_base import get_best_model_weights, init_db
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
# Logique Chargement Modèle & Kafka
# --------------------------------------------------

def init_global_model():
    """Charge les poids initiaux depuis la DB ou le fichier .pth"""
    logger.info("Chargement du modèle global (PyTorch)...")
    weights_bytes, _ = get_best_model_weights()
    
    if weights_bytes is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Fichier modèle introuvable : {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    else:
        buffer = io.BytesIO(weights_bytes)
        state_dict = torch.load(buffer, map_location=torch.device("cpu"))
    
    # Conversion en dictionnaire de listes pour JSON si nécessaire
    model_weights = {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}
    return {"round": 0, "weights": model_weights}

def create_kafka_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8") if isinstance(v, dict) else v
    )

def create_kafka_consumer():
    return KafkaConsumer(
        CLIENT_WEIGHTS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="flower-server-group",
        value_deserializer=lambda v: v # Décodage manuel via decode_kafka_message
    )

def listen_client_updates(consumer):
    """Boucle de lecture Kafka tournant dans un thread séparé"""
    logger.info(f"Kafka Consumer démarré sur le topic : {CLIENT_WEIGHTS_TOPIC}")
    for message in consumer:
        try:
            # Utilise la logique de décodage du fichier kafka_server
            metadata, weights = decode_kafka_message(message.value)
            logger.info(f"Mise à jour reçue du client {metadata.get('client_id')} | Round {metadata.get('round_id')}")
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message Kafka : {e}")

# --------------------------------------------------
# Main : Lancement Flower + Kafka
# --------------------------------------------------

def main():
    # 1. Initialisation
    try:
        init_db()
        global_model_data = init_global_model()
    except Exception as e:
        logger.error(f"Erreur initialisation modèle : {e}")
        return

    producer = create_kafka_producer()
    consumer = create_kafka_consumer()

    # 2. Envoi du modèle initial via Kafka (pour les clients qui écoutent Kafka)
    try:
        # On peut envoyer soit le dictionnaire simple, soit le payload encodé
        producer.send(GLOBAL_MODEL_TOPIC, global_model_data)
        producer.flush()
        logger.info("Modèle global initial diffusé sur Kafka.")
    except Exception as e:
        logger.warning(f"Échec de l'envoi initial Kafka : {e}")

    # 3. Lancer l'écouteur Kafka en arrière-plan
    kafka_thread = threading.Thread(
        target=listen_client_updates,
        args=(consumer,),
        daemon=True
    )
    kafka_thread.start()

    # 4. Configuration de la stratégie Flower
    strategy = CardioStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=get_weighted_average_fn(),
    )

    # 5. Lancement du serveur Flower (gère les rounds et les clients connectés en RPC/gRPC)
    logger.info("--- Serveur Cardio_Federated prêt (Port 8080) ---")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
