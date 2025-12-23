import flwr as fl
from strategy import CardioStrategy, get_weighted_average_fn
from data_base import get_best_model_weights
from kafka import KafkaProducer, KafkaConsumer
import json
import threading
from dotenv import load_dotenv
from pathlib import Path
import os
import io
import torch


BASE_DIR = Path(__file__).resolve().parent.parent  # cardiologie-project/
load_dotenv(BASE_DIR / ".env")

MODEL_PATH = os.getenv("MODEL_PATH")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")


MODEL_PATH = MODEL_PATH


def init_global_model():
    print(" Chargement du modèle global (PyTorch)")
    weights_bytes, model_name=get_best_model_weights()
    if weights_bytes is None:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu")
        )

        model_weights = {
            k: v.cpu().numpy().tolist()
            for k, v in state_dict.items()
        }

        return {
            "round": 0,
            "weights": model_weights
        }
    else :
        buffer = io.BytesIO(weights_bytes)
        state_dict = torch.load(
            buffer,
            map_location=torch.device("cpu")
        )
        return{
             "round": 0,
             "weights": state_dict
        }

def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

def create_consumer():
    return KafkaConsumer(
        "client_weights",
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="flower-server-group"
    )

def listen_client_updates(consumer):
    print(" Kafka Consumer démarré (client_weights)")
    for message in consumer:
        client_update = message.value
        print(" Mise à jour reçue d’un client")
        print(client_update.keys())

def main():


    global_model = init_global_model()


    producer = create_producer()
    consumer = create_consumer()
    
    # Envoi du modèle global initial
    producer.send("global_model", global_model)
    producer.flush()
    print(" Modèle global initial envoyé via Kafka")

    #  Lancer le consumer Kafka dans un thread
    kafka_thread = threading.Thread(
        target=listen_client_updates,
        args=(consumer,),
        daemon=True
    )
    kafka_thread.start()


    strategy = CardioStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=get_weighted_average_fn(),
    )

    print("--- Serveur Cardio_Federated prêt (Port 8080) ---")


    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
