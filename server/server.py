import flwr as fl
from strategy import CardioStrategy, get_weighted_average_fn

def main():
  
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
