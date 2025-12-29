import flwr as fl
from typing import List, Tuple

from flwr.common import Metrics

from .evaluation import evaluate_global_model
from .data_base import save_global_model


class CardioStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_counter = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        """Run after clients finish training."""
        aggregated_weights, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_weights is not None:
            accuracy = evaluate_global_model(aggregated_weights)
            print(f"Round {server_round} : Aggregation complete.")
            model_name = f"model_{self._model_counter}_{server_round}"
            save_global_model(model_name, aggregated_weights, accuracy)
            self._model_counter += 1
        return aggregated_weights, aggregated_metrics


def get_weighted_average_fn():
    """Return a function to aggregate metrics (accuracy, etc.)."""

    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples)}

    return weighted_average
