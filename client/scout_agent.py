# client/scout_agent.py

import logging

logger = logging.getLogger("scout_agent")


def suggest_hyperparameters(history):
    """
    Decide training hyperparameters based on previous rounds.
    """
    # Default values (first round)
    if len(history) < 2:
        logger.info("ScoutAgent: using default hyperparameters")
        return {
            "learning_rate": 1e-3,
            "epochs": 5
        }

    last = history[-1]
    prev = history[-2]

    lr = last["learning_rate"]
    epochs = last["epochs"]

    last_acc = last.get("best_val_acc", 0)
    prev_acc = prev.get("best_val_acc", 0)

    delta = last_acc - prev_acc

    logger.info(f"ScoutAgent: accuracy delta = {delta:.4f}")

    # Rule 1: improving
    if delta > 0.01:
        logger.info("ScoutAgent: performance improving → keep hyperparameters")
        return {
            "learning_rate": lr,
            "epochs": epochs
        }

    # Rule 2: stagnation
    if -0.01 <= delta <= 0.01:
        logger.info("ScoutAgent: stagnation → increasing epochs")
        return {
            "learning_rate": lr,
            "epochs": min(epochs + 2, 15)
        }

    # Rule 3: degradation
    logger.info("ScoutAgent: performance dropped → reducing learning rate")
    return {
        "learning_rate": max(lr * 0.5, 1e-5),
        "epochs": epochs
    }
