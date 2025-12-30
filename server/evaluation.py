import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from client.model import HeartDiseaseModel
from dotenv import load_dotenv
import os
import io



BASE_DIR = Path(__file__).resolve().parent.parent  # cardiologie-project/
load_dotenv(BASE_DIR / ".env")

TEST_CSV_PATH = os.getenv("TEST_PATH")

TEST_CSV_PATH = "server/data_test/test.csv"

def load_test_data(batch_size=32):
    df = pd.read_csv(TEST_CSV_PATH)

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def evaluate_global_model(aggregated_weights):

    model = HeartDiseaseModel()
    model.eval()

   
    state_dict = model.state_dict()
    new_state_dict = {}

    for (key, _), weight in zip(state_dict.items(), aggregated_weights):
        new_state_dict[key] = torch.tensor(weight)

    model.load_state_dict(new_state_dict, strict=True)

    df = pd.read_csv(TEST_CSV_PATH)

    X = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    correct, total = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            preds = torch.sigmoid(logits).round()
            correct += (preds.squeeze() == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    return accuracy

