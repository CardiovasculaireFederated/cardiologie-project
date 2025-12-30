import io
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from client.model import HeartDiseaseModel
from .data_base import get_dataset

BASE_DIR = Path(__file__).resolve().parent.parent  # cardiologie-project/
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger("evaluation")

DEFAULT_TEST_PATH = "server/data_test/test.csv"
TEST_CSV_PATH = os.getenv("TEST_PATH", DEFAULT_TEST_PATH)
PREPROCESSED_TEST_PATH = os.getenv(
    "PREPROCESSED_TEST_PATH", "server/data_test/test_processed.csv"
)
TEST_DATASET_NAME = os.getenv("TEST_DATASET_NAME", "global_test_set")
EVAL_TARGET_COLUMN = os.getenv("EVAL_TARGET_COLUMN", "label")
FALLBACK_TARGET_COLUMN = os.getenv("FALLBACK_TARGET_COLUMN", "Heart Disease Status")
TARGET_DIM = int(os.getenv("TARGET_FEATURE_DIM", "13"))
EVAL_METRIC = os.getenv("EVAL_METRIC", "f1").lower()


def _split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target_col = None
    if EVAL_TARGET_COLUMN in df.columns:
        target_col = EVAL_TARGET_COLUMN
    elif FALLBACK_TARGET_COLUMN in df.columns:
        target_col = FALLBACK_TARGET_COLUMN
    else:
        raise ValueError("Target column not found in test dataset.")

    X_df = df.drop(columns=[target_col])
    y_series = df[target_col]
    return X_df, y_series


def _preprocess_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df = df.dropna()

    X_df, y_series = _split_features_target(df)

    categorical_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))

    if X_df.shape[1] > TARGET_DIM:
        selected_features = X_df.columns[:TARGET_DIM].tolist()
        X_df = X_df[selected_features]

    X = X_df.values.astype(np.float32)
    y = y_series.values
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)
    y = y.astype(np.float32)

    X = np.where(np.isinf(X), np.nan, X)
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)

    return X, y


def _load_test_dataframe() -> pd.DataFrame:
    dataset_bytes = get_dataset(TEST_DATASET_NAME)
    if dataset_bytes:
        logger.info("Using global test dataset from database.")
        return pd.read_csv(io.BytesIO(bytes(dataset_bytes)))

    if os.path.exists(PREPROCESSED_TEST_PATH):
        logger.info("Using preprocessed test dataset from file.")
        return pd.read_csv(PREPROCESSED_TEST_PATH)

    if os.path.exists(TEST_CSV_PATH):
        logger.info("Using raw test dataset from file.")
        return pd.read_csv(TEST_CSV_PATH)

    raise FileNotFoundError("Global test dataset not found.")


def evaluate_global_model(aggregated_weights: Dict) -> float:
    model = HeartDiseaseModel()
    model.eval()

    model.load_state_dict(aggregated_weights, strict=True)

    df = _load_test_dataframe()
    X, y = _preprocess_df(df)
    unique, counts = np.unique(y, return_counts=True)
    label_counts = {int(k): int(v) for k, v in zip(unique, counts)}
    logger.info(f"Global test label distribution: {label_counts}")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    tp = tn = fp = fn = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            preds = (torch.sigmoid(logits) >= 0.5).to(torch.int64).squeeze()
            y_true = (y_batch >= 0.5).to(torch.int64)

            tp += ((preds == 1) & (y_true == 1)).sum().item()
            tn += ((preds == 0) & (y_true == 0)).sum().item()
            fp += ((preds == 1) & (y_true == 0)).sum().item()
            fn += ((preds == 0) & (y_true == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (tpr + tnr) / 2.0

    logger.info(
        "Global test metrics: precision=%.4f recall=%.4f f1=%.4f acc=%.4f bal_acc=%.4f (metric=%s)",
        precision,
        recall,
        f1,
        accuracy,
        balanced_acc,
        EVAL_METRIC,
    )

    if EVAL_METRIC in ("precision", "prec"):
        return precision
    if EVAL_METRIC in ("recall", "rec"):
        return recall
    if EVAL_METRIC in ("f1", "f1_score"):
        return f1
    if EVAL_METRIC in ("acc", "accuracy"):
        return accuracy
    return balanced_acc
