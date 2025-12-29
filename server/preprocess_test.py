"""
Preprocess raw test CSV into a numeric file ready for evaluation.

It tries to use the Spark pipeline defined in client/preprocessing.py.
If Spark is unavailable, it falls back to a lightweight pandas routine
that mirrors the inference-time preprocessing used by the evaluator.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))  # allow importing client modules


def _spark_available() -> bool:
    try:
        import pyspark  # noqa: F401
        return True
    except Exception:
        return False


def _flatten_spark_df(df_spark, output_path: Path) -> None:
    """
    Convert Spark DataFrame with vector "features" (+ optional label)
    into a flat CSV file with f0..fn columns and label.
    """
    pdf = df_spark.toPandas()
    if "features" not in pdf.columns:
        raise ValueError("Spark output missing 'features' column.")

    features_matrix = pd.DataFrame(pdf["features"].apply(lambda v: v.toArray()).tolist())
    features_matrix.columns = [f"f{i}" for i in range(features_matrix.shape[1])]

    if "label" in pdf.columns:
        features_matrix["label"] = pdf["label"].astype(float)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_matrix.to_csv(output_path, index=False)


def preprocess_with_spark(input_path: Path, output_path: Path) -> None:
    from pyspark.sql import SparkSession
    from client.preprocessing import (
        get_heart_schema,
        preprocess_data,
    )

    spark = SparkSession.builder.appName("cardio-test-preprocess").getOrCreate()
    schema = get_heart_schema()
    df_raw = spark.read.csv(str(input_path), header=True, schema=schema)

    df_processed, pipeline_model = preprocess_data(df_raw, pipeline_model=None)

    _flatten_spark_df(df_processed, output_path)
    spark.stop()


def preprocess_with_pandas(input_path: Path, output_path: Path) -> None:
    from server.evaluation import _preprocess_df  # reuse the same logic as evaluation

    df = pd.read_csv(input_path)
    X, y = _preprocess_df(df)
    data = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    data["label"] = y
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess test CSV for evaluation.")
    parser.add_argument(
        "--input",
        default=os.getenv("TEST_PATH", "server/data_test/test.csv"),
        help="Raw test CSV path.",
    )
    parser.add_argument(
        "--output",
        default=os.getenv(
            "PREPROCESSED_TEST_PATH", "server/data_test/test_processed.csv"
        ),
        help="Output preprocessed CSV path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if _spark_available():
        preprocess_with_spark(input_path, output_path)
    else:
        preprocess_with_pandas(input_path, output_path)

    print(f"Preprocessed test data saved to: {output_path}")


if __name__ == "__main__":
    main()
