import logging
import os
import pickle
from datetime import datetime
from typing import Optional, Tuple

import psycopg2

logger = logging.getLogger("database")


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres_db"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        database=os.getenv("POSTGRES_DB", "database_card"),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", "cardio111"),
    )


def init_db() -> None:
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS models_info (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(255),
                weights BYTEA,
                global_accuracy FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE,
                payload BYTEA,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()
        logger.info("Database schema ensured.")
    except Exception as exc:
        logger.error(f"Database init failed: {exc}")
        raise
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def get_best_model_weights() -> Tuple[Optional[bytes], Optional[str]]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT model_name, weights, global_accuracy
            FROM models_info
            ORDER BY global_accuracy DESC
            LIMIT 1;
            """
        )
        result = cur.fetchone()
    finally:
        cur.close()
        conn.close()

    if result is None:
        logger.info("No model found in the database.")
        return None, None

    model_name, weights_bytes, accuracy = result
    logger.info(f"Best model: {model_name} | accuracy={accuracy}")
    return weights_bytes, model_name


def save_global_model(round_num, aggregated_weights, accuracy) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        weights_bytes = pickle.dumps(aggregated_weights)
        model_name = f"global_round_{round_num}"
        cur.execute(
            """
            INSERT INTO models_info (model_name, weights, global_accuracy, created_at)
            VALUES (%s, %s, %s, %s);
            """,
            (
                model_name,
                psycopg2.Binary(weights_bytes),
                accuracy,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

    logger.info(f"[DB] Model saved: {model_name} | acc={accuracy:.4f}")


def save_dataset(name: str, payload: bytes) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO datasets (name, payload, created_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (name) DO NOTHING;
            """,
            (name, psycopg2.Binary(payload), datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()


def get_dataset(name: str) -> Optional[bytes]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT payload
            FROM datasets
            WHERE name = %s
            LIMIT 1;
            """,
            (name,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return row[0]
    finally:
        cur.close()
        conn.close()
