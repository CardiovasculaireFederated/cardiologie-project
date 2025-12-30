import psycopg2
import io
import torch  # si tu utilises PyTorch
from datetime import datetime
import psycopg2
import pickle

def get_connection():
    return psycopg2.connect(
        host="localhost",
        port="5432",
        database="database_card",
        user="admin",
        password="cardio111"
    )

def get_best_model_weights():
    conn=get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT model_name, weights, accuracy
        FROM models_info
        ORDER BY accuracy DESC
        LIMIT 1;
    """)
    
    result = cur.fetchone()
    cur.close()
    conn.close()

    if result is None:
        print("Aucun modèle trouvé dans la base.")
        return None, None

    model_name, weights_bytes, accuracy = result
    print(f"Meilleur modèle : {model_name} | Accuracy : {accuracy}")
    
    return weights_bytes, model_name


def save_global_model(round_num, aggregated_weights, accuracy):

    conn = get_connection()
    cur = conn.cursor()

 
    weights_bytes = pickle.dumps(aggregated_weights)

    model_name = f"global_round_{round_num}"

    cur.execute("""
        INSERT INTO models_info (model_name, weights, global_accuracy,created_at)
        VALUES (%s, %s, %s, %s);
    """, (model_name, psycopg2.Binary(weights_bytes), accuracy,datetime.now().isoformat()))

    conn.commit()
    cur.close()
    conn.close()

    print(f"[DB] Modèle sauvegardé : {model_name} | Acc={accuracy:.4f}")