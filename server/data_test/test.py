import psycopg2
import torch
import io
from datetime import datetime
import os

# 1. Configuration de la connexion
def get_connection():
    return psycopg2.connect(
        host="localhost",  # <--- Indispensable pour un accès hors Docker
        port="5432",
        database="database_card",
        user="admin",
        password="cardio111"
    )
conn=get_connection()
cur = conn.cursor()

# 2. Préparation des données de test
model_name = "Cardio_Model_Alpha_V1"
accuracy = 0.90 # 89.5% de précision

# Simulation de poids PyTorch (dictionnaire de tenseurs)
fake_weights = {
    'layer1.weight': torch.randn(10, 5),
    'layer1.bias': torch.randn(10)
}

# Conversion des poids en format binaire (Bytes)
buffer = io.BytesIO()
torch.save(fake_weights, buffer)
weights_bytes = buffer.getvalue()

# 3. L'exécution de votre commande d'insertion
try:
    cur.execute("""
        INSERT INTO models_info (model_name, weights, global_accuracy, created_at)
        VALUES (%s, %s, %s, %s);
    """, (model_name, psycopg2.Binary(weights_bytes), accuracy, datetime.now().isoformat()))
    
    conn.commit()
    print(f"Succès ! Modèle '{model_name}' inséré.")
except Exception as e:
    print(f"Erreur lors de l'insertion : {e}")
    conn.rollback()

cur.close()
conn.close()