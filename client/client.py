# client/client.py
# Client de Federated Learning avec prétraitement temps réel Spark → PyTorch
# Pont entre le streaming Spark et le modèle PyTorch

import os
import time
import torch
import numpy as np
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT

# Imports locaux
from client.spark_manager import get_spark_session, stop_spark_session
from client.preprocessing import (
    get_heart_schema,
    preprocess_data,
    preprocess_streaming_batch,
    build_preprocessing_pipeline
)
from client.model import HeartDiseaseModel


class FederatedClient:
    """
    Client pour le Federated Learning avec preprocessing en temps réel.
    Lit les batches Spark, les prétraite, et les convertit en tensors PyTorch.
    """

    def __init__(self,
                 client_id="client_1",
                 incoming_dir="data/incoming",
                 processed_dir="data/processed",
                 batch_size=32):
        """
        Args:
            client_id: Identifiant unique du client
            incoming_dir: Répertoire des fichiers CSV entrants
            processed_dir: Répertoire pour sauvegarder les données traitées
            batch_size: Taille des batches PyTorch
        """
        self.client_id = client_id
        self.incoming_dir = incoming_dir
        self.processed_dir = processed_dir
        self.batch_size = batch_size

        # Créer les répertoires s'ils n'existent pas
        Path(self.incoming_dir).mkdir(parents=True, exist_ok=True)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

        # Session Spark
        self.spark = None

        # Pipeline de prétraitement (sera entraîné sur le premier batch)
        self.pipeline_model = None

        # Modèle PyTorch
        self.model = None

        print(f" Client '{client_id}' initialisé")

    def initialize_spark(self):
        """Initialise la session Spark."""
        if self.spark is None:
            print("🔧 Initialisation de Spark...")
            self.spark = get_spark_session(app_name=f"FedClient_{self.client_id}")
            print("✓ Spark session créée")

    def initialize_model(self, input_dim=13):
        """
        Initialise le modèle PyTorch.

        Args:
            input_dim: Dimension d'entrée (nombre de features)
        """
        print(f" Initialisation du modèle PyTorch (input_dim={input_dim})...")
        self.model = HeartDiseaseModel(input_dim=input_dim)
        print("✓ Modèle créé")

    def load_csv_batch(self, csv_path):
        """
        Charge un fichier CSV en DataFrame Spark.

        Args:
            csv_path: Chemin vers le fichier CSV

        Returns:
            PySpark DataFrame
        """
        schema = get_heart_schema()

        df = self.spark.read.csv(
            csv_path,
            header=True,
            schema=schema
        )

        print(f" Chargé: {csv_path} ({df.count()} lignes)")
        return df

    def train_initial_pipeline(self, initial_csv="data/raw/heart_disease.csv"):
        """
        Entraîne le pipeline de prétraitement sur un dataset initial.
        Ce pipeline sera réutilisé pour tous les batches de streaming.

        Args:
            initial_csv: Chemin vers le CSV initial pour fiter le pipeline
        """
        print("\n" + "="*60)
        print(" ENTRAÎNEMENT DU PIPELINE DE PRÉTRAITEMENT")
        print("="*60)

        # Charger le dataset initial
        df_initial = self.load_csv_batch(initial_csv)

        # Prétraiter et fiter le pipeline
        df_processed, self.pipeline_model = preprocess_data(df_initial, pipeline_model=None)

        print("Pipeline de prétraitement entraîné et prêt")
        print("="*60 + "\n")

    def preprocess_batch(self, df):
        """
        Prétraite un batch Spark avec le pipeline pré-entraîné.

        Args:
            df: DataFrame Spark brut

        Returns:
            DataFrame Spark transformé (colonnes: features, label)
        """
        if self.pipeline_model is None:
            raise ValueError("Pipeline non entraîné. Appelez train_initial_pipeline() d'abord.")

        # Prétraiter le batch
        df_processed = preprocess_streaming_batch(df, self.pipeline_model)

        return df_processed

    def spark_to_pytorch(self, df_processed):
        """
        Convertit un DataFrame Spark en tenseurs PyTorch.

        Args:
            df_processed: DataFrame Spark avec colonnes 'features' et 'label'

        Returns:
            tuple: (X_tensor, y_tensor)
                - X_tensor: torch.Tensor de shape (N, num_features)
                - y_tensor: torch.Tensor de shape (N,)
        """
        print(" Conversion Spark → PyTorch...")

        # Collecter les données en Python
        rows = df_processed.collect()

        if len(rows) == 0:
            print("  Batch vide, aucune donnée à convertir")
            return None, None

        # Extraire les features et labels
        X_list = []
        y_list = []

        for row in rows:
            # Convertir le vecteur Spark en numpy array
            features_vector = row['features']

            # Les vecteurs Spark ML peuvent être de type DenseVector ou SparseVector
            if hasattr(features_vector, 'toArray'):
                features_array = features_vector.toArray()
            else:
                features_array = np.array(features_vector)

            X_list.append(features_array)

            # Extraire le label si présent
            if 'label' in row:
                y_list.append(row['label'])

        # Convertir en numpy arrays
        X_numpy = np.array(X_list, dtype=np.float32)

        if y_list:
            y_numpy = np.array(y_list, dtype=np.float32)
        else:
            y_numpy = None

        # Convertir en tenseurs PyTorch
        X_tensor = torch.from_numpy(X_numpy)

        if y_numpy is not None:
            y_tensor = torch.from_numpy(y_numpy)
        else:
            y_tensor = None

        print(f"✓ Conversion terminée: X shape={X_tensor.shape}", end="")
        if y_tensor is not None:
            print(f", y shape={y_tensor.shape}")
        else:
            print()

        return X_tensor, y_tensor

    def process_incoming_files(self, max_files=None):
        """
        Traite tous les fichiers CSV dans le répertoire incoming.

        Args:
            max_files: Nombre max de fichiers à traiter (None = tous)

        Returns:
            list: Liste de tuples (X_tensor, y_tensor) pour chaque fichier
        """
        print("\n" + "="*60)
        print(" TRAITEMENT DES FICHIERS ENTRANTS")
        print("="*60)

        # Lister les fichiers CSV dans incoming/
        incoming_path = Path(self.incoming_dir)
        csv_files = sorted(incoming_path.glob("*.csv"))

        if not csv_files:
            print("  Aucun fichier CSV trouvé dans", self.incoming_dir)
            return []

        if max_files:
            csv_files = csv_files[:max_files]

        print(f" {len(csv_files)} fichiers à traiter\n")

        batches = []

        for i, csv_file in enumerate(csv_files):
            print(f"\n--- Fichier {i+1}/{len(csv_files)}: {csv_file.name} ---")

            # 1. Charger le CSV
            df = self.load_csv_batch(str(csv_file))

            # 2. Prétraiter
            df_processed = self.preprocess_batch(df)

            # 3. Convertir en PyTorch
            X_tensor, y_tensor = self.spark_to_pytorch(df_processed)

            if X_tensor is not None:
                batches.append((X_tensor, y_tensor))

                # Optionnel: Sauvegarder le batch traité
                self.save_processed_batch(X_tensor, y_tensor, batch_id=i)

        print("\n" + "="*60)
        print(f" {len(batches)} batches traités avec succès")
        print("="*60 + "\n")

        return batches

    def save_processed_batch(self, X_tensor, y_tensor, batch_id):
        """
        Sauvegarde un batch traité sur disque (format PyTorch .pt).

        Args:
            X_tensor: Tenseur des features
            y_tensor: Tenseur des labels
            batch_id: Identifiant du batch
        """
        output_path = Path(self.processed_dir) / f"batch_{batch_id}.pt"

        torch.save({
            'X': X_tensor,
            'y': y_tensor
        }, output_path)

        print(f" Batch sauvegardé: {output_path}")

    def train_on_batch(self, X_tensor, y_tensor, epochs=1, learning_rate=0.001):
        """
        Entraîne le modèle sur un batch.

        Args:
            X_tensor: Features (N, num_features)
            y_tensor: Labels (N,)
            epochs: Nombre d'époques
            learning_rate: Taux d'apprentissage

        Returns:
            float: Loss moyenne
        """
        if self.model is None:
            raise ValueError("Modèle non initialisé. Appelez initialize_model() d'abord.")

        self.model.train()

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        total_loss = 0.0

        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X_tensor).squeeze()
            loss = criterion(outputs, y_tensor)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / epochs
        print(f" Loss moyenne: {avg_loss:.4f}")

        return avg_loss

    def shutdown(self):
        """Arrête proprement le client et libère les ressources."""
        print("\n🔌 Arrêt du client...")

        if self.spark:
            stop_spark_session(self.spark)
            print("✓ Spark session arrêtée")

        print(f" Client '{self.client_id}' arrêté proprement")


# ============================
# FONCTION PRINCIPALE (DEMO)
# ============================

def main():
    """Fonction de démonstration du client."""

    print("\n" + "="*70)
    print(" DÉMONSTRATION DU CLIENT FEDERATED LEARNING")
    print("="*70 + "\n")

    # 1. Créer le client
    client = FederatedClient(
        client_id="demo_client",
        incoming_dir="data/incoming",
        processed_dir="data/processed",
        batch_size=32
    )

    try:
        # 2. Initialiser Spark
        client.initialize_spark()

        # 3. Entraîner le pipeline de prétraitement
        client.train_initial_pipeline(initial_csv="data/raw/heart_disease.csv")

        # 4. Initialiser le modèle PyTorch
        # IMPORTANT: La dimension doit correspondre au nombre de features après preprocessing
        # Actuellement: 9 numériques + 11 catégorielles = 20 features
        client.initialize_model(input_dim=20)

        # 5. Traiter les fichiers entrants
        batches = client.process_incoming_files(max_files=5)

        # 6. Entraîner sur les batches
        if batches:
            print("\n" + "="*60)
            print(" ENTRAÎNEMENT SUR LES BATCHES")
            print("="*60 + "\n")

            for i, (X, y) in enumerate(batches):
                print(f"\n--- Batch {i+1}/{len(batches)} ---")
                print(f"Shape: X={X.shape}, y={y.shape}")

                # Entraîner
                loss = client.train_on_batch(X, y, epochs=5, learning_rate=0.001)

            print("\n" + "="*60)
            print(" ENTRAÎNEMENT TERMINÉ")
            print("="*60 + "\n")

    except Exception as e:
        print(f"\n ERREUR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 7. Arrêter proprement
        client.shutdown()


if __name__ == "__main__":
    main()
