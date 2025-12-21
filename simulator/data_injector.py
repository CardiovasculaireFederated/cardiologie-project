# simulator/data_injector.py
# Simulateur de données en temps réel pour le projet Cardio Federated Learning
# Découpe le CSV et l'envoie dans data/incoming/ toutes les 5 secondes

import os
import time
import pandas as pd
from pathlib import Path

class DataInjector:
    """
    Simulateur de streaming de données cardiaques.
    Découpe le dataset CSV et l'envoie progressivement dans le répertoire incoming.
    """

    def __init__(self,
                 source_csv="data/raw/heart_disease.csv",
                 target_dir="data/incoming",
                 batch_size=10,
                 interval_seconds=5):
        """
        Args:
            source_csv: Chemin vers le CSV source
            target_dir: Répertoire de destination pour les chunks
            batch_size: Nombre de lignes par batch
            interval_seconds: Intervalle entre chaque injection
        """
        self.source_csv = source_csv
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.interval_seconds = interval_seconds

        # Créer le répertoire target s'il n'existe pas
        Path(self.target_dir).mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Charge le CSV source."""
        if not os.path.exists(self.source_csv):
            raise FileNotFoundError(f"CSV source introuvable: {self.source_csv}")

        print(f" Chargement du dataset: {self.source_csv}")
        self.df = pd.read_csv(self.source_csv)
        print(f" {len(self.df)} lignes chargées, {self.df.shape[1]} colonnes")
        return self.df

    def clean_target_dir(self):
        """Nettoie le répertoire target avant de commencer."""
        target_path = Path(self.target_dir)
        if target_path.exists():
            for file in target_path.glob("heart_in_*.csv"):
                file.unlink()
            print(f" Répertoire {self.target_dir} nettoyé")

    def start_streaming(self, max_batches=None):
        """
        Démarre le streaming de données.

        Args:
            max_batches: Nombre max de batches à envoyer (None = tous)
        """
        print(f"\n{'='*60}")
        print(" DÉMARRAGE DU SIMULATEUR DE STREAMING")
        print(f"{'='*60}")
        print(f" Batch size: {self.batch_size} lignes")
        print(f"  Intervalle: {self.interval_seconds}s")
        print(f" Destination: {self.target_dir}")
        print(f"{'='*60}\n")

        # Charger les données
        self.load_data()

        # Nettoyer le répertoire target
        self.clean_target_dir()

        total_rows = len(self.df)
        num_batches = (total_rows + self.batch_size - 1) // self.batch_size

        if max_batches:
            num_batches = min(num_batches, max_batches)

        print(f" Nombre total de batches à envoyer: {num_batches}\n")

        batch_counter = 0

        try:
            for i in range(0, total_rows, self.batch_size):
                if max_batches and batch_counter >= max_batches:
                    break

                # Extraire le batch
                end_idx = min(i + self.batch_size, total_rows)
                batch_df = self.df.iloc[i:end_idx]

                # Générer le nom de fichier avec timestamp
                timestamp = int(time.time())
                filename = f"heart_in_{batch_counter}_{timestamp}.csv"
                filepath = os.path.join(self.target_dir, filename)

                # Sauvegarder le batch
                batch_df.to_csv(filepath, index=False)

                batch_counter += 1
                progress = (batch_counter / num_batches) * 100

                print(f" Batch {batch_counter}/{num_batches} envoyé "
                      f"({len(batch_df)} lignes) | Fichier: {filename} | "
                      f"Progression: {progress:.1f}%")

                # Attendre avant le prochain batch (sauf pour le dernier)
                if batch_counter < num_batches:
                    time.sleep(self.interval_seconds)

            print(f"\n{'='*60}")
            print(f" STREAMING TERMINÉ: {batch_counter} batches envoyés")
            print(f"{'='*60}\n")

        except KeyboardInterrupt:
            print(f"\n\n  Streaming interrompu par l'utilisateur")
            print(f" Batches envoyés avant interruption: {batch_counter}")

        except Exception as e:
            print(f"\n ERREUR pendant le streaming: {e}")
            raise

    def send_single_batch(self, batch_number=0):
        """
        Envoie un seul batch (utile pour les tests).

        Args:
            batch_number: Numéro du batch à envoyer
        """
        self.load_data()

        start_idx = batch_number * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.df))

        if start_idx >= len(self.df):
            print(f" Batch {batch_number} hors limite")
            return

        batch_df = self.df.iloc[start_idx:end_idx]

        timestamp = int(time.time())
        filename = f"heart_in_{batch_number}_{timestamp}.csv"
        filepath = os.path.join(self.target_dir, filename)

        batch_df.to_csv(filepath, index=False)
        print(f" Batch {batch_number} envoyé: {filename} ({len(batch_df)} lignes)")


def main():
    """Fonction principale pour exécuter le simulateur."""

    # Configuration
    injector = DataInjector(
        source_csv="data/raw/heart_disease.csv",
        target_dir="data/incoming",
        batch_size=10,          # 10 lignes par batch
        interval_seconds=5       # Toutes les 5 secondes
    )

    # Lancer le streaming (max 20 batches pour le test)
    injector.start_streaming(max_batches=20)


if __name__ == "__main__":
    main()
