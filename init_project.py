import os

PROJECT_NAME = "Cardio_Federated_System"

STRUCTURE = [
    # Data
    "data/raw",
    "data/incoming",
    "data/processed",
    "data/server_test",

    # Client (Hospitals)
    "client",

    # Server
    "server",

    # Shared code
    "common",

    # Simulator
    "simulator",

    # Monitoring
    "monitoring/prometheus",
    "monitoring/grafana/dashboards",

    # Infrastructure
    "infrastructure",
]

FILES = {
    "client": [
        "Dockerfile",
        "requirements.txt",
        "client.py",
        "spark_manager.py",
        "preprocessing.py",
        "trainer.py",
        "model.py",
    ],
    "server": [
        "Dockerfile",
        "requirements.txt",
        "server.py",
        "strategy.py",
        "evaluation.py",
    ],
    "common": [
        "__init__.py",
        "kafka_topics.py",
        "serialization.py",
        "constants.py",
    ],
    "simulator": [
        "data_injector.py",
    ],
    "monitoring/prometheus": [
        "prometheus.yml",
    ],
    "monitoring/grafana": [
        "datasources.yml",
    ],
    "monitoring/grafana/dashboards": [
        "main_dashboard.json",
    ],
    ".": [
        "docker-compose.yml",
        "README.md",
        ".gitignore",
    ],
}

def create_project():
    print(f"🚀 Création du projet : {PROJECT_NAME}\n")

    for folder in STRUCTURE:
        path = os.path.join(PROJECT_NAME, folder)
        os.makedirs(path, exist_ok=True)
        print(f"📁 Dossier créé : {path}")

    for folder, files in FILES.items():
        base_path = PROJECT_NAME if folder == "." else os.path.join(PROJECT_NAME, folder)
        os.makedirs(base_path, exist_ok=True)

        for file in files:
            file_path = os.path.join(base_path, file)
            if not os.path.exists(file_path):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"# {file}\n")
                    f.write("# Projet : Cardio Federated Learning\n")
                    f.write("# TODO: Implémentation à venir\n")
                print(f"    📄 Fichier créé : {file_path}")

    # .gitignore sécurisé
    gitignore_path = os.path.join(PROJECT_NAME, ".gitignore")
    with open(gitignore_path, "w", encoding="utf-8") as f:
        f.write(
            "data/raw/\n"
            "data/incoming/\n"
            "data/processed/\n"
            "data/server_test/\n"
            "venv/\n"
            "__pycache__/\n"
            "*.parquet\n"
            "*.csv\n"
            ".DS_Store\n"
        )

    print("\n✅ Structure du projet créée avec succès.")

if __name__ == "__main__":
    create_project()
