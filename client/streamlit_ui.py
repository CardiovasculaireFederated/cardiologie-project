import os
import subprocess
from typing import List

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cardio Federated Client", layout="wide")

st.title("Client de Cardiologie - Apprentissage Fédéré")

# --- Section 1: Upload de données ---
st.header("1. Préparation des données")
uploaded_file = st.file_uploader("Choisissez un fichier CSV (Données patients)", type="csv")

DATA_PATH = "data/client_data.csv"
LOG_PATH = "logs/client_training.log"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    st.success(f"{len(df)} enregistrements chargés avec succès !")
    st.dataframe(df.head())


def tail_file(path: str, max_lines: int = 200) -> List[str]:
    try:
        with open(path, "r") as handle:
            lines = handle.readlines()
        return lines[-max_lines:]
    except FileNotFoundError:
        return []


# --- Section 2: Contrôle de l'entraînement ---
st.header("2. Entraînement Fédéré")

if "training_pid" not in st.session_state:
    st.session_state.training_pid = None

if st.button("معالجة وبدء التدريب"):
    if os.path.exists(DATA_PATH):
        st.info("Connexion au serveur et démarrage de l'entraînement...")
        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            log_handle = open(LOG_PATH, "a", buffering=1)

            env = os.environ.copy()
            env["CLIENT_DATA_PATH"] = DATA_PATH
            env.setdefault("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

            process = subprocess.Popen(
                ["python", "-m", "client.main"],
                stdout=log_handle,
                stderr=log_handle,
                text=True,
                env=env,
            )
            st.session_state.training_pid = process.pid
            st.success(f"Entraînement lancé (PID {process.pid}).")
        except Exception as e:
            st.error(f"Erreur : {e}")
    else:
        st.warning("Veuillez d'abord uploader un fichier CSV.")

if st.session_state.training_pid:
    st.caption(f"PID actuel: {st.session_state.training_pid}")

logs = tail_file(LOG_PATH, max_lines=200)
if logs:
    st.text_area("Logs du client :", value="".join(logs), height=300)
else:
    st.text_area("Logs du client :", value="Aucun log pour le moment.", height=200)
