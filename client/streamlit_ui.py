import streamlit as st
import pandas as pd
import os
import subprocess
import threading
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "client_data.csv"

st.set_page_config(page_title="Cardio Federated Client", layout="wide")

st.title("Client de Cardiologie - Apprentissage Fédéré")

# --- Section 1: Upload de données ---
st.header("1. Préparation des données")
uploaded_file = st.file_uploader("Choisissez un fichier CSV (Données patients)", type="csv")

if uploaded_file is not None:
    # Sauvegarde du fichier pour que le client Flower puisse l'utiliser
    df = pd.read_csv(uploaded_file)
    df.to_csv(DATA_PATH, index=False)
    st.success(f"✅ {len(df)} enregistrements chargés et sauvegardés dans {DATA_PATH}")
    st.dataframe(df.head())

# --- Section 2: Contrôle de l'entraînement ---
st.header("2. Entraînement Fédéré")

if st.button(" Lancer l'entraînement"):
    if os.path.exists(DATA_PATH):
        st.info("Connexion au broker Kafka et démarrage du cycle d'entraînement...")
        
        try:
            # On définit la variable d'environnement pour que main.py utilise le bon fichier
            env = os.environ.copy()
            env["CLIENT_DATA_PATH"] = DATA_PATH
            
            # Lancement du processus main.py (ManagerAgent)
            # On utilise sys.executable pour garantir d'utiliser le même interpréteur Python
            process = subprocess.Popen(
                [sys.executable, "-m", "client.main"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                env=env
            )
            
            st.warning("Entraînement en cours... Consultez les logs ci-dessous.")
            
            # Zone pour afficher les logs en temps réel
            log_area = st.empty()
            full_logs = ""
            
            # Lecture des logs ligne par ligne
            for line in iter(process.stdout.readline, ""):
                full_logs += line
                log_area.text_area("Flux d'activité du Client :", value=full_logs, height=300)
                
            process.stdout.close()
            return_code = process.wait()
            
            if return_code == 0:
                st.success("✅ Entraînement terminé avec succès !")
            else:
                st.error(f"❌ Le processus s'est arrêté avec le code : {return_code}")
                
        except Exception as e:
            st.error(f"Erreur lors du lancement : {e}")
    else:
        st.warning("⚠️ Veuillez d'abord uploader un fichier CSV pour l'entraînement.")