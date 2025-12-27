import streamlit as st
import pandas as pd
import os
import subprocess
import threading

st.set_page_config(page_title="Cardio Federated Client", layout="wide")

st.title("Client de Cardiologie - Apprentissage Fédéré")

# --- Section 1: Upload de données ---
st.header("1. Préparation des données")
uploaded_file = st.file_uploader("Choisissez un fichier CSV (Données patients)", type="csv")

if uploaded_file is not None:
    # Sauvegarde du fichier pour que le client Flower puisse l'utiliser
    data_path = "data/client_data.csv"
    df = pd.read_csv(uploaded_file)
    df.to_csv(data_path, index=False)
    st.success(f"{len(df)} enregistrements chargés avec succès !")
    st.dataframe(df.head())

# --- Section 2: Contrôle de l'entraînement ---
st.header("2. Entraînement Fédéré")

if st.button("Lancer l'entraînement"):
    if os.path.exists("data/client_data.csv"):
        st.info("Connexion au serveur et démarrage de l'entraînement...")
        
        # On lance le client Flower en arrière-plan ou via un sous-processus
        # Remarque : vous pouvez aussi importer votre classe Client et la lancer ici
        try:
            # Exemple : lancement du script client existant
            process = subprocess.Popen(["python", "client.main"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Affichage des logs en temps réel dans Streamlit
            st.text_area("Logs du client :", value="Entraînement en cours...", height=200)
            st.success("Entraînement terminé ou envoyé au background !")
        except Exception as e:
            st.error(f"Erreur : {e}")
    else:
        st.warning("Veuillez d'abord uploader un fichier CSV.")