#!/usr/bin/env python3
# setup_spark_env.py
# Configuration de l'environnement Spark pour Windows

import os
import sys
import subprocess

def setup_spark_environment():
    """
    Configure les variables d'environnement pour PySpark sur Windows
    """
    print("Configuration de l'environnement Spark...")

    # Détecter le chemin Python actuel
    python_exe = sys.executable
    python_dir = os.path.dirname(python_exe)

    print(f"Python détecté : {python_exe}")

    # Configurer les variables d'environnement PySpark
    os.environ['PYSPARK_PYTHON'] = python_exe
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_exe

    # Essayer de détecter SPARK_HOME
    try:
        import pyspark
        pyspark_location = os.path.dirname(pyspark.__file__)
        spark_home = os.path.abspath(os.path.join(pyspark_location, '..', '..', 'pyspark'))

        # Si le dossier n'existe pas, utiliser le dossier pyspark directement
        if not os.path.exists(spark_home):
            spark_home = pyspark_location

        os.environ['SPARK_HOME'] = spark_home
        print(f"SPARK_HOME défini : {spark_home}")
    except ImportError:
        print("⚠️  PySpark n'est pas installé!")
        return False

    # Désactiver les warnings de compatibilité
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

    print("✅ Configuration terminée!\n")
    return True

if __name__ == "__main__":
    setup_spark_environment()
