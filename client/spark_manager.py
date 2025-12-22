from pyspark.sql import SparkSession
import os

def get_spark_session(app_name="FederatedClient"):
    """
    Crée une session Spark avec des configurations légères optimisées pour les conteneurs Docker.
    Cela évite les erreurs de mémoire (OOM) sur une machine locale (Laptop).

    Args:
        app_name (str): Nom de l'application Spark

    Returns:
        SparkSession: Instance de session Spark configurée
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "512m") \
        .config("spark.driver.memory", "512m") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.driver.bindAddress", "0.0.0.0") \
        .getOrCreate()

    # Réduire le bruit dans les logs (afficher seulement les erreurs importantes)
    spark.sparkContext.setLogLevel("WARN")

    return spark

def stop_spark_session(spark):
    """
    Arrête proprement la session Spark pour libérer les ressources.

    Args:
        spark (SparkSession): Session Spark à arrêter
    """
    if spark:
        spark.stop()
