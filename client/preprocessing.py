from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType

def get_heart_schema():
    """
    Définition manuelle du schéma des données cardiaques.
    Nécessaire pour que Spark Streaming lise correctement les fichiers CSV entrants.

    Schéma basé sur le dataset actuel avec 21 colonnes :
    - Données démographiques : Age, Gender
    - Mesures physiologiques : Blood Pressure, Cholesterol Level, BMI, etc.
    - Facteurs de risque : Smoking, Diabetes, Family Heart Disease
    - Habitudes de vie : Exercise Habits, Alcohol Consumption, Sleep Hours
    - Biomarqueurs : CRP Level, Homocysteine Level, Triglyceride Level, etc.
    - Target : Heart Disease Status (Yes/No)

    Returns:
        StructType: Schéma structuré pour Spark
    """
    return StructType([
        # Données démographiques
        StructField("Age", DoubleType(), True),
        StructField("Gender", StringType(), True),

        # Mesures physiologiques
        StructField("Blood Pressure", DoubleType(), True),
        StructField("Cholesterol Level", DoubleType(), True),
        StructField("BMI", DoubleType(), True),

        # Habitudes de vie
        StructField("Exercise Habits", StringType(), True),
        StructField("Smoking", StringType(), True),
        StructField("Alcohol Consumption", StringType(), True),
        StructField("Sleep Hours", DoubleType(), True),
        StructField("Sugar Consumption", StringType(), True),
        StructField("Stress Level", StringType(), True),

        # Antécédents médicaux
        StructField("Family Heart Disease", StringType(), True),
        StructField("Diabetes", StringType(), True),

        # Conditions médicales
        StructField("High Blood Pressure", StringType(), True),
        StructField("Low HDL Cholesterol", StringType(), True),
        StructField("High LDL Cholesterol", StringType(), True),

        # Biomarqueurs
        StructField("Triglyceride Level", DoubleType(), True),
        StructField("Fasting Blood Sugar", DoubleType(), True),
        StructField("CRP Level", DoubleType(), True),
        StructField("Homocysteine Level", DoubleType(), True),

        # Colonne cible (Target)
        StructField("Heart Disease Status", StringType(), True)
    ])

def get_feature_columns():
    """
    Retourne la liste des colonnes de features (sans la colonne cible).

    Returns:
        list: Liste des noms de colonnes features
    """
    return [
        "Age", "Gender", "Blood Pressure", "Cholesterol Level", "BMI",
        "Exercise Habits", "Smoking", "Alcohol Consumption", "Sleep Hours",
        "Sugar Consumption", "Stress Level", "Family Heart Disease", "Diabetes",
        "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol",
        "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level"
    ]

def get_target_column():
    """
    Retourne le nom de la colonne cible.

    Returns:
        str: Nom de la colonne cible
    """
    return "Heart Disease Status"

def get_categorical_columns():
    """
    Retourne la liste des colonnes catégorielles qui nécessitent un encodage.

    Returns:
        list: Liste des noms de colonnes catégorielles
    """
    return [
        "Gender", "Exercise Habits", "Smoking", "Alcohol Consumption",
        "Sugar Consumption", "Stress Level", "Family Heart Disease",
        "Diabetes", "High Blood Pressure", "Low HDL Cholesterol",
        "High LDL Cholesterol"
    ]

def get_numerical_columns():
    """
    Retourne la liste des colonnes numériques.

    Returns:
        list: Liste des noms de colonnes numériques
    """
    return [
        "Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours",
        "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level"
    ]
