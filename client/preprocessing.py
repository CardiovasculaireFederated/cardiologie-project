# client/preprocessing.py
# Pipeline de prétraitement intelligent pour données cardiaques en temps réel
# Transformation Spark ML: Nettoyage → Encodage → Vectorisation → Normalisation

from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType
from pyspark.sql.functions import col, when, isnan, isnull, mean, lit
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
import numpy as np


# ============================
# SCHÉMA DES DONNÉES
# ============================

def get_heart_schema():
    """
    Schéma structuré pour Spark Streaming.
    Dataset: 21 colonnes (20 features + 1 target)
    """
    return StructType([
        # ORDRE EXACT DU CSV SOURCE
        StructField("Age", DoubleType(), True),
        StructField("Gender", StringType(), True),
        StructField("Blood Pressure", DoubleType(), True),
        StructField("Cholesterol Level", DoubleType(), True),
        StructField("Exercise Habits", StringType(), True),
        StructField("Smoking", StringType(), True),
        StructField("Family Heart Disease", StringType(), True),
        StructField("Diabetes", StringType(), True),
        StructField("BMI", DoubleType(), True),
        StructField("High Blood Pressure", StringType(), True),
        StructField("Low HDL Cholesterol", StringType(), True),
        StructField("High LDL Cholesterol", StringType(), True),
        StructField("Alcohol Consumption", StringType(), True),
        StructField("Stress Level", StringType(), True),
        StructField("Sleep Hours", DoubleType(), True),
        StructField("Sugar Consumption", StringType(), True),
        StructField("Triglyceride Level", DoubleType(), True),
        StructField("Fasting Blood Sugar", DoubleType(), True),
        StructField("CRP Level", DoubleType(), True),
        StructField("Homocysteine Level", DoubleType(), True),
        StructField("Heart Disease Status", StringType(), True)
    ])


def get_feature_columns():
    """Liste des features (sans target)."""
    return [
        "Age", "Gender", "Blood Pressure", "Cholesterol Level", "BMI",
        "Exercise Habits", "Smoking", "Alcohol Consumption", "Sleep Hours",
        "Sugar Consumption", "Stress Level", "Family Heart Disease", "Diabetes",
        "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol",
        "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level"
    ]


def get_target_column():
    """Nom de la colonne target."""
    return "Heart Disease Status"


def get_categorical_columns():
    """Colonnes catégorielles qui nécessitent un encodage."""
    return [
        "Gender", "Exercise Habits", "Smoking", "Alcohol Consumption",
        "Sugar Consumption", "Stress Level", "Family Heart Disease",
        "Diabetes", "High Blood Pressure", "Low HDL Cholesterol",
        "High LDL Cholesterol"
    ]


def get_numerical_columns():
    """Colonnes numériques."""
    return [
        "Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours",
        "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level"
    ]


# ============================
# PRÉTRAITEMENT
# ============================

def clean_missing_values(df):
    """
    Nettoyage des valeurs manquantes.
    Stratégie: Remplir avec la moyenne pour les colonnes numériques.

    Args:
        df: PySpark DataFrame

    Returns:
        DataFrame nettoyé
    """
    numerical_cols = get_numerical_columns()

    print(" Nettoyage des valeurs manquantes...")

    # Pour chaque colonne numérique, remplacer les NaN/null par la moyenne
    for col_name in numerical_cols:
        if col_name in df.columns:
            # Calculer la moyenne en ignorant les NaN
            mean_val = df.select(mean(col(col_name))).collect()[0][0]

            # Si la moyenne est None (toute la colonne est nulle), utiliser 0
            if mean_val is None:
                mean_val = 0.0

            # Remplacer les valeurs nulles
            df = df.withColumn(
                col_name,
                when(col(col_name).isNull() | isnan(col(col_name)), lit(mean_val))
                .otherwise(col(col_name))
            )

    # Pour les colonnes catégorielles, remplacer par "Unknown"
    categorical_cols = get_categorical_columns()
    for col_name in categorical_cols:
        if col_name in df.columns:
            df = df.withColumn(
                col_name,
                when(col(col_name).isNull(), lit("Unknown"))
                .otherwise(col(col_name))
            )

    print("✓ Valeurs manquantes traitées")
    return df


def build_preprocessing_pipeline():
    """
    Construit un pipeline Spark ML complet pour le prétraitement.

    Pipeline:
    1. StringIndexer: Encode les catégorielles en indices numériques
    2. VectorAssembler: Assemble toutes les features en un seul vecteur
    3. StandardScaler: Normalise les features (mean=0, std=1)

    Returns:
        Pipeline Spark ML
    """
    categorical_cols = get_categorical_columns()
    numerical_cols = get_numerical_columns()

    stages = []

    # ========================================
    # ÉTAPE 1: Encoder les colonnes catégorielles
    # ========================================
    indexed_cols = []

    for cat_col in categorical_cols:
        # StringIndexer convertit les strings en indices (0, 1, 2, ...)
        indexer = StringIndexer(
            inputCol=cat_col,
            outputCol=f"{cat_col}_indexed",
            handleInvalid="keep"  # Garder les valeurs inconnues
        )
        stages.append(indexer)
        indexed_cols.append(f"{cat_col}_indexed")

    # ========================================
    # ÉTAPE 2: Assembler toutes les features
    # ========================================
    # Combiner colonnes numériques + catégorielles encodées
    all_feature_cols = numerical_cols + indexed_cols

    assembler = VectorAssembler(
        inputCols=all_feature_cols,
        outputCol="features_raw",
        handleInvalid="skip"  # Ignorer les lignes avec valeurs invalides
    )
    stages.append(assembler)

    # ========================================
    # ÉTAPE 3: Normalisation (StandardScaler)
    # ========================================
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=True,   # Centrer à mean=0
        withStd=True     # Normaliser à std=1
    )
    stages.append(scaler)

    # Créer le pipeline
    pipeline = Pipeline(stages=stages)

    print(f"✓ Pipeline créé avec {len(stages)} étapes:")
    print(f"  - {len(categorical_cols)} colonnes catégorielles encodées")
    print(f"  - {len(all_feature_cols)} features assemblées")
    print(f"  - Normalisation StandardScaler appliquée")

    return pipeline


def preprocess_data(df, pipeline_model=None):
    """
    Fonction principale de prétraitement.

    Args:
        df: PySpark DataFrame brut
        pipeline_model: PipelineModel pré-entraîné (optionnel)
                       Si None, un nouveau pipeline sera fité

    Returns:
        tuple: (df_transformed, pipeline_model)
            - df_transformed: DataFrame avec colonne 'features' normalisée
            - pipeline_model: Pipeline entraîné (pour réutilisation)
    """
    print("\n" + "="*60)
    print(" DÉBUT DU PRÉTRAITEMENT")
    print("="*60)

    # Afficher le nombre de lignes
    row_count = df.count()
    print(f" Nombre de lignes: {row_count}")

    # ========================================
    # ÉTAPE 1: Nettoyage
    # ========================================
    df_clean = clean_missing_values(df)

    # ========================================
    # ÉTAPE 2: Encoder la target (Heart Disease Status)
    # ========================================
    target_col = get_target_column()

    if target_col in df_clean.columns:
        # Encoder Yes/No en 1/0
        df_clean = df_clean.withColumn(
            "label",
            when(col(target_col) == "Yes", 1.0)
            .when(col(target_col) == "No", 0.0)
            .otherwise(0.0)
        )
        print(f"✓ Target '{target_col}' encodée en 'label' (Yes=1, No=0)")

    # ========================================
    # ÉTAPE 3: Appliquer le pipeline
    # ========================================
    if pipeline_model is None:
        # Créer et fiter un nouveau pipeline
        print(" Création d'un nouveau pipeline...")
        pipeline = build_preprocessing_pipeline()
        pipeline_model = pipeline.fit(df_clean)
        print("✓ Pipeline fité sur les données")
    else:
        print("  Utilisation du pipeline pré-entraîné")

    # Transformer les données
    df_transformed = pipeline_model.transform(df_clean)

    print("✓ Transformation appliquée")

    # ========================================
    # ÉTAPE 4: Sélectionner les colonnes finales
    # ========================================
    # Garder uniquement 'features' et 'label'
    if "label" in df_transformed.columns:
        df_final = df_transformed.select("features", "label")
    else:
        df_final = df_transformed.select("features")

    print("\n" + "="*60)
    print(" PRÉTRAITEMENT TERMINÉ")
    print("="*60)
    print(f" Colonnes finales: {df_final.columns}")
    print(f" Shape: ({df_final.count()} lignes)\n")

    return df_final, pipeline_model


def preprocess_streaming_batch(df, pipeline_model):
    """
    Prétraitement optimisé pour les batches de streaming.
    Utilise un pipeline pré-entraîné (pas de fit).

    Args:
        df: DataFrame Spark du batch
        pipeline_model: PipelineModel pré-entraîné

    Returns:
        DataFrame transformé avec colonne 'features'
    """
    # Nettoyage rapide
    df_clean = clean_missing_values(df)

    # Encoder la target
    target_col = get_target_column()
    if target_col in df_clean.columns:
        df_clean = df_clean.withColumn(
            "label",
            when(col(target_col) == "Yes", 1.0)
            .when(col(target_col) == "No", 0.0)
            .otherwise(0.0)
        )

    # Transformer avec le pipeline pré-entraîné
    df_transformed = pipeline_model.transform(df_clean)

    # Sélectionner les colonnes finales
    if "label" in df_transformed.columns:
        df_final = df_transformed.select("features", "label")
    else:
        df_final = df_transformed.select("features")

    return df_final


# ============================
# UTILITAIRES
# ============================

def get_feature_dimension():
    """
    Retourne la dimension finale du vecteur de features après prétraitement.

    Returns:
        int: Nombre de features (doit être 13 pour le modèle PyTorch)
    """
    # 9 numériques + 11 catégorielles encodées = 20 features
    # MAIS le modèle actuel attend 13 features
    # On doit donc sélectionner les 13 features les plus importantes

    # IMPORTANT: Cette fonction est un placeholder
    # La vraie dimension dépend de l'encodage des catégorielles
    # Pour l'instant, on vise 13 features comme requis par le modèle

    return 13  # Dimension cible pour le modèle PyTorch


def select_top_features(df, n_features=13):
    """
    Sélectionne les N features les plus importantes.
    (Optionnel - pour réduire de 20 à 13 features si nécessaire)

    Args:
        df: DataFrame avec colonne 'features'
        n_features: Nombre de features à garder

    Returns:
        DataFrame avec features réduites
    """
    # TODO: Implémenter la sélection de features si nécessaire
    # Utiliser ChiSqSelector ou PCA de Spark ML

    return df
