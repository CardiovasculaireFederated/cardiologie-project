# README.md
# Projet : Cardio Federated Learning
# TODO: Implémentation à venir
# Projet Cardio Federated Learning

Projet d'apprentissage fédéré pour la prédiction des maladies cardiovasculaires utilisant Apache Spark et Flower Framework.

---

## Table des Matières

- [Description du Projet](#description-du-projet)
- [Architecture](#architecture)
- [Installation](#installation)
- [Structure du Projet](#structure-du-projet)
- [Sprint 1 - P2 : Ingénieur Data (Spécialiste Spark)](#sprint-1---p2--ingénieur-data-spécialiste-spark)
- [Utilisation](#utilisation)
- [Dataset](#dataset)
- [Technologies Utilisées](#technologies-utilisées)

---

## Description du Projet

Ce projet implémente un système d'apprentissage fédéré pour la prédiction des maladies cardiovasculaires. L'objectif est de permettre à plusieurs clients (hôpitaux, cliniques) d'entraîner collaborativement un modèle de machine learning sans partager leurs données sensibles.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Serveur Fédéré                      │
│              (Agrégation des modèles)                   │
└─────────────────────────────────────────────────────────┘
                            ▲
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Client 1    │    │  Client 2    │    │  Client N    │
│  (Spark)     │    │  (Spark)     │    │  (Spark)     │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## Installation

### Prérequis

- Python 3.8+
- Apache Spark 3.x
- Java 8 ou 11
- pip

### Étapes d'Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/CardiovasculaireFederated/cardiologie-project.git
cd cardiologie-project

# 2. Créer un environnement virtuel
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Vérifier l'installation de Spark
python test_spark.py
```

---

## Structure du Projet

```
cardiologie-project/
├── client/                     # Code client Spark
│   ├── spark_manager.py       # Gestion de la session Spark
│   ├── preprocessing.py       # Schéma et prétraitement des données
│   ├── model.py               # Définition du modèle ML
│   ├── trainer.py             # Entraînement du modèle
│   ├── client.py              # Client Flower pour FL
│   └── data_splitter.py       # Division des données
│
├── server/                     # Code serveur Flower
│   └── server.py              # Serveur d'agrégation
│
├── data/                       # Données du projet
│   ├── raw/                   # Données brutes
│   │   └── heart_disease.csv
│   └── processed/             # Données traitées
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── test_spark.py              # Tests de configuration Spark
├── requirements.txt           # Dépendances Python
└── README.md                  # Documentation
```

---

## Sprint 1 - P2 : Ingénieur Data (Spécialiste Spark)

### Objectif

Préparer l'environnement Spark et rendre les données accessibles pour le traitement distribué.

### Livrables Complétés

#### 1. Acquisition des Données

- **Fichier** : [data/raw/heart_disease.csv](data/raw/heart_disease.csv)
- **Description** : Dataset cardiovasculaire avec 21 features et 1 target
- **Taille** : Plus de 1000 enregistrements
- **Colonnes** :
  - Démographiques : Age, Gender
  - Physiologiques : Blood Pressure, Cholesterol Level, BMI
  - Habitudes de vie : Exercise Habits, Smoking, Alcohol Consumption
  - Biomarqueurs : CRP Level, Homocysteine Level, Triglyceride Level
  - Target : Heart Disease Status (Yes/No)

#### 2. Configuration du Moteur Spark

- **Fichier** : [client/spark_manager.py](client/spark_manager.py)
- **Fonctions** :
  - `get_spark_session(app_name)` : Crée une session Spark optimisée
  - `stop_spark_session(spark)` : Arrête proprement la session

**Caractéristiques** :
```python
# Configuration optimisée pour Docker/Laptop
- Executor Memory: 512 MB
- Driver Memory: 512 MB
- Shuffle Partitions: 4
- Log Level: WARN (réduit le bruit)
```

#### 3. Définition du Schéma de Données

- **Fichier** : [client/preprocessing.py](client/preprocessing.py)
- **Fonctions** :
  - `get_heart_schema()` : Retourne le schéma Spark structuré
  - `get_feature_columns()` : Liste des 20 features
  - `get_target_column()` : Nom de la colonne cible
  - `get_categorical_columns()` : Liste des features catégorielles
  - `get_numerical_columns()` : Liste des features numériques

**Schéma complet** :
- 20 colonnes de features (11 catégorielles + 9 numériques)
- 1 colonne target (Heart Disease Status)
- Types : DoubleType pour les numériques, StringType pour les catégorielles

#### 4. Script de Test

- **Fichier** : [test_spark.py](test_spark.py)
- **Tests effectués** :
  1. Création de la session Spark
  2. Vérification du schéma de données
  3. Chargement des données CSV
  4. Identification des colonnes features/target

**Exécution** :
```bash
python test_spark.py
```

**Résultat attendu** :
```
  TESTS DE CONFIGURATION SPARK - PROJET CARDIO ML

TEST 1 : Création de la session Spark
Session Spark créée avec succès!

TEST 2 : Vérification du schéma de données
Schéma créé avec succès!
   - Nombre de colonnes : 21

TEST 3 : Chargement des données CSV
Données chargées avec succès!
   - Nombre de lignes : 1000+
   - Aperçu des données...

TEST 4 : Colonnes de features et target
Colonnes identifiées avec succès!
   - Nombre de features : 20
   - Colonne cible : Heart Disease Status

  TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS!
```

---

## Utilisation

### 1. Tester la Configuration Spark

```bash
# Exécuter le script de test complet
python test_spark.py
```

### 2. Utiliser le Gestionnaire Spark dans votre code

```python
from client.spark_manager import get_spark_session, stop_spark_session
from client.preprocessing import get_heart_schema

# Créer une session Spark
spark = get_spark_session("MonApplication")

# Charger les données avec le schéma
schema = get_heart_schema()
df = spark.read.csv("data/raw/heart_disease.csv", header=True, schema=schema)

# Afficher les données
df.show(5)

# Arrêter la session
stop_spark_session(spark)
```

### 3. Accéder aux Métadonnées

```python
from client.preprocessing import (
    get_feature_columns,
    get_target_column,
    get_categorical_columns,
    get_numerical_columns
)

# Récupérer les colonnes
features = get_feature_columns()  # 20 features
target = get_target_column()      # "Heart Disease Status"
categorical = get_categorical_columns()  # 11 colonnes
numerical = get_numerical_columns()      # 9 colonnes
```

---

## Dataset

### Source
Dataset cardiovasculaire personnalisé avec facteurs de risque cliniques.

### Statistiques
- **Nombre d'enregistrements** : 1000+
- **Nombre de features** : 20
- **Target** : Binaire (Yes/No pour Heart Disease Status)
- **Valeurs manquantes** : Quelques colonnes (Fasting Blood Sugar, etc.)

### Features Principales

| Catégorie | Features |
|-----------|----------|
| **Démographiques** | Age, Gender |
| **Physiologiques** | Blood Pressure, Cholesterol, BMI |
| **Habitudes** | Exercise, Smoking, Alcohol, Sleep, Sugar, Stress |
| **Antécédents** | Family Heart Disease, Diabetes |
| **Conditions** | High BP, Low HDL, High LDL |
| **Biomarqueurs** | Triglycerides, Fasting Blood Sugar, CRP, Homocysteine |

---

## Technologies Utilisées

- **Apache Spark** : Traitement distribué des données
- **PySpark** : API Python pour Spark
- **Flower Framework** : Apprentissage fédéré (à venir)
- **Python 3.8+** : Langage de programmation
- **Docker** : Conteneurisation (à venir)

---

## Prochaines Étapes (Sprint 2)

- [ ] P3 : Ingénieur ML - Développement du modèle de classification
- [ ] P4 : Ingénieur DevOps - Containerisation avec Docker
- [ ] P1 : Architecte FL - Intégration Flower Framework
- [ ] Tests d'intégration complète
- [ ] Documentation API

---

## Contributeurs

- **P2 - Ingénieur Data (Spécialiste Spark)** : Configuration Spark et gestion des données

---

## Licence

Ce projet est développé dans un cadre éducatif.

---

## Support

Pour toute question ou problème :
1. Vérifier que Java et Spark sont correctement installés
2. Exécuter `python test_spark.py` pour diagnostiquer
3. Consulter les logs Spark dans `./spark-logs/`

---

**Dernière mise à jour** : Sprint 1 - Configuration Spark complétée
