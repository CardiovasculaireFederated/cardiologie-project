"""
Project-wide constants and configuration.
"""

from pathlib import Path

# Project root path
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_PATHS = {
    "raw": str(PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"),
    "processed": str(PROJECT_ROOT / "data" / "processed"),
    "server_test": str(PROJECT_ROOT / "data" / "server_test"),
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Model configuration
MODEL_CONFIG = {
    "test_size": 0.15,
    "validation_size": 0.15,
    "train_size": 0.70,
    "random_state": RANDOM_SEED,
}

# Kafka configuration
KAFKA_CONFIG = {
    "bootstrap_servers": ["localhost:9092"],
    "topic_prefix": "cardiologie",
}

# Spark configuration
SPARK_CONFIG = {
    "app_name": "cardiologie-ml",
    "master": "local[*]",
}

# Target column name
TARGET_COLUMN = "Heart Disease Status"

# Feature columns (all except target)
FEATURE_COLUMNS = [
    "Age", "Gender", "Blood Pressure", "Cholesterol Level", "Exercise Habits",
    "Smoking", "Family Heart Disease", "Diabetes", "BMI", "High Blood Pressure",
    "Low HDL Cholesterol", "High LDL Cholesterol", "Alcohol Consumption",
    "Stress Level", "Sleep Hours", "Sugar Consumption", "Triglyceride Level",
    "Fasting Blood Sugar", "CRP Level", "Homocysteine Level"
]