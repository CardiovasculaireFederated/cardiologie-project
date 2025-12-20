# client/data_loader.py - VERSION ULTRA-STABLE

import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path, test_size=0.2, val_size=0.1, batch_size=32, random_state=42):
    """
    Charge les données avec gestion robuste des valeurs problématiques.
    """
    
    # ============================================
    # ÉTAPE 1: Charger et nettoyer
    # ============================================
    df = pd.read_csv(file_path)
    
    print(f"Dataset original: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Supprimer les lignes avec valeurs manquantes
    df = df.dropna()
    print(f"Après suppression NaN: {df.shape[0]} lignes")
    
    # ============================================
    # ÉTAPE 2: Target
    # ============================================
    target_column = 'Heart Disease Status'
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # ============================================
    # ÉTAPE 3: Encoder les catégorielles
    # ============================================
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\nFeatures:")
    print(f"  - Numériques: {len(numerical_cols)}")
    print(f"  - Catégorielles: {len(categorical_cols)}")
    
    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # ============================================
    # ÉTAPE 4: Sélectionner 13 features
    # ============================================
    if X.shape[1] > 13:
        selected_features = X.columns[:13].tolist()
        print(f"\n✓ 13 features sélectionnées: {selected_features}")
        X = X[selected_features]
    
    # ============================================
    # ÉTAPE 5: Encoder la target
    # ============================================
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)
    
    # Convertir en numpy
    X = X.values.astype(np.float32)  # float32 au lieu de float64
    y = y.astype(np.float32)
    
    # ============================================
    # ÉTAPE 6: Remplacer inf et valeurs extrêmes
    # ============================================
    # Remplacer inf par NaN puis par la médiane
    X = np.where(np.isinf(X), np.nan, X)
    
    # Clipper les valeurs extrêmes (percentile 1% et 99%)
    for i in range(X.shape[1]):
        col = X[:, i]
        # Enlever les NaN pour calculer les percentiles
        col_clean = col[~np.isnan(col)]
        if len(col_clean) > 0:
            p1, p99 = np.percentile(col_clean, [1, 99])
            X[:, i] = np.clip(col, p1, p99)
    
    # Remplacer les NaN restants par la médiane
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    
    print(f"\n✓ Shape après nettoyage: X={X.shape}, y={y.shape}")
    print(f"✓ Valeurs NaN dans X: {np.isnan(X).sum()}")
    print(f"✓ Valeurs inf dans X: {np.isinf(X).sum()}")
    
    # ============================================
    # ÉTAPE 7: Normalisation ROBUSTE
    # ============================================
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Vérifier après normalisation
    X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)
    
    print(f"✓ Range après normalisation: [{X.min():.2f}, {X.max():.2f}]")
    
    # ============================================
    # ÉTAPE 8: Split
    # ============================================
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    # ============================================
    # ÉTAPE 9: Tenseurs PyTorch
    # ============================================
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # ============================================
    # ÉTAPE 10: DataLoaders
    # ============================================
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n{'='*50}")
    print("DATASET PRÉPARÉ:")
    print(f"  - Train: {len(train_dataset)} échantillons")
    print(f"  - Val:   {len(val_dataset)} échantillons")
    print(f"  - Test:  {len(test_dataset)} échantillons")
    print(f"  - Batch size: {batch_size}")
    print(f"{'='*50}\n")
    
    return train_loader, val_loader, test_loader