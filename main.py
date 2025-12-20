# main.py - VERSION CORRIGÉE

import torch
from client.model import HeartDiseaseModel
from client.trainer import train_model
from client.data_loader import load_data

# ============================================
# CONFIGURATION
# ============================================
DATA_PATH = "D:/cardiologie-project/train.csv"

BATCH_SIZE = 32
LEARNING_RATE = 0.00005
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("ENTRAÎNEMENT DU MODÈLE DE DÉTECTION CARDIAQUE")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print()
    
    # 1. CHARGER LES DONNÉES
    print("Chargement des données...")
    train_loader, val_loader, test_loader = load_data(
        file_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        test_size=0.2,
        val_size=0.1
    )
    print()
    
    # 2. CRÉER LE MODÈLE
    print("Création du modèle...")
    model = HeartDiseaseModel(input_dim=13)  # ← SIMPLIFIÉ
    print(f"Modèle créé: {sum(p.numel() for p in model.parameters())} paramètres")
    print()
    
    # 3. ENTRAÎNER LE MODÈLE
    print("Début de l'entraînement...")
    print("-" * 50)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )
    print("-" * 50)
    print("Entraînement terminé!")
    print()
    
    # 4. ÉVALUER SUR LE TEST SET
    print("Évaluation sur le test set...")
    from client.trainer import evaluate
    import torch.nn as nn
    
    loss_fn = nn.BCEWithLogitsLoss()
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, DEVICE)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print()
    
    # 5. SAUVEGARDER LE MODÈLE
    print("Sauvegarde du modèle...")
    torch.save(model.state_dict(), "heart_disease_model.pth")
    print("Modèle sauvegardé: heart_disease_model.pth")
    
    # 6. VISUALISER L'HISTORIQUE (optionnel)
    print("\n" + "="*50)
    print("RÉSUMÉ DE L'ENTRAÎNEMENT:")
    print(f"  - Meilleure Train Acc: {max(history['train_acc']):.4f}")
    print(f"  - Meilleure Val Acc: {max(history['val_acc']):.4f}")
    print(f"  - Loss finale Train: {history['train_loss'][-1]:.4f}")
    print(f"  - Loss finale Val: {history['val_loss'][-1]:.4f}")
    print("="*50)
    
    print("\n" + "="*50)
    print("TERMINÉ!")
    print("="*50)