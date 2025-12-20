# client/trainer.py - VERSION STABLE

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Entraîne le modèle pour une seule époque."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device).float().unsqueeze(1)
        
        assert X.shape[1] == 13, f"Expected 13 features, got {X.shape[1]}"
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        
        # Vérifier si loss est NaN
        if torch.isnan(loss):
            print(f"⚠️  NaN détecté au batch {batch_idx}! Skip...")
            continue
        
        loss.backward()
        
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        predictions = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (predictions == y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    """Entraîne le modèle sur plusieurs époques."""
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    print(f"\n✓ Meilleure Val Accuracy: {best_val_acc:.4f}")
    
    return history


def evaluate(model, dataloader, loss_fn, device):
    """Évalue le modèle."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device).float().unsqueeze(1)
            
            outputs = model(X)
            loss = loss_fn(outputs, y)
            
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predictions == y).sum().item()
            total += y.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy