# client/model.py - VERSION COMPLÈTE

import torch
import torch.nn as nn

class HeartDiseaseModel(nn.Module):
    """
    Modèle de classification binaire pour la détection des maladies cardiaques.
    """
    
    def __init__(self, input_dim=20, hidden_dim1=64, hidden_dim2=32, hidden_dim3=16):
        super(HeartDiseaseModel, self).__init__()

        # Le modèle accepte maintenant 20 features (9 numériques + 11 catégorielles encodées)
        # Peut être ajusté selon le preprocessing
        
        # Architecture avec BatchNorm
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.bn3 = nn.BatchNorm1d(hidden_dim3)
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(hidden_dim3, 1)
        
        # Initialisation Xavier
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Vérification flexible de la dimension d'entrée
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        
        return x
