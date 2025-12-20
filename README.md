#  Module Serveur - Cardio Federated System

Ce module contient le **Serveur Central (Agrégateur)** du projet. Son rôle est de coordonner l'apprentissage fédéré entre les différents hôpitaux sans jamais accéder à leurs données brutes.

##  État Actuel : Squelette & Flower 

La structure de base du serveur est opérationnelle. Elle permet d'initier la communication gRPC et de gérer l'agrégation des modèles.

###  Fonctionnalités Clés
- **Orchestration Flower** : Gestion des rounds d'entraînement.
- **Stratégie Custom** : Implémentation de `CardioStrategy` (basée sur FedAvg).
- **Agrégation des Métriques** : Calcul de la moyenne pondérée de l'accuracy basée sur le nombre d'échantillons par client.
- **Environnement Isolé** : Configuration optimisée pour Python 3.11 avec Conda.

---

## 📂 Structure du Répertoire
```text
server/
├── requirements.txt   # Dépendances (Flower, PyTorch, NumPy)
├── server.py          # Script principal de lancement du serveur
└── strategy.py        # Logique d'agrégation personnalisée