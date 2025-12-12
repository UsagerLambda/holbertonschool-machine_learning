# Regularization

## Introduction

La régularisation est un ensemble de techniques essentielles pour lutter contre le surajustement (overfitting) en machine learning. Ce module explore les principales méthodes de régularisation : L2, Dropout et Early Stopping.

## Pourquoi la régularisation ?

Le surajustement est l'ennemi numéro un des modèles de machine learning. La régularisation permet de :
- **Réduire la variance** : Diminuer la sensibilité aux données d'entraînement
- **Améliorer la généralisation** : Meilleures performances sur données inconnues
- **Contrôler la complexité** : Éviter que le modèle devienne trop complexe
- **Stabiliser l'entraînement** : Convergence plus robuste
- **Gérer le bruit** : Moins sensible aux données aberrantes

## Comment fonctionne la régularisation ?

### Types de régularisation implémentés :

1. **Régularisation L2 (0-3)** : Pénalise les poids importants
2. **Dropout (4-6)** : Désactive aléatoirement des neurones
3. **Early Stopping (7)** : Arrête l'entraînement au moment optimal

### Principe général :
```
Fonction de coût classique + Terme de régularisation = Fonction de coût régularisée
```

## Quand utiliser chaque technique ?

### Régularisation L2 (Ridge) :
- **Principe** : Ajoute la somme des carrés des poids à la fonction de coût
- **Effet** : Réduit les poids vers zéro, distribue l'importance
- **Quand** : Toujours comme baseline, surtout avec peu de données
- **Paramètre** : λ (lambda) contrôle l'intensité de la régularisation

### Dropout :
- **Principe** : Désactive aléatoirement des neurones pendant l'entraînement
- **Effet** : Force le réseau à ne pas trop dépendre de neurones spécifiques
- **Quand** : Réseaux profonds, grandes datasets, overfitting persistant
- **Paramètre** : keep_prob (probabilité de garder un neurone)

### Early Stopping :
- **Principe** : Surveille la performance sur données de validation
- **Effet** : Arrête l'entraînement quand la validation se dégrade
- **Quand** : Toujours recommandé, simple et efficace
- **Paramètre** : patience (nombre d'epochs sans amélioration)

## Comment choisir les hyperparamètres ?

### Lambda (L2) :
- **Petit (0.01)** : Régularisation légère
- **Moyen (0.1)** : Régularisation modérée
- **Grand (1.0)** : Régularisation forte
- **Validation croisée** : Pour trouver la valeur optimale

### Keep_prob (Dropout) :
- **0.8-0.9** : Couches cachées (drop 10-20%)
- **0.9-0.95** : Couches de sortie (drop 5-10%)
- **Jamais sur la couche de sortie** : En classification

### Patience (Early Stopping) :
- **5-10 epochs** : Datasets petites
- **20-50 epochs** : Datasets moyennes
- **100+ epochs** : Grandes datasets, entraînement long

## Indicateurs de surajustement

- **Écart train/validation** : Accuracy train >> validation
- **Courbes d'apprentissage** : Train continue de descendre, validation remonte
- **Complexité du modèle** : Trop de paramètres par rapport aux données
- **Variance élevée** : Performances très variables selon le split

---

## Lexique des termes utilisés

### B
- **Bias** : Biais du modèle, erreur systématique
- **Bias-Variance Tradeoff** : Compromis entre biais et variance

### D
- **Dropout** : Technique qui désactive aléatoirement des neurones
- **Decay** : Diminution progressive d'un paramètre

### E
- **Early Stopping** : Arrêt précoce de l'entraînement
- **Epoch** : Passage complet sur les données d'entraînement
- **Exponential Moving Average** : Moyenne mobile exponentielle

### G
- **Generalization** : Capacité à bien performer sur données inconnues
- **Gradient** : Dérivée de la fonction de coût

### H
- **Hyperparameter** : Paramètre externe au modèle (lambda, keep_prob)

### I
- **Inverted Dropout** : Technique de normalisation du dropout

### K
- **Keep_prob** : Probabilité qu'un neurone soit conservé en dropout

### L
- **L1 Regularization** : Régularisation avec valeur absolue des poids
- **L2 Regularization** : Régularisation avec carré des poids (Ridge)
- **Lambda (λ)** : Paramètre de régularisation
- **Loss** : Fonction de perte

### M
- **Mask** : Masque binaire pour le dropout
- **Model Complexity** : Complexité du modèle

### O
- **Overfitting** : Surajustement du modèle aux données d'entraînement

### P
- **Patience** : Nombre d'epochs à attendre sans amélioration
- **Penalty** : Pénalité ajoutée à la fonction de coût

### R
- **Regularization** : Régularisation, techniques anti-surajustement
- **Ridge Regression** : Régression avec régularisation L2

### S
- **Shrinkage** : Réduction des poids vers zéro

### T
- **Training** : Phase d'entraînement
- **Test Set** : Jeu de données de test

### U
- **Underfitting** : Sous-ajustement du modèle
- **Unit** : Neurone ou unité dans le réseau

### V
- **Validation** : Données pour évaluer pendant l'entraînement
- **Variance** : Sensibilité du modèle aux variations des données

### W
- **Weights** : Poids du réseau de neurones
- **Weight Decay** : Diminution des poids (autre nom pour L2)

---

## Structure des fichiers

### Régularisation L2 (0-3) :
- `0-l2_reg_cost.py` : Calcul du coût avec régularisation L2
- `1-l2_reg_gradient_descent.py` : Gradient descent avec L2
- `2-l2_reg_cost.py` : L2 avec TensorFlow
- `3-l2_reg_create_layer.py` : Couche avec L2 intégrée

### Dropout (4-6) :
- `4-dropout_forward_prop.py` : Propagation avant avec dropout
- `5-dropout_gradient_descent.py` : Gradient descent avec dropout
- `6-dropout_create_layer.py` : Couche avec dropout intégré

### Early Stopping (7) :
- `7-early_stopping.py` : Implémentation de l'arrêt précoce

Chaque fichier `main.py` démontre l'efficacité de la technique correspondante contre le surajustement.
