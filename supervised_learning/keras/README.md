Tensorflow 2 & # Keras

## Introduction

Keras est une API de haut niveau pour le deep learning, maintenant intégrée dans TensorFlow. Ce module enseigne l'utilisation de Keras pour construire, entraîner et évaluer des réseaux de neurones de manière efficace et intuitive.

## Pourquoi Keras ?

Keras simplifie le deep learning en offrant :
- **API simple et intuitive** : Code lisible et facile à comprendre
- **Modularité** : Composants réutilisables et combinables
- **Extensibilité** : Possibilité d'ajouter des couches personnalisées
- **Performance** : Optimisé et distribué avec TensorFlow
- **Communauté active** : Nombreuses ressources et exemples
- **Production ready** : Déploiement facilité sur différentes plateformes

## Comment utiliser Keras ?

### Progression du module :

1. **Construction de modèles (0-3)** : Architecture et compilation
2. **Entraînement (4-8)** : Différentes stratégies d'apprentissage
3. **Gestion de modèles (9-13)** : Sauvegarde, chargement et prédictions

### Workflow typique :

```python
# 1. Construire le modèle
model = build_model(nx, layers, activations, lambtha, keep_prob)

# 2. Compiler le modèle
model = optimize_model(model, alpha, beta1, beta2)

# 3. Entraîner le modèle
history = train_model(model, data, labels, batch_size, epochs)

# 4. Évaluer et prédire
predictions = predict(model, data)
```

## Quand utiliser quelles techniques ?

### Types de modèles :
- **Sequential** : Pour des architectures linéaires simples
- **Functional API** : Pour des architectures complexes avec branches

### Optimiseurs :
- **Adam** : Bon choix par défaut, adaptatif
- **SGD** : Plus simple, nécessite tuning du learning rate
- **RMSprop** : Bon pour les RNN

### Callbacks et validation :
- **Validation Split** : Division automatique train/validation
- **Early Stopping** : Arrêt anticipé pour éviter le surajustement
- **Model Checkpoint** : Sauvegarde des meilleurs modèles

## Concepts clés de Keras

### Architecture :
- **Layers (Couches)** : Dense, Conv2D, LSTM, etc.
- **Activations** : ReLU, Sigmoid, Tanh, Softmax
- **Regularization** : L1, L2, Dropout

### Compilation :
- **Loss Functions** : Categorical crossentropy, MSE, etc.
- **Optimizers** : Adam, SGD, RMSprop
- **Metrics** : Accuracy, Precision, Recall

### Entraînement :
- **Batch Size** : Taille des lots pour l'entraînement
- **Epochs** : Nombre de passages sur les données
- **Callbacks** : Actions pendant l'entraînement

---

## Lexique des termes utilisés

### A
- **Activation** : Fonction d'activation appliquée à la sortie d'une couche
- **Adam** : Optimiseur adaptatif populaire
- **API** : Application Programming Interface

### B
- **Batch Size** : Taille du lot de données traité simultanément
- **Backend** : Moteur de calcul sous-jacent (TensorFlow)

### C
- **Callback** : Fonction appelée à certains moments de l'entraînement
- **Compile** : Configurer le modèle pour l'entraînement
- **Checkpoint** : Point de sauvegarde du modèle

### D
- **Dense** : Couche entièrement connectée
- **Dropout** : Technique de régularisation

### E
- **Epochs** : Nombre de passages complets sur les données d'entraînement
- **Early Stopping** : Arrêt anticipé de l'entraînement

### F
- **Fit** : Méthode pour entraîner le modèle
- **Functional API** : API de Keras pour créer des modèles complexes

### H
- **History** : Objet contenant l'historique de l'entraînement
- **HDF5** : Format de fichier pour sauvegarder les modèles

### I
- **Input Shape** : Forme des données d'entrée

### K
- **Keras** : API de haut niveau pour le deep learning
- **Kernel** : Matrice de poids dans une couche

### L
- **Layer** : Couche du réseau de neurones
- **Loss** : Fonction de perte à minimiser
- **Learning Rate** : Taux d'apprentissage

### M
- **Model** : Représentation du réseau de neurones
- **Metrics** : Métriques pour évaluer les performances

### O
- **Optimizer** : Algorithme d'optimisation (Adam, SGD, etc.)
- **One-hot** : Encodage binaire des classes

### P
- **Predict** : Faire des prédictions avec le modèle
- **Parameters** : Poids et biais du modèle

### R
- **Regularization** : Techniques pour éviter le surajustement
- **ReLU** : Rectified Linear Unit (max(0, x))

### S
- **Sequential** : Type de modèle Keras en séquence linéaire
- **Shuffle** : Mélanger les données d'entraînement
- **Softmax** : Fonction d'activation pour classification multi-classes

### T
- **TensorFlow** : Framework de machine learning de Google
- **Training** : Phase d'apprentissage du modèle

### V
- **Validation** : Données pour évaluer le modèle pendant l'entraînement
- **Verbose** : Contrôle de l'affichage pendant l'entraînement

### W
- **Weights** : Poids du modèle

---

## Structure des fichiers

### Construction et compilation (0-3) :
- `0-sequential.py` : Création d'un modèle séquentiel
- `1-input.py` : Modèle avec couche d'entrée explicite
- `2-optimize.py` : Configuration de l'optimiseur
- `3-one_hot.py` : Encodage one-hot des labels

### Entraînement (4-8) :
- `4-train.py` : Entraînement basique
- `5-train.py` : Avec validation
- `6-train.py` : Avec early stopping
- `7-train.py` : Avec learning rate decay
- `8-train.py` : Avec sauvegarde du meilleur modèle

### Gestion et utilisation (9-13) :
- `9-model.py` : Sauvegarde et chargement complet
- `10-weights.py` : Sauvegarde et chargement des poids
- `11-config.py` : Configuration du modèle
- `12-test.py` : Évaluation du modèle
- `13-predict.py` : Prédictions

Chaque fichier `main.py` démontre l'utilisation pratique avec des exemples concrets.
