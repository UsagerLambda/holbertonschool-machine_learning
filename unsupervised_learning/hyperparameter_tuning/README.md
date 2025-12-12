# Hyperparameter Tuning - Bayesian Optimization

## Description

Ce projet implémente l'optimisation bayésienne d'hyperparamètres pour un CNN sur CIFAR-10 en utilisant des Gaussian Processes.

## Installation

### Linux / Ubuntu
```bash
# Rendre le script exécutable
chmod +x setup.sh

# Lancer le setup
./setup.sh

# Activer l'environnement virtuel
source venv/bin/activate
```

### macOS
```bash
# Rendre le script exécutable
chmod +x setup-macos.sh

# Lancer le setup
./setup-macos.sh

# Activer l'environnement virtuel
source venv/bin/activate
```

### Installation manuelle
```bash
pip install -r requirements.txt
```
ou
```bash
pip install -r requirements-macos.txt
```

## Structure du Projet

```
.
├── 0-gp.py              # Exercices Gaussian Process
├── 1-gp.py
├── 2-gp.py
├── 3-bayes_opt.py       # Exercices optimisation bayésienne
├── 4-bayes_opt.py
├── 5-bayes_opt.py
├── 6-bayes_opt.py       # ⭐ Optimisation complète CNN CIFAR-10
├── setup.sh             # Setup Linux
├── setup-macos.sh       # Setup macOS
├── requirements.txt     # Dépendances
├── models/              # Modèles sauvegardés (généré)
└── img/                 # Plots de convergence (généré)
```

## Lancer l'Optimisation

### Préparer les dossiers
```bash
mkdir -p models img
```

### Lancer l'entraînement
```bash
python3 6-bayes_opt.py
```

### Paramètres optimisés
Le script optimise automatiquement ces hyperparamètres :
- **Batch size** : 128, 256, 512, 1024
- **Learning rate** : 0.0001 à 0.003
- **Dropout** : 0.1 à 0.4
- **Dense units** : 128, 256
- **L2 regularization** : 0.0001 à 0.005
- **Hidden layers** : 1, 2

### Sorties générées
- `models/model_*.keras` : Meilleurs modèles sauvegardés pour chaque itération
- `img/convergence_plot.png` : Visualisation de la convergence
- `bayes_opt.txt` : Rapport détaillé avec tous les hyperparamètres testés

## Configuration

Vous pouvez modifier les paramètres dans `6-bayes_opt.py` :
```python
# Ligne 273
optimizer = Optimizer(max_iter=25, epochs=50)
```

### Changer l'espace de recherche
Modifiez le dictionnaire `bounds` dans la méthode `run()` (ligne 238) :
```python
bounds = [
    {"name": "batch_size", "type": "discrete", "domain": (128, 256, 512, 1024)},
    {"name": "learning_rate", "type": "continuous", "domain": (0.0001, 0.003)},
    # ...
]
```

## Résultats d'Entraînement

### Exemple de sortie console
```
============================================================
CIFAR-10 HYPERPARAMETER OPTIMIZATION
============================================================

[LOADING] Loading CIFAR-10 dataset...

Dataset shapes:
  Training:   (40000, 32, 32, 3)
  Validation: (10000, 32, 32, 3)
  Test:       (10000, 32, 32, 3)

[TESTING]
  Batch size:      1024
  Learning rate:   0.00119
  Dropout:         0.21
  Dense units:     128
  L2 reg:          0.00335
  Hidden layers:   2
  F1-score:         0.7028
  Time:            45.3s

[TESTING]
  Batch size:      512
  Learning rate:   0.00297
  Dropout:         0.16
  Dense units:     128
  L2 reg:          0.00333
  Hidden layers:   2
  F1-score:         0.7221
  Time:            44.2s

[TESTING]
  Batch size:      128
  Learning rate:   0.00140
  Dropout:         0.24
  Dense units:     128
  L2 reg:          0.00201
  Hidden layers:   1
  F1-score:         0.7572
  Time:            37.0s

[TESTING]
  Batch size:      128
  Learning rate:   0.00085
  Dropout:         0.38
  Dense units:     128
  L2 reg:          0.00374
  Hidden layers:   2
  F1-score:         0.7582
  Time:            49.6s

[TESTING]
  Batch size:      512
  Learning rate:   0.00231
  Dropout:         0.23
  Dense units:     128
  L2 reg:          0.00324
  Hidden layers:   1
  F1-score:         0.7433
  Time:            40.1s

============================================================
BAYESIAN OPTIMIZATION - STARTING
============================================================


[TESTING]
  Batch size:      512
  Learning rate:   0.00249
  Dropout:         0.28
  Dense units:     256
  L2 reg:          0.00344
  Hidden layers:   2
  F1-score:         0.7536
  Time:            46.3s

[TESTING]
  Batch size:      512
  Learning rate:   0.00010
  Dropout:         0.10
  Dense units:     256
  L2 reg:          0.00010
  Hidden layers:   1
  F1-score:         0.6495
  Time:            51.9s

[TESTING]
  Batch size:      128
  Learning rate:   0.00146
  Dropout:         0.29
  Dense units:     128
  L2 reg:          0.00397
  Hidden layers:   2
  F1-score:         0.7546
  Time:            48.3s

[TESTING]
  Batch size:      256
  Learning rate:   0.00142
  Dropout:         0.40
  Dense units:     256
  L2 reg:          0.00327
  Hidden layers:   2
  F1-score:         0.7594
  Time:            50.6s

[TESTING]
  Batch size:      256
  Learning rate:   0.00300
  Dropout:         0.10
  Dense units:     256
  L2 reg:          0.00010
  Hidden layers:   2
  F1-score:         0.7130
  Time:            28.5s

[TESTING]
  Batch size:      128
  Learning rate:   0.00100
  Dropout:         0.32
  Dense units:     128
  L2 reg:          0.00255
  Hidden layers:   1
  F1-score:         0.7599
  Time:            42.6s

[TESTING]
  Batch size:      256
  Learning rate:   0.00010
  Dropout:         0.40
  Dense units:     256
  L2 reg:          0.00500
  Hidden layers:   2
  F1-score:         0.6228
  Time:            54.6s

[TESTING]
  Batch size:      128
  Learning rate:   0.00095
  Dropout:         0.32
  Dense units:     128
  L2 reg:          0.00254
  Hidden layers:   1
  F1-score:         0.7683
  Time:            51.5s

[TESTING]
  Batch size:      128
  Learning rate:   0.00072
  Dropout:         0.32
  Dense units:     128
  L2 reg:          0.00252
  Hidden layers:   1
  F1-score:         0.7511
  Time:            42.8s

[TESTING]
  Batch size:      256
  Learning rate:   0.00194
  Dropout:         0.40
  Dense units:     256
  L2 reg:          0.00258
  Hidden layers:   2
  F1-score:         0.7206
  Time:            45.8s

[TESTING]
  Batch size:      128
  Learning rate:   0.00207
  Dropout:         0.24
  Dense units:     128
  L2 reg:          0.00148
  Hidden layers:   1
  F1-score:         0.7499
  Time:            40.2s

[TESTING]
  Batch size:      128
  Learning rate:   0.00144
  Dropout:         0.32
  Dense units:     128
  L2 reg:          0.00230
  Hidden layers:   1
  F1-score:         0.7605
  Time:            46.6s

[TESTING]
  Batch size:      128
  Learning rate:   0.00121
  Dropout:         0.32
  Dense units:     128
  L2 reg:          0.00312
  Hidden layers:   1
  F1-score:         0.7675
  Time:            45.9s

[TESTING]
  Batch size:      128
  Learning rate:   0.00040
  Dropout:         0.32
  Dense units:     128
  L2 reg:          0.00340
  Hidden layers:   1
  F1-score:         0.7567
  Time:            50.8s

[TESTING]
  Batch size:      128
  Learning rate:   0.00073
  Dropout:         0.24
  Dense units:     128
  L2 reg:          0.00101
  Hidden layers:   1
  F1-score:         0.7610
  Time:            40.1s

[TESTING]
  Batch size:      128
  Learning rate:   0.00107
  Dropout:         0.38
  Dense units:     128
  L2 reg:          0.00421
  Hidden layers:   2
  F1-score:         0.7479
  Time:            38.5s

[TESTING]
  Batch size:      128
  Learning rate:   0.00136
  Dropout:         0.24
  Dense units:     128
  L2 reg:          0.00140
  Hidden layers:   1
  F1-score:         0.7303
  Time:            28.9s

[TESTING]
  Batch size:      128
  Learning rate:   0.00194
  Dropout:         0.38
  Dense units:     128
  L2 reg:          0.00416
  Hidden layers:   2
  F1-score:         0.7408
  Time:            41.0s

[TESTING]
  Batch size:      128
  Learning rate:   0.00060
  Dropout:         0.24
  Dense units:     128
  L2 reg:          0.00143
  Hidden layers:   1
  F1-score:         0.7743
  Time:            51.0s

[TESTING]
  Batch size:      128
  Learning rate:   0.00057
  Dropout:         0.24
  Dense units:     128
  L2 reg:          0.00058
  Hidden layers:   1
  F1-score:         0.7608
  Time:            47.6s

[TESTING]
  Batch size:      128
  Learning rate:   0.00010
  Dropout:         0.24
  Dense units:     128
  L2 reg:          0.00192
  Hidden layers:   1
  F1-score:         0.7004
  Time:            51.3s

[TESTING]
  Batch size:      128
  Learning rate:   0.00029
  Dropout:         0.38
  Dense units:     128
  L2 reg:          0.00296
  Hidden layers:   2
  F1-score:         0.7480
  Time:            53.4s

[TESTING]
  Batch size:      128
  Learning rate:   0.00103
  Dropout:         0.24
  Dense units:     128
  L2 reg:          0.00125
  Hidden layers:   1
  F1-score:         0.7501
  Time:            32.5s

[TESTING]
  Batch size:      256
  Learning rate:   0.00085
  Dropout:         0.40
  Dense units:     256
  L2 reg:          0.00316
  Hidden layers:   2
  F1-score:         0.7569
  Time:            49.9s

[TESTING]
  Batch size:      128
  Learning rate:   0.00010
  Dropout:         0.38
  Dense units:     128
  L2 reg:          0.00423
  Hidden layers:   2
  F1-score:         0.6606
  Time:            53.6s

============================================================
OPTIMIZATION COMPLETED
============================================================

```

### Meilleurs hyperparamètres trouvés
```
[TESTING]
  Batch size:      128
  Learning rate:   0.00060
  Dropout:         0.24
  Dense units:     128
  L2 reg:          0.00143
  Hidden layers:   1
  F1-score:         0.7743
  Time:            51.0s
```

---

## Gaussian Process - Concepts Théoriques

### Le Concept Principal

**Un Gaussian Process prédit la valeur d'un nouveau point en se basant sur les points observés les plus proches.**

### Les 3 Concepts Clés

#### 1. Covariance
Mesure à quel point deux points sont corrélés :
- Points proches → covariance forte → valeurs similaires
- Points loin → covariance faible → valeurs indépendantes

#### 2. Kernel (RBF)
```python
K(x₁, x₂) = σf² × exp(-||x₁ - x₂||² / (2l²))
```
- `l` : vitesse de décroissance de la corrélation avec la distance
- `σf` : amplitude des variations

#### 3. Matrice K
Tableau de covariances entre tous les points observés.
```
        Point1  Point2  Point3
Point1 [ 1.00   0.60    0.10 ]
Point2 [ 0.60   1.00    0.60 ]
Point3 [ 0.10   0.60    1.00 ]
```

### Comment ça Marche ?

**Exemple :** Prédire en X=2.5 avec ces points observés :
```
X = [1, 3, 5]
Y = [2.0, 1.5, 3.0]
```

1. Calcule les covariances avec X=2.5 :
   - Cov(1, 2.5) = 0.325 (faible)
   - Cov(3, 2.5) = 0.882 (FORTE ← plus proche)
   - Cov(5, 2.5) = 0.044 (très faible)

2. Moyenne pondérée : `Prédiction = w₁×2.0 + w₂×1.5 + w₃×3.0`

3. Résultat : Y(2.5) ≈ 1.5 (proche de Y(3) car X=2.5 proche de X=3)

### Application : Hyperparameter Tuning

**Problème :** Grid Search teste 300+ combinaisons (lent)

**Solution :** Le GP teste intelligemment 30-50 combinaisons :
1. Teste 5-10 combinaisons au hasard
2. Le GP prédit où sont les bonnes zones
3. Teste les zones prometteuses
4. Répète jusqu'à trouver l'optimum

**Résultat :** 10x plus rapide que Grid Search !

### Points Clés

- ✅ Interpolation intelligente basée sur la proximité
- ✅ Quantifie l'incertitude des prédictions
- ✅ Gère naturellement le bruit dans les mesures
- ✅ Efficace pour l'optimisation bayésienne

## Dépendances

Voir `requirements.txt` pour les versions exactes :
- TensorFlow (avec support CUDA)
- GPyOpt
- scikit-learn
- matplotlib
- numpy
- scipy
