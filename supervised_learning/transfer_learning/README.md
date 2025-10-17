# Transfer Learning - Entraînement de modèles CIFAR-10

Ce projet implémente un système d'entraînement de modèles de classification d'images utilisant le transfer learning sur le dataset CIFAR-10.

## 📋 Prérequis

### Installation automatique (recommandée)

Le script `run.sh` installe automatiquement toutes les dépendances :

**macOS/Linux :**
```bash
chmod +x run.sh  # Une seule fois
./run.sh train   # Installation automatique + entraînement
```

### Installation manuelle

Si vous préférez installer manuellement :

**Dépendances de base :**
```bash
pip install -r requirements.txt
```

**Optimisations macOS Apple Silicon (M1/M2) :**
```bash
pip install -r requirements-macos.txt
```

### Versions requises
- Python 3.8+
- TensorFlow 2.8+ (+ tensorflow-metal sur macOS Apple Silicon)
- Matplotlib 3.5+
- NumPy 1.21+

## 🚀 Utilisation

### Méthode recommandée : Script interactif

**macOS/Linux :**
```bash
./run.sh
```

Le script vous proposera 3 options :

#### **Option 1 : Entraînement (0-transfer.py)**
- 🎯 **Objectif** : Entraîner un nouveau modèle de classification CIFAR-10
- ⏱️ **Durée** : ~14 minutes (selon votre matériel)
- 💾 **Sortie** : Crée `cifar10.h5` + graphiques dans `img/`
- 🔧 **Quand l'utiliser** : Première fois ou pour réentraîner avec de nouveaux paramètres

#### **Option 2 : Évaluation (0-main.py)**
- 🎯 **Objectif** : Tester les performances d'un modèle déjà entraîné
- ⏱️ **Durée** : 30 secondes - 2 minutes
- 📊 **Sortie** : Affiche l'accuracy finale sur l'ensemble de test complet
- 🔧 **Quand l'utiliser** : Après avoir entraîné un modèle (option 1)
- ⚠️ **Prérequis** : Fichier `cifar10.h5` doit exister

#### **Option 3 : Installation des dépendances**
- 🎯 **Objectif** : Installer/mettre à jour TensorFlow et les dépendances
- 💡 **Auto-détection** : Installe `tensorflow-metal` si macOS Apple Silicon
- 🔧 **Quand l'utiliser** : 
  - Première utilisation du projet
  - Problèmes d'import Python
  - Après mise à jour de Python

### Méthode alternative : Exécution directe

Si vous préférez lancer les scripts manuellement :

**Entraînement :**
```bash
python3 0-transfer.py
```

**Évaluation :**
```bash
python3 0-main.py
```

### Processus d'entraînement complet

1. **Chargement** : Dataset CIFAR-10 (50,000 images d'entraînement)
2. **Prétraitement** : Redimensionnement 32x32 → 224x224 + normalisation
3. **Modèle** : Transfer learning avec EfficientNetV2B1 pré-entraîné
4. **Entraînement** : Fine-tuning avec early stopping
5. **Sauvegarde** : Meilleur modèle dans `cifar10.h5`
6. **Graphiques** : Courbes d'apprentissage dans `img/training_plots.png`

## ⚙️ Configuration

### Paramètres actuels (optimisés pour de bonnes performances)

Les variables dans `0-transfer.py` sont actuellement configurées pour obtenir **~91% de validation accuracy** :

```python
LEARNING_RATE = 0.001         # Taux d'apprentissage optimal pour convergence stable
BATCH_SIZE = 128              # Taille des mini-batchs recommandée
EPOCHS = 5                    # Suffisant avec early stopping
DROPOUT_RATE = 0.3            # Dropout modéré
DENSE_UNITS = 256             # Unités dans la couche dense
TRAIN_SAMPLES = 50000         # Utiliser tout le train set
VAL_SAMPLES = 10000           # Utiliser tout le test set
MODEL_NAME = 'EfficientNetV2B1'  # Modèle à utiliser
PATIENCE = 5
TRAINABLE = True              # Boolean d'activation du dégèle des couches
NB_UNFREEZE_LAYERS = 15       # Nombre de couches à décongeler (si TRAINABLE == True)
PLOT = True                   # Boolean pour afficher les graphs à la fin de l'entrainement
```


### Modèles disponibles

Le code supporte plusieurs architectures pré-entraînées :
- `EfficientNetV2B1` (recommandé)
- `MobileNetV2`
- `ResNet50`

## 📊 Sortie

Après l'entraînement, vous obtiendrez :

### Fichiers générés
- `cifar10.h5` : Le meilleur modèle sauvegardé
- `img/training_plots.png` : Graphiques de suivi de l'entraînement

### Graphiques de suivi
- **Précision** : Évolution de l'accuracy sur train/validation
- **Perte** : Évolution de la loss sur train/validation  
- **Précision en %** : Version en pourcentage de l'accuracy
- **Détection d'overfitting** : Différence train-validation

### Résumé d'entraînement
Le script affiche automatiquement :
- Configuration utilisée
- Résultats finaux (accuracy, loss)
- Diagnostic d'overfitting/underfitting
- Temps d'exécution détaillé

## 📁 Structure des fichiers

```
transfer_learning/
├── 0-transfer.py           # 🎯 Option 1: Script d'entraînement
├── 0-main.py              # 📊 Option 2: Script d'évaluation
├── plot.py                # Affiche des plots avec matplot si la variable PLOT est sur True (pour 0-transfer.py)
├── run.sh                 # 🚀 Script interactif (3 options)
├── requirements.txt       # 📦 Dépendances de base
├── requirements-macos.txt # 🍎 Optimisations macOS (tensorflow-metal)
├── README.md              # 📖 Ce fichier
├── cifar10.h5             # 🤖 Modèle sauvegardé (généré par option 1)
└── img/                   # 📈 Graphiques (créé automatiquement)
    └── training_plots.png # 📊 Courbes d'apprentissage
```

### Menu interactif du script

Quand vous lancez `./run.sh`, vous avez ces choix :

```
Quel script voulez-vous lancer ?
1) 0-transfer.py (Entraînement)   ← Créer/entraîner un modèle
2) 0-main.py (Évaluation)         ← Tester un modèle existant
3) Installer les dépendances      ← Setup/réparation environnement
```

## 🔧 Fonctionnalités

### Transfer Learning
- Utilise des modèles pré-entraînés sur ImageNet
- Support du fine-tuning (dégel de couches)
- Preprocessing automatique selon le modèle choisi

### Monitoring
- Early stopping basé sur la validation accuracy
- Sauvegarde du meilleur modèle automatique
- Graphiques détaillés de l'entraînement

### Optimisations
- Redimensionnement automatique 32x32 → 224x224
- Augmentation de données intégrée
- Gestion mémoire optimisée

## 💡 Conseils d'utilisation

### Pour éviter l'overfitting :
- Augmentez `DROPOUT_RATE`
- Diminuez `EPOCHS`
- Réduisez `TRAIN_SAMPLES`

### Pour améliorer les performances :
- Utilisez `EfficientNetV2B1` ou `EfficientNetB0`
- Activez le fine-tuning avec `TRAINABLE = True`
- Augmentez `NB_UNFREEZE_LAYERS` graduellement

### Pour un entraînement plus rapide :
- Utilisez `MobileNetV2`
- Réduisez `TRAIN_SAMPLES` et `VAL_SAMPLES`
- Augmentez `BATCH_SIZE` si votre GPU le permet

## 📈 Résultats attendus

### Configuration de test optimisée (variables actuelles dans 0-transfer.py)

Les paramètres actuels ont été optimisés et testés avec les résultats suivants :

```python
LEARNING_RATE = 0.001      # Taux d'apprentissage réduit pour stabilité
EPOCHS = 5                 # Plus d'époques pour convergence
DROPOUT_RATE = 0.3         # Dropout modéré
NB_UNFREEZE_LAYERS = 15    # Fine-tuning plus agressif
```


### Performances obtenues (MacBook M1 Pro 16GB)

**Configuration de test :**
- 🖥️ **Matériel** : MacBook M1 Pro avec 16GB RAM
- ⚡ **Accélération** : tensorflow-metal (GPU Apple Silicon)
- 🎯 **Modèle** : EfficientNetV2B1 avec fine-tuning (40 couches dégelées)

**Meilleur résultat expérimental :**
```
============================================================
📊 RÉSUMÉ DE L'ENTRAÎNEMENT
============================================================
🔧 Configuration:
  • Modèle: EfficientNetV2B1
  • Learning Rate: 0.001
  • Batch Size: 128
  • Dropout: 0.3
  • Dense Units: 256
  • Unfreezed layers: 40

📈 Résultats finaux:
  • Training Accuracy: 88.75%
  • Validation Accuracy: 93.79%
  • Meilleure val accuracy: 93.91%
  • Training Loss: 0.3257
  • Validation Loss: 0.1859

🔍 Diagnostic:
  ✅ Bon équilibre! Différence: -5.04%

⏱️  Temps d'exécution:
  • Temps total: 867.73s (14.46min)
  • Temps d'entraînement: 864.73s (14.41min)
  • Temps par époque: 172.95s
============================================================
```
**Évaluation :**
```
313/313 ━━━━━━━━━━━━━━━━━━━━ 36s 100ms/step - accuracy: 0.9391 - loss: 0.1830
```

Vos résultats peuvent varier selon votre configuration :

### Facteurs influençant les performances

- **🔥 RAM disponible** : 8GB minimum, 16GB recommandé
- **⚡ Type de processeur** : Apple Silicon > GPU NVIDIA > CPU moderne > CPU ancien
- **🌡️ Thermique** : Les performances peuvent diminuer si surchauffe
- **💾 Stockage** : SSD plus rapide que HDD pour le chargement des données

## 🖥️ Optimisations par plateforme

### macOS Apple Silicon (M1/M2/M3)
- **Accélération GPU** : Installation automatique de `tensorflow-metal`
- **Performance** : ~2-3x plus rapide qu'en mode CPU uniquement
- **Détection automatique** : Le script détecte l'architecture `arm64`

### macOS Intel
- **TensorFlow standard** : Pas d'optimisations GPU spéciales
- **Compatibilité** : Fonctionne avec TensorFlow classique

### Linux
- **TensorFlow standard** : Compatible avec CUDA si disponible
- **Flexibilité** : Possibilité d'ajouter tensorflow-gpu manuellement

### Windows
- **PowerShell Core** : Compatible avec PowerShell 5.1+ et 7+
- **TensorFlow standard** : Support CUDA possible

## ⚠️ Notes importantes

### Configuration et compatibilité
- Le code gère automatiquement les certificats SSL pour le téléchargement des modèles
- Les images CIFAR-10 sont automatiquement redimensionnées pour correspondre aux modèles pré-entraînés
- Le preprocessing est spécifique à chaque architecture de modèle
- **macOS M1/M2** : tensorflow-metal peut prendre 1-2 minutes pour s'installer au premier lancement

### Performance et résultats
- **Résultats de référence** obtenus avec MacBook M1 Pro 16GB + tensorflow-metal
- **Variabilité** : Vos résultats peuvent différer selon votre matériel (±3-5% d'accuracy)
- **Reproductibilité** : Les résultats peuvent légèrement varier entre les exécutions (nature stochastique de l'entraînement)
- **Recommandation** : Si vos résultats sont < 85%, essayez l'option 3 du menu pour réinstaller tensorflow-metal

### Dépannage courant
- **Erreur de mémoire** : Réduisez `BATCH_SIZE` ou `TRAIN_SAMPLES`
- **Lenteur excessive** : Vérifiez que tensorflow-metal est installé (macOS M1/M2)
- **Accuracy faible** : Augmentez `EPOCHS` ou `NB_UNFREEZE_LAYERS`
