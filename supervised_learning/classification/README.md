# Classification

## Introduction

La classification est un type d'apprentissage supervisé où l'objectif est de prédire la classe ou catégorie d'un échantillon. Ce module explore les concepts fondamentaux des réseaux de neurones pour résoudre des problèmes de classification binaire et multi-classes.

## Pourquoi la classification ?

La classification est cruciale en machine learning car elle permet de :
- **Automatiser les décisions** : Classifier automatiquement des emails comme spam/non-spam
- **Reconnaissance de patterns** : Identifier des objets dans des images
- **Diagnostic médical** : Classifier des symptômes pour diagnostiquer des maladies
- **Analyse de sentiment** : Classer des avis clients comme positifs/négatifs

## Comment ça fonctionne ?

### Progression des concepts :

1. **Neurone simple (0-7)** : Un perceptron basique avec fonction d'activation sigmoïde
2. **Réseau de neurones (8-15)** : Une couche cachée avec plusieurs neurones
3. **Réseau de neurones profond (16-28)** : Plusieurs couches cachées pour des problèmes complexes

### Architecture progressive :

```
Neurone → Réseau 1 couche → Réseau profond
   |            |                |
sigmoid     sigmoid        sigmoid + ReLU
   |            |                |
1 sortie    1 sortie      multiple sorties
```

## Quand utiliser chaque approche ?

- **Neurone simple** : Problèmes linéairement séparables (classification binaire simple)
- **Réseau à 1 couche** : Problèmes avec quelques non-linéarités
- **Réseau profond** : Problèmes complexes avec de nombreuses caractéristiques et patterns

## Fonctions d'activation

- **Sigmoïde** : Sortie entre 0 et 1, idéale pour la classification binaire
- **Softmax** : Normalise les probabilités pour la classification multi-classes
- **ReLU** : Rapide et efficace pour les couches cachées

## Techniques d'optimisation

- **Gradient Descent** : Optimisation itérative des poids
- **Forward Propagation** : Calcul des prédictions
- **Back Propagation** : Calcul et propagation des erreurs
- **One-hot Encoding** : Représentation des classes catégorielles

---

## Lexique des termes utilisés

### A
- **Activation** : Sortie d'un neurone après application de la fonction d'activation
- **Accuracy** : Taux de prédictions correctes

### B
- **Bias** : Terme de biais ajouté à chaque neurone pour améliorer l'ajustement
- **Binary Classification** : Classification en deux classes (0 ou 1)
- **Backpropagation** : Algorithme de rétropropagation pour calculer les gradients

### C
- **Cost Function** : Fonction de coût mesurant l'erreur du modèle
- **Cross-entropy** : Fonction de perte pour la classification

### D
- **Deep Neural Network** : Réseau de neurones avec plusieurs couches cachées

### F
- **Forward Propagation** : Passage des données de l'entrée vers la sortie
- **Feature** : Variable d'entrée ou caractéristique

### G
- **Gradient** : Dérivée partielle de la fonction de coût
- **Gradient Descent** : Algorithme d'optimisation itératif

### H
- **Hidden Layer** : Couche intermédiaire entre l'entrée et la sortie

### L
- **Learning Rate** : Taux d'apprentissage contrôlant la vitesse de convergence
- **Logistic Regression** : Classification binaire utilisant la fonction sigmoïde

### N
- **Neural Network** : Réseau de neurones interconnectés
- **Neuron** : Unité de calcul basique du réseau

### O
- **One-hot Encoding** : Représentation binaire des classes catégorielles

### P
- **Perceptron** : Modèle de neurone artificiel simple
- **Prediction** : Sortie prédite par le modèle

### R
- **ReLU** : Rectified Linear Unit (max(0, x))

### S
- **Sigmoid** : Fonction d'activation en forme de S
- **Softmax** : Fonction d'activation pour classification multi-classes
- **Supervised Learning** : Apprentissage avec des données étiquetées

### T
- **Threshold** : Seuil de décision pour la classification
- **Training** : Processus d'apprentissage du modèle

### W
- **Weights** : Poids des connexions entre neurones

---

## Structure des fichiers

Chaque fichier numéroté implémente une étape progressive :
- `0-neuron.py` à `7-neuron.py` : Construction d'un neurone complet
- `8-neural_network.py` à `15-neural_network.py` : Réseau à une couche cachée  
- `16-deep_neural_network.py` à `28-deep_neural_network.py` : Réseaux profonds
- Les fichiers `main.py` correspondants testent chaque implémentation
