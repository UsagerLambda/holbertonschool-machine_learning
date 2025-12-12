# Decision Tree & Random Forest

## Introduction

Les arbres de décision et les forêts aléatoires sont des algorithmes d'apprentissage supervisé puissants utilisés pour la classification et la régression. Ce module implémente ces algorithmes from scratch, permettant de comprendre leur fonctionnement interne.

## Pourquoi les arbres de décision ?

Les arbres de décision sont populaires car ils :
- **Sont interprétables** : Faciles à visualiser et comprendre
- **Nécessitent peu de préparation** : Pas besoin de normalisation des données
- **Gèrent différents types de données** : Numériques et catégorielles
- **Capturent les interactions** : Relations non-linéaires entre variables
- **Robustes aux outliers** : Moins sensibles aux valeurs aberrantes

## Comment fonctionnent les arbres de décision ?

### Processus de construction :

1. **Sélection de l'attribut** : Choisir la meilleure feature pour diviser
2. **Critère de division** : Utiliser l'entropie, le gain d'information ou Gini
3. **Division récursive** : Répéter le processus sur chaque sous-ensemble
4. **Critères d'arrêt** : Profondeur max, taille min des feuilles, pureté

### Progression dans le module :

- **Fichiers 0-8** : Construction progressive d'un arbre de décision complet
- **Fichiers 9** : Random Forest (ensemble d'arbres)
- **Fichiers 10-11** : Isolation Tree et Isolation Forest (détection d'anomalies)

## Quand utiliser quoi ?

### Arbre de décision simple :
- **Avantages** : Rapide, interprétable, pas d'hypothèses sur les données
- **Inconvénients** : Prone au surajustement, instable
- **Utilisation** : Problèmes simples, analyse exploratoire, règles métier

### Random Forest :
- **Avantages** : Réduit le surajustement, plus stable, mesure d'importance
- **Inconvénients** : Moins interprétable, plus lent
- **Utilisation** : Problèmes complexes, besoin de performance élevée

### Isolation Forest :
- **Avantages** : Efficace pour la détection d'anomalies
- **Utilisation** : Détection de fraude, monitoring système, contrôle qualité

## Concepts clés

### Mesures de pureté :
- **Entropie** : Mesure du désordre dans un ensemble
- **Gain d'information** : Réduction de l'entropie après division
- **Index de Gini** : Mesure d'impureté alternative

### Techniques d'ensemble :
- **Bagging** : Bootstrap Aggregating (Random Forest)
- **Bootstrap sampling** : Échantillonnage avec remise
- **Feature randomness** : Sélection aléatoire de features

---

## Lexique des termes utilisés

### A
- **Attribut** : Variable ou feature utilisée pour la division
- **Anomalie** : Point de données qui dévie significativement de la normale

### B
- **Bootstrap** : Méthode d'échantillonnage avec remise
- **Bagging** : Bootstrap Aggregating, technique d'ensemble
- **Branch** : Branche de l'arbre reliant les nœuds

### C
- **Classification** : Prédiction de classes catégorielles
- **Critère d'arrêt** : Conditions pour arrêter la croissance de l'arbre

### D
- **Decision Tree** : Arbre de décision, modèle en forme d'arbre
- **Depth** : Profondeur de l'arbre (nombre de niveaux)

### E
- **Entropy** : Mesure du désordre ou de l'incertitude
- **Ensemble** : Combinaison de plusieurs modèles

### F
- **Feature** : Variable d'entrée ou attribut
- **Forest** : Forêt, collection d'arbres de décision

### G
- **Gini Index** : Mesure d'impureté pour la division des nœuds
- **Gain** : Amélioration de la pureté après une division

### I
- **Information Gain** : Gain d'information, réduction de l'entropie
- **Impurity** : Impureté, mesure du mélange des classes
- **Isolation** : Technique de séparation pour détecter les anomalies

### L
- **Leaf** : Feuille, nœud terminal de l'arbre
- **Left Child** : Enfant gauche d'un nœud

### M
- **Max Depth** : Profondeur maximale autorisée
- **Min Population** : Taille minimale d'un nœud pour être divisé

### N
- **Node** : Nœud, point de décision dans l'arbre
- **Numpy** : Bibliothèque Python pour le calcul numérique

### O
- **Outlier** : Valeur aberrante
- **Overfitting** : Surajustement du modèle

### P
- **Purity** : Pureté, homogénéité des classes dans un nœud
- **Prediction** : Prédiction du modèle
- **Pruning** : Élagage, réduction de la taille de l'arbre

### R
- **Random Forest** : Forêt aléatoire, ensemble d'arbres
- **Root** : Racine, nœud principal de l'arbre
- **Right Child** : Enfant droit d'un nœud

### S
- **Split** : Division d'un nœud en sous-nœuds
- **Subset** : Sous-ensemble de données
- **Supervised Learning** : Apprentissage supervisé

### T
- **Threshold** : Seuil de division pour les variables continues
- **Tree** : Arbre, structure hiérarchique de décisions
- **Target** : Variable cible à prédire

### V
- **Variance** : Mesure de la variabilité du modèle
- **Voting** : Vote majoritaire pour la prédiction finale

---

## Structure des fichiers

### Construction de l'arbre (0-8) :
- `0-build_decision_tree.py` : Classes Node et Leaf de base
- `1-build_decision_tree.py` : Calcul de la profondeur
- `2-build_decision_tree.py` : Vérification des feuilles
- `3-build_decision_tree.py` : Prédictions
- `4-build_decision_tree.py` : Calcul de l'entropie
- `5-build_decision_tree.py` : Gain d'information
- `6-build_decision_tree.py` : Meilleure division
- `7-build_decision_tree.py` : Construction récursive
- `8-build_decision_tree.py` : Arbre complet avec fit

### Ensemble et détection d'anomalies (9-11) :
- `9-random_forest.py` : Forêt aléatoire
- `10-isolation_tree.py` : Arbre d'isolation
- `11-isolation_forest.py` : Forêt d'isolation

Chaque fichier `main.py` teste l'implémentation correspondante.
