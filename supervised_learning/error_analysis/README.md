# Error Analysis

## Introduction

L'analyse d'erreur est une étape cruciale dans l'évaluation des modèles de machine learning. Elle permet de comprendre où et pourquoi un modèle fait des erreurs, fournissant des insights précieux pour l'amélioration des performances.

## Pourquoi l'analyse d'erreur ?

L'analyse d'erreur permet de :
- **Évaluer objectivement** : Mesurer précisément les performances du modèle
- **Identifier les faiblesses** : Détecter les classes mal prédites
- **Guider l'amélioration** : Savoir où concentrer les efforts d'optimisation
- **Comparer les modèles** : Choisir objectivement entre différentes approches
- **Valider la robustesse** : Tester la fiabilité sur différents types de données

## Comment analyser les erreurs ?

### Progression des métriques :

1. **Matrice de confusion (0)** : Vision globale des erreurs de classification
2. **Sensibilité/Recall (1)** : Capacité à détecter les vrais positifs
3. **Précision (2)** : Exactitude des prédictions positives
4. **Spécificité (3)** : Capacité à éviter les faux positifs
5. **F1-Score (4)** : Harmonie entre précision et recall
6. **Analyse comparative (5-6)** : Comparaison et gestion d'erreurs

## Quand utiliser chaque métrique ?

### Matrice de confusion :
- **Usage** : Vue d'ensemble des performances
- **Avantage** : Visualisation claire des erreurs par classe
- **Quand** : Toujours en premier pour comprendre la distribution des erreurs

### Sensibilité (Recall) :
- **Usage** : Importance de capturer tous les vrais positifs
- **Exemple** : Diagnostic médical (ne pas manquer une maladie)
- **Formule** : TP / (TP + FN)

### Précision :
- **Usage** : Importance d'éviter les faux positifs
- **Exemple** : Détection de spam (éviter de classer un email important comme spam)
- **Formule** : TP / (TP + FP)

### Spécificité :
- **Usage** : Capacité à identifier correctement les négatifs
- **Exemple** : Tests de dépistage (éviter les fausses alertes)
- **Formule** : TN / (TN + FP)

### F1-Score :
- **Usage** : Équilibre entre précision et recall
- **Quand** : Classes déséquilibrées ou compromise nécessaire
- **Formule** : 2 × (Précision × Recall) / (Précision + Recall)

## Types d'erreurs

### Classification binaire :
- **Vrai Positif (TP)** : Prédiction positive correcte
- **Vrai Négatif (TN)** : Prédiction négative correcte
- **Faux Positif (FP)** : Erreur de type I (prédire positif quand négatif)
- **Faux Négatif (FN)** : Erreur de type II (prédire négatif quand positif)

### Classification multi-classes :
- Extension des concepts binaires à chaque classe
- Analyse classe par classe ou micro/macro moyennes

---

## Lexique des termes utilisés

### A
- **Accuracy** : Exactitude, taux de prédictions correctes
- **Analysis** : Analyse, étude détaillée des performances

### C
- **Confusion Matrix** : Matrice de confusion, tableau croisant vraies et prédites classes
- **Class** : Classe ou catégorie à prédire

### E
- **Error** : Erreur, différence entre prédiction et réalité
- **Evaluation** : Évaluation, mesure des performances

### F
- **F1-Score** : Moyenne harmonique de précision et recall
- **False Negative (FN)** : Faux négatif, erreur de type II
- **False Positive (FP)** : Faux positif, erreur de type I

### G
- **Ground Truth** : Vérité terrain, vraies étiquettes

### M
- **Metric** : Métrique, mesure de performance
- **Macro Average** : Moyenne non pondérée des métriques par classe
- **Micro Average** : Moyenne pondérée par la fréquence des classes

### P
- **Precision** : Précision, exactitude des prédictions positives
- **Positive** : Classe positive (généralement la classe d'intérêt)
- **Predicted** : Prédit par le modèle

### R
- **Recall** : Rappel ou sensibilité, taux de vrais positifs détectés
- **ROC** : Receiver Operating Characteristic

### S
- **Sensitivity** : Sensibilité, synonyme de recall
- **Specificity** : Spécificité, taux de vrais négatifs
- **Score** : Score, mesure de performance

### T
- **True Negative (TN)** : Vrai négatif, prédiction négative correcte
- **True Positive (TP)** : Vrai positif, prédiction positive correcte
- **Threshold** : Seuil de décision

### U
- **Underfitting** : Sous-ajustement, modèle trop simple
- **Overfitting** : Surajustement, modèle trop complexe

---

## Structure des fichiers

### Métriques de base (0-4) :
- `0-create_confusion.py` : Création de la matrice de confusion
- `1-sensitivity.py` : Calcul de la sensibilité (recall)
- `2-precision.py` : Calcul de la précision
- `3-specificity.py` : Calcul de la spécificité
- `4-f1_score.py` : Calcul du F1-score

### Analyses avancées (5-6) :
- `5-error_handling` : Gestion d'erreurs dans les métriques
- `6-compare_and_contrast` : Comparaison de modèles

Chaque fichier `main.py` teste l'implémentation correspondante avec des exemples concrets.

# VP Vrai positif (TP true positive) = Éléments en diagonale
# FN Faux négatif (FN False negative) = Somme de la ligne moins l'élément diagonal
# FP Faux positif (FP False negative) = Somme de la colonne moins l'élément diagonal
# VN Vrai négatif (TN True negative) = Somme totale - VP - FN - FP
