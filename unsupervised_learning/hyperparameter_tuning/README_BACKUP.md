# Gaussian Process - L'Essentiel

## Le Concept

**Un Gaussian Process prédit la valeur d'un nouveau point en se basant sur les points observés les plus proches.**

## Les 3 Concepts Clés

### 1. Covariance
Mesure à quel point deux points sont corrélés :
- Points proches → covariance forte → valeurs similaires
- Points loin → covariance faible → valeurs indépendantes

### 2. Kernel (RBF)
```python
K(x₁, x₂) = σf² × exp(-||x₁ - x₂||² / (2l²))
```
- `l` : vitesse de décroissance de la corrélation avec la distance
- `σf` : amplitude des variations

### 3. Matrice K
Tableau de covariances entre tous les points observés.
```
        Point1  Point2  Point3
Point1 [ 1.00   0.60    0.10 ]
Point2 [ 0.60   1.00    0.60 ]
Point3 [ 0.10   0.60    1.00 ]
```

## Comment ça Marche ?

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

## Application : Hyperparameter Tuning

**Problème :** Grid Search teste 300+ combinaisons (lent)

**Solution :** Le GP teste intelligemment 30-50 combinaisons :
1. Teste 5-10 combinaisons au hasard
2. Le GP prédit où sont les bonnes zones
3. Teste les zones prometteuses
4. Répète jusqu'à trouver l'optimum

**Résultat :** 10x plus rapide que Grid Search !

## Points Clés

- ✅ Interpolation intelligente basée sur la proximité
- ✅ Quantifie l'incertitude des prédictions
- ✅ Gère naturellement le bruit dans les mesures
- ✅ Efficace pour l'optimisation bayésienne
