Ce dossier explore l’algèbre linéaire, pilier central du Machine Learning.

# Guide linéaire : vecteurs, matrices, produit scalaire, NumPy

## 1. Introduction aux vecteurs
Un vecteur est une entité mathématique (une « flèche ») définie par une magnitude (longueur) et une direction.  
En notation algébrique, pour **a** = $a₁, a₂, …, aₙ$ et **b** = $b₁, b₂, …, bₙ$ :

$ a ⋅ b = ∑ᵢ aᵢ bᵢ = ||a|| · ||b|| · cos(θ) $

où θ est l’angle entre **a** et **b**.  
Quand **a** ⟂ **b**, alors $ a ⋅ b = 0 $.

---

## 2. Qu’est-ce qu’une matrice ?
Une matrice est un tableau à deux dimensions (m lignes, n colonnes) contenant des nombres.  
Elle permet de représenter des transformations linéaires.

---

## 3. Transposée
La **transposée** d’une matrice A, notée [ Aᵀ ], est obtenue en échangeant les lignes et les colonnes :  
$ (Aᵀ)ᵢⱼ = Aⱼᵢ $

---

## 4. Le produit scalaire (dot product)
Voir section 1 : c’est [ a ⋅ b ], qui donne un résultat scalaire. Il reflète la projection de l’un sur l’autre, ou leur alignement (θ=0 → max, θ=90° → nul).

---

## 5. Multiplication de matrices
Soient A de taille (m×n) et B de taille (n×p), alors leur produit C = A·B est de taille (m×p), avec :

$ Cᵢⱼ = ∑ₖ Aᵢₖ ⋅ Bₖⱼ $

Ce produit revient à faire le **produit scalaire** entre la i‑ème ligne de A et la j‑ème colonne de B.

---

## 6. Relation entre produit scalaire et multiplication matricielle
La multiplication matricielle est une série de produits scalaires.  
Chaque élément $ Cᵢⱼ $ est le dot product de la ligne i de A et de la colonne j de B.  
Exemple : $ aᵀ·b = a ⋅ b $, mais [ a·aᵀ ] est une matrice (produit extérieur).

---

## 7. Produit scalaire, multiplication matricielle & magie des matrices orthogonales (avancé)
Une **matrice orthogonale** Q est une matrice carrée réelle dont les lignes ET colonnes sont des vecteurs orthonormés :

$ Qᵀ·Q = Q·Qᵀ = I $
$ ⇔ Qᵀ = Q⁻¹ $

Cela signifie que Q préserve le produit scalaire entre vecteurs :  
$ ||Qx|| = ||x|| $

Donc Q est une **isométrie** (ex. : rotation, symétrie).

---

## 8. Tutoriel NumPy (jusqu’à shape manipulation exclu)

### numpy basics (jusqu'à universal functions inclus)
- **`numpy.ndarray`** : tableau N‑dimensions.
- **`.shape`** : dimensions de l’array.
- **Transposition** :
  - `np.transpose(arr)` ou `arr.T`
- **Produit scalaire** :
  - `np.dot(a, b)` : scalaire si a & b sont 1‑D, matrice si 2‑D.
- **Multiplication de matrices** :
  - `np.matmul(A, B)` ou `A @ B`

---

## 9. Indexation de tableaux
- `arr[i]` : i‑ème ligne  
- `arr[i, j]` : élément ligne i, colonne j  
- `arr[start:stop]` : tranches  
- Slicing multi‑dimensions : `arr[:, 1:3]`, etc.

---

## 10. Opérations numériques sur tableaux
- Éléments : `+`, `-`, `*`, `/`, `**`, etc.  
- Fonctions universelles : `np.sin(arr)`, `np.exp(arr)`, etc.

---

## 11. Broadcasting
Permet d’appliquer des opérations entre tableaux de dimensions différentes mais compatibles, sans duplication de données.

---

## 12. Mutations et broadcasting
- Exemple : `arr *= 2` modifie directement l’array.
- Le broadcasting permet d’appliquer l’opération sur toute une dimension sans boucle.

---

## 13. Propriétés de `numpy.ndarray`
- Objet principal de NumPy.
- `.shape`, `.dtype`, `.ndim` sont utiles.
- Transposition : `arr.T` ou `arr.transpose()`.

---

## 14. Résumé : `np.matmul`
- Produit matriciel.
- Gère les dimensions >2 avec broadcasting (ex. batchs).
- Meilleur que `np.dot` pour clarté.

---

### ✅ En résumé
- **Vecteur** : direction + norme
- **Matrice** : tableau 2D pour représenter transformations
- **Transpose** : échange lignes / colonnes
- **Dot product** : mesure d’alignement, produit scalaire
- **Multiplication matricielle** : composition de transformations
- **NumPy** : outils puissants pour manipuler vecteurs, matrices, avec efficacité et élégance

---

### 📝 Petit encart orthographe
Aucune faute détectée dans ton prompt initial.
