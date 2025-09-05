#!/usr/bin/env python3
"""Implémentation d'un arbre de décision simple."""

import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """
    Arbre d'isolation aléatoire pour la détection d'anomalies.

    Cette classe implémente un arbre de décision utilisé dans les algorithmes
    d'isolation pour détecter les anomalies dans les données. L'arbre effectue
    des divisions aléatoires sur les caractéristiques pour isoler les points
    de données anormaux.
    """

    def __init__(self, max_depth=10, seed=0, root=None):
        """
        Initialise un arbre d'isolation aléatoire.

        Args:
            max_depth (int): Profondeur maximale de l'arbre (défaut: 10)
            seed (int): Graine pour la génération de
                nombres aléatoires (défaut: 0)
            root (Node): Nœud racine existant
                (défaut: None, crée un nouveau nœud)
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """
        Retourne une représentation textuelle de l'arbre.

        Returns:
            str: Représentation en chaîne de caractères de l'arbre
        """
        return self.root.__str__()

    def depth(self):
        """
        Calculate the maximum depth of the tree.

        Returns:
            int: La profondeur maximale de l'arbre
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Compte le nombre de nœuds dans l'arbre.

        Args:
            only_leaves (bool): Si True,
                compte seulement les feuilles (défaut: False)

        Returns:
            int: Le nombre de nœuds (ou de feuilles si only_leaves=True)
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """
        Update the bounds of all nodes in the tree.

        Necessary for the proper functioning of predictions.
        """
        self.root.update_bounds_below()

    def get_leaves(self):
        """
        Récupère toutes les feuilles de l'arbre.

        Returns:
            list: Liste de tous les nœuds feuilles de l'arbre
        """
        return self.root.get_leaves_below()

    def update_predict(self):
        """
        Update the prediction function of the tree.

        Configure les indicateurs des feuilles et crée une fonction lambda
        qui calcule les prédictions en sommant les
            contributions de chaque feuille.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: sum(
            leaf.indicator(A) * leaf.value
            for leaf in leaves)

    def np_extrema(self, arr):
        """
        Calculate the minimum and maximum values of an array.

        Args:
            arr (numpy.array): Tableau dont on veut les extrema

        Returns:
            tuple: (valeur_minimale, valeur_maximale)
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Generate a random split criterion for a node.

        Choisit aléatoirement une caractéristique et un seuil de division.

        Args:
            node (Node): Le nœud pour lequel générer le critère de division

        Returns:
            tuple: (feature, threshold) -
                l'indice de la caractéristique et le seuil
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        Crée un nœud feuille enfant avec la sous-population donnée.

        Args:
            node (Node): Le nœud parent
            sub_population (numpy.array): Masque booléen de la sous-population

        Returns:
            Leaf: Le nœud feuille créé
        """
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Crée un nœud enfant (non-feuille) avec la sous-population donnée.

        Args:
            node (Node): Le nœud parent
            sub_population (numpy.array): Masque booléen de la sous-population

        Returns:
            Node: Le nœud enfant créé
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """
        Recursively fit a node by creating its children.

        Détermine s'il faut créer des feuilles ou continuer
            la division selon la profondeur
        et la taille de la population.

        Args:
            node (Node): Le nœud à ajuster
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = (node.sub_population) & (
            self.explanatory[:, node.feature] > node.threshold)
        right_population = (node.sub_population) & (
            self.explanatory[:, node.feature] <= node.threshold)

        # Is left node a leaf ?
        is_left_leaf = (node.depth >= self.max_depth - 1) or (
            left_population.sum() <= 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (node.depth >= self.max_depth - 1) or (
            right_population.sum() <= 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """
        Train the isolation tree on the provided explanatory data.

        Construit l'arbre complet et prépare la fonction de prédiction.

        Args:
            explanatory (numpy.array): Données d'entraînement (
                caractéristiques)
            verbose (int): Niveau de verbosité (
                0=silencieux, 1=affiche les statistiques)
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones_like(
            explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
