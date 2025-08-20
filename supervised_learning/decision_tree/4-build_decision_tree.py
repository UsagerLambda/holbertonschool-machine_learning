#!/usr/bin/env python3
"""Implémentation d'un arbre de décision simple."""

import numpy as np


class Node:
    """Classe représentant un nœud interne d'un arbre de décision."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialise un nœud interne de l'arbre de décision."""
        self.feature = feature              # Index de la feature pour split
        self.threshold = threshold          # Seuil de split pour la feature
        self.left_child = left_child        # Sous-arbre gauche (Node ou Leaf)
        self.right_child = right_child      # Sous-arbre droit (Node ou Leaf)
        self.is_leaf = False                # Indique si ce nœud est une leaf
        self.is_root = is_root              # Indique si ce nœud est la racine
        self.sub_population = None
        self.depth = depth                  # Profondeur du nœud dans l'arbre

    def max_depth_below(self):
        """Retourne la profondeur maximale.

        Méthode récursive qui compte la profondeur max d'un arbre
        à partir d'un nœud donné

        Returns:
            int: La plus grande valeur des résultats de la récursion
        """
        # Liste vide pour récupérer la profondeur max de chaque branche
        result = []
        # Si l'enfant gauche n'est pas vide
        if self.left_child is not None:
            # Recursion dans l'enfant gauche et append le resultat
            result.append(self.left_child.max_depth_below())
        if self.right_child is not None:  # Pareil pour la droite
            result.append(self.right_child.max_depth_below())

        if not result:
            return 0

        return max(result)  # Renvoie le plus grand résultat de la liste

    def count_nodes_below(self, only_leaves=False):
        """Compte les nodes enfant d'un node donné.

        Args:
            only_leaves (bool, optional):
            indique si le retour dois être le nombre de node ou de feuille.
            Defaults to False.

        Returns:
            int: nombre de node enfant ou de feuilles
        """
        count = 0

        # Parcourt les branches gauche récursivement
        # tant que le node visé n'est pas None
        if self.left_child is not None:
            count += self.left_child.count_nodes_below(only_leaves)

        # Parcourt les branches droite récursivement
        # tant que le node visé n'est pas None
        if self.right_child is not None:
            count += self.right_child.count_nodes_below(only_leaves)

        # Si on souhaite compter uniquement les feuilles
        if only_leaves:
            # Un nœud est une feuille s'il n'a pas d'enfants gauche ni droit
            if self.left_child is None and self.right_child is None:
                return 1  # Ce nœud est une feuille
            else:
                # Sinon, retourne la somme des feuilles
                # trouvées dans les sous-arbres
                return count

        # Si self.left_child/right_child == None alors ajoute 1
        # au nombre de node trouvé
        return 1 + count

    def get_leaves_below(self):
        """
        Retourne la liste de toutes les feuilles sous ce nœud (récursif).

        Returns:
            list: Liste des objets feuilles descendants.
        """
        result = []
        if self.left_child is not None:
            result.extend(self.left_child.get_leaves_below())
        if self.right_child is not None:
            result.extend(self.right_child.get_leaves_below())

        if self.left_child is None and self.right_child is None:
            return [self]

        return result

    def update_bounds_below(self):
        """
        Met à jour récursivement les bornes (upper/lower) pour chaque nœud.

        Initialise les bornes à la racine, puis propage les contraintes
        de split à chaque enfant selon la branche (gauche/droite).
        """
        if self.is_root:
            self.upper = {0: np.inf}  # {0: +∞}
            self.lower = {0: -1 * np.inf}  # {0: -∞}
            # Racine : -∞ < X < +∞

        for child in [self.left_child, self.right_child]:
            if child is None:
                continue

            # Copie les bornes du parent (upper & lower)
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()

            if child == self.left_child:
                # Enfant gauche : feature <= threshold
                # MAJ de la borne supérieure
                child.upper[self.feature] = self.threshold
            else:
                # Enfant droit : feature > threshold
                # MAJ de la borne inférieure
                child.lower[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()  # Récursion


class Leaf(Node):
    """Classe représentant une feuille de l'arbre de décision."""

    def __init__(self, value, depth=None):
        """Initialise une feuille de l'arbre de décision."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Retourne la profondeur de la feuille.

        Returns:
            int: Profondeur de la feuille.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Compte le nombre de nœuds ou de feuilles."""
        return 1

    def get_leaves_below(self):
        """
        Retourne une liste contenant cette feuille.

        Returns:
            list: Liste contenant uniquement cette feuille.
        """
        return [self]

    def __str__(self):
        """
        Retourne une représentation lisible de la feuille pour l'affichage.

        Returns:
            str: Description de la feuille avec sa valeur.
        """
        return f"-> leaf [value={self.value}]"

    def update_bounds_below(self):
        """
        Méthode présente pour compatibilité avec Node.

        Ne fait rien pour une feuille.
        """
        pass


class Decision_Tree():
    """Classe principale pour l'arbre de décision."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialise un arbre de décision."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Retourne la profondeur maximale de l'arbre.

        Returns:
            int: Profondeur maximale de l'arbre.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Compte les nœuds ou feuilles de l'arbre."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Retourne une représentation lisible de l'arbre (racine).

        Returns:
            str: Affichage de la racine de l'arbre.
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        Retourne la liste de toutes les feuilles de l'arbre.

        Returns:
            list: Liste des objets feuilles de l'arbre.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Met à jour les bornes (upper/lower) pour tout l'arbre.

        Lance la propagation à partir de la racine.
        """
        self.root.update_bounds_below()
