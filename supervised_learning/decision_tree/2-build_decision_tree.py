#!/usr/bin/env python3
"""Implémentation d'un arbre de décision simple."""

import numpy as np


class Node:
    """Classe représentant un nœud interne d'un arbre de décision."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialise un nœud interne de l'arbre de décision."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

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

    def __str__(self):
        """Affiche le nœud et ses enfants sous forme de chaîne."""
        s = f"-> node [feature={self.feature}, threshold={self.threshold}]"
        if self.is_root is True:
            s = f"root [feature={self.feature}, threshold={self.threshold}]"

        if self.left_child is not None:
            s += "\n" + self.left_child_add_prefix(str(self.left_child))

        if self.right_child is not None:
            s += self.right_child_add_prefix(str(self.right_child))

        return s

    def left_child_add_prefix(self, text):
        """Ajoute un préfixe pour afficher le sous-arbre gauche."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:-1]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Ajoute un préfixe pour afficher le sous-arbre droit."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:-1]:
            new_text += ("       "+x) + "\n"
        return (new_text)


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

    def __str__(self):
        """Affiche la feuille sous forme de chaîne."""
        return (f"-> leaf [value={self.value}]")


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
        """Affiche l'abre sous forme de chaîne."""
        return self.root.__str__()
