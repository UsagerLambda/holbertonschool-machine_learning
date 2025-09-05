#!/usr/bin/env python3
"""Implémentation d'un arbre de décision simple."""

import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """Classe pour implémenter une forêt d'arbres d'isolation aléatoires."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initialise la forêt d'isolation aléatoire.

        Args:
            n_trees (int): Nombre d'arbres dans la forêt
            max_depth (int): Profondeur maximale des arbres
            min_pop (int): Population minimale pour diviser un nœud
            seed (int): Graine pour la génération aléatoire
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """
        Prédit les profondeurs moyennes pour les données d'entrée.

        Args:
            explanatory: Données explicatives

        Returns:
            Moyennes des prédictions de tous les arbres
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """
        Entraîne la forêt d'isolation sur les données.

        Args:
            explanatory: Données d'entraînement
            n_trees (int): Nombre d'arbres à entraîner
            verbose (int): Niveau de verbosité
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth,
                                      seed=self.seed+i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            mean_depth = np.array(depths).mean()
            mean_nodes = np.array(nodes).mean()
            mean_leaves = np.array(leaves).mean()
            print(f"""  Training finished.
    - Mean depth                     : {mean_depth}
    - Mean number of nodes           : {mean_nodes}
    - Mean number of leaves          : {mean_leaves}""")

    def suspects(self, explanatory, n_suspects):
        """
        Retourne les lignes avec les plus petites profondeurs moyennes.

        Args:
            explanatory: Données explicatives
            n_suspects (int): Nombre de suspects à retourner

        Returns:
            Les indices des n_suspects lignes avec les plus petites profondeurs
        """
        depths = self.predict(explanatory)
        # Obtenir les indices triés par ordre croissant des profondeurs
        sorted_i = np.argsort(depths)
        # Retourner les n_suspects premiers indices (plus petites profondeurs)
        suspect_i = sorted_i[:n_suspects]
        # Retourner aussi les profondeurs correspondantes
        suspect_d = depths[suspect_i]
        return suspect_i, suspect_d
