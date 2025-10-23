#!/usr/bin/env python3
"""Implémentation de la classe Yolo pour la détection d'objets avec YOLO."""

from tensorflow import keras as K
import numpy as np


class Yolo:
    """Classe Yolo pour la détection d'objets avec le modèle YOLO."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialise la classe Yolo.

        Args:
            model_path (str): chemin vers le modèle Keras Darknet.
            classes_path (str): chemin vers la liste des noms de classes
                utilisées par le modèle Darknet, listées dans l'ordre des
                index.
            class_t (float): seuil de score pour le filtrage initial des
                boîtes.
            nms_t (float): seuil IOU pour la suppression non maximale.
            anchors (numpy.ndarray): tableau de forme (outputs, anchor_boxes,
                2) contenant toutes les boîtes d'ancrage :
                outputs : nombre de sorties (prédictions) du modèle Darknet.
                anchor_boxes : nombre de boîtes d'ancrage utilisées pour
                chaque prédiction.
                2 => [largeur_boîte, hauteur_boîte].
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Convertit les sorties brutes du modèle YOLO.

        en coordonnées exploitables (x1, y1, x2, y2) et en scores de
        confiance.

        Args:
            outputs (list of np.ndarray): prédictions du modèle, une par
                échelle (grille).
            image_size (np.ndarray): [hauteur, largeur] de l'image
                originale.

        Returns:
            (boxes, box_confidences, box_class_probs)
                - boxes : coordonnées des boîtes converties (x1, y1, x2, y2)
                - box_confidences : confiance de chaque boîte
                - box_class_probs : probas pour chaque classe dans chaque boîte
        """
        def sigmoid(x):
            """Fonction sigmoïde pour ramener les valeurs entre 0 et 1."""
            return 1 / (1 + np.exp(-x))

        input_height = self.model.input.shape[1]
        input_width = self.model.input.shape[2]

        # Dimensions réelles de l'image
        image_height, image_width = image_size

        boxes = []  # Stocke les boîtes finales
        box_confidences = []  # Stock les confiances des boîtes
        box_class_probs = []  # Stock les probabilité de classe dans les boîtes

        # On traite chaque sortie
        for i, output in enumerate(outputs):
            # grid_h et grid_w: nombre
            # de cellules en hauteur et largeur de la grille
            # anchor_boxes: nombre d'ancres par cellule (boîtes de
            # base prédéfinies)
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # décalage horizontal & vertical
            # du centre de la boxe dans la cellule
            t_x = output[..., 0]
            t_y = output[..., 1]

            # Facteur d'ajustement de la hauteur et largeur de la boîte
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Coordonnées (x, y) du coin haut gauche de chaque
            # cellule de la grille (voir image)
            c_x = np.tile(np.arange(grid_w).reshape(1, grid_w, 1),
                          (grid_h, 1, anchor_boxes))
            c_y = np.tile(np.arange(grid_h).reshape(grid_h, 1, 1),
                          (1, grid_w, anchor_boxes))

            # On récupère les ancres correspondant à cette échelle
            anchors = self.anchors[i]

            # Calcule la position normalisée (entre 0 et 1) du centre
            # de la boîte dans l'image.
            # t_x est un offset relatif (en fraction de cellule) prédit
            # par le réseau, ramené entre 0 et 1 via sigmoid.
            # c_x est l'indice de colonne de la cellule dans la grille.
            # sigmoid(t_x) + c_x donne la position absolue en unités de
            # cellules, puis division par grid_w normalise à l'échelle
            # de l'image.
            # Pareil pour b_y
            b_x = (sigmoid(t_x) + c_x) / grid_w  # Position en x du centre
            b_y = (sigmoid(t_y) + c_y) / grid_h  # Position en y du centre

            # Calcule la largeur et la hauteur de la boîte.
            # Multiplie la largeur/hauteur de l'ancre par l'exponentiel
            # de t_w/t_h, puis divise par la largeur/hauteur de l'image
            # pour normaliser (entre 0 et 1).
            b_w = anchors[:, 0] * np.exp(t_w) / input_width
            b_h = anchors[:, 1] * np.exp(t_h) / input_height

            # Coin supérieur gauche
            x1 = (b_x - b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            # Coin inférieur droit
            x2 = (b_x + b_w / 2) * image_width
            y2 = (b_y + b_h / 2) * image_height

            # On empile toutes les coordonnées
            # ensemble dans un seul tableau
            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            box_confidences.append(sigmoid(output[..., 4:5]))
            box_class_probs.append(sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs
