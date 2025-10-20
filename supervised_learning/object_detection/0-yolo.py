#!/usr/bin/env python3
"""Implémentation de la classe Yolo pour la détection d'objets avec YOLO."""

from tensorflow import keras as K


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
