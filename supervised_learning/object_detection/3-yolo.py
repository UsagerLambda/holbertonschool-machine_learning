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

        if isinstance(self.model.input, list):
            input_width = self.model.input[0].shape[1]
            input_height = self.model.input[0].shape[2]
        else:
            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]

        # Dimensions réelles de l'image
        image_height, image_width = image_size[0], image_size[1]

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
            c_x = np.arange(grid_w).reshape(1, grid_w, 1)
            c_x = np.tile(c_x, (grid_h, 1, anchor_boxes))

            c_y = np.arange(grid_h).reshape(grid_h, 1, 1)
            c_y = np.tile(c_y, (1, grid_w, anchor_boxes))

            # On récupère les ancres correspondant à cette échelle
            current_anchors = self.anchors[i]
            pw = current_anchors[:, 0].reshape(1, 1, anchor_boxes)
            ph = current_anchors[:, 1].reshape(1, 1, anchor_boxes)

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
            b_w = (pw * np.exp(t_w)) / input_width
            b_h = (ph * np.exp(t_h)) / input_height

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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filtre les boîtes en fonction du seuil de classe.

        Args:
            boxes (list of numpy.ndarrays): liste de tableaux numpy de forme
                (grid_height, grid_width, anchor_boxes, 4) contenant les
                boîtes de délimitation traitées pour chaque sortie,
                respectivement.
            box_confidences (list of numpy.ndarrays): liste de tableaux numpy
                de forme (grid_height, grid_width, anchor_boxes, 1)
                contenant les confiances des boîtes traitées pour chaque
                sortie, respectivement.
            box_class_probs (list of numpy.ndarrays): liste de tableaux numpy
                de forme (grid_height, grid_width, anchor_boxes, classes)
                contenant les probabilités de classe des boîtes traitées pour
                chaque sortie, respectivement.

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
                - filtered_boxes: numpy.ndarray de forme (?, 4) contenant
                  toutes les boîtes de délimitation filtrées.
                - box_classes: numpy.ndarray de forme (?,) contenant le numéro
                  de classe que chaque boîte dans filtered_boxes prédit,
                  respectivement.
                - box_scores: numpy.ndarray de forme (?) contenant les scores
                  des boîtes pour chaque boîte dans filtered_boxes,
                  respectivement.
        """
        # Rappel : class_t (float): seuil de score pour
        # le filtrage initial des boîtes.

        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            # Calculate box scores: confidence * class_probability
            scores = box_confidences[i] * box_class_probs[i]

            # Récupère la meilleure classe par boîte (argument + valeur)
            box_class = np.argmax(scores, axis=-1)
            box_score = np.max(scores, axis=-1)

            # Mask = True si la condition est vrai sinon False
            mask = box_score >= self.class_t

            # Ajoute dans filtered_boxes les boîtes où mask=True
            filtered_boxes.append(boxes[i][mask])
            # Ajoute dans box_classes les classes où mask=True
            box_classes.append(box_class[mask])
            # Ajoute dans box_scores les scores où mask=True
            box_scores.append(box_score[mask])

        # Concatenate les résulats
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Effectue la suppression non maximale (NMS) sur les boîtes filtrées.

        Args:
            filtered_boxes (numpy.ndarray): tableau de forme (?, 4)
            contenant toutes les boîtes de délimitation filtrées.
            box_classes (numpy.ndarray): tableau de forme (?,) contenant
            le numéro de classe prédit pour chaque boîte dans
            filtered_boxes.
            box_scores (numpy.ndarray): tableau de forme (?) contenant
            les scores des boîtes pour chaque boîte dans
            filtered_boxes.

        Returns:
            tuple: (box_predictions, predicted_box_classes,
            predicted_box_scores)
            - box_predictions: numpy.ndarray de forme (?, 4)
              contenant toutes les boîtes de délimitation
              prédites, ordonnées par classe et score de boîte.
            - predicted_box_classes: numpy.ndarray de forme (?,)
              contenant le numéro de classe pour chaque boîte dans
              box_predictions, ordonné par classe et score de boîte.
            - predicted_box_scores: numpy.ndarray de forme (?) contenant
              les scores des boîtes pour chaque boîte dans
              box_predictions, ordonné par classe et score de boîte.
        """
        # self.nms_t pourcentage minimal de confiance

        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Récupère les classes uniques
        unique_classes = np.unique(box_classes)

        # Pour chaque classe récupéré
        for cls in unique_classes:
            # Indice des boxes de la classe courante
            mask = box_classes == cls
            # Récupère les zones ayant les indices de la classe idxs_boxes
            cls_boxes = filtered_boxes[mask]
            # Pareil pour les scores
            cls_scores = box_scores[mask]

            # Trie les indices des valeurs de cls_scores dans
            # l'ordre décroissant
            sorted_indices = np.argsort(-cls_scores)
            # Organise les boxes selon l'ordre des scores
            cls_boxes = cls_boxes[sorted_indices]
            # Organise les scores selon l'ordre des scores
            cls_scores = cls_scores[sorted_indices]

            # Liste pour sauvegarder la boxe avec le score le plus élevé
            keep = []

            # Parcourir les boxes
            i = 0
            while i < len(cls_boxes):
                # Ajoute l'index traité dans la liste
                keep.append(i)

                # Casse la boucle si c'est le dernier indice
                if i == len(cls_scores) - 1:
                    break

                # Récupère la boite à l'indice itéré
                current_box = cls_boxes[i]
                # Récupère les boites qui suivent la boite de l'indice itéré
                remaining_boxes = cls_boxes[i + 1:]

                # Coordonnées de l'intersection entre les boîtes
                # max des X gauches
                x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
                # max des Y hauts
                y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
                # min des X droits
                x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
                # min des Y bas
                y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter_area = inter_w * inter_h

                # Calcul l'aire des boxes avec
                # la current box - l'aire de chevauchement
                # Aire de la box
                box_area = (current_box[2] - current_box[0]) * \
                           (current_box[3] - current_box[1])
                # Aires des autres boxes
                other_areas = (
                    remaining_boxes[:, 2] - remaining_boxes[:, 0]) * \
                    (remaining_boxes[:, 3] - remaining_boxes[:, 1])
                # Aire total des boxes avec current_box - aire de chevauchement
                union_area = box_area + other_areas - inter_area

                # IoU (pourcentage de chevauchement)
                # Exemple :
                # inter_area = 3000 px²  (zone commune)
                # box_area = 10000 px²
                # other_areas = 10000 px²
                # union_area = 10000 + 10000 - 3000 = 17000 px²
                # IoU = 3000 / 17000 = 0.176  ← 17.6% de chevauchement

                iou = inter_area / union_area

                # On garde seulement les boxes avec un IoU inférieur à nms_t
                # keep_mask est un tableau de boolean
                keep_mask = iou <= self.nms_t

                # Garde les indices des boxes par rapport au masque (keep_mask)
                # i + 1 sert à correspondre au tableau cls_boxes
                # Exemple:
                # i = 1  # On traite la boîte à l'indice 1
                # current_box = cls_boxes[i]           # Boîte à l'indice 1
                # remaining_boxes = cls_boxes[i + 1:]  # Commence à l'indice 2
                remaining_indices = np.where(keep_mask)[0] + i + 1

                # Reconstruit les tableaux cls_boxes/scores
                # en supprimant les boxes indésirables (< IoU)
                cls_boxes = np.concatenate([cls_boxes[:i + 1],
                                           cls_boxes[remaining_indices]])
                cls_scores = np.concatenate([cls_scores[:i + 1],
                                            cls_scores[remaining_indices]])

                i += 1  # Passe à la prochaine box

            # Récupérer les indices originaux des boîtes gardées
            original_indices = sorted_indices[keep]

            # Récupérer les boîtes originales via le masque
            cls_indices = np.where(mask)[0]
            final_indices = cls_indices[original_indices]

            # Ajouter aux résultats
            box_predictions.extend(filtered_boxes[final_indices])
            predicted_box_classes.extend([cls] * len(final_indices))
            predicted_box_scores.extend(box_scores[final_indices])

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores
