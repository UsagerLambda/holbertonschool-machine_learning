#!/usr/bin/env python3
"""Implémentation de la classe Yolo pour la détection d'objets avec YOLO."""

from tensorflow import keras as K
import numpy as np
import cv2
import os


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
            # Masque pour extraire les boîtes de la classe courante
            mask = box_classes == cls
            # Récupère les boîtes de la classe courante
            cls_boxes = filtered_boxes[mask]
            # Récupère les scores de la classe courante
            cls_scores = box_scores[mask]

            # Trie par score décroissant
            sorted_indices = np.argsort(-cls_scores)
            cls_boxes = cls_boxes[sorted_indices]
            cls_scores = cls_scores[sorted_indices]

            # Listes pour stocker les boîtes et scores gardés
            keep_boxes = []
            keep_scores = []

            # Parcourir les boxes avec la méthode simplifiée
            while len(cls_boxes) > 0:
                # Garder la première boîte (meilleur score restant)
                keep_boxes.append(cls_boxes[0])
                keep_scores.append(cls_scores[0])

                # Si c'était la dernière boîte, terminer
                if len(cls_boxes) == 1:
                    break

                # Récupère la boite courante (première)
                current_box = cls_boxes[0]
                # Récupère les boites restantes
                remaining_boxes = cls_boxes[1:]

                # Coordonnées de l'intersection entre les boîtes
                # max des X gauches
                x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
                # max des Y hauts
                y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
                # min des X droits
                x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
                # min des Y bas
                y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

                # Aire d'intersection (0 si pas de chevauchement)
                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter_area = inter_w * inter_h

                # Aire de la box courante
                box_area = (current_box[2] - current_box[0]) * \
                           (current_box[3] - current_box[1])

                # Aires des autres boxes
                other_areas = (
                    remaining_boxes[:, 2] - remaining_boxes[:, 0]) * \
                    (remaining_boxes[:, 3] - remaining_boxes[:, 1])

                # Aire totale des boxes - aire de chevauchement
                union_area = box_area + other_areas - inter_area

                # IoU (pourcentage de chevauchement)
                # Exemple :
                # inter_area = 3000 px²  (zone commune)
                # box_area = 10000 px²
                # other_areas = 10000 px²
                # union_area = 10000 + 10000 - 3000 = 17000 px²
                # IoU = 3000 / 17000 = 0.176  ← 17.6% de chevauchement
                iou = inter_area / union_area

                # Garde seulement les boxes avec un IoU inférieur à nms_t
                # (pas de chevauchement excessif)
                keep_mask = iou < self.nms_t

                # Mettre à jour les tableaux : retire la première boîte
                # et garde uniquement celles avec IoU faible
                cls_boxes = cls_boxes[1:][keep_mask]
                cls_scores = cls_scores[1:][keep_mask]

            # Ajouter les résultats de cette classe
            if len(keep_boxes) > 0:
                box_predictions.append(np.array(keep_boxes))
                predicted_box_classes.append(np.full(len(keep_boxes), cls))
                predicted_box_scores.append(np.array(keep_scores))

        # Concaténer tous les résultats
        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Affiche une image.

        Args:
            folder_path (tuple): chemin des images à charger.

        Return: Un tuple avec une liste des images, et leurs chemin.
        """
        images = []
        image_paths = []

        # Parcours du dossier
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
                image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """Redimension et normalise les images.

        Args:
            images (numpy.ndarray): liste d'images.

        Returns:
            Tuple: (pimages, image_shapes)
        """
        if isinstance(self.model.input, list):
            input_w = self.model.input[0].shape[1]
            input_h = self.model.input[0].shape[2]
        else:
            input_w = self.model.input.shape[1]
            input_h = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            # Sauvegarder la forme originale
            image_shapes.append(img.shape[:2])

            # Redimensionnement avec interpolation bicubique
            resized = cv2.resize(
                img, (input_w, input_h), interpolation=cv2.INTER_CUBIC
            )

            # Normalisation [0, 1]
            normalized = resized / 255.0

            pimages.append(normalized)

        # Conversion en numpy arrays
        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Affiche les boîtes de détection sur une image.

        Args:
            image (numpy.ndarray): image d'origine (H, W, C) en BGR.
            boxes (numpy.ndarray): tableau de boîtes de forme (n, 4)
            contenant les coordonnées [x1, y1, x2, y2].
            box_classes (numpy.ndarray): tableau d'entiers de forme (n,)
            indiquant l'indice de la classe prédite pour chaque boîte.
            box_scores (numpy.ndarray): tableau de floats de forme (n,)
            contenant le score (confiance) associé à chaque boîte.
            file_name (str): nom ou chemin utilisé pour la fenêtre OpenCV et
            pour la sauvegarde éventuelle de l'image (si l'utilisateur
            appuie sur 's' ou 'S').

        Notes:
            - La fonction affiche l'image annotée dans une fenêtre OpenCV et
              attend l'appui d'une touche.
            - Si l'utilisateur appuie sur 's' ou 'S', l'image annotée est
              sauvegardée dans le dossier "detections" (créé si nécessaire).
            - Aucun objet n'est retourné.
        """
        # Copie des images
        img_display = image.copy()

        # Parcourt les boxes
        for i, box in enumerate(boxes):
            # Recupère les coordonnées de la box itéré
            x1, y1, x2, y2 = box.astype(int)

            # Dessine un rectangle bleu d'épaisseur 2 avec les coordonnées
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Récupère l'index de la classe
            class_idx = int(box_classes[i])
            # Récupère le nom de la classe grâce à l'index
            class_name = self.class_names[class_idx]
            # Récupère le score de la box
            score = box_scores[i]

            # Prépare le label
            label = f"{class_name} {score:.2f}"

            # Calcul la position du label
            text_x = x1
            text_y = y1 - 5

            # Affiche le label en rouge
            cv2.putText(
                img_display,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        # Affiche l'image
        cv2.imshow(file_name, img_display)

        # Écoute la pression d'une touche
        key = cv2.waitKey(0)

        # Si la touche pressé est 's' ou 'S'
        if key == ord('s') or key == ord('S'):
            # Créer/assigne le dossier nommé "detections"
            # comme répertoire de sauvegarde
            detections_dir = 'detections'
            if not os.path.exists(detections_dir):
                os.makedirs(detections_dir)

            # Sauvegarde l'image
            save_path = os.path.join(
                detections_dir, os.path.basename(file_name))
            cv2.imwrite(save_path, img_display)

    # Ferme l'image
    cv2.destroyAllWindows()
