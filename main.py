import cv2
import os
import numpy as np
from PIL import Image
from utils import model, tools
import torch

# Point d'entrée principal du script
if __name__ == "__main__":

    # Définir le répertoire source et de sortie
    source_path_dir = "images"
    test_path_subdir = "images/test"
    output_path_dir = "output"
    output_test_path_subdir = "output/test"

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_path_dir, exist_ok=True)

    # Charger le modèle et appliquer les transformations à l'image
    seg_model, transforms = model.get_model()

    # Parcourir les sous-répertoires "cam0" et "cam1"
    for subdir in ["cam0", "cam1"]:
        subdir_path = os.path.join(source_path_dir, subdir)

        # Vérifier si le sous-répertoire existe
        if not os.path.exists(subdir_path):
            continue

        # Créer le sous-répertoire de sortie s'il n'existe pas
        output_subdir = os.path.join(output_path_dir, subdir)
        os.makedirs(output_subdir, exist_ok=True)

        # Parcourir les fichiers dans le sous-répertoire
        for filename in os.listdir(subdir_path):
            if filename.endswith(".png"):  # Vérifier si le fichier est une image PNG
                # Chemin complet de l'image
                image_path = os.path.join(subdir_path, filename)
                
                # Ouvrir l'image et appliquer les transformations
                image = Image.open(image_path)
                transformed_img = transforms(image)
                
                # Effectuer l'inférence sur l'image transformée sans calculer les gradients
                with torch.no_grad():
                    output = seg_model([transformed_img])

                # Traiter le résultat de l'inférence
                result = tools.process_inference(output, image, filename)
                
                # Enregistrer le résultat
                result.save(os.path.join(output_subdir, filename))

    # bounding_boxes_info = []

    # for filename in os.listdir(test_path_subdir):
    #     if filename.endswith(".png"):
    #         # Lire l'image
    #         image = cv2.imread(filename)

    #         x, y, w, h = cv2.boundingRect(image)

    #         bounding_boxes_info.append([filename.split('.')[0], x, y, w, h])

    #          # Écrire les informations des boîtes englobantes dans un fichier texte
    #         with open('bounding_boxes_test.txt', 'a') as f:
    #             for info in bounding_boxes_info:
    #                 f.write(','.join(str(x) for x in info) + '\n')


#  # Ouvrir l'image et appliquer les transformations
#     for filename in os.listdir(test_path_subdir):
#         if filename.endswith(".png"):
#             image = Image.open(os.path.join(test_path_subdir, filename))
#             transformed_img = transforms(image)
                        
#             # Effectuer l'inférence sur l'image transformée sans calculer les gradients
#             with torch.no_grad():
#                 output = seg_model([transformed_img])

#             # Traiter le résultat de l'inférence
#             result = tools.process_inference(output, image, filename)
                        
#             # Enregistrer le résultat
#             result.save(os.path.join(output_test_path_subdir, filename))

    # image = cv2.imread("person_1.png")

    # x, y, w, h = cv2.boundingRect(image)

    # bounding_boxes_info.append(["person_1.png".split('.')[0], x, y, w, h])

    # # Écrire les informations des boîtes englobantes dans un fichier texte
    # with open('bounding_boxes_test.txt', 'a') as f:
    #     for info in bounding_boxes_info:
    #         f.write(','.join(str(x) for x in info) + '\n')