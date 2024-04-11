import cv2
import numpy as np
from utils.config import THRESHOLD, PERSON_LABEL, ALPHA, MASK_COLOR
from PIL import Image

# Function to calculate bounding box from mask
def calculate_bounding_box_from_mask(mask):
    # Convert mask to numpy array and apply thresholding
    mask_np = mask[0].mul(255).byte().cpu().numpy() > (THRESHOLD * 255)
    
    # Find coordinates of pixels where the mask is True
    person_pixels = np.argwhere(mask_np)
    
    # Calculate bounding box coordinates using min and max of x and y
    x_min = np.min(person_pixels[:, 1])
    y_min = np.min(person_pixels[:, 0])
    x_max = np.max(person_pixels[:, 1])
    y_max = np.max(person_pixels[:, 0])
    
    # Return bounding box coordinates
    return x_min, y_min, x_max - x_min, y_max - y_min


# # Fonction pour traiter les sorties d'inférence du modèle
# def process_inference(model_output, image, filename):
#     # Extraire les masques, les scores et les étiquettes de la sortie du modèle
#     masks = model_output[0]['masks']
#     scores = model_output[0]['scores']
#     labels = model_output[0]['labels']

#     # Convertir l'image en tableau NumPy
#     img_np = np.array(image)

#     # Initialiser une liste pour stocker les coordonnées de chaque personne
#     person_coordinates = []

#     # Initialiser une liste pour stocker les informations des boîtes englobantes
#     bounding_boxes_info = []

#     # Itérer sur chaque prédiction pour appliquer le seuillage et l'étiquetage
#     for i, (mask, score, label) in enumerate(zip(masks, scores, labels)):
#         # Vérifier si la détection correspond à une personne
#         if score > THRESHOLD and label == PERSON_LABEL:
#             # Convertir le masque en un tableau NumPy et appliquer le seuillage
#             mask_np = mask[0].mul(255).byte().cpu().numpy() > (THRESHOLD * 255)
            
#             # Trouver les coordonnées des pixels où le masque est True
#             person_pixels = np.argwhere(mask_np)
            
#             # Ajouter les coordonnées à la liste
#             person_coordinates.append(person_pixels.tolist())

#             # Calculer les coordonnées de la boîte englobante
#             x, y, w, h = cv2.boundingRect(person_pixels)
            
#             # Ajouter les informations de la boîte englobante à la liste
#             bounding_boxes_info.append([filename.split('.')[0], x, y, w, h])

#             # Appliquer le masque à l'image
#             for c in range(3):
#                 img_np[:, :, c] = np.where(mask_np, 
#                                             ALPHA * MASK_COLOR[c] + (1 - ALPHA) * img_np[:, :, c],
#                                             img_np[:, :, c])

#     # Écrire les informations des boîtes englobantes dans un fichier texte
#     with open('bounding_boxes.txt', 'a') as f:
#         for info in bounding_boxes_info:
#             f.write(','.join(str(x) for x in info) + '\n')

#     # Convertir le tableau NumPy en une image
#     processed_image = Image.fromarray(img_np.astype(np.uint8))

#     # Retourner l'image traitée
#     return processed_image


def process_inference(model_output, image, filename):
    # Extract masks, scores, and labels from model output
    masks = model_output[0]['masks']
    scores = model_output[0]['scores']
    labels = model_output[0]['labels']

    # Convert the image to a NumPy array
    img_np = np.array(image)

    # Initialize lists to store person coordinates and bounding box info
    bounding_boxes_info = []

    # Iterate over each prediction to apply thresholding and labeling
    for i, (mask, score, label) in enumerate(zip(masks, scores, labels)):
        # Check if the detection corresponds to a person
        if score > THRESHOLD and label == PERSON_LABEL:
            # Calculate bounding box from mask
            x, y, w, h = calculate_bounding_box_from_mask(mask)
            
            # Add bounding box info to the list
            bounding_boxes_info.append([filename.split('.')[0], x, y, w, h])

            # Apply the mask to the image
            for c in range(3):
                img_np[:, :, c] = np.where(mask[0], 
                                            ALPHA * MASK_COLOR[c] + (1 - ALPHA) * img_np[:, :, c],
                                            img_np[:, :, c])

    # Write bounding box info to a text file
    with open('bounding_boxes.txt', 'a') as f:
        for info in bounding_boxes_info:
            f.write(','.join(str(x) for x in info) + '\n')

    # Convert the NumPy array back to an image
    processed_image = Image.fromarray(img_np.astype(np.uint8))

    # Return the processed image
    return processed_image
