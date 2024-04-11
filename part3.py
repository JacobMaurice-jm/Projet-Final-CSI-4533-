import cv2
import numpy as np
import os
from shutil import copyfile
from PIL import Image

def create_image_dict(folder_path):
    """
    Creates a dictionary mapping image names to PIL.Image objects. 

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        dict: Dictionary mapping image names to PIL.Image objects.
    """
    image_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_name = os.path.splitext(filename)[0]
            image_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(image_path)
                image_dict[image_name] = image
            except Exception as e:
                print(f"Failed to load image {image_path}: {e}")
    return image_dict

def extract_bounding_box(image_dict, data_file):
    """
    Extracts bounding boxes from images based on data provided in a text file.

    Args:
        image_dict (dict): Dictionary mapping image names to PIL.Image objects.
        data_file (file): Text file containing bounding box data.

    Returns:
        list: List of cropped images based on the bounding boxes.
    """
    result_images = []
    for line in data_file:
        parts = line.strip().split(',')
        image_name = parts[0]
        if image_name in image_dict:
            image = image_dict[image_name]
            x, y, width, height = map(int, parts[1:])
            bbox = (x, y, x + width, y + height)
            # We discard any bounding boxes with an area less than 2500, as they are considered too small to represent individuals in the foreground.
            if width * height >= 2500:
                cropped_image = image.crop(bbox)
                result_images.append(cropped_image)
    return result_images

def extract_upper_half_bounding_box(image_dict, data_file):
    """
    Extracts the upper half of bounding boxes from images based on data provided in a text file.

    Args:
        image_dict (dict): Dictionary mapping image names to PIL.Image objects.
        data_file (file): Text file containing bounding box data.

    Returns:
        list: List of cropped images representing the upper half of bounding boxes.
    """
    result_images = []
    for line in data_file:
        parts = line.strip().split(',')
        image_name = parts[0]
        if image_name in image_dict:
            image = image_dict[image_name]
            x, y, width, height = map(int, parts[1:])
            bbox = (x, y, x + width, y + height)
            if width * height >= 2500:
                cropped_image = image.crop((x, y, x + width, y + height // 2))
                result_images.append(cropped_image)
    return result_images


def calc_histogram(image):
    """
    Calculates the histogram of an image in HSV color space considering all three channels: Hue, Saturation, and Value.

    Args:
        image (PIL.Image): Input image.

    Returns:
        numpy.ndarray: Histogram of the image.
    """
    image_np = np.array(image)
    hsb_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    # Splitting the image into individual channels
    h_channel = hsb_image[:, :, 0]
    s_channel = hsb_image[:, :, 1]
    v_channel = hsb_image[:, :, 2]
    
    # Calculating histograms for each channel with 256 bins each
    h_hist = cv2.calcHist([h_channel], [0], None, [256], [0, 256])
    s_hist = cv2.calcHist([s_channel], [0], None, [256], [0, 256])
    v_hist = cv2.calcHist([v_channel], [0], None, [256], [0, 256])
    
    # Concatenating histograms for all three channels
    hist = np.concatenate((h_hist, s_hist, v_hist), axis=0)
    
    # Normalizing the histogram
    hist = cv2.normalize(hist, hist)
    
    return hist

'''

# The following function offers another formulation of the calc_histogram method. It makes use of the RGB greyscale distributed over 256 bins

def calc_histogram(image):

    # Converting the PIL image to a NumPy array
    image_np = np.array(image)

    # Converting the image to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Calculating the histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Normalizing the histogram
    hist = cv2.normalize(hist, hist)

    return hist

'''

def compare_histograms(hist1, hist2):
    """
    Compares two histograms using the intersection method.

    Args:
        hist1 (numpy.ndarray): First histogram.
        hist2 (numpy.ndarray): Second histogram.

    Returns:
        float: Similarity score between the histograms.
    """
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)

def crop_image(lst):
    """
    Crops an image according to the provided coordinates of a bounding box.

    Args:
        lst (list): List that includes the image's filename along with the bounding box coordinates arranged as follows:
        [file_name, x_coordinate, y_coordinate, width, height]

    Returns:
        PIL.Image: Cropped image.
    """
    original_image = Image.open(str(lst[0]))
    cropped_image = original_image.crop((lst[1], lst[2], lst[1] + lst[3], lst[2] + lst[4]))  
    return cropped_image

def crop_upper_half(lst):
    """
    Crops the upper half of an image according to provided coordinates of a bounding box.

    Args:
        lst (list): List that includes the image's filename along with the bounding box coordinates arranged as follows:
        [file_name, x_coordinate, y_coordinate, width, height]

    Returns:
        PIL.Image: Cropped upper half image.
    """
    original_image = Image.open(str(lst[0]))
    new_y = lst[2]
    new_height = lst[4] // 2
    cropped_image = original_image.crop((lst[1], new_y, lst[1] + lst[3], new_y + new_height))  
    return cropped_image

def compare_histogram_pairs(hist_test, bounding_boxes):
    """
    Compares the histogram of an individual in the foreground of a test image with the histogram of all the individuals in the foreground of all other images.

    Args:
        hist_test (numpy.ndarray): Histogram of an individual in a test image.
        bounding_boxes (list): List of bounding box images (This parameter can either be a list of bounding boxes or a list of the upper half of bounding boxes).

    Returns:
        list: List of similarity scores between the test histogram and bounding box histograms arranged as a list of lists (where the each inner-list is comprised of the similarity
        score at index 0 and the image of said bounding box at index 1)
    """
    similarity_scores = []
    for bbox in bounding_boxes:
        bbox_hist = calc_histogram(bbox)
        similarity_score = compare_histograms(hist_test, bbox_hist)
        similarity_scores.append([similarity_score, bbox])
    return similarity_scores

def compare_max_similarity_scores(comp1, comp2, comp3, comp4):
    """
    Compares maximum similarity scores between four sets of comparison results.

    Args:
        comp1 (list): First set of comparison results arranged as a list of lists containing a similarity score at index 0 and their corresponding image at index 1.
        comp2 (list): Second set of comparison results arranged as a list of lists containing a similarity score at index 0 and their corresponding image at index 1.
        comp3 (list): Third set of comparison results arranged as a list of lists containing a similarity score at index 0 and their corresponding image at index 1.
        comp4 (list): Fourth set of comparison results arranged as a list of lists containing a similarity score at index 0 and their corresponding image at index 1.

    Returns:
        list: List of lists containing the maximum similarity scores (index 0) and their corresponding images (index 1).
    """
    max_similarity_scores = []
    for c1, c2, c3, c4 in zip(comp1, comp2, comp3, comp4):
        max_score = max(c1[0], c2[0], c3[0], c4[0])
        max_image = None
        for comp in [c1, c2, c3, c4]:
            if comp[0] == max_score:
                max_image = comp[1]
                break
        max_similarity_scores.append([max_score, max_image])
    return max_similarity_scores

def get_top_100_images(max_similarity_scores_all):
    """
    Finds the top 100 images based on maximum similarity scores.

    Args:
        max_similarity_scores_all (list): List of lists containing the maximum similarity scores at index 0 and their corresponding images at index 1.

    Returns:
        list: List of the top 100 images.
    """
    # The similarity scores are sorted in descending order and the images associated with the first 100 are appended to a new list
    sorted_scores = sorted(max_similarity_scores_all, key=lambda x: x[0], reverse=True)
    top_100_images = [score[1] for score in sorted_scores[:100]]
    return top_100_images

def save_top_100_images(top_100_images, destination_folder):
    """
    Saves top 100 images to a destination folder.

    Args:
        top_100_images (list): List of top 100 images.
        destination_folder (str): Path to the destination folder.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for i, image in enumerate(top_100_images):
        image_filename = f"image_{i+1}.png"
        image_path = os.path.join(destination_folder, image_filename)
        image.save(image_path)

import os

import os

# Function to save images from a list into a destination folder
def save_images_to_folder(images, destination_folder):
    """
    Saves images from a list to a destination folder.

    Args:
        images (list): List of PIL.Image objects.
        destination_folder (str): Path to the destination folder.
    """
    # Ensure the destination folder exists, create if it doesn't
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Save each image to the destination folder
    for i, image in enumerate(images):
        image_path = os.path.join(destination_folder, f"image_{i+1}.png")
        image.save(image_path)

'''

MAIN EXECUTION

'''
        

# Initializing a dictionnary of images (with their names as keys) from the cam_0 and cam_1 folder. 
image_folder_cam0 = 'images/cam0'
# image_folder_cam1 = 'images/cam1'

image_dict = create_image_dict(image_folder_cam0)
# image_dict_cam1 = create_image_dict(image_folder_cam1)

# image_dict = {**image_dict_cam0, **image_dict_cam1}

# Loading the labels text file and extracting both a list of all bounding boxes and a list of the upper-half of said bounding boxes
data_file_path = 'bounding_boxes.txt'
with open(data_file_path, 'r') as data_file:
    bounding_boxes = extract_bounding_box(image_dict, data_file)
with open(data_file_path, 'r') as data_file:
    upper_bounding_boxes = extract_upper_half_bounding_box(image_dict, data_file)

# Test Images

# The following lists denote the image name as well as the coordinates of the bounding box of the persons of interest in the video sequence arranged as follows: 
# [file_name, x_coordinate, y_coordinate, width, height]
        
# Boy in a white t-shirt
p1 = ["person_1.png",284,24,46,180] 
# 1637433806859709600,119,66,75,247 
# 1637433806859709600,356,10,61,221
# 1637433806859709600,184,21,70,267
# 1637433806859709600,284,24,46,180
# 1637433806859709600,279,210,133,175
# 1637433806859709600,578,33,69,147
# 1637433806859709600,572,221,120,164
# 1637433806859709600,238,21,38,110
# 1637433806859709600,314,1,34,32
# Man in a black jacket
p2 = ["person_2.png",300,2,98,257]
# 1637434087168850600,300,2,98,257 
# 1637434087168850600,1,166,106,192
# 1637434087168850600,110,105,97,149
# 1637434087168850600,139,219,109,233
# 1637434087168850600,293,301,91,102
# 1637434087168850600,105,8,51,95
# Boy in Cargo Hoodie
p3 = ["person_3.png",251,26,71,210]
# 1637433822031079200,4,142,89,212 
# 1637433822031079200,566,206,112,180
# 1637433822031079200,251,26,71,210
# 1637433822031079200,250,232,131,186
# 1637433822031079200,345,11,87,179
# 1637433822031079200,290,4,57,188
# 1637433822031079200,362,89,58,185
# Man in a grey long-sleeve
p4 = ["person_4.png",423,57,101,281]
# 1637434154979087500,423,57,101,281
# 1637434154979087500,150,125,118,250
# 1637434154979087500,286,2,82,215
# 1637434154979087500,5,201,136,192
# 1637434154979087500,282,194,87,188
# 1637434154979087500,285,43,82,328
# Man in a marine blue jacket
p5 = ["person_5.png",504,24,80,245]
# 1637434045404623100,504,24,80,245
# 1637434045404623100,72,150,106,217
# 1637434045404623100,181,253,81,193
# 1637434045404623100,259,293,81,118
# 1637434045404623100,19,111,93,240

# Defining the lists of coordinates of the bounding box of each individual in the foreground of the two test images
coordinates = [p1, p2, p3, p4, p5]

# Initializing the lists to store all cropped images (of bounding boxes) and their associated histograms
cropped_images = []
cropped_images_upper_half = []
histograms = []
histograms_upper_half = []

# Looping through the list of bounding boxes from the two test images and croping them according to their coordinates. They are then stored in their respective lists 
# (one for the entire bounding box and another for their upper-half) 

for i, coords in enumerate(coordinates, start=1):
     
     cropped_img_upper_half = crop_upper_half(coords)
     cropped_images_upper_half.append(cropped_img_upper_half)  

     # Retrieve the image filename
     image_filename = coords[0]

     # Load the image using PIL
     image_obj = Image.open(image_filename)
     cropped_images.append(image_obj)

     # Calculate histograms for the image and its upper half
     hist = calc_histogram(image_obj)
     histograms.append(hist)
     hist_upper_half = calc_histogram(cropped_img_upper_half)
     histograms_upper_half.append(hist_upper_half)

    #  cropped_img = crop_image(coords)
    #  cropped_images.append(cropped_img)
    
    #  cropped_img_upper_half = crop_upper_half(coords)
    #  cropped_images_upper_half.append(cropped_img_upper_half)
    
    #  # Calculating the associated histograms of each image from both lists.
    #  hist = calc_histogram(cropped_img)
    #  histograms.append(hist)
    
    #  hist_upper_half = calc_histogram(cropped_img_upper_half)
    #  histograms_upper_half.append(hist_upper_half)

 # Comparing histograms by intersection and preserving the top 100 images with the highest similarity scores for each individual in the test images
for i, (hist_test, hist_test_upper_half) in enumerate(zip(histograms, histograms_upper_half), start=1):
     # comp1 compares the histogram of an individual from a test image to the histogram of an individual from another image
     comp1 = compare_histogram_pairs(hist_test, bounding_boxes)
     # comp2 compares the histogram of an individual from a test image to the histogram of the upper-half of an individual from another image
     comp2 = compare_histogram_pairs(hist_test, upper_bounding_boxes)
     # comp3 compares the histogram of the upper-half of an individual from a test image to the histogram of an individual from another image
     comp3 = compare_histogram_pairs(hist_test_upper_half, bounding_boxes)
     # comp4 compares the histogram of the upper-half of an individual from a test image to the histogram of the upper-half of an individual from another image
     comp4 = compare_histogram_pairs(hist_test_upper_half, upper_bounding_boxes)
    
     # The maximum similarity score between the four comparison instances is then preserved
     max_similarity_scores_all = compare_max_similarity_scores(comp1, comp2, comp3, comp4)

     # Finally, the top 100 images from the list of all maximum similarity scores for each individual in the test images are saved to a new folder
     top_100_images = get_top_100_images(max_similarity_scores_all)
     destination_folder = f'result_person_{i}'
     save_top_100_images(top_100_images, destination_folder)




 

# cropped_img_upper_half = crop_upper_half(p1)

# # Retrieve the image filename
# image_filename = p1[0]

# # Load the image using PIL
# image_obj = Image.open("person_1.png")

# # Calculate histograms for the image and its upper half
# hist = calc_histogram(image_obj)
# hist_upper_half = calc_histogram(cropped_img_upper_half)

# # Comparing histograms by intersection and preserving the top 100 images with the highest similarity scores
# comp1 = compare_histogram_pairs(hist, bounding_boxes)
# comp2 = compare_histogram_pairs(hist, upper_bounding_boxes)
# comp3 = compare_histogram_pairs(hist_upper_half, bounding_boxes)
# comp4 = compare_histogram_pairs(hist_upper_half, upper_bounding_boxes)

# max_similarity_scores_all = compare_max_similarity_scores(comp1, comp2, comp3, comp4)

# top_100_images = get_top_100_images(max_similarity_scores_all)
# destination_folder = f'result_person_{1}'
# save_top_100_images(top_100_images, destination_folder)

# # Path to the destination folder
# destination_folder = "bounding_box_images"

#  # Call the function to save images
# save_images_to_folder(bounding_boxes, destination_folder)
