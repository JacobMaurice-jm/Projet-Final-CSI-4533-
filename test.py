import cv2
import os

def extract_bounding_boxes(image_name, parameters):
    # Create a directory to store the extracted bounding boxes
    output_folder = 'extracted_boxes'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(os.path.abspath(image_name))
    # Read the input image
    image = cv2.imread(image_name)

    # Check if image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image {image_name}")
        return

    # Iterate over each set of parameters
    for param in parameters:
        # Parse parameters
        values = param.split(',')
        if len(values) != 5:
            print(f"Error: Invalid parameter format - {param}")
            continue
        try:
            x, y, width, height = map(int, values[1:])
        except ValueError:
            print(f"Error: Invalid parameter values - {param}")
            continue

        # Check if coordinates are within image dimensions
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            print(f"Error: Invalid bounding box coordinates - {param}")
            continue
        if x + width > image.shape[1] or y + height > image.shape[0]:
            print(f"Error: Bounding box out of image bounds - {param}")
            continue

        # Extract bounding box
        box = image[y:y+height, x:x+width]

        # Save the bounding box as a separate image
        box_name = os.path.join(output_folder, f'box_{values[1]}.png')
        cv2.imwrite(box_name, box)

        print(f"Bounding box {values[0]} saved as {box_name}")

# if __name__ == "__main__":
#     image_name = "1637433772063999700.png"
#     parameters = [
#         "1637433772063999700,197,259,192,132",
#         "1637433772063999700,18,185,273,74",
#         "1637433772063999700,154,71,207,87",
#         "1637433772063999700,213,576,183,111",
#         "1637433772063999700,68,117,212,72",
#         "1637433772063999700,41,2,170,53",
#         "1637433772063999700,7,270,191,52",
#         "1637433772063999700,4,247,127,38",
#         "1637433772063999700,3,215,52,29",
#         "1637433772063999700,1,315,34,33"
#     ]
#     extract_bounding_boxes(image_name, parameters)

# if __name__ == "__main__":
#      image_name = "1637433806859709600.png"
#      parameters = [
#     "1637433806859709600,119,66,75,247",
#     "1637433806859709600,356,10,61,221",
#     "1637433806859709600,184,21,70,267",
#     "1637433806859709600,284,24,46,180",
#     "1637433806859709600,279,210,133,175",
#     "1637433806859709600,578,33,69,147",
#     "1637433806859709600,572,221,120,164",
#     "1637433806859709600,238,21,38,110",
#     "1637433806859709600,314,1,34,32"
#     ]

#      extract_bounding_boxes(image_name, parameters)

# if __name__ == "__main__":
#      image_name = "1637434087168850600.png"
#      parameters = [
#     "1637434087168850600,300,2,98,257",
#     "1637434087168850600,1,166,106,192",
#     "1637434087168850600,110,105,97,149",
#     "1637434087168850600,139,219,109,233",
#     "1637434087168850600,293,301,91,102",
#     "1637434087168850600,105,8,51,95"
# ]
#      extract_bounding_boxes(image_name, parameters)

# if __name__ == "__main__":
#      image_name = "1637433822031079200.png"
#      parameters = [
#     "1637433822031079200,4,142,89,212",
#     "1637433822031079200,566,206,112,180",
#     "1637433822031079200,251,26,71,210",
#     "1637433822031079200,250,232,131,186",
#     "1637433822031079200,345,11,87,179",
#     "1637433822031079200,290,4,57,188",
#     "1637433822031079200,362,89,58,185"
# ]
#      extract_bounding_boxes(image_name, parameters)


# if __name__ == "__main__":
#      image_name = "1637434154979087500.png"
#      parameters = [
#     "1637434154979087500,423,57,101,281",
#     "1637434154979087500,150,125,118,250",
#     "1637434154979087500,286,2,82,215",
#     "1637434154979087500,5,201,136,192",
#     "1637434154979087500,282,194,87,188",
#     "1637434154979087500,285,43,82,328"
# ]

#      extract_bounding_boxes(image_name, parameters)


# S
