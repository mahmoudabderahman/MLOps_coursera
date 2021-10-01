from IPython.display import Image, display
import os
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

image_files = [
    'apple.jpg',
    'clock.jpg',
    'oranges.jpg',
    'car.jpg'
]

for image_file in image_files:
    print(f"\nDisplaying image: {image_file}")
    display(Image(filename=f"../images/{image_file}"))

dir_name = "../images_with_boxes"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def detect_and_draw_box(filename, model="yolov3-tiny", confidence=0.5):
    """Detects common objects on an image and creates a new image with bounding boxes.

    Args:
        filename (str): Filename of the image.
        model (str): Either "yolov3" or "yolov3-tiny". Defaults to "yolov3-tiny".
        confidence (float, optional: Desired confidence level. Defaults to 0.5.
    """

    # images are stored under the images/ directory
    image_filepath = f'../images/{filename}'

    # Read the image into a numpy array
    img = cv2.imread(image_filepath)

    # Perform the object detection
    bbox, label, conf = cv.detect_common_objects(img, confidence=confidence, model=model)

    # Print current image's filename
    print(f"=============================\nImage processed: {filename}\n")

    # Print detected objects with confidence level
    for l, c in zip(label, conf):
        print(f"Detected object: {l} with confidence level of {c}\n")

    # Create a new image that includes the bounding boxes
    output_image = draw_bbox(img, bbox, label, conf)

    # Save the image in the directory images_with_boxes
    cv2.imwrite(f'../images_with_boxes/{filename}', output_image)

    # Display the image with bounding boxes
    display(Image(f'../images_with_boxes/{filename}'))


for image_file in image_files:
    detect_and_draw_box(image_file)


detect_and_draw_box("fruits.jpg", confidence=0.2)