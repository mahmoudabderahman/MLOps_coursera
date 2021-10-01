import os
import io
import cv2
import requests
import numpy as np
from IPython.display import Image, display

base_url = 'http://127.0.0.1:8000'
endpoint = '/predict'
model = 'yolov3-tiny'
confidence = '0.5'

url_with_endpoint_no_params = base_url + endpoint
print(url_with_endpoint_no_params)

full_url = url_with_endpoint_no_params + "?model=" + model + "&confidence=" + confidence
print(full_url)


def response_from_server(url, image_file, verbose=True):
    """Makes a POST request to the server and returns the response.

    :param url (str): URL that the request is sent to.
    :param image_file (_io.BufferedReader): File to upload,
    :param verbose (bool): True if the status of the response should be printed, False otherwise.
    :return:
        requests.models.Response: Response from the server.
    """
    files = {'file': image_file}
    response = requests.post(url, files=files)
    status_code = response.status_code
    if verbose:
        msg = "Everything went well!" if status_code == 200 else "There was an error when handling the request."
        print(msg)
    return response


with open("../images/fruits.jpg", "rb") as image_file:
    prediction = response_from_server(full_url, image_file)

dir_name = "../images_predicted"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def display_image_from_response(response):
    """Display image within server's response.

    :param response (requests.models.Response): The response from the server after object detection.
    """

    # Read image as a stream of bytes
    image_stream = io.BytesIO(response.content)
    # Start the stream from the beginning (position zero)
    image_stream.seek(0)
    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    filename = "image_with_objects.jpeg"
    cv2.imwrite(f'../images_predicted/{filename}', image)
    #display(image(f'../images_predicted/{filename}'))


display_image_from_response(prediction)

image_files = [
    'car2.jpg',
    'clock3.jpg',
    'apples.jpg'
]

for image_file in image_files:
    with open(f'../images/{image_file}', 'rb') as image_file:
        prediction = response_from_server(full_url, image_file, verbose=False)

    display_image_from_response(prediction)