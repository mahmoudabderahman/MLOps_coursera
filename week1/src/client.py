import os
import io
import cv2
import requests
import numpy as np
from IPython.display import Image, display

base_url = 'http://localhost:8000'
endpoint = '/predict'
model='yolov3-tiny'

url_with_endpoint_no_params = base_url + endpoint
print(url_with_endpoint_no_params)

full_url = url_with_endpoint_no_params + "?model=" + model
print(full_url)

