import os

import cv2
import numpy as np
from deepface import DeepFace


def save_image(img, path, filename):
    img.save(os.path.join(path, filename))


def get_encoding_from_image(img):
    rgb_img = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2RGB)

    try:
        result = DeepFace.represent(rgb_img, model_name="VGG-Face", enforce_detection=False)
        embedding = result[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"Error extracting face encoding: {e}")
        return None
