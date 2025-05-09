import os

import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace


def save_image(img, path, filename):
    if isinstance(img, np.ndarray):  # Check if image is a numpy array
        img = Image.fromarray(img)  # Convert to PIL Image
    img.save(os.path.join(path, filename))


def get_encoding_from_image(img):
    rgb_img = np.array(img)
    try:
        result = DeepFace.represent(rgb_img, model_name="VGG-Face", enforce_detection=False)
        embedding = result[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"Error extracting face encoding: {e}")
        return None


def convert_pair_to_grayscale_uint8(pair):
    def to_gray_uint8(img):
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray

    return to_gray_uint8(pair[0]), to_gray_uint8(pair[1])


def ensure_rgb(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img
