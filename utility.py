import os

import cv2
import hdbscan
import numpy as np
import torch
from PIL import Image
from deepface import DeepFace


def save_image(img, path, filename):
    if isinstance(img, np.ndarray):  # Check if image is a numpy array
        img = Image.fromarray(img)  # Convert to PIL Image
    img.save(os.path.join(path, filename))


def get_ui_clusters_hdbscan(unrecognizable_training_images, min_cluster_size):
    unrec_embs = [enc for _, enc, _, _ in unrecognizable_training_images if enc is not None]
    ui_centroids = []
    if len(unrec_embs) > 0:
        X = np.array(unrec_embs)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(X)
        unique_cluster_ids = sorted(set(cluster_labels) - {-1})
        for cluster_id in unique_cluster_ids:
            cluster_points = X[cluster_labels == cluster_id]
            centroid = cluster_points.mean(axis=0)
            ui_centroids.append(centroid)
    return ui_centroids


def get_encoding_from_image(img, method, embedding_cache, image_key, detector, embedder, device):
    if method == "deepface":
        rgb_img = np.array(img)
        try:
            result = DeepFace.represent(rgb_img, model_name="VGG-Face", enforce_detection=False, detector_backend="mtcnn")
            embedding = result[0]["embedding"]
            return np.array(embedding)
        except Exception as e:
            print(f"Error extracting face encoding: {e}")
            return None
    elif method == "MTCNN":
        if image_key in embedding_cache:
            return np.array(embedding_cache[image_key])
        #detector = MTCNN()
        rgb_img = np.array(img)
        if rgb_img.dtype != np.uint8:
            rgb_img = (rgb_img * 255).clip(0, 255).astype(np.uint8)
        try:
            detections = detector.detect_faces(rgb_img)
            if len(detections) == 0:
                return None

            # Use first detected face
            box = detections[0]['box']
            x, y, w, h = box
            x, y = max(0, x), max(0, y)
            face = rgb_img[y:y + h, x:x + w]

            # Resize to 224x224 if needed (VGG-Face default)
            face = Image.fromarray(face).resize((224, 224))
            face_array = np.array(face)

            result = DeepFace.represent(face_array, model_name="VGG-Face", enforce_detection=False)
            embedding = result[0]["embedding"]
            embedding_array = np.array(embedding)
            if embedding_array.size == 1 and np.isnan(embedding_array[0]) and np.isnan(embedding_array).any():
                return None
            embedding_cache[image_key] = embedding_array
            return embedding_array

        except Exception as e:
            print(f"[MTCNN] Error extracting face encoding: {e}")
            return None
    elif method == "facenet_pytorch":
        if image_key in embedding_cache:
            return embedding_cache[image_key]

        try:
            if not isinstance(img, Image.Image):
                if img.dtype != np.uint8:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img)
            # Detect face (returns cropped face tensor if detected)
            face_tensor = detector(img)

            if face_tensor is None:
                return None  # No face detected

            face_tensor = face_tensor.unsqueeze(0).to(device)  # Add batch dimension

            # Get embedding
            with torch.no_grad():
                embedding = embedder(face_tensor).cpu().numpy().flatten()

            embedding_cache[image_key] = embedding
            return embedding

        except Exception as e:
            print(f"[PyTorch] Error extracting embedding: {e}")
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
