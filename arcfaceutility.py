import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from insightface.app.common import Face

# Modelleri RAM'de tutmak için sözlük (Böylece aynı model 2 kez yüklenmez)
_loaded_models = {}

def get_insightface_app(model_name):
    """Modeli sadece ihtiyaç duyulduğunda yükler ve RAM'de tutar."""
    if model_name not in _loaded_models:
        print(f"⏳ InsightFace ({model_name}) modeli yükleniyor...")
        app = FaceAnalysis(name=model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _loaded_models[model_name] = app
        print(f"✅ {model_name} hazır.")
    return _loaded_models[model_name]

def ensure_rgb(img):
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# DİKKAT: Artık fonksiyon 'model_name' parametresi istiyor!
def get_encoding_from_image(img, model_name, cache=None, cache_key=None, detector=None, embedder=None, device=None):
    if cache is not None and cache_key in cache:
        return cache[cache_key], None

    if isinstance(img, torch.Tensor): img = img.cpu().numpy()
    if not isinstance(img, np.ndarray): return None, None

    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.5 else img.astype(np.uint8)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    embedding, bbox = None, None

    # 🌟 SİHİRLİ SATIR: Modeli dinamik olarak çağır
    app = get_insightface_app(model_name)

    try:
        faces = app.get(img_bgr)
        if len(faces) > 0:
            face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
            embedding = face.embedding
            bbox = face.bbox
        else:
            img_112 = cv2.resize(img_bgr, (112, 112))
            face = Face(bbox=np.array([0, 0, img_bgr.shape[1], img_bgr.shape[0]]))
            face.kps = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                                 [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
            face.det_score = 1.0

            if hasattr(app, 'models') and 'recognition' in app.models:
                app.models['recognition'].get(img_112, face)
                embedding = face.embedding
                bbox = face.bbox
    except Exception as e:
        return None, None

    if embedding is None: return None, None
    if cache is not None and cache_key is not None: cache[cache_key] = embedding.copy()
    return embedding, bbox