import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from insightface.app.common import Face

# ==========================================
# INSIGHTFACE (ARCFACE) MODEL YÜKLEME
# ==========================================
print("⏳ InsightFace (ArcFace) modeli yükleniyor...")
# GPU varsa kullan, yoksa CPU
#app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print("✅ InsightFace modeli hazır.")


def ensure_rgb(img):
    """Resim BGR ise RGB'ye çevirir."""
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_encoding_from_image(img, model_name=None, cache=None, cache_key=None, detector=None, embedder=None, device=None):
    """
    InsightFace (ArcFace) kullanarak embedding alır.
    Fallback mekanizması ile yüz bulunamayan (zaten crop olan) resimleri de işler.
    """
    # 1. Cache Kontrolü
    if cache is not None and cache_key in cache:
        return cache[cache_key], None

    # 2. Görüntü Hazırlığı ve Tip Dönüşümü
    if isinstance(img, torch.Tensor):  # Eğer torch tensor gelirse numpy'a çevir
        img = img.cpu().numpy()

    if not isinstance(img, np.ndarray):
        return None, None

    # Float -> Uint8 Dönüşümü
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.5:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # RGB -> BGR (InsightFace standardı)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 3. Yüz Tespiti ve Embedding
    embedding = None
    bbox = None

    try:
        # app objesi global olarak tanımlı varsayılıyor (arcfaceutility.py başında)
        faces = app.get(img_bgr)

        if len(faces) > 0:
            # Yüz bulundu, en büyüğünü al
            face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
            embedding = face.embedding
            bbox = face.bbox
        else:
            # --- FALLBACK MECHANISM ---
            # Dedektör yüz bulamadı. (LFW crop resimleri için)

            # 1. 112x112 Resize (ArcFace standardı)
            img_112 = cv2.resize(img_bgr, (112, 112))

            # 2. Dummy Face Objesi Yarat
            # Face sınıfı import edilmiş olmalı: from insightface.app.common import Face
            face = Face(bbox=np.array([0, 0, img_bgr.shape[1], img_bgr.shape[0]]))

            # Keypoints (Landmarks) - Standart hizalama için
            face.kps = np.array([
                [38.2946, 51.6963],  # Sol Göz
                [73.5318, 51.5014],  # Sağ Göz
                [56.0252, 71.7366],  # Burun
                [41.5493, 92.3655],  # Dudak Sol
                [70.7299, 92.2041]  # Dudak Sağ
            ], dtype=np.float32)
            face.det_score = 1.0

            # 3. Recognition Modelini Bul ve Çalıştır
            rec_model = None
            # app.models listesinden recognition modelini çek
            if hasattr(app, 'models') and 'recognition' in app.models:
                rec_model = app.models['recognition']

            if rec_model is not None:
                # get fonksiyonu resmi keypointlere göre hizalar ve embedding üretir
                rec_model.get(img_112, face)
                embedding = face.embedding
                bbox = face.bbox

    except Exception as e:
        # Hata bastırma, logla
        # import traceback eklemeyi unutma
        return None, None

    if embedding is None:
        return None, None

    # 4. Cache'e Yaz (Eğer cache aktifse)
    if cache is not None and cache_key is not None:
        cache[cache_key] = embedding.copy()

    return embedding, bbox