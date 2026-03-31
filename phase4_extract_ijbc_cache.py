import os
import cv2
import pickle
from tqdm import tqdm
from arcfaceutility import get_encoding_from_image
from insightface.app import FaceAnalysis

#MODEL_NAME = 'buffalo_l'
MODEL_NAME = 'antelopev2'

#MODEL_NAME = 'buffalo_s'
#MODEL_NAME = 'r100_ms1mv3'
#MODEL_NAME = 'r34_glint360k'
#MODEL_NAME = 'r50_vgg2'
#MODEL_NAME = 'r100_ms1mv2'
#MODEL_NAME = 'r50_cisia'
#MODEL_NAME = 'r50_glintasia'
#MODEL_NAME = 'ms1m_megaface_r50'
#MODEL_NAME = 'ms1mv2_r50'
#MODEL_NAME = 'ms1mv3_r50'
#MODEL_NAME = 'r34_ms1mv3'
#MODEL_NAME = 'r18_ms1mv3'
#MODEL_NAME = 'buffalo_l'
#MODEL_NAME = 'r100_glint360k'

IJBC_IMAGES_DIR = "datasets/ijb-testsuite/ijb/IJBC/loose_crop"
IJBC_PAIRS_FILE = "datasets/ijb-testsuite/ijb/IJBC/pairs.txt"
OUTPUT_CACHE = f"ijbc_embeddings_cache_{MODEL_NAME}.pkl"

# Modeli Yükle
app = FaceAnalysis(name=MODEL_NAME, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Benzersiz resim isimlerini bul (Sadece gerekli resimleri işlemek için)
unique_images = set()
with open(IJBC_PAIRS_FILE, 'r') as f:
    for line in f.readlines()[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            unique_images.add(parts[0].strip())
            unique_images.add(parts[1].strip())

print(f"📸 Toplam {len(unique_images)} benzersiz resim işlenecek...")

embeddings_dict = {}
ram_cache = {}

for img_name in tqdm(list(unique_images), desc=f"Cache Çıkarılıyor ({MODEL_NAME})"):
    full_path = os.path.join(IJBC_IMAGES_DIR, img_name)
    img = cv2.imread(full_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        emb, _ = get_encoding_from_image(img, cache=ram_cache, cache_key=img_name)
        if emb is not None:
            embeddings_dict[img_name] = emb

# Diske Kaydet
with open(OUTPUT_CACHE, 'wb') as f:
    pickle.dump(embeddings_dict, f)

print(f"✅ Başarılı! Cache dosyası oluşturuldu: {OUTPUT_CACHE}")