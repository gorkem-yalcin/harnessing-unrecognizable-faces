import os
import cv2
import pickle
from tqdm import tqdm

# 🌟 KONTROL MERKEZİNDEN AYARLARI ÇEK
from config import *
from arcfaceutility import get_encoding_from_image, get_insightface_app

# ==========================================
# ⚙️ PARAMETRELER (ARTIK CONFIG'DEN GELİYOR)
# ==========================================
# Bu dosyayı hangi model için çalıştırıyorsan config.py'daki MATCHER_MODEL'i ona ayarla!
MODEL_NAME = MATCHER_MODEL

IJBC_IMAGES_DIR = "datasets/ijb-testsuite/ijb/IJBC/loose_crop"
IJBC_PAIRS_FILE = "datasets/ijb-testsuite/ijb/IJBC/pairs.txt"
OUTPUT_CACHE = f"ijbc_embeddings_cache_{MODEL_NAME}.pkl"


def main():
    print(f"--- 🚀 IJB-C CACHE ÇIKARICI (Model: {MODEL_NAME}) ---")

    # Modeli önceden yükle (RAM'e alıp hazır bekletiyoruz)
    app = get_insightface_app(MODEL_NAME)

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

            # 🌟 BURASI GÜNCELLENDİ: model_name parametresi eklendi
            emb, _ = get_encoding_from_image(img, model_name=MODEL_NAME, cache=ram_cache, cache_key=img_name)
            if emb is not None:
                embeddings_dict[img_name] = emb

    # Diske Kaydet
    with open(OUTPUT_CACHE, 'wb') as f:
        pickle.dump(embeddings_dict, f)

    print(f"✅ Başarılı! Cache dosyası oluşturuldu: {OUTPUT_CACHE}")


if __name__ == "__main__":
    main()