import os
import random
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.datasets import fetch_lfw_pairs

# Mevcut scriptlerinden importlar
from degradations import degradation_pool
from arcfaceutility import ensure_rgb, get_encoding_from_image

# ==========================================
# ⚙️ PARAMETRELER (PYCHARM'DAN DÜZENLE)
# ==========================================
MODEL_NAME = 'buffalo_l'  # Hangi modeli kullanıyoruz?
DECISION_THRESHOLD = 0.25  # BUNDAN BÜYÜKSE = 1 (Recognizable)

# Dosya Yolları
BASE_CACHE_FILE = f"lfw_base_embeddings_{MODEL_NAME}.pkl"  # AĞIR İŞÇİLİK BURAYA KAYDEDİLECEK
FINAL_DATA_FILE = f"lfw_train_data_{MODEL_NAME}_thresh{DECISION_THRESHOLD}.pkl"

# Degradation (Bozma) Ayarları
DEGRADATION_STRENGTH_MIN = 1
DEGRADATION_STRENGTH_MAX = 2.5
APPLY_DEGRADATION_PROB = 1

# RAM Cache (Sadece o anki çalışma sırasında temiz resimlerin tekrarını önlemek için)la
RAM_CACHE = {}


# ==========================================
# 🛠️ YARDIMCI FONKSİYONLAR
# ==========================================
def process_image_array(img_array, img_id, apply_deg=False):
    """Sklearn'den gelen Numpy resim dizisini işler, istenirse bozar ve embedding alır."""
    if img_array.dtype == np.float32:
        img_uint8 = (img_array * 255).astype(np.uint8)
    else:
        img_uint8 = img_array.copy()

    deg_name = "Original"
    deg_str = 0.0

    if apply_deg:
        strength = random.uniform(DEGRADATION_STRENGTH_MIN, DEGRADATION_STRENGTH_MAX)
        fn = random.choice(degradation_pool)
        degraded = fn(img_uint8, strength=strength)

        if not isinstance(degraded, np.ndarray):
            degraded = np.array(degraded)

        img_uint8 = degraded
        deg_name = fn.__name__
        deg_str = strength

    img_uint8 = ensure_rgb(img_uint8)

    # RAM Cache kontrolü (Özellikle Original resimler için çok hızlandırır)
    cache_key = f"{MODEL_NAME}_{img_id}_{deg_name}_{deg_str:.2f}"
    emb, _ = get_encoding_from_image(img_uint8, cache=RAM_CACHE, cache_key=cache_key)

    return emb, deg_name, deg_str


# ==========================================
# 🚀 AŞAMA 1: AĞIR İŞÇİLİK (DISK CACHE OLUŞTURMA)
# ==========================================
def generate_base_embeddings():
    print(f"\n--- ⏳ AŞAMA 1: Base Embedding'ler Çıkarılıyor (Model: {MODEL_NAME}) ---")

    lfw_dataset = fetch_lfw_pairs(subset='10_folds', color=True, resize=1.0)
    pairs = lfw_dataset.pairs

    base_records = []

    # Tüm pairlerdeki (Match/Mismatch fark etmez) resimleri tek bir havuza alıyoruz
    for i in tqdm(range(len(pairs)), desc="Resimler İşleniyor"):
        # Birinci resim
        img1_array = pairs[i][0]
        img1_id = f"pair_{i}_img1"

        # İkinci resim
        img2_array = pairs[i][1]
        img2_id = f"pair_{i}_img2"

        # Her iki resim için de AYNI İŞLEMİ yapıyoruz: Kendisinin Temizi vs Kendisinin Bozuğu
        for img_array, img_id in [(img1_array, img1_id), (img2_array, img2_id)]:
            # Kendisinin Temiz Hali
            emb_clean, _, _ = process_image_array(img_array, img_id, apply_deg=False)

            # Kendisinin Bozuk Hali
            should_degrade = random.random() < APPLY_DEGRADATION_PROB
            emb_deg, deg_name, deg_str = process_image_array(img_array, img_id, apply_deg=should_degrade)

            if emb_clean is not None and emb_deg is not None:
                base_records.append({
                    'clean_embedding': emb_clean,
                    'degraded_embedding': emb_deg,
                    'deg_type': deg_name,
                    'deg_strength': deg_str
                })

    df_base = pd.DataFrame(base_records)
    with open(BASE_CACHE_FILE, 'wb') as f:
        pickle.dump(df_base, f)
    print(f"✅ Ağır işçilik bitti! Base veriler '{BASE_CACHE_FILE}' dosyasına kaydedildi.")
    return df_base


# ==========================================
# ⚡ AŞAMA 2: HIZLI ETİKETLEME (THRESHOLD UYGULAMA)
# ==========================================
def apply_threshold_and_save(df_base):
    print(f"\n--- ⚡ AŞAMA 2: Threshold ({DECISION_THRESHOLD}) Uygulanıyor ---")

    dataset_records = []

    for _, row in tqdm(df_base.iterrows(), total=len(df_base), desc="Kosinüs & Label Hesaplanıyor"):
        emb_clean = row['clean_embedding']
        emb_deg = row['degraded_embedding']

        # Kosinüs Benzerliği
        norm_clean = emb_clean / np.linalg.norm(emb_clean)
        norm_deg = emb_deg / np.linalg.norm(emb_deg)
        cos_sim = np.dot(norm_clean, norm_deg)

        # Hard Label
        hard_label = 1 if cos_sim >= DECISION_THRESHOLD else 0

        # Classifier için sadece bozuk embedding ve label lazım
        dataset_records.append({
            'degraded_embedding': emb_deg,
            'original_cos_sim': cos_sim,
            'hard_label': hard_label,
            'deg_type': row['deg_type'],
            'deg_strength': row['deg_strength']
            # pair_is_match satırı tamamen silindi!
        })

    df_final = pd.DataFrame(dataset_records)
    with open(FINAL_DATA_FILE, 'wb') as f:
        pickle.dump(df_final, f)

    print(f"✅ İşlem Tamam! Eğitim verisi '{FINAL_DATA_FILE}' olarak kaydedildi.")
    print(f"📊 Sınıf Dağılımı (1=Tanınabilir, 0=Tanınamaz):")
    print(df_final['hard_label'].value_counts())

# ==========================================
# 🎯 ANA KONTROL BLOĞU
# ==========================================
print(f"🚀 FAZ 1 BAŞLIYOR | Model: {MODEL_NAME} | Threshold: {DECISION_THRESHOLD}")

# 1. Disk Cache Var Mı Kontrol Et
if os.path.exists(BASE_CACHE_FILE):
    print(f"📦 Disk Cache bulundu: '{BASE_CACHE_FILE}'. Çıkartma işlemi atlanıyor!")
    with open(BASE_CACHE_FILE, 'rb') as f:
        df_base_data = pickle.load(f)
else:
    print(f"⚠️ Disk Cache bulunamadı. Baştan hesaplanacak...")
    df_base_data = generate_base_embeddings()

# 2. Hızlı Etiketlemeyi Çalıştır
apply_threshold_and_save(df_base_data)