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

# 🌟 KONTROL MERKEZİNDEN AYARLARI ÇEK
from config import *

# ==========================================
# ⚙️ PARAMETRELER (ARTIK CONFIG'DEN GELİYOR)
# ==========================================
# QUALITY_MODEL, DECISION_THRESHOLD, LFW_BASE_CACHE, LFW_TRAIN_DATA
# değişkenleri otomatik olarak config.py dosyasından alındı!

# Degradation (Bozma) Ayarları
DEGRADATION_STRENGTH_MIN = 1
DEGRADATION_STRENGTH_MAX = 2.5
APPLY_DEGRADATION_PROB = 1

# RAM Cache
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

    # 🌟 BURASI GÜNCELLENDİ: QUALITY_MODEL kullanılıyor
    cache_key = f"{QUALITY_MODEL}_{img_id}_{deg_name}_{deg_str:.2f}"
    emb, _ = get_encoding_from_image(img_uint8, model_name=QUALITY_MODEL, cache=RAM_CACHE, cache_key=cache_key)

    return emb, deg_name, deg_str


# ==========================================
# 🚀 AŞAMA 1: AĞIR İŞÇİLİK (DISK CACHE OLUŞTURMA)
# ==========================================
def generate_base_embeddings():
    print(f"\n--- ⏳ AŞAMA 1: Base Embedding'ler Çıkarılıyor (Hakem Model: {QUALITY_MODEL}) ---")

    lfw_dataset = fetch_lfw_pairs(subset='10_folds', color=True, resize=1.0)
    pairs = lfw_dataset.pairs

    base_records = []

    for i in tqdm(range(len(pairs)), desc="Resimler İşleniyor"):
        img1_array = pairs[i][0]
        img1_id = f"pair_{i}_img1"

        img2_array = pairs[i][1]
        img2_id = f"pair_{i}_img2"

        for img_array, img_id in [(img1_array, img1_id), (img2_array, img2_id)]:
            emb_clean, _, _ = process_image_array(img_array, img_id, apply_deg=False)

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
    with open(LFW_BASE_CACHE, 'wb') as f:
        pickle.dump(df_base, f)
    print(f"✅ Ağır işçilik bitti! Base veriler '{LFW_BASE_CACHE}' dosyasına kaydedildi.")
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

        norm_clean = emb_clean / np.linalg.norm(emb_clean)
        norm_deg = emb_deg / np.linalg.norm(emb_deg)
        cos_sim = np.dot(norm_clean, norm_deg)

        hard_label = 1 if cos_sim >= DECISION_THRESHOLD else 0

        dataset_records.append({
            'degraded_embedding': emb_deg,
            'original_cos_sim': cos_sim,
            'hard_label': hard_label,
            'deg_type': row['deg_type'],
            'deg_strength': row['deg_strength']
        })

    df_final = pd.DataFrame(dataset_records)
    with open(LFW_TRAIN_DATA, 'wb') as f:
        pickle.dump(df_final, f)

    print(f"✅ İşlem Tamam! Eğitim verisi '{LFW_TRAIN_DATA}' olarak kaydedildi.")
    print(f"📊 Sınıf Dağılımı (1=Tanınabilir, 0=Tanınamaz):")
    print(df_final['hard_label'].value_counts())


# ==========================================
# 🎯 ANA KONTROL BLOĞU
# ==========================================
print(f"🚀 FAZ 1 BAŞLIYOR | Hakem Model: {QUALITY_MODEL} | Threshold: {DECISION_THRESHOLD}")

if os.path.exists(LFW_BASE_CACHE):
    print(f"📦 Disk Cache bulundu: '{LFW_BASE_CACHE}'. Çıkartma işlemi atlanıyor!")
    with open(LFW_BASE_CACHE, 'rb') as f:
        df_base_data = pickle.load(f)
else:
    print(f"⚠️ Disk Cache bulunamadı. Baştan hesaplanacak...")
    df_base_data = generate_base_embeddings()

apply_threshold_and_save(df_base_data)