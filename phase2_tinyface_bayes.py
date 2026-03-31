import os
import cv2
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# Mevcut scriptlerinden import
from arcfaceutility import ensure_rgb, get_encoding_from_image

# ==========================================
# ⚙️ PARAMETRELER
# ==========================================
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

TINYFACE_IMAGES_DIR = "datasets/tinyface/images"
TINYFACE_PAIRS_FILE = "datasets/tinyface/pairs.txt"

# Çıktılar
BAYES_MODEL_FILE = f"tinyface_bayes_model_{MODEL_NAME}.pkl"
PLOT_OUTPUT_FILE = f"tinyface_distributions_{MODEL_NAME}.png"

CACHE = {}


# ==========================================
# 🛠️ YARDIMCI FONKSİYONLAR
# ==========================================
def parse_tinyface_pairs(pairs_path):
    pairs = []
    with open(pairs_path, 'r') as f:
        lines = f.readlines()[1:]  # Header'ı atla

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            # Match
            name = parts[0]
            img1 = os.path.join(name, parts[1])
            img2 = os.path.join(name, parts[2])
            pairs.append((img1, img2, 1))
        elif len(parts) == 4:
            # Mismatch
            name1, name2 = parts[0], parts[2]
            img1 = os.path.join(name1, parts[1])
            img2 = os.path.join(name2, parts[3])
            pairs.append((img1, img2, 0))
    return pairs


def process_clean_image(img_path):
    full_path = os.path.join(TINYFACE_IMAGES_DIR, img_path)
    img = cv2.imread(full_path)
    if img is None: return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cache_key = f"{MODEL_NAME}_{img_path}"
    emb, _ = get_encoding_from_image(img, cache=CACHE, cache_key=cache_key)
    return emb


# ==========================================
# 🚀 ANA İŞLEM
# ==========================================
def main():
    print(f"--- 🚀 FAZ 2: TinyFace Bayes Dağılımı ({MODEL_NAME}) ---")

    pairs = parse_tinyface_pairs(TINYFACE_PAIRS_FILE)
    print(f"👥 Okunan TinyFace Çifti: {len(pairs)}")

    genuine_scores = []  # Aynı kişiler (1)
    imposter_scores = []  # Farklı kişiler (0)

    for img1_path, img2_path, is_match in tqdm(pairs, desc="TinyFace İşleniyor"):
        emb1 = process_clean_image(img1_path)
        emb2 = process_clean_image(img2_path)

        if emb1 is not None and emb2 is not None:
            norm1 = emb1 / np.linalg.norm(emb1)
            norm2 = emb2 / np.linalg.norm(emb2)
            cos_sim = np.dot(norm1, norm2)

            if is_match == 1:
                genuine_scores.append(cos_sim)
            else:
                imposter_scores.append(cos_sim)

    print("\n📊 Dağılımlar Çıkarılıyor (KDE)...")
    # Kernel Density Estimation (PDF - Olasılık Yoğunluk Fonksiyonu)
    kde_genuine = gaussian_kde(genuine_scores)
    kde_imposter = gaussian_kde(imposter_scores)

    # Bayes Modelini Kaydet (Sözlük olarak)
    # P(G) = P(I) = 0.5 kabul ediyoruz çünkü pairs dosyasında 3000 match, 3000 mismatch var.
    bayes_model = {
        'kde_genuine': kde_genuine,
        'kde_imposter': kde_imposter,
        'p_genuine': len(genuine_scores) / (len(genuine_scores) + len(imposter_scores)),
        'p_imposter': len(imposter_scores) / (len(genuine_scores) + len(imposter_scores))
    }

    with open(BAYES_MODEL_FILE, 'wb') as f:
        pickle.dump(bayes_model, f)
    print(f"💾 Bayes KDE Modeli Kaydedildi: {BAYES_MODEL_FILE}")

    # ==========================================
    # 📈 GRAFİK ÇİZİMİ (Hocaya Göstermelik)
    # ==========================================
    x_vals = np.linspace(-0.2, 1.0, 500)
    y_gen = kde_genuine(x_vals)
    y_imp = kde_imposter(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_gen, color='green', label='Genuine', linewidth=2)
    plt.fill_between(x_vals, y_gen, alpha=0.3, color='green')

    plt.plot(x_vals, y_imp, color='red', label='Imposter', linewidth=2)
    plt.fill_between(x_vals, y_imp, alpha=0.3, color='red')

    plt.title("TinyFace Cosine Similarity Distributions - " + MODEL_NAME, fontsize=14)
    plt.xlabel("Cosine Similarity Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(PLOT_OUTPUT_FILE, dpi=300)
    print(f"📈 Dağılım Grafiği Çizildi: {PLOT_OUTPUT_FILE}")
    print("✅ Faz 2 Tamamlandı!")


if __name__ == "__main__":
    main()