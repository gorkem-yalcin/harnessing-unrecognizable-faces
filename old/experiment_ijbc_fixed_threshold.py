import os
import cv2
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics.pairwise import cosine_similarity

# Kendi modüllerin
from arcfaceutility import ensure_rgb, get_encoding_from_image

# ==========================================
# 1. AYARLAR
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# Klasör Yolları
IJBC_ROOT = "datasets/ijb-testsuite/ijb/IJBC"
PAIRS_PATH = os.path.join(IJBC_ROOT, "pairs.txt")
IMG_DIR = os.path.join(IJBC_ROOT, "loose_crop")

# --- KRİTİK AYAR: ROC Analizinden Gelen Best Threshold ---
FIXED_THRESHOLD = 123.6058
print(f"🏆 Testing with FIXED THRESHOLD: {FIXED_THRESHOLD}")

# ==========================================
# 2. IJB-C PRE-COMPUTATION (Aynen Kalıyor)
# ==========================================
print("\n--- PHASE A: IJB-C Pre-computation ---")
print("1. Pairs dosyası okunuyor...")
df_pairs = pd.read_csv(PAIRS_PATH)
print(f"   Toplam Çift Sayısı: {len(df_pairs)}")

unique_imgs = pd.unique(df_pairs[['img1', 'img2']].values.ravel('K'))
print(f"   Benzersiz Resim Sayısı: {len(unique_imgs)}")

embedding_map = {}

print("2. Resimler işleniyor ve RAM'e alınıyor...")
# Daha önce çalıştıysa cache'den okumasını sağlayabiliriz ama
# temiz kurulum için tekrar hesaplatıyoruz (Hızlı sürüyor zaten)
for img_name in tqdm(unique_imgs):
    full_path = os.path.join(IMG_DIR, img_name)

    img_arr = ensure_rgb(full_path)
    if img_arr is None: continue

    emb, _ = get_encoding_from_image(img_arr, "", None, None)

    if emb is not None:
        embedding_map[img_name] = emb

print(f"✅ {len(embedding_map)} resim başarıyla işlendi.")

# ==========================================
# 3. EVALUATION (FIXED THRESHOLD)
# ==========================================
print("\n--- PHASE B: Running Evaluation on 15M Pairs (Fixed Threshold) ---")

tp, tn, fp, fn = 0, 0, 0, 0
skipped = 0

pairs_numpy = df_pairs.values

for row in tqdm(pairs_numpy, total=len(pairs_numpy)):
    img1, img2, label = row[0], row[1], int(row[2])

    if img1 not in embedding_map or img2 not in embedding_map:
        skipped += 1
        continue

    emb1 = embedding_map[img1]
    emb2 = embedding_map[img2]

    # Similarity Calculation
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    # --- KARAR MEKANİZMASI (BASİTLEŞTİRİLDİ) ---
    # Artık Adaptive değil, sabit ROC threshold kullanıyoruz.
    pred = 1 if sim >= FIXED_THRESHOLD else 0

    if pred == 1 and label == 1:
        tp += 1
    elif pred == 0 and label == 0:
        tn += 1
    elif pred == 1 and label == 0:
        fp += 1
    elif pred == 0 and label == 1:
        fn += 1

# ==========================================
# 4. SONUÇLAR
# ==========================================
total_valid = tp + tn + fp + fn
acc = (tp + tn) / total_valid if total_valid > 0 else 0
f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
far = fp / (fp + tn) if (fp + tn) > 0 else 0
frr = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"\n=== IJB-C FINAL RESULTS (Threshold: {FIXED_THRESHOLD}) ===")
print(f"Total Pairs: {total_valid}")
print(f"Accuracy   : {acc:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"FAR        : {far:.6f}")
print(f"FRR        : {frr:.4f}")

# Sonuçları kaydet
with open("results_ijbc_fixed.txt", "w") as f:
    f.write(f"IJB-C Results with Fixed Threshold\n")
    f.write(f"Threshold: {FIXED_THRESHOLD}\n")
    f.write(f"F1: {f1:.4f}\n")
    f.write(f"FAR: {far:.6f}\n")
    f.write(f"FRR: {frr:.4f}\n")

print("✅ Sonuçlar 'results_ijbc_fixed.txt' dosyasına kaydedildi.")