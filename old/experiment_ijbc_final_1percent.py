import os
import cv2
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# ==========================================
# 1. AYARLAR
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

IJBC_ROOT = "datasets/ijb-testsuite/ijb/IJBC"
PAIRS_PATH = os.path.join(IJBC_ROOT, "pairs.txt")
IMG_DIR = os.path.join(IJBC_ROOT, "loose_crop")

# HEDEF FAR (Makalenin Table 11 değeri)
TARGET_FAR = 0.01  # %1


# ==========================================
# 2. MODEL (SOTA Hybrid Classifier - 2049 Dim)
# ==========================================
class SOTAClassifier(nn.Module):
    def __init__(self, input_dim=2049):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x): return self.net(x)


model_path = "final_sota_classifier.pth"
model = SOTAClassifier().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print("✅ SOTA Classifier Yüklendi.")

# ==========================================
# 3. VERİ HAZIRLIĞI (Embedding Cache)
# ==========================================
# (Hızlıca cache doldurma veya yükleme)
from arcfaceutility import ensure_rgb, get_encoding_from_image

print("Pairs ve Embeddingler Yükleniyor...")
df_pairs = pd.read_csv(PAIRS_PATH)
unique_imgs = pd.unique(df_pairs[['img1', 'img2']].values.ravel('K'))
embedding_map = {}

# NOT: Eğer notebook kullanıyorsan ve embedding_map zaten hafızadaysa
# bu döngüyü atlayabilirsin. Sıfırdan çalıştırıyorsan gerekli.
for img_name in tqdm(unique_imgs):
    full_path = os.path.join(IMG_DIR, img_name)
    img_arr = ensure_rgb(full_path)
    if img_arr is not None:
        emb, _ = get_encoding_from_image(img_arr, "", None, None)
        if emb is not None: embedding_map[img_name] = emb

# ==========================================
# 4. TÜM SKORLARI TOPLAMA
# ==========================================
print("\n--- Calculating All Scores ---")
all_scores = []
all_labels = []

BATCH_SIZE = 4096
pairs_numpy = df_pairs.values

for i in tqdm(range(0, len(pairs_numpy), BATCH_SIZE)):
    batch = pairs_numpy[i: i + BATCH_SIZE]
    batch_features = []
    current_labels = []
    batch_sims = []  # Cosine Sim (Ham)

    for row in batch:
        img1, img2, label = row[0], row[1], int(row[2])
        if img1 not in embedding_map or img2 not in embedding_map: continue

        enc1, enc2 = embedding_map[img1], embedding_map[img2]

        diff = np.abs(enc1 - enc2)
        mult = enc1 * enc2
        sim = np.dot(enc1, enc2)

        feat = np.concatenate([enc1, enc2, diff, mult, [sim]])
        batch_features.append(feat)
        current_labels.append(label)
        batch_sims.append(sim)

    if not batch_features: continue

    X_batch = torch.tensor(np.array(batch_features), dtype=torch.float32).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(X_batch)).cpu().numpy().flatten()

    # HYBRID SCORE (Sim * Prob)
    sims = np.array(batch_sims)
    hybrid_scores = sims * probs

    all_scores.extend(hybrid_scores)
    all_labels.extend(current_labels)

all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

# ==========================================
# 5. FIND THRESHOLD @ FAR = 1%
# ==========================================
print("\n--- Finding Optimal Threshold for FAR = 1% ---")

# Negatif (Eşleşmeyen) Skorları Al
neg_scores = all_scores[all_labels == 0]
neg_scores.sort()  # Küçükten büyüğe sırala

# Negatiflerin en yüksek %1'lik kısmını kesen nokta
# Örnek: 100 negatif varsa, en yüksek 1 tanesi hata (False Accept) olsun.
# index = len - (len * 0.01)
target_idx = int(len(neg_scores) * (1 - TARGET_FAR))
optimal_threshold = neg_scores[target_idx]

print(f"🎯 Target FAR: {TARGET_FAR * 100}%")
print(f"🔑 Calculated Threshold: {optimal_threshold:.4f}")

# ==========================================
# 6. CALCULATE FRR @ THAT THRESHOLD
# ==========================================
pos_scores = all_scores[all_labels == 1]
# Threshold'un altında kalan pozitifler (Yanlışlıkla reddedilenler)
false_rejects = np.sum(pos_scores < optimal_threshold)
frr = false_rejects / len(pos_scores)

# Gerçekleşen FAR'ı da kontrol edelim (Double Check)
false_accepts = np.sum(neg_scores >= optimal_threshold)
actual_far = false_accepts / len(neg_scores)

print(f"\n=== FINAL SOTA COMPARISON (Aligned FAR) ===")
print(f"Threshold used: {optimal_threshold:.4f}")
print(f"Actual FAR    : {actual_far:.4f} (Hedef: {TARGET_FAR})")
print(f"🏆 FINAL FRR  : {frr:.4f}")

print("-" * 30)
print(f"Makale (SOTA) : 0.4718")
print(f"Bizim Sonuç   : {frr:.4f}")
if frr < 0.4718:
    print(f"🚀 FARK: -{0.4718 - frr:.4f} (Daha iyi!)")