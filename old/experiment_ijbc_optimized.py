import os
import cv2
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_lfw_pairs

# Kendi modüllerin
from degradations import degradation_pool
from arcfaceutility import ensure_rgb, get_encoding_from_image

# ==========================================
# 1. AYARLAR
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# Klasör yolu (Debug ile doğruladığın yol)
IJBC_ROOT = "datasets/ijb-testsuite/ijb/IJBC"
PAIRS_PATH = os.path.join(IJBC_ROOT, "pairs.txt")
IMG_DIR = os.path.join(IJBC_ROOT, "loose_crop")

THRESHOLD_MAP_PATH = "results_tinyface/tinyface_learned_thresholds.pkl"

if not os.path.exists(THRESHOLD_MAP_PATH):
    raise FileNotFoundError("❌ TinyFace threshold dosyası yok! Önce experiment_tinyface.py çalışmalı.")

with open(THRESHOLD_MAP_PATH, "rb") as f:
    LEARNED_THRESHOLDS = pickle.load(f)
print("✅ TinyFace Thresholds yüklendi.")

# ==========================================
# 2. CLASSIFIER MODELİ (LFW-Synthetic ile Hızlı Eğitim)
# ==========================================
print("\n--- PHASE A: Training Classifier (LFW-Synthetic) ---")

# Embedding Cache
if os.path.exists("embedding_cache_lfw.pkl"):
    with open("embedding_cache_lfw.pkl", "rb") as f:
        cache = pickle.load(f)
else:
    cache = {}

lfw_train = fetch_lfw_pairs(subset='train', color=True, resize=1.0)
train_pairs = [(ensure_rgb(p[0]), ensure_rgb(p[1])) for p in lfw_train.pairs]
train_labels = lfw_train.target
train_match_pairs = [p for p, l in zip(train_pairs, train_labels) if l == 1]

X_train_list, y_train_list = [], []
centroid_list = []

print("Generating synthetic data for classifier...")
for i in tqdm(range(len(train_match_pairs))):
    img1, img2 = train_match_pairs[i]

    # Clean Ref
    verif_enc, _ = get_encoding_from_image(img2, "", cache, f"train_verif_{i}")
    if verif_enc is None: continue

    # Clean Probe
    orig_enc, _ = get_encoding_from_image(img1, "", cache, f"train_orig_{i}")
    if orig_enc is not None:
        X_train_list.append(orig_enc)
        y_train_list.append(1.0)

    # Degraded Probe
    deg_fn = degradation_pool[i % len(degradation_pool)]
    deg_img = deg_fn(img1.copy(), strength=np.random.randint(3, 6))
    deg_enc, _ = get_encoding_from_image(deg_img, "", {}, "temp")

    if deg_enc is not None:
        sim = cosine_similarity([verif_enc], [deg_enc])[0][0]
        lbl = 1.0 if sim > 0.25 else 0.0
        X_train_list.append(deg_enc)
        y_train_list.append(lbl)
        if lbl == 0.0: centroid_list.append(deg_enc)


# Classifier Definition
class Classifier(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LeakyReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(), nn.BatchNorm1d(64), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x): return self.net(x)


X_tr = np.array(X_train_list)
if len(X_tr) > 0: X_tr = X_tr / np.linalg.norm(X_tr, axis=1, keepdims=True)
y_tr = np.array(y_train_list)

model = Classifier().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()
dset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
ldr = DataLoader(dset, batch_size=256, shuffle=True)

model.train()
for epoch in range(15):
    for xb, yb in ldr:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
model.eval()
print("✅ Classifier hazır.")

# ==========================================
# 3. IJB-C OPTIMIZASYON (PRE-COMPUTE)
# ==========================================
print("\n--- PHASE B: IJB-C Pre-computation ---")
print("1. Pairs dosyası okunuyor...")
df_pairs = pd.read_csv(PAIRS_PATH)
print(f"   Toplam Çift Sayısı: {len(df_pairs)}")

unique_imgs = pd.unique(df_pairs[['img1', 'img2']].values.ravel('K'))
print(f"   Benzersiz Resim Sayısı: {len(unique_imgs)}")

embedding_map = {}
prob_map = {}

print("2. Resimler işleniyor ve RAM'e alınıyor...")
for img_name in tqdm(unique_imgs):
    full_path = os.path.join(IMG_DIR, img_name)

    # 1. ÖNCE RESMİ OKU (ensure_rgb path'i kabul eder ve array döner)
    img_arr = ensure_rgb(full_path)

    if img_arr is None:
        continue  # Resim bozuksa veya yoksa atla

    # 2. ARRAY'İ FONKSİYONA VER
    emb, _ = get_encoding_from_image(img_arr, "", None, None)

    if emb is not None:
        embedding_map[img_name] = emb

        with torch.no_grad():
            enc_norm = emb / np.linalg.norm(emb)
            t_tensor = torch.tensor([enc_norm], dtype=torch.float32).to(device)
            prob = torch.sigmoid(model(t_tensor)).item()
            prob_map[img_name] = prob

print(f"✅ {len(embedding_map)} resim başarıyla işlendi.")

# ==========================================
# 4. EVALUATION
# ==========================================
print("\n--- PHASE C: Running Evaluation on 15M Pairs ---")


def get_threshold_for_score(score, threshold_map):
    sorted_bins = sorted(threshold_map.keys())
    for bid in sorted_bins:
        vals = threshold_map[bid]
        if vals['min_score'] <= score <= vals['max_score']:
            return vals['thresh']
    if score < threshold_map[sorted_bins[0]]['min_score']:
        return threshold_map[sorted_bins[0]]['thresh']
    return threshold_map[sorted_bins[-1]]['thresh']


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
    prob1 = prob_map[img1]  # Probe quality

    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    # Adaptive Threshold
    adaptive_thresh = get_threshold_for_score(prob1, LEARNED_THRESHOLDS)

    pred = 1 if sim >= adaptive_thresh else 0

    if pred == 1 and label == 1:
        tp += 1
    elif pred == 0 and label == 0:
        tn += 1
    elif pred == 1 and label == 0:
        fp += 1
    elif pred == 0 and label == 1:
        fn += 1

# ==========================================
# 5. SONUÇLAR
# ==========================================
total_valid = tp + tn + fp + fn
acc = (tp + tn) / total_valid if total_valid > 0 else 0
f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
far = fp / (fp + tn) if (fp + tn) > 0 else 0
frr = fn / (fn + tp) if (fn + tp) > 0 else 0

print("\n=== IJB-C FINAL RESULTS (Cross-Dataset) ===")
print(f"Total Pairs Processed: {total_valid} (Skipped: {skipped})")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"FAR     : {far:.6f}")
print(f"FRR     : {frr:.4f}")

with open("results_ijbc_summary.txt", "w") as f:
    f.write(f"IJB-C Evaluation Results\n")
    f.write(f"Model: ArcFace (ResNet50)\n")
    f.write(f"Method: Classifier Adaptive Threshold (Trained on TinyFace)\n")
    f.write(f"F1: {f1:.4f}\n")
    f.write(f"FAR: {far:.6f}\n")
    f.write(f"FRR: {frr:.4f}\n")

print("✅ Sonuçlar 'results_ijbc_summary.txt' dosyasına kaydedildi.")