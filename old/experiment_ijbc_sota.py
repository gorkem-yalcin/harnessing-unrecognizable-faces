import os
import cv2
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.spatial.distance import euclidean

# Kendi modüllerin
from arcfaceutility import ensure_rgb, get_encoding_from_image

# ==========================================
# 1. AYARLAR
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

IJBC_ROOT = "datasets/ijb-testsuite/ijb/IJBC"
PAIRS_PATH = os.path.join(IJBC_ROOT, "pairs.txt")
IMG_DIR = os.path.join(IJBC_ROOT, "loose_crop")

# TinyFace'ten gelen SOTA Threshold
FIXED_THRESHOLD = 124.4503
print(f"🏆 Testing with SOTA HYBRID THRESHOLD: {FIXED_THRESHOLD}")

# ==========================================
# 2. MODEL TANIMI VE YÜKLEME
# ==========================================
# TinyFace'te kullanılan modelin AYNISI (input_dim=2049)
class SOTAClassifier(nn.Module):
    def __init__(self, input_dim=2049): # DÜZELTİLDİ: 1537 -> 2049
        super().__init__()
        # TinyFace'teki mimariyle aynı olmalı:
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),  # TinyFace kodunda 1024 ile başlıyor
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

    def forward(self, x):
        return self.net(x)

# Modeli Yükle
model_path = "final_sota_classifier.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ 'final_sota_classifier.pth' bulunamadı! Önce experiment_tinyface.py çalışmalı.")

model = SOTAClassifier().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print("✅ SOTA Classifier Modeli Yüklendi.")

# ==========================================
# 3. IJB-C PRE-COMPUTATION
# ==========================================
print("\n--- PHASE A: IJB-C Pre-computation ---")
# Pairs dosyasını oku
df_pairs = pd.read_csv(PAIRS_PATH)
print(f"   Toplam Çift Sayısı: {len(df_pairs)}")

# Benzersiz resimleri bul
unique_imgs = pd.unique(df_pairs[['img1', 'img2']].values.ravel('K'))
print(f"   Benzersiz Resim Sayısı: {len(unique_imgs)}")

embedding_map = {}

print("2. Resimler işleniyor ve RAM'e alınıyor...")
# Embeddingleri çıkar (ResNet50 - Buffalo)
# arcfaceutility.py dosyasında CUSTOM_R100_PATH = None olduğundan emin ol!
for img_name in tqdm(unique_imgs):
    full_path = os.path.join(IMG_DIR, img_name)

    img_arr = ensure_rgb(full_path)
    if img_arr is None: continue

    emb, _ = get_encoding_from_image(img_arr, "", None, None)

    if emb is not None:
        embedding_map[img_name] = emb

print(f"✅ {len(embedding_map)} resim başarıyla işlendi.")

# ==========================================
# 4. EVALUATION (HYBRID SCORE)
# ==========================================
print("\n--- PHASE B: Running Evaluation on 15M Pairs (Hybrid Score) ---")

tp, tn, fp, fn = 0, 0, 0, 0
skipped = 0

# Veriyi Batch Halinde İşlemek Hızlandırır (PyTorch için)
BATCH_SIZE = 4096
pairs_numpy = df_pairs.values

# Batch döngüsü
for i in tqdm(range(0, len(pairs_numpy), BATCH_SIZE)):
    batch = pairs_numpy[i: i + BATCH_SIZE]

    batch_features = []
    batch_labels = []
    batch_sims = []
    valid_indices = []

    # Batch içindeki her bir çift için Feature Engineering
    for idx, row in enumerate(batch):
        img1, img2, label = row[0], row[1], int(row[2])

        if img1 not in embedding_map or img2 not in embedding_map:
            skipped += 1
            continue

        enc1 = embedding_map[img1]
        enc2 = embedding_map[img2]

        # --- FEATURE ENGINEERING (TinyFace ile Birebir Aynı) ---
        diff = np.abs(enc1 - enc2)
        mult = enc1 * enc2
        sim = np.dot(enc1, enc2)  # Cosine Sim (Zaten normlanmışsa dot yeterli)

        # [enc1, enc2, diff, mult, sim] -> 512+512+512+512+1 = 2049 boyutlu?
        # DİKKAT: TinyFace kodunda "enc1, enc2, diff, mult, [sim]" yazıyor.
        # enc1(512), enc2(512), diff(512), mult(512), sim(1) = 2049 Boyut.
        # Ama yukarıda model tanımında input_dim=1537 dedim.
        # TinyFace'teki Classifier sınıfının input_dim değerini kontrol etmeliyiz.
        # Genelde diff+mult+sim kullanılır (1025). Ama koduna göre full feature.
        # Eğer hata alırsan Classifier input_dim'ini 2049 yap.

        # Senin TinyFace koduna göre:
        feat = np.concatenate([enc1, enc2, diff, mult, [sim]])

        batch_features.append(feat)
        batch_labels.append(label)
        batch_sims.append(sim)
        valid_indices.append(idx)

    if not batch_features:
        continue

    # PyTorch Inference
    X_batch = torch.tensor(np.array(batch_features), dtype=torch.float32).to(device)

    with torch.no_grad():
        # Modelden olasılık al
        probs = torch.sigmoid(model(X_batch)).cpu().numpy().flatten()

    # Hybrid Score Hesapla ve Karar Ver
    sims = np.array(batch_sims)
    labels = np.array(batch_labels)

    hybrid_scores = sims * probs

    preds = (hybrid_scores >= FIXED_THRESHOLD).astype(int)

    # Metrikleri Güncelle
    tp += np.sum((preds == 1) & (labels == 1))
    tn += np.sum((preds == 0) & (labels == 0))
    fp += np.sum((preds == 1) & (labels == 0))
    fn += np.sum((preds == 0) & (labels == 1))

# ==========================================
# 5. SONUÇLAR
# ==========================================
total_valid = tp + tn + fp + fn
acc = (tp + tn) / total_valid if total_valid > 0 else 0
f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
far = fp / (fp + tn) if (fp + tn) > 0 else 0
frr = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"\n=== IJB-C FINAL SOTA RESULTS ===")
print(f"Total Pairs: {total_valid} (Skipped: {skipped})")
print(f"Accuracy   : {acc:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"FAR        : {far:.6f}")
print(f"FRR        : {frr:.4f}")

with open("results_ijbc_sota.txt", "w") as f:
    f.write(f"IJB-C Results with SOTA Hybrid Score\n")
    f.write(f"Threshold: {FIXED_THRESHOLD}\n")
    f.write(f"F1: {f1:.4f}\n")
    f.write(f"FAR: {far:.6f}\n")
    f.write(f"FRR: {frr:.4f}\n")

print("✅ Sonuçlar 'results_ijbc_sota.txt' dosyasına kaydedildi.")