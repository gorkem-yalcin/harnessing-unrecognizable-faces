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

# BASE THRESHOLD (SOTA Skorumuzdan gelen)
BASE_THRESH = 123.6058

# DINAMIK AYARLAR (Bu değerlerle oynayacağız)
# Güven yüksekse threshold düşer (kurtarmak için), düşükse artar (güvenlik için)
THRESH_EASY = BASE_THRESH * 0.90  # Prob > 0.8 (Kolay Örnekler)
THRESH_MID = BASE_THRESH * 1.00  # Prob 0.4 - 0.8 (Orta)
THRESH_HARD = BASE_THRESH * 1.10  # Prob < 0.4 (Zor/Bulanık)

print(f"🧪 Dynamic Threshold Experiment")
print(f"   Easy (>0.8): {THRESH_EASY:.4f}")
print(f"   Mid  (0.4-0.8): {THRESH_MID:.4f}")
print(f"   Hard (<0.4): {THRESH_HARD:.4f}")


# ==========================================
# 2. MODEL TANIMI (Classifier - 2049 Dim)
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
print("✅ Classifier Yüklendi.")

# ==========================================
# 3. VERİ HAZIRLIĞI (Embedding Cache)
# ==========================================
# (Hızlı çalışması için embedding alma kısmını özet geçiyorum,
# ama senin kodunda embedding_map dolu olmalı. Önceki koddan kopyalayabilirsin.)
# ... Buraya previous koddaki "Pre-computation" kısmını eklediğini varsayıyorum ...
# Veya hızlı test için embeddingleri pickle ile kaydedip yüklemek mantıklı olurdu.
# Şimdilik tam akışı yazıyorum:

from arcfaceutility import ensure_rgb, get_encoding_from_image

df_pairs = pd.read_csv(PAIRS_PATH)
unique_imgs = pd.unique(df_pairs[['img1', 'img2']].values.ravel('K'))
embedding_map = {}

print("Memory Cache Dolduruluyor...")
# BURASI UZUN SÜREBİLİR, EĞER ÖNCEKİ ÇALIŞMADAN EMBEDDINGLER RAM'DEYSE
# JUPYTER/PYTHON CONSOLE KULLANIYORSAN TEKRAR YÜKLEMENE GEREK YOK.
# SIFIRDAN ÇALIŞTIRACAKSAN BU DÖNGÜ GEREKLİ:
for img_name in tqdm(unique_imgs):
    full_path = os.path.join(IMG_DIR, img_name)
    img_arr = ensure_rgb(full_path)
    if img_arr is not None:
        emb, _ = get_encoding_from_image(img_arr, "", None, None)
        if emb is not None: embedding_map[img_name] = emb

# ==========================================
# 4. DİNAMİK EVALUATION
# ==========================================
print("\n--- Running Dynamic Threshold Evaluation ---")
tp, tn, fp, fn = 0, 0, 0, 0
skipped = 0

BATCH_SIZE = 4096
pairs_numpy = df_pairs.values

for i in tqdm(range(0, len(pairs_numpy), BATCH_SIZE)):
    batch = pairs_numpy[i: i + BATCH_SIZE]
    batch_features = []
    batch_labels = []
    batch_sims = []

    for row in batch:
        img1, img2, label = row[0], row[1], int(row[2])
        if img1 not in embedding_map or img2 not in embedding_map:
            skipped += 1
            continue

        enc1, enc2 = embedding_map[img1], embedding_map[img2]

        diff = np.abs(enc1 - enc2)
        mult = enc1 * enc2
        sim = np.dot(enc1, enc2)

        feat = np.concatenate([enc1, enc2, diff, mult, [sim]])
        batch_features.append(feat)
        batch_labels.append(label)
        batch_sims.append(sim)

    if not batch_features: continue

    X_batch = torch.tensor(np.array(batch_features), dtype=torch.float32).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(X_batch)).cpu().numpy().flatten()

    sims = np.array(batch_sims)
    labels = np.array(batch_labels)
    hybrid_scores = sims * probs

    # --- DİNAMİK KARAR MEKANİZMASI ---
    preds = []
    for prob, score in zip(probs, hybrid_scores):
        # 1. Kural: Model Çok Eminse (>0.80) -> Kapıyı Gevşet (Easy Thresh)
        if prob > 0.80:
            decision = 1 if score >= THRESH_EASY else 0

        # 2. Kural: Model Şüpheliyse (<0.40) -> Kapıyı Sıkı Tut (Hard Thresh)
        elif prob < 0.40:
            decision = 1 if score >= THRESH_HARD else 0

        # 3. Kural: Normal Durum -> Standart Threshold
        else:
            decision = 1 if score >= THRESH_MID else 0

        preds.append(decision)

    preds = np.array(preds)

    tp += np.sum((preds == 1) & (labels == 1))
    tn += np.sum((preds == 0) & (labels == 0))
    fp += np.sum((preds == 1) & (labels == 0))
    fn += np.sum((preds == 0) & (labels == 1))

# ==========================================
# 5. SONUÇLAR
# ==========================================
total = tp + tn + fp + fn
far = fp / (fp + tn) if (fp + tn) > 0 else 0
frr = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"\n=== IJB-C DYNAMIC THRESHOLD RESULTS ===")
print(f"FAR: {far:.6f}")
print(f"FRR: {frr:.4f}")

# Karşılaştırma
print(f"\nPrevious Best (Fixed): FRR 0.4487")
diff = 0.4487 - frr
if diff > 0:
    print(f"🚀 İyileştirme: +{diff:.4f} (Daha az hata!)")
else:
    print(f"🔻 Gerileme: {diff:.4f} (Dynamic ayarları tune edilmeli)")