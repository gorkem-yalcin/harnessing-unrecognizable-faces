import os
import cv2
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from arcfaceutility import ensure_rgb, get_encoding_from_image

# ==========================================
# 1. AYARLAR
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# Dataset Yolları
DATASET_ROOT = "datasets"
TINYFACE_PATH = os.path.join(DATASET_ROOT, "tinyface")
TINY_PAIRS = os.path.join(TINYFACE_PATH, "pairs.txt")
TINY_IMG_DIR = os.path.join(TINYFACE_PATH, "images")

IJBC_ROOT = "datasets/ijb-testsuite/ijb/IJBC"
IJBC_PAIRS = os.path.join(IJBC_ROOT, "pairs.txt")
IJBC_IMG_DIR = os.path.join(IJBC_ROOT, "loose_crop")


# ==========================================
# 2. MODELİ YÜKLE (Mevcut En İyi Model)
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
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Model yok! Önce experiment_tinyface.py çalıştır.")

model = SOTAClassifier().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print("✅ Model Yüklendi.")


# ==========================================
# 3. FONKSİYONLAR
# ==========================================
def get_hybrid_scores(pairs_path, img_dir, dataset_name):
    print(f"\n--- Processing {dataset_name} ---")

    # Pairs Okuma
    if dataset_name == "TinyFace":
        matches, mismatches = [], []
        with open(pairs_path, 'r') as f:
            lines = f.readlines()[1:]
        for line in lines:
            p = line.strip().split('\t')
            if len(p) == 3:
                matches.append((p[0], p[1], p[2]))
            elif len(p) == 4:
                mismatches.append((p[0], p[1], p[2], p[3]))

        # TinyFace formatını IJB-C gibi tek listeye çevir
        pairs = []
        for p in matches: pairs.append([p[0], p[0], 1, p[1], p[2]])  # img1, img2 path logic
        for p in mismatches: pairs.append([p[0], p[2], 0, p[1], p[3]])
    else:
        # IJB-C Formatı
        df = pd.read_csv(pairs_path)
        pairs = df.values

    # Embedding Cache (Hız için basit versiyon)
    embedding_map = {}
    print("   Embeddingler hazırlanıyor...")

    # Unique resimleri bul
    if dataset_name == "TinyFace":
        # TinyFace path mantığı biraz karışık, direkt döngüde halledelim
        pass
    else:
        unique_imgs = pd.unique(df[['img1', 'img2']].values.ravel('K'))
        for img_name in tqdm(unique_imgs):
            full_path = os.path.join(img_dir, img_name)
            emb, _ = get_encoding_from_image(ensure_rgb(full_path), "", None, None)
            if emb is not None: embedding_map[img_name] = emb

    y_true, y_scores = [], []

    print("   Skorlar hesaplanıyor...")
    BATCH_SIZE = 2048
    batch_feats, batch_lbls, batch_sims = [], [], []

    for i, row in tqdm(enumerate(pairs), total=len(pairs)):
        # Path Mantığı
        if dataset_name == "TinyFace":
            name1, name2, label, f1, f2 = row
            path1 = os.path.join(img_dir, name1, f1)
            path2 = os.path.join(img_dir, name2, f2)
            enc1, _ = get_encoding_from_image(ensure_rgb(path1), "", None, None)
            enc2, _ = get_encoding_from_image(ensure_rgb(path2), "", None, None)
        else:
            img1, img2, label = row[0], row[1], int(row[2])
            if img1 not in embedding_map or img2 not in embedding_map: continue
            enc1, enc2 = embedding_map[img1], embedding_map[img2]

        if enc1 is None or enc2 is None: continue

        diff = np.abs(enc1 - enc2)
        mult = enc1 * enc2
        sim = np.dot(enc1, enc2)
        feat = np.concatenate([enc1, enc2, diff, mult, [sim]])

        batch_feats.append(feat)
        batch_lbls.append(label)
        batch_sims.append(sim)

        if len(batch_feats) >= BATCH_SIZE or i == len(pairs) - 1:
            X_b = torch.tensor(np.array(batch_feats), dtype=torch.float32).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(X_b)).cpu().numpy().flatten()

            sims = np.array(batch_sims)
            hybrids = sims * probs

            y_true.extend(batch_lbls)
            y_scores.extend(hybrids)

            batch_feats, batch_lbls, batch_sims = [], [], []

    return np.array(y_true), np.array(y_scores)


# ==========================================
# 4. TINYFACE: BEST THRESHOLD BULMA
# ==========================================
y_tiny_true, y_tiny_scores = get_hybrid_scores(TINY_PAIRS, TINY_IMG_DIR, "TinyFace")

# ROC Curve Hesapla
fpr, tpr, thresholds = roc_curve(y_tiny_true, y_tiny_scores)
roc_auc = auc(fpr, tpr)

# Youden's J Statistic (En iyi denge noktası)
# J = Sensitivity (TPR) + Specificity (1-FPR) - 1
# J'nin en büyük olduğu threshold "Best Threshold"dur.
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]

print(f"\n🏆 TINYFACE ANALİZİ")
print(f"   ROC AUC: {roc_auc:.4f}")
print(f"   Best Threshold (Youden's J): {best_thresh:.4f}")
print(f"   Bu noktada -> TPR: {tpr[ix]:.4f}, FPR: {fpr[ix]:.4f}")

# Grafiği Çiz
plt.figure()
plt.plot(fpr, tpr, label=f'TinyFace ROC (AUC = {roc_auc:.2f})')
plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label=f'Best Thresh={best_thresh:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('TinyFace ROC & Best Threshold')
plt.legend()
plt.savefig("tinyface_best_threshold_roc.png")
print("   Grafik kaydedildi: tinyface_best_threshold_roc.png")

# ==========================================
# 5. IJB-C: CROSS-DATASET TEST
# ==========================================
print(f"\n🚀 IJB-C TESTİ BAŞLIYOR (Threshold: {best_thresh:.4f})")
print("Hocanın isteği: TinyFace'te bulunan threshold ile IJB-C'yi test et.")

y_ijbc_true, y_ijbc_scores = get_hybrid_scores(IJBC_PAIRS, IJBC_IMG_DIR, "IJB-C")

# FRR Hesapla
# Threshold'dan küçük olan Pozitifler (False Reject)
pos_scores = y_ijbc_scores[y_ijbc_true == 1]
neg_scores = y_ijbc_scores[y_ijbc_true == 0]

fn = np.sum(pos_scores < best_thresh)
fp = np.sum(neg_scores >= best_thresh)
tp = np.sum(pos_scores >= best_thresh)
tn = np.sum(neg_scores < best_thresh)

frr = fn / (fn + tp)
far = fp / (fp + tn)

print(f"\n=== IJB-C CROSS-VALIDATION RESULT ===")
print(f"Threshold Used : {best_thresh:.4f} (From TinyFace)")
print(f"Accuracy       : {(tp + tn) / len(y_ijbc_true):.4f}")
print(f"FAR (Hata)     : {far:.6f}")
print(f"FRR (Başarı)   : {frr:.4f}")

if frr < 0.4718:
    print("✅ SOTA'yı (0.4718) geçiyoruz!")
else:
    print("⚠️ SOTA'nın gerisindeyiz, Optuna şart.")