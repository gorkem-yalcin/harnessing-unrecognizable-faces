import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
from arcfaceutility import ensure_rgb, get_encoding_from_image

# ==========================================
# 1. AYARLAR & MODEL MİMARİSİ
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 SISTEMATIK RAPORLAMA BAŞLIYOR (A/B TESTI) ON: {device}")

# Model Parametreleri ve Sınıfı
with open("best_hyperparams.pkl", "rb") as f:
    best_params = pickle.load(f)


class OptimizedClassifier(nn.Module):
    def __init__(self, params):
        super().__init__()
        layers = []
        input_dim = 2049
        current_dim = params['hidden_start']
        for i in range(params['n_layers']):
            layers.append(nn.Linear(input_dim, current_dim))
            layers.append(nn.BatchNorm1d(current_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(params['dropout']))
            input_dim = current_dim
            current_dim = max(current_dim // 2, 64)
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Modeli Yükle
model = OptimizedClassifier(best_params).to(device)
model.load_state_dict(torch.load("final_optimized_model.pth"))
model.eval()

# ==========================================
# 2. VERİ YÜKLEME (IJB-C)
# ==========================================
IJBC_ROOT = "datasets/ijb-testsuite/ijb/IJBC"
IJBC_PAIRS = os.path.join(IJBC_ROOT, "pairs.txt")
IJBC_IMG = os.path.join(IJBC_ROOT, "loose_crop")

df = pd.read_csv(IJBC_PAIRS)
unique_imgs = pd.unique(df[['img1', 'img2']].values.ravel('K'))

print(f"   Caching {len(unique_imgs)} IJB-C images...")
emb_cache = {}
for img_name in tqdm(unique_imgs):
    full_path = os.path.join(IJBC_IMG, img_name)
    emb, _ = get_encoding_from_image(ensure_rgb(full_path), "", None, None)
    if emb is not None:
        # ÖNEMLİ: Baseline Cosine Similarity için vektörleri Normalize ediyoruz.
        emb_norm = emb / np.linalg.norm(emb)
        emb_cache[img_name] = {'raw': emb, 'norm': emb_norm}

# ==========================================
# 3. SKOR HESAPLAMA (BASELINE vs HYBRID)
# ==========================================
labels = []
baseline_scores = []  # Saf Cosine Similarity (A Yöntemi)
hybrid_scores = []  # Bizim Yöntem (B Yöntemi)

batch_feats = []
batch_sims = []
batch_idxs = []

print(f"   Analiz ediliyor: Baseline vs Hybrid...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    data1 = emb_cache.get(row['img1'])
    data2 = emb_cache.get(row['img2'])

    if data1 is None or data2 is None: continue

    label = int(row['label'])
    labels.append(label)

    # --- YÖNTEM A: BASELINE (Normalized Cosine Similarity) ---
    cos_sim = np.dot(data1['norm'], data2['norm'])
    baseline_scores.append(cos_sim)

    # --- YÖNTEM B: HYBRID İÇİN HAZIRLIK ---
    raw1, raw2 = data1['raw'], data2['raw']
    diff = np.abs(raw1 - raw2)
    mult = raw1 * raw2
    raw_sim = np.dot(raw1, raw2)

    feat = np.concatenate([raw1, raw2, diff, mult, [raw_sim]])
    batch_feats.append(feat)
    batch_sims.append(raw_sim)
    batch_idxs.append(len(baseline_scores) - 1)  # Sırayı tutmak için

    # Batch işleme
    if len(batch_feats) >= 2048:
        X_b = torch.tensor(np.array(batch_feats), dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(X_b)).cpu().numpy().flatten()

        hybrids = np.array(batch_sims) * probs
        for i, h_score in zip(batch_idxs, hybrids):
            hybrid_scores.append(h_score)

        batch_feats, batch_sims, batch_idxs = [], [], []

# Kalan batch
if batch_feats:
    X_b = torch.tensor(np.array(batch_feats), dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(X_b)).cpu().numpy().flatten()
    hybrids = np.array(batch_sims) * probs
    hybrid_scores.extend(hybrids)

labels = np.array(labels)
baseline_scores = np.array(baseline_scores)
hybrid_scores = np.array(hybrid_scores)


# ==========================================
# 4. FAR/FRR TABLOSU OLUŞTURMA (Hocanın İstediği Format)
# ==========================================
def get_frr_at_fars(y_true, y_scores, target_fars):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    frrs = {}
    for target_far in target_fars:
        # Hedef FAR'a en yakın noktayı bul
        idx = np.argmin(np.abs(fpr - target_far))
        frrs[target_far] = 1 - tpr[idx]
    return frrs, fpr, tpr


target_fars = [1e-2, 1e-3, 1e-4, 1e-5]

base_frrs, base_fpr, base_tpr = get_frr_at_fars(labels, baseline_scores, target_fars)
hyb_frrs, hyb_fpr, hyb_tpr = get_frr_at_fars(labels, hybrid_scores, target_fars)

print("\n" + "=" * 50)
print("📊 HOCANIN İSTEDİĞİ SİSTEMATİK RAPOR TABLOSU (IJB-C Single-Image)")
print("=" * 50)
print(f"{'FAR Değeri':<15} | {'Baseline (Sadece ResNet50)':<30} | {'Bizim Yöntem (ResNet50+Hybrid)':<30} | {'İyileşme (Fark)'}")
print("-" * 100)
for far in target_fars:
    base_val = base_frrs[far]
    hyb_val = hyb_frrs[far]
    diff = base_val - hyb_val
    print(f"{far:<15} | {base_val:<30.4f} | {hyb_val:<30.4f} | %{(diff / base_val) * 100:+.2f} ({(diff):+.4f})")

print("\n🏆 ROC AUC Karşılaştırması:")
print(f"Baseline AUC : {auc(base_fpr, base_tpr):.4f}")
print(f"Hybrid AUC   : {auc(hyb_fpr, hyb_tpr):.4f}")

# ==========================================
# 5. ROC CURVE ÇİZİMİ (Yan Yana)
# ==========================================
plt.figure(figsize=(10, 8))
# Logaritmik scale kullanmak IJB-C için (1e-5 vs) daha profesyoneldir
plt.plot(base_fpr, base_tpr, color='darkorange', lw=2, label=f'Baseline ResNet-50 (AUC = {auc(base_fpr, base_tpr):.4f})')
plt.plot(hyb_fpr, hyb_tpr, color='blue', lw=2, label=f'Proposed Hybrid Method (AUC = {auc(hyb_fpr, hyb_tpr):.4f})')

plt.xlim([1e-6, 1.0])
plt.ylim([0.0, 1.05])
plt.xscale('log')  # Hocaların görmeye alışık olduğu Logaritmik FAR skalası
plt.xlabel('False Positive Rate (FAR) - Log Scale')
plt.ylabel('True Positive Rate (1 - FRR)')
plt.title('IJB-C Verification Performance: Baseline vs Proposed Method')
plt.legend(loc="lower right")
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.savefig("systematic_report_roc.png", dpi=300, bbox_inches='tight')
print("✅ Hocanın istediği ROC grafiği kaydedildi: systematic_report_roc.png")