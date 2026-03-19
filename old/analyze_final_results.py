import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from arcfaceutility import ensure_rgb, get_encoding_from_image

# ==========================================
# 1. AYARLAR & MODEL MİMARİSİ
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 FINAL ANALYSIS STARTED ON: {device}")

# Optuna Parametrelerini Yükle (Mimariyi kurmak için şart)
if os.path.exists("best_hyperparams.pkl"):
    with open("best_hyperparams.pkl", "rb") as f:
        best_params = pickle.load(f)
    print(f"🏆 Mimari Parametreleri: {best_params}")
else:
    raise FileNotFoundError("❌ best_hyperparams.pkl bulunamadı! Model mimarisi bilinemiyor.")


# --- Model Sınıfı (Eğitimdekiyle AYNI olmalı) ---
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
            current_dim = current_dim // 2
            if current_dim < 64: current_dim = 64

        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Modeli Başlat ve Ağırlıkları Yükle
model = OptimizedClassifier(best_params).to(device)
model_path = "final_optimized_model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("✅ Final Model Başarıyla Yüklendi.")
else:
    raise FileNotFoundError(f"❌ {model_path} bulunamadı!")


# ==========================================
# 2. YARDIMCI FONKSİYONLAR
# ==========================================
def get_hybrid_scores(model, pairs_path, img_dir, dataset_name):
    print(f"\n--- Processing {dataset_name} ---")

    pairs_list = []

    # --- TINYFACE PARSING (Hatayı düzelten kısım) ---
    if dataset_name == "TinyFace":
        with open(pairs_path, 'r') as f:
            lines = f.readlines()[1:]  # Header'ı atla

        for line in lines:
            p = line.strip().split('\t')
            # LFW Formatı:
            # 3 kolon: name, img1, img2 -> (Aynı Kişi - Match)
            # 4 kolon: name1, img1, name2, img2 -> (Farklı Kişi - Mismatch)

            if len(p) == 3:
                # Match: folder/img1, folder/img2
                # TinyFace yapısında genelde: folder_name, img1_name, img2_name
                path1 = os.path.join(img_dir, p[0], p[1])
                path2 = os.path.join(img_dir, p[0], p[2])
                pairs_list.append({'p1': path1, 'p2': path2, 'label': 1})

            elif len(p) == 4:
                # Mismatch: folder1, img1, folder2, img2
                path1 = os.path.join(img_dir, p[0], p[1])
                path2 = os.path.join(img_dir, p[2], p[3])
                pairs_list.append({'p1': path1, 'p2': path2, 'label': 0})

    # --- IJB-C PARSING ---
    else:
        # IJB-C pairs.txt formatı (img1, img2, label)
        df = pd.read_csv(pairs_path)
        # Unique resimleri önceden cacheleyelim (Hız için kritik)
        unique_imgs = pd.unique(df[['img1', 'img2']].values.ravel('K'))
        emb_cache = {}
        print(f"   Caching {len(unique_imgs)} IJB-C images...")
        for img_name in tqdm(unique_imgs):
            full_path = os.path.join(img_dir, img_name)
            emb, _ = get_encoding_from_image(ensure_rgb(full_path), "", None, None)
            if emb is not None: emb_cache[img_name] = emb

        # Listeye çevir
        for idx, row in df.iterrows():
            pairs_list.append({
                'p1': row['img1'],
                'p2': row['img2'],
                'label': int(row['label']),
                'use_cache': True,
                'cache': emb_cache
            })

    # --- SKOR HESAPLAMA DÖNGÜSÜ ---
    scores, labels = [], []
    batch_feats, batch_lbls, batch_sims = [], [], []

    print(f"   Calculating scores for {len(pairs_list)} pairs...")

    for item in tqdm(pairs_list):
        if item.get('use_cache'):
            enc1 = item['cache'].get(item['p1'])
            enc2 = item['cache'].get(item['p2'])
        else:
            enc1, _ = get_encoding_from_image(ensure_rgb(item['p1']), "", None, None)
            enc2, _ = get_encoding_from_image(ensure_rgb(item['p2']), "", None, None)

        if enc1 is None or enc2 is None: continue

        diff = np.abs(enc1 - enc2)
        mult = enc1 * enc2
        sim = np.dot(enc1, enc2)
        feat = np.concatenate([enc1, enc2, diff, mult, [sim]])

        batch_feats.append(feat)
        batch_lbls.append(item['label'])
        batch_sims.append(sim)

        if len(batch_feats) >= 2048:
            X_b = torch.tensor(np.array(batch_feats), dtype=torch.float32).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(X_b)).cpu().numpy().flatten()

            hybrids = np.array(batch_sims) * probs
            scores.extend(hybrids)
            labels.extend(batch_lbls)
            batch_feats, batch_lbls, batch_sims = [], [], []

    # Kalan son batch
    if batch_feats:
        X_b = torch.tensor(np.array(batch_feats), dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(X_b)).cpu().numpy().flatten()
        hybrids = np.array(batch_sims) * probs
        scores.extend(hybrids)
        labels.extend(batch_lbls)

    return np.array(labels), np.array(scores)


# ==========================================
# 3. ÇALIŞTIRMA (PATH AYARLARI)
# ==========================================
DATASET_ROOT = "datasets"
TINY_PAIRS = os.path.join(DATASET_ROOT, "tinyface", "pairs.txt")
TINY_IMG = os.path.join(DATASET_ROOT, "tinyface", "images")

IJBC_ROOT = "datasets/ijb-testsuite/ijb/IJBC"
IJBC_PAIRS = os.path.join(IJBC_ROOT, "pairs.txt")
IJBC_IMG = os.path.join(IJBC_ROOT, "loose_crop")

try:
    # A. TINYFACE ANALİZİ
    y_tiny_true, y_tiny_scores = get_hybrid_scores(model, TINY_PAIRS, TINY_IMG, "TinyFace")

    fpr, tpr, thresholds = roc_curve(y_tiny_true, y_tiny_scores)
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_thresh = thresholds[best_idx]

    print(f"\n🏆 TINYFACE RESULT")
    print(f"   Best Threshold: {best_thresh:.4f}")
    print(f"   ROC AUC: {auc(fpr, tpr):.4f}")

    # B. IJB-C TESTİ (KÖR TEST)
    y_ijbc_true, y_ijbc_scores = get_hybrid_scores(model, IJBC_PAIRS, IJBC_IMG, "IJB-C")

    # FRR Hesapla
    pos_scores = y_ijbc_scores[y_ijbc_true == 1]
    neg_scores = y_ijbc_scores[y_ijbc_true == 0]

    fn = np.sum(pos_scores < best_thresh)
    fp = np.sum(neg_scores >= best_thresh)
    tp = np.sum(pos_scores >= best_thresh)
    tn = np.sum(neg_scores < best_thresh)

    frr = fn / (fn + tp)
    far = fp / (fp + tn)

    print(f"\n██████ FINAL SOTA REPORT ██████")
    print(f"Model Architecture : {best_params}")
    print(f"Threshold Used     : {best_thresh:.4f} (Derived from TinyFace)")
    print(f"IJB-C Actual FAR   : {far:.6f} (Target: ~0.01)")
    print(f"IJB-C FRR (Error)  : {frr:.4f}")
    print(f"Baseline (Previous): 0.2508")

    if frr < 0.2508:
        print(f"🚀 GELİŞTİRME BAŞARILI! Fark: -{0.2508 - frr:.4f}")
    else:
        print(f"⚠️ Gelişme yok (Overfitting olabilir).")

except Exception as e:
    print(f"❌ HATA OLUŞTU: {e}")
    import traceback

    traceback.print_exc()