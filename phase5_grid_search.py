import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# ==========================================
# ⚙️ PARAMETRELER
# ==========================================
MODEL_NAME = 'buffalo_l'
THRESHOLDS = ['0.2', '0.25', '0.3', '0.4', '0.5', '0.6']
FUSION_NAMES = ['F1 (Pm * Pc)', 'F2 (Pm + 0.2Pc)', 'F3 (Pm + 0.4Pc)', 'F4 (Pm + 0.6Pc)', 'F5 (Pm + 0.8Pc)']

IJBC_BASE_DIR = "datasets/ijb-testsuite/ijb/IJBC"
IJBC_PAIRS_FILE = os.path.join(IJBC_BASE_DIR, "pairs.txt")
BAYES_MODEL_PATH = f"tinyface_bayes_model_{MODEL_NAME}.pkl"
IJBC_CACHE_FILE = f"ijbc_embeddings_cache_{MODEL_NAME}.pkl"

RESULTS_CSV = f"ijbc_grid_search_results_{MODEL_NAME}.csv"
HEATMAP_FILE = f"ijbc_grid_search_heatmap_{MODEL_NAME}.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 🧠 SINIFLANDIRICI MİMARİSİ
# ==========================================
class RecognizabilityClassifier(nn.Module):
    def __init__(self, input_dim=512):
        super(RecognizabilityClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


def parse_ijbc_pairs(pairs_path):
    pairs = []
    if not os.path.exists(pairs_path): return []
    with open(pairs_path, 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            pairs.append((parts[0].strip(), parts[1].strip(), int(parts[2].strip())))
    return pairs


def get_tpr_at_fpr(y_true, y_scores, target_fpr=1e-3):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    valid_idx = np.where(fpr <= target_fpr)[0]
    return tpr[valid_idx[-1]] if len(valid_idx) > 0 else 0.0


# ==========================================
# 🚀 ANA İŞLEM (25 KOMBINASYON)
# ==========================================
def main():
    print("--- 🚀 FAZ 5: 25 Kombinasyonluk Dev Grid Search ---")

    pairs = parse_ijbc_pairs(IJBC_PAIRS_FILE)
    if not pairs: return

    # 1. CACHE YÜKLE
    print(f"📦 Disk Cache Yükleniyor...")
    with open(IJBC_CACHE_FILE, 'rb') as f:
        embeddings_dict = pickle.load(f)

    with open(BAYES_MODEL_PATH, 'rb') as f:
        bayes_model = pickle.load(f)

    # 2. SABİT DEĞERLERİ HESAPLA (Kosinüs ve Bayes sadece 1 kere hesaplanır)
    print("⚡ Kosinüs ve Bayes (Pm) Skorları Hesaplanıyor...")
    y_true = np.zeros(len(pairs), dtype=np.int8)
    cos_sims = np.zeros(len(pairs), dtype=np.float32)

    valid_idx = 0
    for img1_name, img2_name, is_match in tqdm(pairs, desc="Pair Okuma"):
        emb1 = embeddings_dict.get(img1_name)
        emb2 = embeddings_dict.get(img2_name)
        if emb1 is not None and emb2 is not None:
            # Hızlı olması için baştan normalize edilmiş varsayıyoruz
            emb1_norm = emb1 / np.linalg.norm(emb1)
            emb2_norm = emb2 / np.linalg.norm(emb2)
            y_true[valid_idx] = is_match
            cos_sims[valid_idx] = np.dot(emb1_norm, emb2_norm)
            valid_idx += 1

    y_true = y_true[:valid_idx]
    cos_sims = cos_sims[:valid_idx]

    # Bayes LUT
    x_lut = np.linspace(-1.0, 1.0, 10000)
    f_gen = np.interp(cos_sims, x_lut, bayes_model['kde_genuine'](x_lut))
    f_imp = np.interp(cos_sims, x_lut, bayes_model['kde_imposter'](x_lut))
    p_g, p_i = bayes_model['p_genuine'], bayes_model['p_imposter']
    scores_pm = (f_gen * p_g) / (f_gen * p_g + f_imp * p_i + 1e-10)

    # Baseline Başarısını Not Al
    baseline_tpr = get_tpr_at_fpr(y_true, cos_sims)
    print(f"🎯 Baseline (Sadece Kosinüs) TPR @ FPR=1e-3: {baseline_tpr:.4f}\n")

    # 3. 25 KOMBINASYON İÇİN MATRİS OLUŞTUR
    results_matrix = np.zeros((len(THRESHOLDS), len(FUSION_NAMES)))

    for i, thresh in enumerate(THRESHOLDS):
        print(f"🔍 Sınıflandırıcı Yükleniyor: Threshold {thresh}")
        model_path = f"classifier_{MODEL_NAME}_thresh{thresh}.pth"

        if not os.path.exists(model_path):
            print(f"⚠️ Uyarı: {model_path} bulunamadı, bu satır atlanıyor.")
            continue

        classifier = RecognizabilityClassifier().to(DEVICE)
        classifier.load_state_dict(torch.load(model_path, map_location=DEVICE))
        classifier.eval()

        # Bu baraj için Kalite (Pc) Skorlarını Hesapla
        quality_dict = {}
        for img_name, emb in embeddings_dict.items():
            emb_norm = emb / np.linalg.norm(emb)
            emb_tensor = torch.tensor(emb_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                quality_dict[img_name] = torch.sigmoid(classifier(emb_tensor)).item()

        pc_scores = np.zeros(valid_idx, dtype=np.float32)
        idx = 0
        for img1_name, img2_name, _ in pairs:
            if img1_name in quality_dict and img2_name in quality_dict:
                pc_scores[idx] = quality_dict[img1_name] * quality_dict[img2_name]
                idx += 1

        # 5 Fusion Stratejisini Uygula
        fusion_scores = [
            scores_pm * pc_scores,  # F1
            scores_pm + (0.2 * pc_scores),  # F2
            scores_pm + (0.4 * pc_scores),  # F3
            scores_pm + (0.6 * pc_scores),  # F4
            scores_pm + (0.8 * pc_scores)  # F5
        ]

        # Her biri için TPR hesapla ve matrise yaz
        for j, f_score in enumerate(fusion_scores):
            tpr = get_tpr_at_fpr(y_true, f_score)
            results_matrix[i, j] = tpr

    # ==========================================
    # 📊 SONUÇLARI GÖRSELLEŞTİR VE KAYDET
    # ==========================================
    print("\n🏆 GRID SEARCH SONUÇLARI (TPR @ FPR=1e-3) 🏆")
    print(f"Baseline TPR: {baseline_tpr:.4f}\n")

    # Konsola Tablo Yazdır
    header = "Thresh | " + " | ".join(FUSION_NAMES)
    print(header)
    print("-" * len(header))
    for i, thresh in enumerate(THRESHOLDS):
        row_str = f"  {thresh}  | " + " | ".join([f"{val:.4f}" for val in results_matrix[i]])
        print(row_str)

    # Heatmap (Isı Haritası) Çiz
    plt.figure(figsize=(10, 6))
    sns.heatmap(results_matrix, annot=True, fmt=".4f", cmap="YlGnBu",
                xticklabels=FUSION_NAMES, yticklabels=THRESHOLDS,
                cbar_kws={'label': 'TPR @ FPR=1e-3'})
    plt.title(f'Grid Search: TPR at FPR=1e-3\n(Baseline: {baseline_tpr:.4f})', fontsize=14)
    plt.xlabel('Fusion Strategy', fontsize=12)
    plt.ylabel('Classifier Threshold', fontsize=12)
    plt.tight_layout()
    plt.savefig(HEATMAP_FILE, dpi=300)
    print(f"\n✅ Isı Haritası Kaydedildi: {HEATMAP_FILE}")


if __name__ == "__main__":
    main()
