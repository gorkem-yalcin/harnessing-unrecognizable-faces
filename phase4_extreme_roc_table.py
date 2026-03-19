import os
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ==========================================
# ⚙️ PARAMETRELER (ŞAMPİYON MODEL)
# ==========================================
MODEL_NAME = 'buffalo_l'
THRESHOLD_NAME = '0.25'  # Şampiyon barajımız

IJBC_BASE_DIR = "datasets/ijb-testsuite/ijb/IJBC"
IJBC_PAIRS_FILE = os.path.join(IJBC_BASE_DIR, "pairs.txt")
CLASSIFIER_MODEL_PATH = f"classifier_{MODEL_NAME}_thresh{THRESHOLD_NAME}.pth"
BAYES_MODEL_PATH = f"tinyface_bayes_model_{MODEL_NAME}.pkl"
IJBC_CACHE_FILE = f"ijbc_embeddings_cache_{MODEL_NAME}.pkl"

EXTREME_ROC_FILE = f"ijbc_extreme_roc_{MODEL_NAME}_T{THRESHOLD_NAME}.png"
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


def get_frr_at_fpr(y_true, y_scores, target_fpr):
    """Belirli bir FPR noktasında FRR (1 - TPR) değerini bulur."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0:
        return 1.0  # Eğer o kadar düşük FPR'a inemiyorsa hata %100'dür
    best_tpr = tpr[valid_idx[-1]]
    return 1.0 - best_tpr  # FRR = 1 - TPR


# ==========================================
# 🚀 ANA İŞLEM
# ==========================================
def main():
    print(f"--- 🚀 HOCANIN İSTEDİĞİ EXTREME FPR TABLOSU (Thresh: {THRESHOLD_NAME}) ---")

    pairs = parse_ijbc_pairs(IJBC_PAIRS_FILE)
    if not pairs: return

    # 1. CACHE VE MODELLERİ YÜKLE
    print(f"📦 Veriler yükleniyor...")
    with open(IJBC_CACHE_FILE, 'rb') as f:
        embeddings_dict = pickle.load(f)
    with open(BAYES_MODEL_PATH, 'rb') as f:
        bayes_model = pickle.load(f)

    classifier = RecognizabilityClassifier().to(DEVICE)
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE))
    classifier.eval()

    # 2. SKORLARI HESAPLA
    print("⚡ 15.6 Milyon Eşleşme Matrislere Aktarılıyor...")
    y_true = np.zeros(len(pairs), dtype=np.int8)
    cos_sims = np.zeros(len(pairs), dtype=np.float32)
    quality_joint_scores = np.zeros(len(pairs), dtype=np.float32)

    # Kaliteleri bir kere hesapla
    quality_dict = {}
    for img_name, emb in embeddings_dict.items():
        emb_norm = emb / np.linalg.norm(emb)
        embeddings_dict[img_name] = emb_norm
        emb_tensor = torch.tensor(emb_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            quality_dict[img_name] = torch.sigmoid(classifier(emb_tensor)).item()

    valid_idx = 0
    for img1_name, img2_name, is_match in tqdm(pairs, desc="Skorlar Çıkarılıyor"):
        emb1 = embeddings_dict.get(img1_name)
        emb2 = embeddings_dict.get(img2_name)
        if emb1 is not None and emb2 is not None:
            y_true[valid_idx] = is_match
            cos_sims[valid_idx] = np.dot(emb1, emb2)
            quality_joint_scores[valid_idx] = quality_dict[img1_name] * quality_dict[img2_name]
            valid_idx += 1

    y_true = y_true[:valid_idx]
    cos_sims = cos_sims[:valid_idx]
    quality_joint_scores = quality_joint_scores[:valid_idx]

    # Bayes (Pm)
    x_lut = np.linspace(-1.0, 1.0, 10000)
    f_gen = np.interp(cos_sims, x_lut, bayes_model['kde_genuine'](x_lut))
    f_imp = np.interp(cos_sims, x_lut, bayes_model['kde_imposter'](x_lut))
    p_g, p_i = bayes_model['p_genuine'], bayes_model['p_imposter']
    scores_pm = (f_gen * p_g) / (f_gen * p_g + f_imp * p_i + 1e-10)

    # Şampiyon Formül: F1 (Pm * Pc)
    scores_proposed = scores_pm * quality_joint_scores
    scores_baseline = cos_sims

    # 3. HOCANIN İSTEDİĞİ FRR TABLOSUNU OLUŞTUR
    target_fprs = [1e-6, 1e-5, 1e-4, 1e-3]

    print("\n" + "=" * 50)
    print(f"📊 FRR KARŞILAŞTIRMA TABLOSU (Matcher: {MODEL_NAME})")
    print("=" * 50)
    print(f"{'FPR':<10} | {'Baseline FRR':<15} | {'Proposed FRR (F1)':<15}")
    print("-" * 50)

    for fpr_val in target_fprs:
        base_frr = get_frr_at_fpr(y_true, scores_baseline, fpr_val)
        prop_frr = get_frr_at_fpr(y_true, scores_proposed, fpr_val)
        # FRR değerleri genelde % (yüzde) olarak veya ondalık olarak verilir. Tabloya ondalık basıyoruz.
        print(f"{fpr_val:<10.0e} | {base_frr:<15.4f} | {prop_frr:<15.4f}")
    print("=" * 50 + "\n")

    # 4. EXTREME ROC GRAFİĞİNİ ÇİZ ($10^{-6}$'dan başlatıyoruz)
    print("📉 Extreme ROC Grafiği Çiziliyor...")
    plt.figure(figsize=(10, 6))

    fpr_base, tpr_base, _ = roc_curve(y_true, scores_baseline)
    fpr_prop, tpr_prop, _ = roc_curve(y_true, scores_proposed)

    plt.plot(fpr_base, tpr_base, color='black', linestyle='--', lw=2, label=f'Baseline (Cosine Sim)')
    plt.plot(fpr_prop, tpr_prop, color='red', lw=2, label=f'Proposed (F1: Pm * Pc)')

    plt.xscale('log')
    # İşte hocanın görmek istediği asıl arena!
    plt.xlim([1e-6, 1e-2])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR) - Log Scale', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(f'IJB-C ROC Comparison (Threshold {THRESHOLD_NAME})', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # 1e-6, 1e-5 noktalarını dikey çizgilerle (grid) vurgula
    plt.axvline(x=1e-6, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=1e-5, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=1e-4, color='gray', linestyle=':', alpha=0.5)

    plt.savefig(EXTREME_ROC_FILE, dpi=300)
    print(f"✅ Grafik Kaydedildi: {EXTREME_ROC_FILE}")


if __name__ == "__main__":
    main()