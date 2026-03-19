import os
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# ==========================================
# ⚙️ ÇAPRAZ TEST PARAMETRELERİ (CROSS-MATCHING)
# ==========================================
QUALITY_MODEL = 'buffalo_l'  # 🧠 Hakem (Kaliteyi bu ölçecek)
THRESHOLD_NAME = '0.25'

MATCHER_MODEL = 'antelopev2'  # 👷 İşçi (Eşleştirmeyi bu yapacak)

IJBC_BASE_DIR = "datasets/ijb-testsuite/ijb/IJBC"
IJBC_PAIRS_FILE = os.path.join(IJBC_BASE_DIR, "pairs.txt")

# Kalite (Quality) Dosyaları
CLASSIFIER_PATH = f"classifier_{QUALITY_MODEL}_thresh{THRESHOLD_NAME}.pth"
QUALITY_CACHE = f"ijbc_embeddings_cache_{QUALITY_MODEL}.pkl"

# Eşleşme (Matcher) Dosyaları
MATCHER_CACHE = f"ijbc_embeddings_cache_{MATCHER_MODEL}.pkl"
BAYES_PATH = f"tinyface_bayes_model_{MATCHER_MODEL}.pkl"

OUTPUT_PLOT = f"ijbc_extreme_crossmatch_{MATCHER_MODEL}_with_{QUALITY_MODEL}.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RecognizabilityClassifier(nn.Module):
    def __init__(self, input_dim=512):
        super(RecognizabilityClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x): return self.net(x)


def parse_ijbc_pairs(pairs_path):
    pairs = []
    with open(pairs_path, 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 3: pairs.append((parts[0].strip(), parts[1].strip(), int(parts[2].strip())))
    return pairs


def get_frr_at_fpr(y_true, y_scores, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0: return 1.0
    return 1.0 - tpr[valid_idx[-1]]


def main():
    print(f"--- 🚀 GERÇEK CROSS-MATCHING EXTREME TESTİ ---")
    print(f"Hakem: {QUALITY_MODEL} | İşçi: {MATCHER_MODEL}")

    pairs = parse_ijbc_pairs(IJBC_PAIRS_FILE)

    # 1. HER İKİ CACHE'İ DE YÜKLE (Kritik nokta burası!)
    with open(QUALITY_CACHE, 'rb') as f:
        quality_embeddings = pickle.load(f)
    with open(MATCHER_CACHE, 'rb') as f:
        matcher_embeddings = pickle.load(f)
    with open(BAYES_PATH, 'rb') as f:
        bayes_model = pickle.load(f)

    classifier = RecognizabilityClassifier().to(DEVICE)
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    classifier.eval()

    # 2. HAKEM (buffalo_l) İLE KALİTELERİ ÖLÇ
    quality_dict = {}
    for img_name, emb in quality_embeddings.items():
        emb_norm = emb / np.linalg.norm(emb)
        emb_tensor = torch.tensor(emb_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            quality_dict[img_name] = torch.sigmoid(classifier(emb_tensor)).item()

    # 3. İŞÇİ (antelopev2) İLE EŞLEŞTİR
    y_true, cos_sims, joint_qualities = [], [], []
    for img1, img2, is_match in tqdm(pairs, desc="Çapraz Eşleşme"):
        emb1_m = matcher_embeddings.get(img1)
        emb2_m = matcher_embeddings.get(img2)

        if emb1_m is not None and emb2_m is not None and img1 in quality_dict and img2 in quality_dict:
            emb1_m_norm = emb1_m / np.linalg.norm(emb1_m)
            emb2_m_norm = emb2_m / np.linalg.norm(emb2_m)

            y_true.append(is_match)
            cos_sims.append(np.dot(emb1_m_norm, emb2_m_norm))
            joint_qualities.append(quality_dict[img1] * quality_dict[img2])

    y_true = np.array(y_true, dtype=np.int8)
    cos_sims = np.array(cos_sims, dtype=np.float32)
    joint_qualities = np.array(joint_qualities, dtype=np.float32)

    # 4. BAYES & FUSION (F1)
    x_lut = np.linspace(-1.0, 1.0, 10000)
    f_gen = np.interp(cos_sims, x_lut, bayes_model['kde_genuine'](x_lut))
    f_imp = np.interp(cos_sims, x_lut, bayes_model['kde_imposter'](x_lut))
    scores_pm = (f_gen * bayes_model['p_genuine']) / (f_gen * bayes_model['p_genuine'] + f_imp * bayes_model['p_imposter'] + 1e-10)

    scores_proposed = scores_pm * joint_qualities
    scores_baseline = cos_sims

    # 5. TABLO VE GRAFİK ÇIKTISI
    target_fprs = [1e-6, 1e-5, 1e-4, 1e-3]
    print("\n" + "=" * 60)
    print(f"📊 CROSS-MATCHING FRR TABLOSU (Matcher: {MATCHER_MODEL})")
    print("=" * 60)
    print(f"{'FPR':<10} | {'Baseline FRR':<15} | {'Proposed FRR (F1)':<15}")
    print("-" * 60)
    for fpr_val in target_fprs:
        base_frr = get_frr_at_fpr(y_true, scores_baseline, fpr_val)
        prop_frr = get_frr_at_fpr(y_true, scores_proposed, fpr_val)
        print(f"{fpr_val:<10.0e} | {base_frr:<15.4f} | {prop_frr:<15.4f}")
    print("=" * 60 + "\n")

    plt.figure(figsize=(10, 6))
    fpr_base, tpr_base, _ = roc_curve(y_true, scores_baseline)
    fpr_prop, tpr_prop, _ = roc_curve(y_true, scores_proposed)

    plt.plot(fpr_base, tpr_base, color='black', linestyle='--', lw=2, label=f'Baseline ({MATCHER_MODEL})')
    plt.plot(fpr_prop, tpr_prop, color='red', lw=2, label=f'Proposed ({QUALITY_MODEL})')

    plt.xscale('log')
    plt.xlim([1e-6, 1e-2])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR) - Log Scale', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(f'Cross-Matching ROC\n(Matcher: ResNet100 - {MATCHER_MODEL} | Quality: ResNet50 - {QUALITY_MODEL})', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"✅ Gerçek Çapraz Test Grafiği Kaydedildi: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()