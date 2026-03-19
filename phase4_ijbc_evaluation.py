import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from arcfaceutility import ensure_rgb, get_encoding_from_image

# ==========================================
# ⚙️ PARAMETRELER
# ==========================================
QUALITY_MODEL_NAME = 'buffalo_l'
THRESHOLD_NAME = '0.25'  # Hangi turdaysan ona göre güncelle (0.3, 0.4, 0.5, 0.6)
MATCHING_MODEL_NAME = 'antelopev2'

IJBC_BASE_DIR = "datasets/ijb-testsuite/ijb/IJBC"
IJBC_IMAGES_DIR = os.path.join(IJBC_BASE_DIR, "loose_crop")
IJBC_PAIRS_FILE = os.path.join(IJBC_BASE_DIR, "pairs.txt")

CLASSIFIER_MODEL_PATH = f"classifier_{QUALITY_MODEL_NAME}_thresh{THRESHOLD_NAME}.pth"
BAYES_MODEL_PATH = f"tinyface_bayes_model_{MATCHING_MODEL_NAME}.pkl"

# IJB-C Embedding Cache Dosyası
IJBC_CACHE_FILE = f"ijbc_embeddings_cache_{MATCHING_MODEL_NAME}.pkl"

ROC_OUTPUT_FILE = f"ijbc_roc_curves_Match{MATCHING_MODEL_NAME}_Qual{QUALITY_MODEL_NAME}_T{THRESHOLD_NAME}.png"
EDC_OUTPUT_FILE = f"ijbc_edc_curve_Match{MATCHING_MODEL_NAME}_Qual{QUALITY_MODEL_NAME}_T{THRESHOLD_NAME}.png"

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


# ==========================================
# 🛠️ YARDIMCI FONKSİYONLAR
# ==========================================
def parse_ijbc_pairs(pairs_path):
    pairs = []
    if not os.path.exists(pairs_path): return []
    with open(pairs_path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line: continue
        parts = line.split(',')
        if len(parts) >= 3:
            pairs.append((parts[0].strip(), parts[1].strip(), int(parts[2].strip())))
    return pairs


def calculate_edc_fnmr(y_true, match_scores, quality_scores, discard_fractions, target_fmr=1e-3):
    y_true, match_scores, quality_scores = np.array(y_true), np.array(match_scores), np.array(quality_scores)
    sorted_indices = np.argsort(quality_scores)
    fnmr_list = []
    for fraction in discard_fractions:
        discard_count = int(len(sorted_indices) * fraction)
        keep_indices = sorted_indices[discard_count:]
        if len(keep_indices) == 0:
            fnmr_list.append(1.0)
            continue
        current_y_true = y_true[keep_indices]
        current_scores = match_scores[keep_indices]
        if sum(current_y_true) == 0:
            fnmr_list.append(1.0)
            continue
        fpr, tpr, _ = roc_curve(current_y_true, current_scores)
        valid_idx = np.where(fpr <= target_fmr)[0]
        best_tpr = tpr[valid_idx[-1]] if len(valid_idx) > 0 else 0.0
        fnmr_list.append(1.0 - best_tpr)
    return fnmr_list


# ==========================================
# 🚀 ANA İŞLEM
# ==========================================
def main():
    print(f"--- 🚀 FAZ 4: IJB-C Final Evaluation (AKILLI MOD) ---")

    pairs = parse_ijbc_pairs(IJBC_PAIRS_FILE)
    if not pairs: return

    # 1. BENZERSİZ RESİMLERİ BUL
    unique_images = set()
    for img1, img2, _ in pairs:
        unique_images.add(img1)
        unique_images.add(img2)
    print(f"📸 Benzersiz Resim Sayısı: {len(unique_images)}")

    # 2. CACHE KONTROLÜ VE OLUŞTURMA
    embeddings_dict = {}
    if os.path.exists(IJBC_CACHE_FILE):
        print(f"📦 Disk Cache Bulundu! {IJBC_CACHE_FILE} yükleniyor...")
        with open(IJBC_CACHE_FILE, 'rb') as f:
            embeddings_dict = pickle.load(f)
    else:
        print("⏳ Ağır İşçilik: Cache bulunamadı. Benzersiz resimlerin özellikleri çıkarılıyor (~45 dk)...")
        ram_cache = {}
        for img_name in tqdm(list(unique_images), desc="Embedding Çıkarılıyor"):
            full_path = os.path.join(IJBC_IMAGES_DIR, img_name)
            img = cv2.imread(full_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                emb, _ = get_encoding_from_image(img, cache=ram_cache, cache_key=img_name)
                if emb is not None:
                    embeddings_dict[img_name] = emb

        with open(IJBC_CACHE_FILE, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print("✅ Ağır İşçilik Tamamlandı ve Cache Kaydedildi!")

    # 3. MODELLERİ YÜKLE
    print("📦 Değerlendirme Modelleri yükleniyor...")
    with open(BAYES_MODEL_PATH, 'rb') as f:
        bayes_model = pickle.load(f)
    classifier = RecognizabilityClassifier().to(DEVICE)
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE))
    classifier.eval()

    # --- HIZLANDIRMA ADIMI 1: EMBEDDING'LERİ NORMALIZE ET VE KALİTELERİ ÖNCEDEN HESAPLA ---
    print("⚡ 23.124 resmin kalitesi ve normları tek seferde hesaplanıyor...")
    quality_dict = {}

    for img_name, emb in tqdm(embeddings_dict.items(), desc="Kalite Skoru & Normalize"):
        norm_emb = emb / np.linalg.norm(emb)
        embeddings_dict[img_name] = norm_emb

        emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            quality_dict[img_name] = torch.sigmoid(classifier(emb_tensor)).item()

    # --- HIZLANDIRMA ADIMI 2: 15.6 MİLYON İŞLEMİ NUMPY İLE YAP ---
    y_true = np.zeros(len(pairs), dtype=np.int8)
    cos_sims = np.zeros(len(pairs), dtype=np.float32)
    quality_joint_scores = np.zeros(len(pairs), dtype=np.float32)

    print("⚡ 15.6 Milyon Eşleşme Matrislere Aktarılıyor...")
    valid_idx = 0
    for img1_name, img2_name, is_match in tqdm(pairs, desc="Pair Okuma"):
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

    # --- HIZLANDIRMA ADIMI 3: BAYES LOOKUP TABLE (LUT) ---
    print("📈 KDE (Dağılım) Numpy Lookup Table ile hesaplanıyor...")
    x_lut = np.linspace(-1.0, 1.0, 10000)
    f_gen_lut = bayes_model['kde_genuine'](x_lut)
    f_imp_lut = bayes_model['kde_imposter'](x_lut)

    f_gen = np.interp(cos_sims, x_lut, f_gen_lut)
    f_imp = np.interp(cos_sims, x_lut, f_imp_lut)

    p_g = bayes_model['p_genuine']
    p_i = bayes_model['p_imposter']

    scores_pm = (f_gen * p_g) / (f_gen * p_g + f_imp * p_i + 1e-10)

    print("🧠 Fusion Stratejileri Uygulanıyor...")
    scores_baseline = cos_sims
    scores_f1 = scores_pm * quality_joint_scores
    scores_f2 = scores_pm + (0.2 * quality_joint_scores)
    scores_f3 = scores_pm + (0.4 * quality_joint_scores)
    scores_f4 = scores_pm + (0.6 * quality_joint_scores)
    scores_f5 = scores_pm + (0.8 * quality_joint_scores)

    # ==========================================
    # 📉 GRAFİK ÇİZİMLERİ
    # ==========================================
    print("\n📉 EDC Eğrisi Çiziliyor...")
    discard_fractions = np.linspace(0.0, 0.50, 11)
    fnmr_values = calculate_edc_fnmr(y_true, scores_f4, quality_joint_scores, discard_fractions, target_fmr=1e-3)

    plt.figure(figsize=(10, 6))
    plt.plot(discard_fractions * 100, fnmr_values, marker='o', color='purple', linewidth=2, label='FNMR @ FMR=1e-3')
    plt.xlabel('Discarded Low-Quality Images (%)', fontsize=12)
    plt.ylabel('False Non-Match Rate (FNMR)', fontsize=12)
    plt.title(f'EDC Curve (Matcher: {MATCHING_MODEL_NAME}, Quality: {QUALITY_MODEL_NAME}, Thresh: {THRESHOLD_NAME})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(EDC_OUTPUT_FILE, dpi=300)

    print("📊 IJB-C ROC Eğrileri Çiziliyor...")
    plt.figure(figsize=(12, 8))

    def plot_roc(y_labels, y_scores, label, color, linestyle='-'):
        fpr, tpr, _ = roc_curve(y_labels, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, linestyle=linestyle, label=f'{label} (AUC = {roc_auc:.4f})')

    plot_roc(y_true, scores_baseline, 'Baseline (Cosine Sim)', 'black', '--')
    plot_roc(y_true, scores_pm, 'Bayes Only (Pm)', 'gray', '--')
    plot_roc(y_true, scores_f1, 'Fusion 1 (Pm * Pc)', 'red')
    plot_roc(y_true, scores_f2, 'Fusion 2 (Pm + 0.2*Pc)', 'blue')
    plot_roc(y_true, scores_f3, 'Fusion 3 (Pm + 0.4*Pc)', 'green')
    plot_roc(y_true, scores_f4, 'Fusion 4 (Pm + 0.6*Pc)', 'purple')
    plot_roc(y_true, scores_f5, 'Fusion 5 (Pm + 0.8*Pc)', 'orange')

    plt.xscale('log')
    plt.xlim([1e-3, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(f'IJB-C Final ROC Comparison (Threshold {THRESHOLD_NAME})', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(ROC_OUTPUT_FILE, dpi=300)

    print(f"🎉 Tüm işlemler bitti! Grafikler kaydedildi!")


if __name__ == "__main__":
    main()