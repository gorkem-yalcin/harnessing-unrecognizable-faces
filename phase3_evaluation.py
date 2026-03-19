import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Mevcut scriptlerinden importlar
from arcfaceutility import ensure_rgb, get_encoding_from_image

# ==========================================
# ⚙️ PARAMETRELER
# ==========================================
MODEL_NAME = 'buffalo_l'
THRESHOLD_NAME = '0.25' # Eğittiğin modele göre değiştir

TINYFACE_IMAGES_DIR = "datasets/tinyface/images"
TINYFACE_PAIRS_FILE = "datasets/tinyface/pairs.txt"

CLASSIFIER_MODEL_PATH = f"classifier_{MODEL_NAME}_thresh{THRESHOLD_NAME}.pth"
BAYES_MODEL_PATH = f"tinyface_bayes_model_{MODEL_NAME}.pkl"

# 🚨 İŞTE BURAYI GÜNCELLEDİK (Artık overwrite yapmayacak)
ROC_OUTPUT_FILE = f"fusion_roc_curves_{MODEL_NAME}_thresh{THRESHOLD_NAME}.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = {}


# ==========================================
# 🧠 1. SINIFLANDIRICI MİMARİSİ (Yeniden Tanımlama)
# ==========================================
class RecognizabilityClassifier(nn.Module):
    def __init__(self, input_dim=512):
        super(RecognizabilityClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 🛠️ YARDIMCI FONKSİYONLAR
# ==========================================
def parse_tinyface_pairs(pairs_path):
    pairs = []
    with open(pairs_path, 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            pairs.append((os.path.join(parts[0], parts[1]), os.path.join(parts[0], parts[2]), 1))
        elif len(parts) == 4:
            pairs.append((os.path.join(parts[0], parts[1]), os.path.join(parts[2], parts[3]), 0))
    return pairs


def get_image_embedding(img_path):
    full_path = os.path.join(TINYFACE_IMAGES_DIR, img_path)
    img = cv2.imread(full_path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cache_key = f"{MODEL_NAME}_{img_path}"
    emb, _ = get_encoding_from_image(img, cache=CACHE, cache_key=cache_key)
    return emb


def calculate_bayes_prob(cos_sim, bayes_model):
    """Kosinüs skorunu Bayes ile Eşleşme Olasılığına (Pm) çevirir."""
    p_g = bayes_model['p_genuine']
    p_i = bayes_model['p_imposter']

    # KDE fonksiyonları array bekler, [cos_sim] şeklinde veriyoruz
    f_gen = bayes_model['kde_genuine']([cos_sim])[0]
    f_imp = bayes_model['kde_imposter']([cos_sim])[0]

    # Sıfıra bölünme hatasını önlemek için eps ekliyoruz
    eps = 1e-10
    prob = (f_gen * p_g) / ((f_gen * p_g) + (f_imp * p_i) + eps)
    return float(prob)


def get_quality_prob(embedding, model):
    """Embedding'i Sınıflandırıcıya sokar ve Kalite Olasılığını (Pc) döndürür."""
    # Numpy array -> PyTorch Tensor
    emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(emb_tensor)
        prob = torch.sigmoid(logits).item()  # Sigmoid ile 0-1 arasına çekiyoruz
    return prob


# ==========================================
# 🚀 ANA İŞLEM (EVALUATION & FUSION)
# ==========================================
def main():
    print(f"--- 🚀 FAZ 3: Fusion & Evaluation Başlıyor ---")

    # 1. Modelleri Yükle
    print("📦 Modeller yükleniyor...")
    with open(BAYES_MODEL_PATH, 'rb') as f:
        bayes_model = pickle.load(f)

    classifier = RecognizabilityClassifier().to(DEVICE)
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE))
    classifier.eval()
    print("✅ Modeller başarıyla yüklendi.")

    pairs = parse_tinyface_pairs(TINYFACE_PAIRS_FILE)

    y_true = []
    scores_baseline = []  # Sadece Cosine Similarity
    scores_pm = []  # Sadece Bayes (Pm)

    # Fusion Skorları Listeleri (Hocanın 5 Formülü)
    scores_f1 = []  # Pm * Pc
    scores_f2 = []  # Pm + (0.2 * Pc)
    scores_f3 = []  # Pm + (0.4 * Pc)
    scores_f4 = []  # Pm + (0.6 * Pc)
    scores_f5 = []  # Pm + (0.8 * Pc)

    print("⚡ Eşleşmeler değerlendiriliyor...")
    for img1_path, img2_path, is_match in tqdm(pairs):
        emb1 = get_image_embedding(img1_path)
        emb2 = get_image_embedding(img2_path)

        if emb1 is not None and emb2 is not None:
            # 1. Kosinüs Benzerliği (Baseline)
            norm1 = emb1 / np.linalg.norm(emb1)
            norm2 = emb2 / np.linalg.norm(emb2)
            cos_sim = np.dot(norm1, norm2)

            # 2. Bayes Olasılığı (Pm)
            p_m = calculate_bayes_prob(cos_sim, bayes_model)

            # 3. Kalite Olasılığı (Pc)
            q1 = get_quality_prob(emb1, classifier)
            q2 = get_quality_prob(emb2, classifier)
            p_c = q1 * q2  # Ortak kalite olasılığı

            # 4. Listelere Ekleme
            y_true.append(is_match)
            scores_baseline.append(cos_sim)
            scores_pm.append(p_m)

            # 5 FUSION STRATEJİSİ
            scores_f1.append(p_m * p_c)
            scores_f2.append(p_m + (0.2 * p_c))
            scores_f3.append(p_m + (0.4 * p_c))
            scores_f4.append(p_m + (0.6 * p_c))
            scores_f5.append(p_m + (0.8 * p_c))

    # ==========================================
    # 📈 ROC EĞRİLERİ ÇİZİMİ
    # ==========================================
    print("\n📊 ROC Eğrileri Çiziliyor...")
    plt.figure(figsize=(12, 8))

    # Çizim Fonksiyonu
    def plot_roc(y_labels, y_scores, label, color, linestyle='-'):
        fpr, tpr, _ = roc_curve(y_labels, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, linestyle=linestyle,
                 label=f'{label} (AUC = {roc_auc:.4f})')

    # Baseline'lar
    plot_roc(y_true, scores_baseline, 'Baseline (Cosine Sim)', 'black', '--')
    plot_roc(y_true, scores_pm, 'Bayes Only (Pm)', 'gray', '--')

    # 5 Fusion Stratejisi
    plot_roc(y_true, scores_f1, 'Fusion 1 (Pm * Pc)', 'red')
    plot_roc(y_true, scores_f2, 'Fusion 2 (Pm + 0.2*Pc)', 'blue')
    plot_roc(y_true, scores_f3, 'Fusion 3 (Pm + 0.4*Pc)', 'green')
    plot_roc(y_true, scores_f4, 'Fusion 4 (Pm + 0.6*Pc)', 'purple')
    plot_roc(y_true, scores_f5, 'Fusion 5 (Pm + 0.8*Pc)', 'orange')

    # IJB-C / TinyFace standartları için X eksenini Logaritmik yapıyoruz (Zorlu FAR bölgeleri)
    plt.xscale('log')
    plt.xlim([1e-3, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Fusion Strategies ROC Comparison (Log Scale)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.savefig(ROC_OUTPUT_FILE, dpi=300)
    print(f"🎉 Tüm işlemler bitti! Grafik kaydedildi: {ROC_OUTPUT_FILE}")


if __name__ == "__main__":
    main()