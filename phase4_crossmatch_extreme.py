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
QUALITY_MODEL  = 'antelopev2'
MATCHER_MODEL  = 'antelopev2'
THRESHOLD_NAME = '0.25'

#MATCHER_MODEL = 'antelopev2'
#MATCHER_MODEL = 'r100_ms1mv3'
#MATCHER_MODEL = 'r34_glint360k'
#MATCHER_MODEL = 'r50_vgg2'
#MATCHER_MODEL = 'r100_ms1mv2'
#MATCHER_MODEL = 'r50_cisia'
#MATCHER_MODEL = 'r50_glintasia'
#MATCHER_MODEL = 'ms1m_megaface_r50'
#MATCHER_MODEL = 'ms1mv2_r50'
#MATCHER_MODEL = 'ms1mv3_r50'
#MATCHER_MODEL = 'r34_ms1mv3'
#MATCHER_MODEL = 'r18_ms1mv3'
#MATCHER_MODEL = 'buffalo_l'
#MATCHER_MODEL = 'r100_glint360k'

IJBC_BASE_DIR = "datasets/ijb-testsuite/ijb/IJBC"
IJBC_PAIRS_FILE = os.path.join(IJBC_BASE_DIR, "pairs.txt")

# Kalite (Quality) Dosyaları
CLASSIFIER_PATH = f"classifier_{QUALITY_MODEL}_thresh{THRESHOLD_NAME}.pth"
QUALITY_CACHE = f"ijbc_embeddings_cache_{QUALITY_MODEL}.pkl"

# Eşleşme (Matcher) Dosyaları
MATCHER_CACHE = f"ijbc_embeddings_cache_{MATCHER_MODEL}.pkl"
BAYES_PATH = f"tinyface_bayes_model_{MATCHER_MODEL}.pkl"

# Çıktı Dosyaları
OUTPUT_ROC_PLOT = f"ijbc_extreme_crossmatch_ROC_{MATCHER_MODEL}_with_{QUALITY_MODEL}.png"
OUTPUT_EDC_PLOT = f"ijbc_extreme_crossmatch_EDC_{MATCHER_MODEL}_with_{QUALITY_MODEL}.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 🧠 MİMARİ VE YARDIMCI FONKSİYONLAR
# ==========================================
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


def calculate_edc_fnmr(y_true, y_scores, quality_scores, discard_fractions, target_fpr=1e-5):
    """Belirli bir FPR noktasında, kalitesiz resimler atıldıkça oluşan FNMR (FRR) değerlerini hesaplar."""
    sorted_indices = np.argsort(quality_scores)  # Kalitesi en düşükten en yükseğe sırala
    fnmr_list = []

    for fraction in discard_fractions:
        discard_count = int(len(sorted_indices) * fraction)
        keep_indices = sorted_indices[discard_count:]  # Sadece iyileri tut

        if len(keep_indices) == 0:
            fnmr_list.append(1.0)
            continue

        current_y_true = y_true[keep_indices]
        current_scores = y_scores[keep_indices]

        if sum(current_y_true) == 0:
            fnmr_list.append(1.0)
            continue

        fnmr = get_frr_at_fpr(current_y_true, current_scores, target_fpr)
        fnmr_list.append(fnmr)

    return fnmr_list


# ==========================================
# 🚀 ANA İŞLEM
# ==========================================
def main():
    print(f"--- 🚀 GERÇEK CROSS-MATCHING EXTREME TESTİ ---")
    print(f"Hakem: {QUALITY_MODEL} | İşçi: {MATCHER_MODEL}")
    print(f"Cihaz: {DEVICE}")

    pairs = parse_ijbc_pairs(IJBC_PAIRS_FILE)

    # 1. HER İKİ CACHE'İ DE YÜKLE
    print("📦 Önbellekler (Cache) yükleniyor...")
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
    print("🧠 Hakem modeli resim kalitelerini puanlıyor...")
    quality_dict = {}
    for img_name, emb in quality_embeddings.items():
        emb_norm = emb / np.linalg.norm(emb)
        emb_tensor = torch.tensor(emb_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            quality_dict[img_name] = torch.sigmoid(classifier(emb_tensor)).item()

    # 3. İŞÇİ İLE EŞLEŞTİR
    print("👷 İşçi modeli eşleştirme yapıyor...")
    y_true, cos_sims, joint_qualities = [], [], []
    valid_img_pairs = []  # 🌟 EKLENEN SATIR: Resim isimlerini aklımızda tutmalıyız!

    for img1, img2, is_match in tqdm(pairs, desc="Çapraz Eşleşme"):
        emb1_m = matcher_embeddings.get(img1)
        emb2_m = matcher_embeddings.get(img2)

        if emb1_m is not None and emb2_m is not None and img1 in quality_dict and img2 in quality_dict:
            emb1_m_norm = emb1_m / np.linalg.norm(emb1_m)
            emb2_m_norm = emb2_m / np.linalg.norm(emb2_m)

            y_true.append(is_match)
            cos_sims.append(np.dot(emb1_m_norm, emb2_m_norm))
            min_quality = min(quality_dict[img1], quality_dict[img2])
            joint_qualities.append(min_quality)

            valid_img_pairs.append((img1, img2))  # 🌟 EKLENEN SATIR: İsimleri indekse paralel olarak kaydet

    y_true = np.array(y_true, dtype=np.int8)
    cos_sims = np.array(cos_sims, dtype=np.float32)
    joint_qualities = np.array(joint_qualities, dtype=np.float32)

    # 4. BAYES ÇEVİRİSİ (Dokunmuyoruz, aynen kalıyor!)
    print("⚡ Skorlar Bayes olasılığına çevriliyor...")
    x_lut = np.linspace(-1.0, 1.0, 10000)
    f_gen = np.interp(cos_sims, x_lut, bayes_model['kde_genuine'](x_lut))
    f_imp = np.interp(cos_sims, x_lut, bayes_model['kde_imposter'](x_lut))
    scores_pm = (f_gen * bayes_model['p_genuine']) / (f_gen * bayes_model['p_genuine'] + f_imp * bayes_model['p_imposter'] + 1e-10)

    # ==========================================
    # 🛡️ SÜREKLİ FÜZYON (Continuous Min-Quality Fusion)
    # ==========================================
    print("🛡️ Threshold Olmadan 'Min-Quality' Çarpımı Uygulanıyor...")

    # 1. Baseline her zamanki gibi ham kosinüs skoru
    scores_baseline = cos_sims

    # 2. joint_qualities dizisi yukarıda min(Pc1, Pc2) olarak hesaplanmıştı!
    # YENİ KURAL: Bayes olasılığı ile eşleşmedeki en zayıf resmin kalitesini çarpıyoruz.
    scores_proposed = scores_pm * joint_qualities

    # ==========================================
    # 📊 5. ÇIKTILAR VE GRAFİKLER
    # ==========================================

    # --- TABLO ---
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

    # --- ROC GRAFİĞİ ---
    print("📈 ROC Grafiği Çiziliyor...")
    plt.figure(figsize=(10, 6))
    fpr_base, tpr_base, _ = roc_curve(y_true, scores_baseline)
    fpr_prop, tpr_prop, _ = roc_curve(y_true, scores_proposed)

    plt.plot(fpr_base, tpr_base, color='black', linestyle='--', lw=2, label=f'Baseline ({MATCHER_MODEL})')
    plt.plot(fpr_prop, tpr_prop, color='red', lw=2, label=f'Proposed ($P_m \\times P_c$)')

    plt.xscale('log')
    plt.xlim([1e-6, 1e-2])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR) - Log Scale', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.plot(fpr_prop, tpr_prop, color='red', lw=2, label=r'Proposed ($P_m \times \min(P_{c1}, P_{c2})$)')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(OUTPUT_ROC_PLOT, dpi=300)

    # --- ÇOKLU FPR İLE EDC GRAFİĞİ ---
    print("📉 Çoklu FPR noktaları için EDC Grafiği Çiziliyor...")
    discard_fractions = np.linspace(0.0, 0.40, 9)  # %0'dan %40'a

    # Hangi FPR noktalarında EDC çizeceğiz ve renkleri ne olacak?
    target_fprs_for_edc = [1e-4, 1e-5, 1e-6]
    colors = ['#2ca02c', '#1f77b4', '#d62728']  # Yeşil (10^-4), Mavi (10^-5), Kırmızı (10^-6)

    plt.figure(figsize=(10, 8))  # Grafiği biraz daha uzun yaptık ki çizgiler rahat sığsın

    for fpr_val, color in zip(target_fprs_for_edc, colors):
        # Etiketi IEEE formatına uygun yazdır (Örn: 10^-5)
        fpr_label = f"$10^{{{int(np.log10(fpr_val))}}}$"

        # 1. Baseline EDC Hesapla
        edc_base = calculate_edc_fnmr(y_true, scores_baseline, joint_qualities, discard_fractions, target_fpr=fpr_val)
        plt.plot(discard_fractions * 100, edc_base, color=color, linestyle='--', marker='o', alpha=0.6,
                 label=f'Baseline FNMR @ FPR={fpr_label}')

        # 2. Proposed (Bizim) EDC Hesapla
        edc_prop = calculate_edc_fnmr(y_true, scores_proposed, joint_qualities, discard_fractions, target_fpr=fpr_val)
        plt.plot(discard_fractions * 100, edc_prop, color=color, linestyle='-', marker='s', linewidth=2.5,
                 label=f'Proposed FNMR @ FPR={fpr_label}')

    plt.xlabel('Discarded Low-Quality Images (%)', fontsize=12, fontweight='bold')
    plt.ylabel('False Non-Match Rate (FNMR)', fontsize=12, fontweight='bold')
    plt.title(f'Multi-FPR EDC Curves\n(Matcher: {MATCHER_MODEL} | Quality: {QUALITY_MODEL})', fontsize=14, fontweight='bold')

    plt.grid(True, linestyle=':', alpha=0.8)

    # Legend çok kalabalık olacağı için grafiğin dışına, sağ tarafa alıyoruz
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.9)
    plt.tight_layout()  # Sağdaki legend kesilmesin diye layoutu toparlar

    plt.savefig(OUTPUT_EDC_PLOT, dpi=300, bbox_inches='tight')
    print(f"✅ Çoklu EDC Grafiği Kaydedildi: {OUTPUT_EDC_PLOT}")
    # ==========================================
    # 🕵️ 6. SOMUT KANIT: HATALARI CIMBIZLA ÇEKME (DİNAMİK VERSİYON)
    # ==========================================
    print("\n" + "=" * 60)
    print("🕵️ SOMUT KANIT ANALİZİ BAŞLIYOR... (DİNAMİK YÖNTEM)")
    print("=" * 60)

    # 1. SIZAN SAHTELER (High-Quality Impostors)
    impostor_idx = np.where(y_true == 0)[0]
    impostors = []
    for idx in impostor_idx:
        impostors.append((valid_img_pairs[idx][0], valid_img_pairs[idx][1], cos_sims[idx], scores_pm[idx], joint_qualities[idx]))

    # Bayes olasılığına (Pm) göre BÜYÜKTEN KÜÇÜĞE sırala (Modelin 'kesin aynı kişi' diyerek en çok aldandığı sahteler)
    impostors.sort(key=lambda x: x[3], reverse=True)

    sizan_sahteler = []
    # Modelin en kötü (en yüksek skorlu) ilk 2000 sahtesine bak, kalitesi "kötü olmayanları (Pc > 0.40)" cımbızla
    for imp in impostors[:2000]:
        if imp[4] > 0.40:  # Hakemin "Bu resim fena değil" dedikleri
            sizan_sahteler.append(imp)
            if len(sizan_sahteler) >= 5: break

    print(f"\n🚨 Filtreden Sızan 'Net' Resimli SAHTE Eşleşmeler (Kapasite Sorunu):")
    if not sizan_sahteler:
        print("   Bulunamadı.")
    else:
        for i, (img1, img2, cos_val, pm_val, qual_val) in enumerate(sizan_sahteler):
            print(f"   {i + 1}. Resim 1: {img1:<25} | Resim 2: {img2:<25} | Kosinüs: {cos_val:.3f} | Pm: {pm_val:.4f} | Kalite: {qual_val:.3f}")

    # 2. KATLEDİLEN GERÇEKLER (Sacrificed Genuines)
    genuine_idx = np.where(y_true == 1)[0]
    genuines = []
    for idx in genuine_idx:
        genuines.append((valid_img_pairs[idx][0], valid_img_pairs[idx][1], cos_sims[idx], scores_pm[idx], joint_qualities[idx]))

    # Kaliteye göre KÜÇÜKTEN BÜYÜĞE sırala (Hakemin "Bu resimler çöp" diyerek dibe vurdukları)
    genuines.sort(key=lambda x: x[4])

    katledilen_gercekler = []
    # En kötü kaliteli ilk 1000 gerçeğe bak, İşçinin kosinüs skoru az çok iyi olanları (örn: cos > 0.20) cımbızla
    for gen in genuines[:1000]:
        if gen[2] > 0.20:  # İşçi az çok benzetmiş ama filtrenin acımasızlığına kurban gitmiş
            katledilen_gercekler.append(gen)
            if len(katledilen_gercekler) >= 5: break

    print(f"\n☠️ Filtrenin Çöpe Attığı GERÇEK Eşleşmeler (İkincil Hasar):")
    if not katledilen_gercekler:
        print("   Bulunamadı.")
    else:
        for i, (img1, img2, cos_val, pm_val, qual_val) in enumerate(katledilen_gercekler):
            print(f"   {i + 1}. Resim 1: {img1:<25} | Resim 2: {img2:<25} | Kosinüs: {cos_val:.3f} | Pm: {pm_val:.4f} | Kalite: {qual_val:.3f}")
    print("=" * 60 + "\n")

    # ==========================================
    # 📊 7. HOCAYA GÖNDERİLECEK KANIT GRAFİĞİ (SCATTER PLOT)
    # ==========================================
    print("📈 Hocaya sunulacak Dağılım Grafiği (Scatter Plot) çiziliyor...")
    plt.figure(figsize=(10, 8))

    # Çok fazla nokta olduğu için grafiğin şişmesini engellemek adına alt küme (subset) alıyoruz
    # Gerçeklerin hepsini, Sahtelerin ise rastgele 50,000 tanesini alalım
    np.random.seed(42)
    sample_impostors = np.random.choice(impostor_idx, min(50000, len(impostor_idx)), replace=False)

    # 1. Sahteleri (Kırmızı) Çiz
    plt.scatter(joint_qualities[sample_impostors], scores_pm[sample_impostors],
                color='red', alpha=0.1, s=10, label='Impostors (Sahteler)')

    # 2. Gerçekleri (Yeşil) Çiz
    plt.scatter(joint_qualities[genuine_idx], scores_pm[genuine_idx],
                color='green', alpha=0.3, s=15, label='Genuines (Gerçekler)')

    # Kritik Bölgeleri İşaretle (Hocanın dikkatini çekecek yerler)
    plt.axvline(x=0.40, color='blue', linestyle='--', linewidth=2, label='Kalite Filtresi Sınırı (Örnek)')
    plt.axhline(y=0.80, color='purple', linestyle='--', linewidth=2, label='Kabul Barajı (Örnek)')

    # Sızan Sahteler (Tehlike Bölgesi - Sağ Üst)
    plt.text(0.70, 0.90, "Sızan HD Sahteler\n(Kapasite Sorunu)", color='darkred',
             fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

    # Katledilen Gerçekler (İkincil Hasar - Sol Alt/Orta)
    plt.text(0.10, 0.30, "Çöpe Atılan Gerçekler\n(İkincil Hasar)", color='darkgreen',
             fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

    plt.xlabel('Joint Quality Score $min(P_{c1}, P_{c2})$', fontsize=12, fontweight='bold')
    plt.ylabel('Matcher Bayes Probability ($P_m$)', fontsize=12, fontweight='bold')
    plt.title(f'Failure Analysis: Quality vs Probability\nMatcher: {MATCHER_MODEL} | Quality: {QUALITY_MODEL}', fontsize=14, fontweight='bold')

    # Legend'ı düzenle
    leg = plt.legend(loc='upper left', markerscale=3)
    for lh in leg.legend_handles: lh.set_alpha(1)  # Legenddaki renkleri tam opak yap
    plt.grid(True, linestyle=':', alpha=0.7)
    scatter_output = f"ijbc_failure_scatter_{MATCHER_MODEL}.png"
    plt.savefig(scatter_output, dpi=300, bbox_inches='tight')
    print(f"✅ Kanıt Grafiği Kaydedildi: {scatter_output}")
if __name__ == "__main__":
    main()