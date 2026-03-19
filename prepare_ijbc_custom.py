import os
import pandas as pd
from tqdm import tqdm

# --- AYARLAR ---
# İndirdiğin klasörü datasets/ijb-c içine koyduğunu varsayıyorum
IJBC_ROOT = "datasets/ijb-testsuite/ijb/IJBC"
META_DIR = os.path.join(IJBC_ROOT, "meta")
IMG_DIR = os.path.join(IJBC_ROOT, "loose_crop")
OUTPUT_PAIRS = os.path.join(IJBC_ROOT, "pairs.txt")

# Senin bahsettiğin dosya isimleri
METADATA_FILE = os.path.join(META_DIR, "ijbc_face_tid_mid.txt")
PAIR_FILE = os.path.join(META_DIR, "ijbc_template_pair_label.txt")


def prepare_ijbc():
    print("1. Metadata (Template -> Image) haritası çıkarılıyor...")
    # Bu dosya genelde boşlukla ayrılmıştır: img_name template_id media_id
    # Header yoksa header=None kullanıyoruz.
    df_meta = pd.read_csv(METADATA_FILE, sep=' ', header=None, names=['img_name', 'template_id', 'media_id'])

    # Her Template ID için İLK resmi seçelim (Basitleştirilmiş Protokol)
    # IJB-C normalde "Set-to-Set"tir ama bizim pipeline "Image-to-Image".
    # Şimdilik her şablonun en temsili (ilk) resmini alıyoruz.
    template_to_img = df_meta.groupby('template_id').first()['img_name'].to_dict()

    print(f"   Toplanan Template Sayısı: {len(template_to_img)}")

    print("2. Eşleşme Listesi (Pairs) oluşturuluyor...")
    # Bu dosya: template_id_1 template_id_2 label
    df_pairs = pd.read_csv(PAIR_FILE, sep=' ', header=None, names=['t1', 't2', 'label'])

    final_pairs = []

    for idx, row in tqdm(df_pairs.iterrows(), total=len(df_pairs)):
        t1, t2, label = row['t1'], row['t2'], row['label']

        # Eğer iki template'in de resmi varsa listeye ekle
        if t1 in template_to_img and t2 in template_to_img:
            img1 = template_to_img[t1]
            img2 = template_to_img[t2]

            # loose_crop klasöründe uzantı kontrolü (genelde .jpg olur ama listede olmayabilir)
            if not img1.endswith(".jpg"): img1 += ".jpg"
            if not img2.endswith(".jpg"): img2 += ".jpg"

            # Label: IJB-C'de 1=Match, 0=Mismatch'tir.
            final_pairs.append(f"{img1},{img2},{int(label)}")

    print(f"3. Kaydediliyor: {OUTPUT_PAIRS}")
    with open(OUTPUT_PAIRS, "w") as f:
        f.write("img1,img2,label\n")  # Header
        for line in final_pairs:
            f.write(line + "\n")

    print(f"✅ Hazır! Toplam {len(final_pairs)} çift oluşturuldu.")


if __name__ == "__main__":
    if os.path.exists(METADATA_FILE) and os.path.exists(PAIR_FILE):
        prepare_ijbc()
    else:
        print("❌ Dosyalar bulunamadı. Yolları kontrol et.")