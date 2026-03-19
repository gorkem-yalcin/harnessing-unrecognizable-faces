import os
import pandas as pd

# Senin ayarların
IJBC_ROOT = "datasets/ijb-testsuite/ijb/IJBC"
PAIRS_PATH = os.path.join(IJBC_ROOT, "pairs.txt")
IMG_DIR = os.path.join(IJBC_ROOT, "loose_crop")

print(f"--- PATH DEBUGGER ---")
print(f"1. Pairs dosyası okunuyor: {PAIRS_PATH}")

if not os.path.exists(PAIRS_PATH):
    print("❌ HATA: Pairs dosyası bulunamadı!")
else:
    df = pd.read_csv(PAIRS_PATH)
    print(f"✅ Pairs okundu. İlk satır:\n{df.iloc[0]}")

    first_img_name = df.iloc[0]['img1']
    print(f"\n2. İlk resim adı analiz ediliyor: '{first_img_name}'")

    full_path = os.path.join(IMG_DIR, first_img_name)
    abs_path = os.path.abspath(full_path)

    print(f"3. Kodun oluşturduğu tam yol:\n   -> {abs_path}")

    if os.path.exists(abs_path):
        print("\n✅ BAŞARILI: Dosya diskte bulundu!")
    else:
        print("\n❌ HATA: Dosya diskte BULUNAMADI.")
        print("   Olası sebepler:")
        print("   - Klasör yolunda hata var.")
        print("   - Pairs dosyasında resim adının içinde boşluk veya tırnak kalmış.")

        print("\n4. 'loose_crop' klasöründeki gerçek dosyalardan örnekler:")
        try:
            real_files = os.listdir(IMG_DIR)[:5]
            for f in real_files:
                print(f"   - {f}")
        except FileNotFoundError:
            print("❌ HATA: loose_crop klasörü hiç bulunamadı!")