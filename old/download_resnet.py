import os
import requests
from tqdm import tqdm

# Hedef Klasör ve Dosya
MODEL_DIR = os.path.expanduser("~/.insightface/models/arcface_r100_v1")
FILE_NAME = "arcface_r100.onnx"
SAVE_PATH = os.path.join(MODEL_DIR, FILE_NAME)
# GitHub Raw Linki (LFS için en güvenilir link formatı)
URL = "https://media.githubusercontent.com/media/onnx/models/main/vision/body_analysis/arcface/model/arcface-resnet100-8.onnx"


def fix_model():
    # 1. Varsa bozuk dosyayı sil
    if os.path.exists(SAVE_PATH):
        print(f"🗑️ Eski/Bozuk dosya siliniyor: {SAVE_PATH}")
        os.remove(SAVE_PATH)

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"🚀 İndiriliyor (Bu işlem internet hızına göre 1-5 dk sürebilir)...")
    try:
        response = requests.get(URL, stream=True)
        response.raise_for_status()  # Link kırık ise hata ver

        total_size = int(response.headers.get('content-length', 0))

        with open(SAVE_PATH, 'wb') as file, tqdm(
                desc="İlerleme",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                bar.update(size)

        # 2. Boyut Kontrolü (Kritik Adım)
        file_size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
        print(f"\n✅ İndirme Tamamlandı. Dosya Boyutu: {file_size_mb:.2f} MB")

        if file_size_mb < 50:
            print("❌ HATA: Dosya çok küçük! Muhtemelen inmedi veya GitHub engelledi.")
            print("Lütfen şu linkten manuel indirip klasöre atın:")
            print(f"Link: {URL}")
            print(f"Hedef Klasör: {MODEL_DIR}")
        else:
            print("✅ Dosya sağlam görünüyor. Şimdi experiment kodunu çalıştırabilirsin.")

    except Exception as e:
        print(f"\n❌ İndirme sırasında hata oluştu: {e}")


if __name__ == "__main__":
    fix_model()