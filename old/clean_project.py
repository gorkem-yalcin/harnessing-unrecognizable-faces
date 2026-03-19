import os
import shutil

# Silinecek Dosyalar (Eğitilmiş Modeller ve Cache)
files_to_delete = [
    "final_sota_classifier.pth",  # Eğitilen model
    "embedding_cache_lfw.pkl",  # LFW cache
    "tinyface_learned_thresholds.pkl",  # Threshold haritası
    "results_ijbc_fixed.txt",  # Eski sonuçlar
    "results_ijbc_sota.txt",  # Eski sonuçlar
    "results_ijbc_dynamic.txt"  # Eski sonuçlar
]

# Temizlenecek Klasörler
folders_to_clean = [
    "results_tinyface"  # ROC eğrileri ve loglar
]


def clean_project():
    print("🧹 Temizlik Başlıyor...")

    # 1. Dosyaları Sil
    for f in files_to_delete:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"✅ Silindi: {f}")
            except Exception as e:
                print(f"❌ Silinemedi {f}: {e}")
        else:
            print(f"ℹ️ Zaten yok: {f}")

    # 2. Klasörleri Temizle
    for folder in folders_to_clean:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                os.makedirs(folder)  # Klasörü boş olarak tekrar oluştur
                print(f"✅ Klasör Sıfırlandı: {folder}")
            except Exception as e:
                print(f"❌ Klasör Hatası {folder}: {e}")
        else:
            os.makedirs(folder)
            print(f"✅ Klasör Oluşturuldu: {folder}")

    print("\n✨ Proje tertemiz! 'experiment_tinyface.py' ile sıfırdan başlayabilirsin.")


if __name__ == "__main__":
    clean_project()