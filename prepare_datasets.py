import os
import random
import glob

# KLASÖR YOLLARI
DATASET_ROOT = "datasets"
XQLFW_PATH = os.path.join(DATASET_ROOT, "xqlfw")
TINYFACE_PATH = os.path.join(DATASET_ROOT, "tinyface")


def check_dataset(name, path):
    print(f"\n--- Checking {name} ---")
    if not os.path.exists(path):
        print(f"❌ Klasör bulunamadı: {path}")
        return False

    img_dir = os.path.join(path, "images")
    if not os.path.exists(img_dir):
        # Bazen 'aligned_images' adıyla gelir, kontrol edelim
        if os.path.exists(os.path.join(path, "aligned_images")):
            print(f"⚠️ 'images' klasörü yok ama 'aligned_images' bulundu. Lütfen adını 'images' yapın.")
        else:
            print(f"❌ 'images' klasörü bulunamadı: {img_dir}")
        return False

    identities = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    print(f"✅ Klasör yapısı doğru görünüyor.")
    print(f"📊 Toplam Kişi Sayısı (Identity): {len(identities)}")

    total_images = sum([len(files) for r, d, files in os.walk(img_dir)])
    print(f"📊 Toplam Resim Sayısı: {total_images}")

    # Pairs Check
    pairs_path = os.path.join(path, "pairs.txt")
    if os.path.exists(pairs_path):
        print(f"✅ pairs.txt mevcut.")
        with open(pairs_path, 'r') as f:
            print(f"📄 Pairs satır sayısı: {len(f.readlines())}")
    else:
        print(f"⚠️ pairs.txt MEVCUT DEĞİL.")
        if name == "tinyface":
            print("⚙️ TinyFace için pairs.txt otomatik oluşturuluyor...")
            generate_pairs(img_dir, pairs_path)


def generate_pairs(img_dir, output_path, num_pairs=3000):
    """
    Eğer TinyFace pairs dosyası yoksa, LFW formatında (3000 match, 3000 mismatch)
    rastgele bir çift listesi oluşturur.
    """
    identities = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    # Sadece en az 2 resmi olan kişileri al (Match üretebilmek için)
    valid_idents = [d for d in identities if len(os.listdir(os.path.join(img_dir, d))) > 1]

    matches = []
    mismatches = []

    print("   Generating Match Pairs...")
    # Generate Matches
    while len(matches) < num_pairs:
        idn = random.choice(valid_idents)
        files = os.listdir(os.path.join(img_dir, idn))
        if len(files) < 2: continue
        img1, img2 = random.sample(files, 2)
        matches.append(f"{idn}\t{img1}\t{img2}")

    print("   Generating Mismatch Pairs...")
    # Generate Mismatches
    while len(mismatches) < num_pairs:
        id1, id2 = random.sample(identities, 2)
        files1 = os.listdir(os.path.join(img_dir, id1))
        files2 = os.listdir(os.path.join(img_dir, id2))
        if not files1 or not files2: continue
        img1 = random.choice(files1)
        img2 = random.choice(files2)
        mismatches.append(f"{id1}\t{img1}\t{id2}\t{img2}")

    # Save (LFW Format: header line -> 10 300)
    with open(output_path, "w") as f:
        f.write(f"10\t{num_pairs}\n")  # Header
        for line in matches:
            f.write(line + "\n")
        for line in mismatches:
            f.write(line + "\n")
    print(f"✅ pairs.txt oluşturuldu: {output_path}")


# --- RUN CHECKS ---
check_dataset("tinyface", TINYFACE_PATH)
check_dataset("xqlfw", XQLFW_PATH)