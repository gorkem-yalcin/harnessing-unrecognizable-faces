import os
import cv2
import matplotlib.pyplot as plt

# AYARLAR
DATASET_ROOT = "datasets"
XQLFW_PATH = os.path.join(DATASET_ROOT, "xqlfw")
PAIRS_PATH = os.path.join(XQLFW_PATH, "pairs.txt")
IMG_DIR = os.path.join(XQLFW_PATH, "images")


def parse_pairs(pairs_path):
    """
    Standart LFW pairs.txt formatını okur.
    Return: matches (list), mismatches (list)
    """
    matches = []
    mismatches = []

    with open(pairs_path, 'r') as f:
        lines = f.readlines()

    # İlk satır header (10 300) olabilir, onu atlayalım
    if len(lines[0].strip().split('\t')) == 2:
        print(f"Header found and skipped: {lines[0].strip()}")
        lines = lines[1:]

    for line in lines:
        parts = line.strip().split('\t')

        if len(parts) == 3:  # MATCH: Name, ID1, ID2
            name = parts[0]
            id1, id2 = parts[1], parts[2]
            matches.append((name, id1, id2))

        elif len(parts) == 4:  # MISMATCH: Name1, ID1, Name2, ID2
            name1, id1 = parts[0], parts[1]
            name2, id2 = parts[2], parts[3]
            mismatches.append((name1, id1, name2, id2))

    return matches, mismatches


def get_image_path(root_dir, name, img_id):
    """
    İsim ve ID'den tam dosya yolunu oluşturur (0001 formatına dikkat ederek).
    Örn: Marcelo_Ebrard + 3 -> .../Marcelo_Ebrard/Marcelo_Ebrard_0003.jpg
    """
    # XQLFW dosya formatı genelde: Name_0001.jpg şeklindedir.
    # zfill(4) ile 3 -> 0003 yapılır.
    filename = f"{name}_{str(img_id).zfill(4)}.jpg"
    return os.path.join(root_dir, name, filename)


# --- TEST ---
print(f"Reading pairs from: {PAIRS_PATH}")
matches, mismatches = parse_pairs(PAIRS_PATH)
print(f"Found {len(matches)} matches and {len(mismatches)} mismatches.")

# Rastgele bir Match kontrol edelim
if len(matches) > 0:
    name, id1, id2 = matches[0]
    p1 = get_image_path(IMG_DIR, name, id1)
    p2 = get_image_path(IMG_DIR, name, id2)

    print(f"\nSample Match Check:")
    print(f"1: {p1} -> Exists? {os.path.exists(p1)}")
    print(f"2: {p2} -> Exists? {os.path.exists(p2)}")

# Rastgele bir Mismatch kontrol edelim
if len(mismatches) > 0:
    name1, id1, name2, id2 = mismatches[0]
    p1 = get_image_path(IMG_DIR, name1, id1)
    p2 = get_image_path(IMG_DIR, name2, id2)

    print(f"\nSample Mismatch Check:")
    print(f"1: {p1} -> Exists? {os.path.exists(p1)}")
    print(f"2: {p2} -> Exists? {os.path.exists(p2)}")

if os.path.exists(p1):
    img = cv2.imread(p1)
    if img is not None:
        print(f"\nImage Loaded Successfully. Size: {img.shape}")
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.show()
    else:
        print("\n❌ Error: Image path exists but cv2 could not read it.")
else:
    print("\n❌ Error: Sample image path does not exist. Check folder structure.")