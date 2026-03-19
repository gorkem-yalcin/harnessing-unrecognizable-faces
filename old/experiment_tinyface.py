import os
import cv2
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity

# Kendi modüllerin
from facenet_pytorch.models.mtcnn import MTCNN
from degradations import degradation_pool
from arcfaceutility import ensure_rgb, get_encoding_from_image

# ==========================================
# 1. AYARLAR
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

DATASET_ROOT = "datasets"
TINYFACE_PATH = os.path.join(DATASET_ROOT, "tinyface")
PAIRS_PATH = os.path.join(TINYFACE_PATH, "pairs.txt")
IMG_DIR = os.path.join(TINYFACE_PATH, "images")
os.makedirs("../results_tinyface", exist_ok=True)

# Cache
CACHE_PATH = "embedding_cache_lfw.pkl"
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        embedding_cache = pickle.load(f)
else:
    embedding_cache = {}


# ==========================================
# 2. YARDIMCI FONKSİYONLAR
# ==========================================
def parse_pairs(pairs_path):
    matches, mismatches = [], []
    if not os.path.exists(pairs_path): return [], []
    with open(pairs_path, 'r') as f:
        lines = f.readlines()
    if len(lines[0].strip().split('\t')) == 2: lines = lines[1:]
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            matches.append((parts[0], parts[1], parts[2]))
        elif len(parts) == 4:
            mismatches.append((parts[0], parts[1], parts[2], parts[3]))
    return matches, mismatches


def get_image_path(root_dir, name, img_name):
    return os.path.join(root_dir, name, img_name)


def read_image(path):
    if not os.path.exists(path): return None
    img = cv2.imread(path)
    if img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ==========================================
# 3. PHASE A: CLASSIFIER EĞİTİMİ (GÜÇLENDİRİLMİŞ)
# ==========================================
print("\n--- PHASE A: Training SOTA Classifier ---")
lfw_train = fetch_lfw_pairs(subset='train', color=True, resize=1.0)
train_pairs = [(ensure_rgb(p[0]), ensure_rgb(p[1])) for p in lfw_train.pairs]
train_labels = lfw_train.target
train_match_pairs = [p for p, l in zip(train_pairs, train_labels) if l == 1]

X_train_list, y_train_list = [], []

# TRICK 1: DAHA FAZLA VERİ
AUGMENT_FACTOR = 10
print(f"Generating synthetic data (x{AUGMENT_FACTOR} augmentation)...")

for i in tqdm(range(len(train_match_pairs))):
    img1, img2 = train_match_pairs[i]

    # Orijinal High-Res Embedding
    verif_enc, _ = get_encoding_from_image(img2, "", embedding_cache, f"train_verif_{i}")
    if verif_enc is None: continue

    # Bozulmamış Eş (Referans)
    orig_enc, _ = get_encoding_from_image(img1, "", embedding_cache, f"train_orig_{i}")

    if orig_enc is not None:
        diff = np.abs(orig_enc - verif_enc)
        mult = orig_enc * verif_enc
        cosine = np.dot(orig_enc, verif_enc)

        # Feature: [Emb1, Emb2, Diff, Mult, Cosine]
        feat = np.concatenate([orig_enc, verif_enc, diff, mult, [cosine]])
        X_train_list.append(feat)
        y_train_list.append(1.0)

    # Sentetik Bozma
    for k in range(AUGMENT_FACTOR):
        deg_fn = degradation_pool[(i + k) % len(degradation_pool)]
        strength = np.random.randint(2, 7)
        deg_img = deg_fn(img1.copy(), strength=strength)

        deg_enc, _ = get_encoding_from_image(deg_img, "", {}, "temp")

        if deg_enc is not None:
            sim = np.dot(verif_enc, deg_enc)
            lbl = 1.0 if sim > 0.30 else 0.0

            diff = np.abs(deg_enc - verif_enc)
            mult = deg_enc * verif_enc

            feat = np.concatenate([deg_enc, verif_enc, diff, mult, [sim]])
            X_train_list.append(feat)
            y_train_list.append(lbl)

# Veriyi Hazırla
X_tr = np.array(X_train_list, dtype=np.float32)
y_tr = np.array(y_train_list, dtype=np.float32)

print(f"Eğitim Verisi Boyutu: {X_tr.shape}")


# TRICK 3: DAHA GÜÇLÜ MODEL
class SOTAClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)


model = SOTAClassifier(input_dim=X_tr.shape[1]).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()

dataset = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr).unsqueeze(1))
loader = DataLoader(dataset, batch_size=512, shuffle=True)

print("Training SOTA Classifier...")
model.train()
for epoch in range(100):
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0: # Log kirliliği olmasın diye 10'da bir yazsın
        print(f"Epoch {epoch}: Loss {total_loss / len(loader):.4f}")

# --- KRİTİK EKLEME: MODELİ KAYDET ---
model.eval()
torch.save(model.state_dict(), "final_sota_classifier.pth")
print("✅ Final Model Saved to 'final_sota_classifier.pth'")
# ------------------------------------

# ==========================================
# 4. PHASE D: PROCESS TINYFACE
# ==========================================
print("\n--- PHASE D: Processing TinyFace ---")
matches, mismatches = parse_pairs(PAIRS_PATH)
tinyface_data = []


def process_pair(img1_path, img2_path, label):
    img1 = read_image(img1_path)
    img2 = read_image(img2_path)
    if img1 is None or img2 is None: return

    enc1, _ = get_encoding_from_image(img1)
    enc2, _ = get_encoding_from_image(img2)

    if enc1 is None or enc2 is None: return

    diff = np.abs(enc1 - enc2)
    mult = enc1 * enc2
    sim = np.dot(enc1, enc2)

    feat = np.concatenate([enc1, enc2, diff, mult, [sim]])
    t_tensor = torch.tensor([feat], dtype=torch.float32).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(t_tensor)).item()

    hybrid_score = sim * prob
    tinyface_data.append({'prob': prob, 'sim': sim, 'hybrid': hybrid_score, 'label': label})

print("Processing Matches...")
for name, img1, img2 in tqdm(matches):
    process_pair(get_image_path(IMG_DIR, name, img1), get_image_path(IMG_DIR, name, img2), 1.0)
print("Processing Mismatches...")
for name1, img1, name2, img2 in tqdm(mismatches):
    process_pair(get_image_path(IMG_DIR, name1, img1), get_image_path(IMG_DIR, name2, img2), 0.0)

df_tiny = pd.DataFrame(tinyface_data)

# ==========================================
# 5. PHASE E: SOTA BENCHMARKING
# ==========================================
print("\n--- PHASE E: SOTA Benchmarking ---")

def calculate_frr_at_far(y_true, y_scores, target_far):
    neg_scores = y_scores[y_true == 0]
    neg_scores_sorted = np.sort(neg_scores)[::-1]
    far_idx = int(target_far * len(neg_scores))
    threshold = neg_scores_sorted[far_idx]

    pos_scores = y_scores[y_true == 1]
    false_rejects = np.sum(pos_scores < threshold)
    frr = false_rejects / len(pos_scores)
    return frr, threshold

scores = df_tiny['hybrid'].values
labels = df_tiny['label'].values

print(f"{'FAR':<10} | {'Threshold':<10} | {'FRR (OURS)':<15} | {'SOTA (Goal)':<15}")
print("-" * 60)

fars = [1e-2, 1e-3, 1e-4]
for far in fars:
    frr, thresh = calculate_frr_at_far(labels, scores, far)
    goal = "0.4718" if far == 1e-2 else "-"
    print(f"{far:<10} | {thresh:.4f}     | {frr:.4f}          | {goal}")

fpr, tpr, _ = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)
print(f"\n🏆 Hybrid ROC AUC: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='purple', lw=2, label=f'Hybrid ROC (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (ResNet50 + SOTA Classifier)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("results_tinyface/roc_curve_sota.png")