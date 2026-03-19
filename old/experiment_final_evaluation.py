import os
import cv2
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim  # <-- EKSİK OLAN KISIM EKLENDİ
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.spatial.distance import euclidean
from sklearn.metrics import precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity

from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN

# Kendi modüllerin
from arcfaceutility import ensure_rgb, get_encoding_from_image
from sklearn.datasets import fetch_lfw_pairs
from degradations import degradation_pool

# ==========================================
# 1. AYARLAR & MODEL
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

detector = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
embedder = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

# Yollar
DATASET_ROOT = "datasets"
XQLFW_PATH = os.path.join(DATASET_ROOT, "xqlfw")
PAIRS_PATH = os.path.join(XQLFW_PATH, "pairs.txt")
IMG_DIR = os.path.join(XQLFW_PATH, "images")

# Öğrenilmiş Threshold Dosyası (TinyFace'ten geliyor)
THRESHOLD_MAP_PATH = "results_tinyface/tinyface_learned_thresholds.pkl"

if not os.path.exists(THRESHOLD_MAP_PATH):
    raise FileNotFoundError("❌ Threshold map bulunamadı! Önce experiment_tinyface.py çalışmalı.")

with open(THRESHOLD_MAP_PATH, "rb") as f:
    LEARNED_THRESHOLDS = pickle.load(f)

print("✅ Learned Thresholds loaded from TinyFace.")

# ==========================================
# 2. QUICK RETRAIN OF CLASSIFIER (LFW-SYNTHETIC)
# ==========================================
print("\n--- PHASE A: Re-Training Quality Classifier (Consistency Check) ---")

# Cache Load (Varsa)
if os.path.exists("embedding_cache_lfw.pkl"):
    with open("embedding_cache_lfw.pkl", "rb") as f:
        cache = pickle.load(f)
else:
    cache = {}

lfw_train = fetch_lfw_pairs(subset='train', color=True, resize=1.0)
train_pairs = [(ensure_rgb(p[0]), ensure_rgb(p[1])) for p in lfw_train.pairs]
train_labels = lfw_train.target
train_match_pairs = [p for p, l in zip(train_pairs, train_labels) if l == 1]

X_train_list, y_train_list = [], []
centroid_list = []

print("Generating synthetic data...")
for i in tqdm(range(len(train_match_pairs))):
    img1, img2 = train_match_pairs[i]

    # Clean Reference
    verif_enc, _ = get_encoding_from_image(img2, "facenet_pytorch", cache, f"train_verif_{i}", detector, embedder, device)
    if verif_enc is None: continue

    # Clean Probe (Label 1)
    orig_enc, _ = get_encoding_from_image(img1, "facenet_pytorch", cache, f"train_orig_{i}", detector, embedder, device)
    if orig_enc is not None:
        X_train_list.append(orig_enc)
        y_train_list.append(1.0)

    # Degraded Probe (Label 0)
    deg_fn = degradation_pool[i % len(degradation_pool)]
    deg_img = deg_fn(img1.copy(), strength=np.random.randint(3, 6))
    deg_enc, _ = get_encoding_from_image(deg_img, "facenet_pytorch", {}, "temp", detector, embedder, device)

    if deg_enc is not None:
        # Labeling logic
        sim = cosine_similarity([verif_enc], [deg_enc])[0][0]
        lbl = 1.0 if sim > 0.25 else 0.0
        X_train_list.append(deg_enc)
        y_train_list.append(lbl)
        if lbl == 0.0: centroid_list.append(deg_enc)


# Classifier Model
class Classifier(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LeakyReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(), nn.BatchNorm1d(64), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x): return self.net(x)


X_tr = np.array(X_train_list)
if len(X_tr) > 0: X_tr = X_tr / np.linalg.norm(X_tr, axis=1, keepdims=True)
y_tr = np.array(y_train_list)

model = Classifier().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()
dset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
ldr = DataLoader(dset, batch_size=256, shuffle=True)

print("Training Classifier...")
model.train()
for epoch in range(15):
    for xb, yb in ldr:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
model.eval()
print("Classifier Re-Trained.")

if centroid_list:
    ui_centroid = np.mean(centroid_list, axis=0)
else:
    ui_centroid = np.zeros(512)

# ==========================================
# 3. EVALUATION ON XQLFW
# ==========================================
print("\n--- PHASE B: Final Evaluation on XQLFW ---")


def parse_pairs(pairs_path):
    matches, mismatches = [], []
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


def get_image_path(root_dir, name, img_id):
    filename = f"{name}_{str(img_id).zfill(4)}.jpg"
    return os.path.join(root_dir, name, filename)


def read_image(path):
    if not os.path.exists(path): return None
    img = cv2.imread(path)
    if img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


matches, mismatches = parse_pairs(PAIRS_PATH)
test_data = []


def process_pair(img1_path, img2_path, label):
    img1 = read_image(img1_path)
    img2 = read_image(img2_path)
    if img1 is None or img2 is None: return

    # Cache yok, taze hesapla
    enc1, _ = get_encoding_from_image(img1, "facenet_pytorch", {}, "temp", detector, embedder, device)
    enc2, _ = get_encoding_from_image(img2, "facenet_pytorch", {}, "temp", detector, embedder, device)
    if enc1 is None or enc2 is None: return

    sim = cosine_similarity([enc1], [enc2])[0][0]
    ers = euclidean(ui_centroid.flatten(), enc1.flatten())

    with torch.no_grad():
        enc_norm = enc1 / np.linalg.norm(enc1)
        prob = torch.sigmoid(model(torch.tensor([enc_norm], dtype=torch.float32).to(device))).item()

    test_data.append({'ers': ers, 'prob': prob, 'sim': sim, 'label': label})


print(f"Processing XQLFW Pairs ({len(matches)} matches, {len(mismatches)} mismatches)...")
for n, i1, i2 in tqdm(matches):
    process_pair(get_image_path(IMG_DIR, n, i1), get_image_path(IMG_DIR, n, i2), 1.0)
for n1, i1, n2, i2 in tqdm(mismatches):
    process_pair(get_image_path(IMG_DIR, n1, i1), get_image_path(IMG_DIR, n2, i2), 0.0)

df_test = pd.DataFrame(test_data)
print(f"Processed {len(df_test)} samples.")

# ==========================================
# 4. APPLY LEARNED THRESHOLDS
# ==========================================
print("\n--- PHASE C: Applying TinyFace Thresholds ---")


def apply_thresholds(df, score_col, threshold_map):
    sorted_bins = sorted(threshold_map.keys())
    y_pred = []

    scores = df[score_col].values
    sims = df['sim'].values

    for i in range(len(scores)):
        score = scores[i]
        sim = sims[i]

        selected_thresh = 0.5  # Default
        found = False

        # Hangi bin?
        for bid in sorted_bins:
            vals = threshold_map[bid]
            if vals['min_score'] <= score <= vals['max_score']:
                selected_thresh = vals['thresh']
                found = True
                break

        # Aralık dışı kontrolü
        if not found:
            if score < threshold_map[sorted_bins[0]]['min_score']:
                selected_thresh = threshold_map[sorted_bins[0]]['thresh']
            else:
                selected_thresh = threshold_map[sorted_bins[-1]]['thresh']

        y_pred.append(1.0 if sim >= selected_thresh else 0.0)

    return np.array(y_pred)


y_true = df_test['label'].values
y_pred_prob = apply_thresholds(df_test, 'prob', LEARNED_THRESHOLDS)

# Metrics
tp = np.sum((y_pred_prob == 1) & (y_true == 1))
tn = np.sum((y_pred_prob == 0) & (y_true == 0))
fp = np.sum((y_pred_prob == 1) & (y_true == 0))
fn = np.sum((y_pred_prob == 0) & (y_true == 1))

acc = (tp + tn) / len(y_true)
f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
far = fp / (fp + tn) if (fp + tn) > 0 else 0
frr = fn / (fn + tp) if (fn + tp) > 0 else 0

print("\n=== FINAL CROSS-DATASET EVALUATION ===")
print("Train: LFW-Synthetic | Val: TinyFace | Test: XQLFW")
print(f"Method: Classifier Probability")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"FAR     : {far:.4f}")
print(f"FRR     : {frr:.4f}")

# Save detailed results
df_test['pred_prob'] = y_pred_prob
df_test.to_csv("results_xqlfw/final_cross_dataset_results.csv", index=False)
print("Results saved.")