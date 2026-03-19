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

from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr00
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN

# Kendi modüllerin
from degradations import degradation_pool
from arcfaceutility import ensure_rgb, get_encoding_from_image

# ==========================================
# 1. AYARLAR & MODEL
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# MTCNN (Yüz Tespiti)
detector = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
# Facenet (Embedding)
embedder = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

# Klasör Yolları
DATASET_ROOT = "datasets"
XQLFW_PATH = os.path.join(DATASET_ROOT, "xqlfw")
PAIRS_PATH = os.path.join(XQLFW_PATH, "pairs.txt")
IMG_DIR = os.path.join(XQLFW_PATH, "images")

# Sonuç Klasörü
os.makedirs("../results_xqlfw", exist_ok=True)

# Cache (Sentetik eğitim için LFW cache'i, XQLFW için kullanılmayacak)
CACHE_PATH = "embedding_cache_lfw.pkl"
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        embedding_cache = pickle.load(f)
else:
    embedding_cache = {}


# ==========================================
# 2. DATA LOADER (XQLFW)
# ==========================================
def parse_pairs(pairs_path):
    matches, mismatches = [], []
    with open(pairs_path, 'r') as f:
        lines = f.readlines()

    # Header check
    if len(lines[0].strip().split('\t')) == 2:
        lines = lines[1:]

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


# ==========================================
# 3. PHASE A: TRAINING QUALITY CLASSIFIER (SYNTHETIC LFW)
# ==========================================
print("\n--- PHASE A: Training Quality Classifier (on Synthetic LFW) ---")
# Modeli hala LFW üzerinde eğitiyoruz çünkü "Kalite" kavramını buradan öğrenmesini istiyoruz.
# Bakalım sentetik eğitim, gerçek XQLFW bozulmalarını yakalayacak mı?

lfw_train_subset = fetch_lfw_pairs(subset='train', color=True, resize=1.0)
train_pairs_for_clf = [(ensure_rgb(p[0]), ensure_rgb(p[1])) for p in lfw_train_subset.pairs]
train_labels_for_clf = lfw_train_subset.target
train_match_pairs = [p for p, l in zip(train_pairs_for_clf, train_labels_for_clf) if l == 1]

X_train_list = []
y_train_list = []
unrecognizable_embeddings_for_centroid = []

print("Generating synthetic training data...")
# Hızlı eğitim için (Exhaustive değil)
for i in tqdm(range(len(train_match_pairs))):
    img1, img2 = train_match_pairs[i]

    # Clean Ref
    verif_enc, _ = get_encoding_from_image(img2, "facenet_pytorch", embedding_cache, f"train_verif_{i}", detector, embedder, device)
    if verif_enc is None: continue

    # Clean Probe (Label 1)
    orig_enc, _ = get_encoding_from_image(img1, "facenet_pytorch", embedding_cache, f"train_orig_{i}", detector, embedder, device)
    if orig_enc is not None:
        X_train_list.append(orig_enc)
        y_train_list.append(1.0)

    # Degraded Probe (Label 0)
    deg_fn = degradation_pool[i % len(degradation_pool)]
    deg_img = deg_fn(img1.copy(), strength=np.random.randint(3, 6))
    deg_enc, _ = get_encoding_from_image(deg_img, "facenet_pytorch", {}, "temp", detector, embedder, device)

    if deg_enc is not None:
        sim = cosine_similarity([verif_enc], [deg_enc])[0][0]
        label = 1.0 if sim > 0.25 else 0.0
        X_train_list.append(deg_enc)
        y_train_list.append(label)
        if label == 0.0: unrecognizable_embeddings_for_centroid.append(deg_enc)


# Model
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


X_train = np.array(X_train_list)
y_train = np.array(y_train_list)
if len(X_train) > 0:
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)

X_tr_t = torch.tensor(X_train, dtype=torch.float32)
y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

model = Classifier().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()
dataset = TensorDataset(X_tr_t, y_tr_t)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

print("Training Classifier...")
model.train()
for epoch in range(15):
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
model.eval()
print("Classifier Trained.")

if unrecognizable_embeddings_for_centroid:
    ui_centroid = np.mean(unrecognizable_embeddings_for_centroid, axis=0)
else:
    # Fallback (Çok nadir)
    ui_centroid = np.zeros(512)

# ==========================================
# 4. PHASE D: PROCESS XQLFW DATA
# ==========================================
print("\n--- PHASE D: Processing XQLFW Dataset ---")
matches, mismatches = parse_pairs(PAIRS_PATH)
print(f"XQLFW Pairs: {len(matches)} matches, {len(mismatches)} mismatches")

xqlfw_data = []


# Helper: Tek bir pair işle
def process_pair(img1_path, img2_path, label):
    img1 = read_image(img1_path)
    img2 = read_image(img2_path)

    if img1 is None or img2 is None: return  # Dosya okuma hatası

    # Cache kullanmıyoruz çünkü XQLFW yeni veri
    enc1, _ = get_encoding_from_image(img1, "facenet_pytorch", {}, "temp", detector, embedder, device)
    enc2, _ = get_encoding_from_image(img2, "facenet_pytorch", {}, "temp", detector, embedder, device)

    if enc1 is None or enc2 is None: return  # Yüz bulunamadı

    sim = cosine_similarity([enc1], [enc2])[0][0]
    ers = euclidean(ui_centroid.flatten(), enc1.flatten())

    with torch.no_grad():
        enc_norm = enc1 / np.linalg.norm(enc1)
        t_tensor = torch.tensor([enc_norm], dtype=torch.float32).to(device)
        prob = torch.sigmoid(model(t_tensor)).item()

    xqlfw_data.append({'ers': ers, 'prob': prob, 'sim': sim, 'label': label})


# Process Matches
for name, id1, id2 in tqdm(matches, desc="XQLFW Matches"):
    p1 = get_image_path(IMG_DIR, name, id1)
    p2 = get_image_path(IMG_DIR, name, id2)
    process_pair(p1, p2, 1.0)

# Process Mismatches
for name1, id1, name2, id2 in tqdm(mismatches, desc="XQLFW Mismatches"):
    p1 = get_image_path(IMG_DIR, name1, id1)
    p2 = get_image_path(IMG_DIR, name2, id2)
    process_pair(p1, p2, 0.0)

df_xqlfw = pd.DataFrame(xqlfw_data)
print(f"Total processed XQLFW samples: {len(df_xqlfw)}")
df_xqlfw.to_csv("results_xqlfw/xqlfw_raw_data.csv", index=False)

# ==========================================
# 5. PHASE E: 10-FOLD CV ON XQLFW
# ==========================================
print("\n--- PHASE E: Running 10-Fold CV on XQLFW ---")
# LFW protokolü gibi, 10 parçaya bölüp test ediyoruz.
# XQLFW'nin yapısı LFW ile aynı olduğu için direkt KFold uygulayabiliriz.

kf = KFold(n_splits=10, shuffle=True, random_state=42)  # Shuffle True çünkü pair listesi sıralı

results_ers = []
results_prob = []


def get_optimal_threshold_map(train_df, score_col, max_far=0.01):
    try:
        train_df['bin'] = pd.qcut(train_df[score_col], q=5, labels=False, duplicates='drop')
    except:
        train_df['bin'] = pd.cut(train_df[score_col], bins=5, labels=False)

    threshold_map = {}
    grouped = train_df.groupby('bin')[score_col].agg(['min', 'max'])

    for bin_id in sorted(train_df['bin'].unique()):
        subset = train_df[train_df['bin'] == bin_id]
        if len(subset) < 10: continue

        p, r, t = precision_recall_curve(subset['label'], subset['sim'])
        f1 = np.nan_to_num(2 * (p * r) / (p + r))

        best_f1 = -1
        best_thresh = t[-1]

        negatives = subset[subset['label'] == 0]
        n_neg = len(negatives)

        if n_neg > 0:
            sorted_idx = np.argsort(f1)[::-1]
            valid_found = False
            for idx in sorted_idx:
                if idx >= len(t): continue
                curr_thresh = t[idx]
                fp = len(negatives[negatives['sim'] >= curr_thresh])
                if (fp / n_neg) <= max_far:
                    best_thresh = curr_thresh
                    best_f1 = f1[idx]
                    valid_found = True
                    break
            if not valid_found: best_thresh = t[-1]

        threshold_map[bin_id] = {
            'thresh': best_thresh,
            'min_score': grouped.loc[bin_id, 'min'],
            'max_score': grouped.loc[bin_id, 'max']
        }
    return threshold_map


def evaluate_fold(test_df, threshold_map, score_col):
    sorted_bins = sorted(threshold_map.keys())
    y_true = test_df['label'].values
    scores = test_df[score_col].values
    sims = test_df['sim'].values
    y_pred = []

    for i in range(len(scores)):
        score = scores[i]
        sim = sims[i]
        selected_thresh = 0.5
        found = False
        for bid in sorted_bins:
            vals = threshold_map[bid]
            if vals['min_score'] <= score <= vals['max_score']:
                selected_thresh = vals['thresh']
                found = True
                break
        if not found:
            if score < threshold_map[sorted_bins[0]]['min_score']:
                selected_thresh = threshold_map[sorted_bins[0]]['thresh']
            else:
                selected_thresh = threshold_map[sorted_bins[-1]]['thresh']
        y_pred.append(1.0 if sim >= selected_thresh else 0.0)

    y_pred = np.array(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    acc = (tp + tn) / len(y_true)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {'acc': acc, 'f1': f1, 'far': far, 'frr': frr}


# Run CV
for fold, (train_idx, test_idx) in enumerate(kf.split(df_xqlfw)):
    train_df = df_xqlfw.iloc[train_idx].copy()
    test_df = df_xqlfw.iloc[test_idx].copy()

    # ERS
    map_ers = get_optimal_threshold_map(train_df, 'ers', max_far=0.01)
    res_ers = evaluate_fold(test_df, map_ers, 'ers')
    results_ers.append(res_ers)

    # Prob
    map_prob = get_optimal_threshold_map(train_df, 'prob', max_far=0.01)
    res_prob = evaluate_fold(test_df, map_prob, 'prob')
    results_prob.append(res_prob)

    print(f"Fold {fold + 1}: ERS F1={res_ers['f1']:.4f}, Prob F1={res_prob['f1']:.4f}")


# ==========================================
# 6. RESULTS & PLOTS
# ==========================================
def print_summary(metrics, name):
    f1s = [m['f1'] for m in metrics]
    fars = [m['far'] for m in metrics]
    frrs = [m['frr'] for m in metrics]
    print(f"\n--- {name} Results (XQLFW) ---")
    print(f"Avg F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Avg FAR: {np.mean(fars):.4f} ± {np.std(fars):.4f}")
    print(f"Avg FRR: {np.mean(frrs):.4f} ± {np.std(frrs):.4f}")


print_summary(results_ers, "ERS")
print_summary(results_prob, "Classifier")

# Correlation Plot (Real World Check)
plt.figure(figsize=(8, 6))
plt.scatter(df_xqlfw['ers'], df_xqlfw['prob'], c=df_xqlfw['prob'], cmap='viridis', alpha=0.5, s=10)
corr, _ = pearsonr(df_xqlfw['ers'], df_xqlfw['prob'])
plt.title(f"Real-World XQLFW: ERS vs Prob\nCorrelation: {corr:.4f}")
plt.xlabel("ERS Score")
plt.ylabel("Classifier Probability")
plt.grid(True, alpha=0.3)
plt.savefig("results_xqlfw/xqlfw_correlation.png")
print("\nAnalysis Complete. Check 'results_xqlfw/' folder.")