import os
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
from scipy.stats import pearsonr
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN

# Kendi modüllerin
from degradations import degradation_pool
from utility import get_encoding_from_image, ensure_rgb

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

detector = MTCNN(image_size=160, margin=0, device=device)
embedder = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

# Config
min_degradation_strength = 0
max_degradation_strength = 6
face_detection_method = "facenet_pytorch"

# Folders
os.makedirs("../results", exist_ok=True)

# Cache
CACHE_PATH = "embedding_cache.pkl"
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        embedding_cache = pickle.load(f)
else:
    embedding_cache = {}

# ==========================================
# 2. DATA LOADING (OFFICIAL 10-FOLDS)
# ==========================================
print("Loading LFW dataset (Official 10 Folds)...")
lfw_10_folds = fetch_lfw_pairs(subset='10_folds', color=True, resize=1.0)
all_pairs = [(ensure_rgb(p[0]), ensure_rgb(p[1])) for p in lfw_10_folds.pairs]
all_labels = lfw_10_folds.target
print(f"LFW 10-Folds Loaded. Total Pairs: {len(all_pairs)}")

# ==========================================
# 3. PHASE A: TRAINING QUALITY CLASSIFIER
# ==========================================
print("\n--- PHASE A: Training Quality Classifier ---")
lfw_train_subset = fetch_lfw_pairs(subset='train', color=True, resize=1.0)
train_pairs_for_clf = [(ensure_rgb(p[0]), ensure_rgb(p[1])) for p in lfw_train_subset.pairs]
train_labels_for_clf = lfw_train_subset.target
train_match_pairs = [p for p, l in zip(train_pairs_for_clf, train_labels_for_clf) if l == 1]

X_train_list = []
y_train_list = []
unrecognizable_embeddings_for_centroid = []

print("Generating synthetic data for classifier training...")
for i in tqdm(range(len(train_match_pairs))):
    img1, img2 = train_match_pairs[i]
    verif_enc, _ = get_encoding_from_image(img2, face_detection_method, embedding_cache, f"train_verif_{i}", detector, embedder, device)
    if verif_enc is None: continue

    # 1. Clean
    orig_enc, _ = get_encoding_from_image(img1, face_detection_method, embedding_cache, f"train_orig_{i}", detector, embedder, device)
    if orig_enc is not None:
        X_train_list.append(orig_enc)
        y_train_list.append(1.0)

    # 2. Degraded (Random)
    deg_fn = degradation_pool[i % len(degradation_pool)]
    deg_img = deg_fn(img1.copy(), strength=np.random.randint(3, 6))
    deg_enc, _ = get_encoding_from_image(deg_img, face_detection_method, {}, "temp", detector, embedder, device)
    if deg_enc is not None:
        sim = cosine_similarity([verif_enc], [deg_enc])[0][0]
        label = 1.0 if sim > 0.25 else 0.0
        X_train_list.append(deg_enc)
        y_train_list.append(label)
        if label == 0.0: unrecognizable_embeddings_for_centroid.append(deg_enc)


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
for epoch in range(20):
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
    ui_centroid = None

# ==========================================
# 4. PHASE D: EXHAUSTIVE DEGRADED DATASET GEN
# ==========================================
print("\n--- PHASE D: Pre-Calculating Degraded Dataset for CV (Exhaustive) ---")
all_data_records = []

for i in tqdm(range(len(all_pairs)), desc="Generating Exhaustive Data"):
    img1, img2 = all_pairs[i]
    label = all_labels[i]

    enc2, _ = get_encoding_from_image(img2, face_detection_method, embedding_cache, f"lfw_10fold_{i}_2", detector, embedder, device)
    if enc2 is None: continue

    for deg_fn in degradation_pool:
        for strength in range(min_degradation_strength, max_degradation_strength):
            deg_img = deg_fn(img1.copy(), strength=strength)
            enc1, _ = get_encoding_from_image(deg_img, face_detection_method, {}, "temp", detector, embedder, device)

            if enc1 is None: continue

            sim = cosine_similarity([enc1], [enc2])[0][0]
            ers = euclidean(ui_centroid.flatten(), enc1.flatten()) if ui_centroid is not None else 0.0

            with torch.no_grad():
                enc_norm = enc1 / np.linalg.norm(enc1)
                t_tensor = torch.tensor([enc_norm], dtype=torch.float32).to(device)
                prob = torch.sigmoid(model(t_tensor)).item()

            all_data_records.append({
                'ers': ers, 'prob': prob, 'sim': sim, 'label': label,
                'pair_index': i, 'deg_type': deg_fn.__name__, 'strength': strength
            })

df_all = pd.DataFrame(all_data_records)
print(f"Total processed samples: {len(df_all)}")

with open(CACHE_PATH, "wb") as f:
    pickle.dump(embedding_cache, f)

# ==========================================
# 5. 10-FOLD CROSS VALIDATION LOGIC
# ==========================================
print("\n--- PHASE E: Running 10-Fold Cross Validation ---")

pair_indices = np.arange(len(all_pairs))
kf = KFold(n_splits=10, shuffle=False)

results_ers = []
results_prob = []
all_test_data_for_plot = []


def get_optimal_threshold_map(train_df, score_col, num_bins=5, max_far=0.01):
    try:
        train_df['bin'] = pd.qcut(train_df[score_col], q=num_bins, labels=False, duplicates='drop')
    except:
        train_df['bin'] = pd.cut(train_df[score_col], bins=num_bins, labels=False)

    threshold_map = {}
    grouped = train_df.groupby('bin')[score_col].agg(['min', 'max'])

    for bin_id in sorted(train_df['bin'].unique()):
        subset = train_df[train_df['bin'] == bin_id]
        if len(subset) < 10: continue

        p, r, t = precision_recall_curve(subset['label'], subset['sim'])
        f1 = np.nan_to_num(2 * (p * r) / (p + r))

        best_f1 = -1
        best_thresh = 0.0
        negatives = subset[subset['label'] == 0]
        n_neg = len(negatives)

        if n_neg == 0:
            best_thresh = subset['sim'].min()
        else:
            sorted_indices = np.argsort(f1)[::-1]
            valid_found = False
            for idx in sorted_indices:
                if idx >= len(t): continue
                curr_thresh = t[idx]
                fp = len(negatives[negatives['sim'] >= curr_thresh])
                far = fp / n_neg
                if far <= max_far:
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
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return {'acc': acc, 'f1': f1, 'far': far, 'frr': frr}


for fold, (train_pair_idx, test_pair_idx) in enumerate(kf.split(pair_indices)):
    train_df = df_all[df_all['pair_index'].isin(train_pair_idx)].copy()
    test_df = df_all[df_all['pair_index'].isin(test_pair_idx)].copy()

    # Store test data WITH SIMILARITY for plotting later
    all_test_data_for_plot.append(test_df[['ers', 'prob', 'label', 'sim']].copy())

    # 1. ERS
    map_ers = get_optimal_threshold_map(train_df, 'ers', num_bins=5, max_far=0.01)
    res_ers = evaluate_fold(test_df, map_ers, 'ers')
    results_ers.append(res_ers)

    # 2. Classifier
    map_prob = get_optimal_threshold_map(train_df, 'prob', num_bins=5, max_far=0.01)
    res_prob = evaluate_fold(test_df, map_prob, 'prob')
    results_prob.append(res_prob)

    print(f"Fold {fold + 1}/10 -> ERS F1: {res_ers['f1']:.4f} | Prob F1: {res_prob['f1']:.4f}")

# ==========================================
# 6. FINAL REPORTING & PLOTS
# ==========================================
print("\n--- FINAL RESULTS ---")


def print_summary(metrics, name):
    f1s = [m['f1'] for m in metrics]
    fars = [m['far'] for m in metrics]
    frrs = [m['frr'] for m in metrics]
    accs = [m['acc'] for m in metrics]
    print(f"\nMethod: {name}")
    print(f"Avg Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Avg F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Avg FAR     : {np.mean(fars):.4f} ± {np.std(fars):.4f}")
    print(f"Avg FRR     : {np.mean(frrs):.4f} ± {np.std(frrs):.4f}")

    # Save to CSV
    return pd.DataFrame(metrics)


df_res_ers = print_summary(results_ers, "ERS (UI Distance)")
df_res_prob = print_summary(results_prob, "Classifier Probability")
df_res_ers.to_csv("results/results_ers_folds.csv")
df_res_prob.to_csv("results/results_prob_folds.csv")

# --- CONSOLIDATED PLOTS ---
print("\nGenerating Final Plots...")
final_df = pd.concat(all_test_data_for_plot)


# 1. Generate Adaptive Threshold Curves (Visual Proof)
def generate_adaptive_plots(df, score_col, title, higher_better=True):
    try:
        df['bin'] = pd.qcut(df[score_col], q=5, labels=False, duplicates='drop')
    except:
        df['bin'] = pd.cut(df[score_col], bins=5, labels=False)

    bins = sorted(df['bin'].unique())
    if not higher_better: bins = bins[::-1]

    plot_data = []
    print(f"\nCurve Data: {title}")

    for b in bins:
        subset = df[df['bin'] == b]
        if len(subset) < 10: continue

        p, r, t = precision_recall_curve(subset['label'], subset['sim'])
        f1 = np.nan_to_num(2 * (p * r) / (p + r))

        # Max FAR Constraint (0.01) for plotting consistency
        best_f1 = -1
        best_thresh = t[-1]

        negatives = subset[subset['label'] == 0]
        n_neg = len(negatives)

        if n_neg > 0:
            sorted_idx = np.argsort(f1)[::-1]
            for idx in sorted_idx:
                if idx >= len(t): continue
                th = t[idx]
                fp = len(negatives[negatives['sim'] >= th])
                if (fp / n_neg) <= 0.01:
                    best_thresh = th
                    best_f1 = f1[idx]
                    break

        min_s, max_s = subset[score_col].min(), subset[score_col].max()
        center = (min_s + max_s) / 2
        print(f"Bin {b}: Range {min_s:.2f}-{max_s:.2f} -> Thresh {best_thresh:.4f} (F1: {best_f1:.4f})")
        plot_data.append({'center': center, 'thresh': best_thresh})

    if plot_data:
        plt.figure(figsize=(6, 4))
        plt.plot([x['center'] for x in plot_data], [x['thresh'] for x in plot_data], marker='o', linewidth=2)
        plt.title(f"Adaptive Threshold: {title}")
        plt.xlabel(f"{title} Score")
        plt.ylabel("Optimal Threshold (FAR<=0.01)")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"results/adaptive_curve_{score_col}.png")
        print(f"Saved: results/adaptive_curve_{score_col}.png")


generate_adaptive_plots(final_df, 'ers', "ERS", higher_better=False)
generate_adaptive_plots(final_df, 'prob', "Classifier Prob", higher_better=True)

# 2. Scatter Plot
if len(final_df) > 100:
    corr, _ = pearsonr(final_df['ers'], final_df['prob'])
    print(f"Overall Correlation: {corr:.4f}")

    plot_sample = final_df.sample(min(len(final_df), 10000))
    plt.figure(figsize=(8, 6))
    plt.scatter(plot_sample['ers'], plot_sample['prob'], c=plot_sample['prob'], cmap='viridis', alpha=0.3, s=5)
    plt.colorbar(label="Classifier Probability")
    plt.title(f"Degraded LFW (10-Fold Aggregated): ERS vs Prob\nCorrelation: {corr:.4f}")
    plt.xlabel("ERS Score")
    plt.ylabel("Classifier Probability")
    plt.grid(True, alpha=0.2)
    plt.savefig("results/final_correlation.png")
    print("Saved: results/final_correlation.png")

print("\nALL DONE.")