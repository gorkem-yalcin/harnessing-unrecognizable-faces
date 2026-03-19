import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN

# Kendi modüllerin
from degradations import degradation_pool
from utility import get_encoding_from_image, save_image, ensure_rgb

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# Models
detector = MTCNN(image_size=160, margin=0, device=device)
embedder = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

# Config
verification_threshold = 0.25
ers_threshold = 1.0
min_degradation_strength = 1
max_degradation_strength = 6
face_detection_method = "facenet_pytorch"

# Folders
os.makedirs("../output/recognizable", exist_ok=True)
os.makedirs("../output/unrecognizable", exist_ok=True)
os.makedirs("../output/no_embedding", exist_ok=True)
os.makedirs("../results", exist_ok=True)

# Cache
CACHE_PATH = "embedding_cache.pkl"
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        embedding_cache = pickle.load(f)
else:
    embedding_cache = {}

# ==========================================
# 2. DATA LOADING
# ==========================================
print("Loading LFW dataset...")
lfw_train = fetch_lfw_pairs(subset='train', color=True, resize=1.0)
lfw_test = fetch_lfw_pairs(subset='test', color=True, resize=1.0)

# Train Pairs
train_pairs = [(ensure_rgb(p[0]), ensure_rgb(p[1])) for p in lfw_train.pairs]
train_labels = lfw_train.target
train_match_pairs = [p for p, l in zip(train_pairs, train_labels) if l == 1]

# Test Pairs (CLEAN)
test_pairs = [(ensure_rgb(p[0]), ensure_rgb(p[1])) for p in lfw_test.pairs]
test_labels = lfw_test.target
test_match_pairs = [p for p, l in zip(test_pairs, test_labels) if l == 1]
test_mismatch_pairs = [p for p, l in zip(test_pairs, test_labels) if l == 0]

print(f"LFW Loaded. Train Match: {len(train_match_pairs)} | Test Match: {len(test_match_pairs)} | Test Mismatch: {len(test_mismatch_pairs)}")

# ==========================================
# 3. PHASE A: TRAINING DATA GENERATION (DEGRADED)
# ==========================================
print("\n--- PHASE A: Generating Training Data (Degraded) ---")

X_train_list = []
y_train_list = []
unrecognizable_embeddings_for_centroid = []

training_stats = defaultdict(int)

for i in tqdm(range(len(train_match_pairs)), desc="Processing Train Pairs"):
    image1, image2 = train_match_pairs[i]

    # 1. Get Clean Embeddings (Base)
    orig_enc, _ = get_encoding_from_image(image1, face_detection_method, embedding_cache, f"train_orig_{i}", detector, embedder, device)
    verif_enc, _ = get_encoding_from_image(image2, face_detection_method, embedding_cache, f"train_verif_{i}", detector, embedder, device)

    if orig_enc is None or verif_enc is None:
        continue

    # 2. Apply Degradations
    for deg_fn in degradation_pool:
        for strength in range(min_degradation_strength, max_degradation_strength):
            deg_img = deg_fn(image1.copy(), strength=strength)
            deg_key = f"train_deg_{i}_{deg_fn.__name__}_s{strength}"

            deg_enc, _ = get_encoding_from_image(deg_img, face_detection_method, embedding_cache, deg_key, detector, embedder, device)

            if deg_enc is None:
                training_stats['no_face'] += 1
                continue

            sim = cosine_similarity([verif_enc], [deg_enc])[0][0]

            if sim > verification_threshold:
                X_train_list.append(deg_enc)
                y_train_list.append(1.0)
                training_stats['recognizable'] += 1
            else:
                X_train_list.append(deg_enc)
                y_train_list.append(0.0)
                unrecognizable_embeddings_for_centroid.append(deg_enc)
                training_stats['unrecognizable'] += 1

print(f"Generation Complete. Stats: {dict(training_stats)}")

# ==========================================
# 4. PHASE B: CLASSIFIER TRAINING
# ==========================================
print("\n--- PHASE B: Training Recognizability Classifier ---")

X_train = np.array(X_train_list)
y_train = np.array(y_train_list)

if len(X_train) > 0:
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, shuffle=True)


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


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        loss = self.bce(inputs, targets)
        pt = torch.exp(-loss)
        return (self.alpha * (1 - pt) ** self.gamma * loss).mean()


X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

model = Classifier().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = FocalLoss(alpha=0.7)

batch_size = 256
dataset = TensorDataset(X_tr_t, y_tr_t)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

best_val_acc = 0.0
best_model_path = "results/best_classifier.pt"

print("Training Classifier...")
for epoch in range(50):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_val_t))
        val_acc = ((val_probs > 0.5).float() == y_val_t).float().mean().item()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}: Val Acc = {val_acc:.4f} (Best: {best_val_acc:.4f})")

print(f"Training Finished. Loading Best Model (Acc: {best_val_acc:.4f})...")
model.load_state_dict(torch.load(best_model_path))
model.eval()
print("Classifier Ready (Best Weights Loaded).")

# ==========================================
# 5. PHASE C: UI CENTROID CALCULATION
# ==========================================
print("\n--- PHASE C: Calculating UI Centroid ---")
if len(unrecognizable_embeddings_for_centroid) > 0:
    ui_centroid = np.mean(unrecognizable_embeddings_for_centroid, axis=0)
    print("UI Centroid Calculated.")
else:
    print("Error: No unrecognizable embeddings found.")
    ui_centroid = None

# ==========================================
# 6. PHASE D: TESTING & ANALYSIS DATA COLLECTION
# ==========================================
print("\n--- PHASE D: Testing & Analysis Data Collection ---")

full_analysis_data = []


def process_test_pairs(pair_list, label_val, desc):
    count = 0
    for i in tqdm(range(len(pair_list)), desc=desc):
        img1, img2 = pair_list[i]

        # 1. Get Embeddings & DETECTION PROBS (Updated)
        # Artık det_prob'u da alıyoruz (önceden _ idi)
        enc1, det_prob1 = get_encoding_from_image(img1, face_detection_method, embedding_cache, f"test_{desc}_{i}_1", detector, embedder, device)
        enc2, det_prob2 = get_encoding_from_image(img2, face_detection_method, embedding_cache, f"test_{desc}_{i}_2", detector, embedder, device)

        if enc1 is None or enc2 is None:
            continue

        count += 1

        # 2. Calculate Similarity
        sim = cosine_similarity([enc1], [enc2])[0][0]

        # 3. Calculate ERS
        if ui_centroid is not None:
            ers_score = euclidean(ui_centroid.flatten(), enc1.flatten())
        else:
            ers_score = 0.0

        # 4. Calculate Classifier Probability
        with torch.no_grad():
            enc_norm = enc1 / np.linalg.norm(enc1)
            enc_tensor = torch.tensor([enc_norm], dtype=torch.float32).to(device)
            clf_prob = torch.sigmoid(model(enc_tensor)).item()

        # 5. Handle Detection Prob (Safe unpacking)
        # Bazen liste dönebilir, float olduğundan emin olalım
        if isinstance(det_prob1, (list, np.ndarray)):
            d_prob = det_prob1[0]
        else:
            d_prob = det_prob1

        # 6. Store ALL Data
        full_analysis_data.append({
            'ers': ers_score,
            'clf_prob': clf_prob,  # Bizim Classifier
            'det_prob': d_prob,  # MTCNN Dedektör (YENİ EKLENDİ)
            'similarity': sim,
            'label': label_val
        })
    return count


count_match = process_test_pairs(test_match_pairs, 1, "Match Pairs")
count_mismatch = process_test_pairs(test_mismatch_pairs, 0, "Mismatch Pairs")

print(f"Testing Complete. Collected {len(full_analysis_data)} data points.")

with open(CACHE_PATH, "wb") as f:
    pickle.dump(embedding_cache, f)

# ==========================================
# 7. PHASE E: FINAL COMPARATIVE ANALYSIS
# ==========================================
print("\n--- PHASE E: Method Comparison (ERS vs Classifier vs Detector) ---")


def analyze_adaptive_thresholds(data_dicts, score_key, score_name, higher_is_better=True, num_bins=5):
    df = pd.DataFrame(data_dicts)
    df['score'] = df[score_key]

    # None temizliği (Dedektör bazen None dönebilir)
    df = df.dropna(subset=['score'])

    try:
        df['bin'] = pd.qcut(df['score'], q=num_bins, labels=False, duplicates='drop')
    except:
        df['bin'] = pd.cut(df['score'], bins=num_bins, labels=False)

    unique_bins = sorted(df['bin'].unique())
    if not higher_is_better:
        unique_bins = unique_bins[::-1]

    results = []
    print(f"\nAnalysis for: {score_name}")
    print(f"{'Bin':<5} | {'Range':<15} | {'Opt Thresh':<10} | {'Max F1':<8} | {'FAR'}")
    print("-" * 60)

    for b in unique_bins:
        subset = df[df['bin'] == b]
        if len(subset) < 10 or len(subset['label'].unique()) < 2:
            continue

        y_true = subset['label']
        y_scores = subset['similarity']

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            f1_scores = np.nan_to_num(f1_scores)

        if len(f1_scores) == 0: continue

        best_idx = np.argmax(f1_scores)
        idx = min(best_idx, len(thresholds) - 1)
        best_thresh = thresholds[idx]
        best_f1 = f1_scores[best_idx]

        mismatches = subset[subset['label'] == 0]
        far = 0.0
        if len(mismatches) > 0:
            far = len(mismatches[mismatches['similarity'] >= best_thresh]) / len(mismatches)

        min_s, max_s = subset['score'].min(), subset['score'].max()
        print(f"{b:<5} | {min_s:.2f}-{max_s:.2f}     | {best_thresh:.4f}     | {best_f1:.4f}   | {far:.4f}")
        results.append({'center': (min_s + max_s) / 2, 'thresh': best_thresh})

    if results:
        centers = [r['center'] for r in results]
        threshs = [r['thresh'] for r in results]
        plt.figure(figsize=(6, 4))
        plt.plot(centers, threshs, marker='o')
        plt.title(f"Adaptive Threshold: {score_name}")
        plt.xlabel(score_name)
        plt.ylabel("Optimal Threshold")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"adaptive_{score_key}.png")
        print(f"Graph saved: adaptive_{score_key}.png")


# 1. ERS Analysis
analyze_adaptive_thresholds(full_analysis_data, 'ers', "ERS (UI Distance)", higher_is_better=False)

# 2. Classifier Probability Analysis
analyze_adaptive_thresholds(full_analysis_data, 'clf_prob', "Classifier Probability", higher_is_better=True)

# ... (Kodunun üst kısımları aynen kalıyor) ...

# 2. Classifier Probability Analysis
analyze_adaptive_thresholds(full_analysis_data, 'clf_prob', "Classifier Probability", higher_is_better=True)

# ==============================================================================
# BURAYA YAPIŞTIR: Face Detection Analizi (Eksik Olan Kısım)
# ==============================================================================
print("\n--- Generating Face Detection Correlation Plot ---")

# Veriyi ayıkla (None olanları filtrele)
ers_vals = [d['ers'] for d in full_analysis_data if d['det_prob'] is not None]
det_vals = [d['det_prob'] for d in full_analysis_data if d['det_prob'] is not None]

if len(ers_vals) > 10:
    # Korelasyon hesapla
    corr_det, _ = pearsonr(ers_vals, det_vals)
    print(f"Correlation (ERS vs Face Det Prob): {corr_det:.4f}")

    # Grafiği çiz
    plt.figure(figsize=(8, 6))
    plt.scatter(ers_vals, det_vals, alpha=0.5, s=10)
    plt.title(f"Clean Test Set: ERS vs Face Detection Prob (r={corr_det:.2f})")
    plt.xlabel("ERS (Distance to UI Centroid) [Higher is Better]")
    plt.ylabel("Face Detection Probability")
    plt.grid(True)

    # Dedektör genelde 0.99 verdiği için grafiğin üst kısmına odaklanalım
    plt.ylim(0.5, 1.05)

    plt.savefig("correlation_ers_vs_facedet.png")
    print("Graph saved: correlation_ers_vs_facedet.png")
else:
    print("Not enough data for Face Detection analysis.")
# ==============================================================================

# ==========================================
# 8. PHASE F: STRESS TEST
# ==========================================
# ... (Kodunun geri kalanı aynen devam ediyor) ...


# 3. Face Detection Probability Analysis (Geri Getirilen Kısım)
# Önce korelasyon grafiğini çizelim (Hocanın istediği scatter plot)
print("\n--- Generating Face Detection Correlation Plot ---")
ers_vals = [d['ers'] for d in full_analysis_data if d['det_prob'] is not None]
det_vals = [d['det_prob'] for d in full_analysis_data if d['det_prob'] is not None]

if len(ers_vals) > 10:
    corr_det, _ = pearsonr(ers_vals, det_vals)
    print(f"Correlation (ERS vs Face Det Prob): {corr_det:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(ers_vals, det_vals, alpha=0.5, s=10)
    plt.title(f"Clean Test Set: ERS vs Face Detection Prob (r={corr_det:.2f})")
    plt.xlabel("ERS (Distance to UI Centroid) [Higher is Better]")
    plt.ylabel("Face Detection Probability")
    plt.grid(True)
    plt.ylim(0.5, 1.05)  # Genelde 0.99 olduğu için üst tarafı görelim
    plt.savefig("correlation_ers_vs_facedet.png")
    print("Graph saved: correlation_ers_vs_facedet.png")
else:
    print("Not enough data for Face Detection analysis.")

# ==========================================
# 8. PHASE F: STRESS TEST
# ==========================================
print("\n--- PHASE F: Stress Test (Visualizing Correlation on Degraded Data) ---")
print("Applying degradations to a subset of test data to visualize ERS vs Classifier alignment...")

stress_samples = 200
stress_ers_scores = []
stress_prob_scores = []

for i in tqdm(range(min(stress_samples, len(test_match_pairs))), desc="Running Stress Test"):
    img1, _ = test_match_pairs[i]

    for deg_fn in degradation_pool:
        for strength in range(min_degradation_strength, max_degradation_strength):

            bad_img = deg_fn(img1.copy(), strength=strength)
            enc, _ = get_encoding_from_image(bad_img, face_detection_method, {}, "temp_stress", detector, embedder, device)

            if enc is None: continue

            if ui_centroid is not None:
                ers = euclidean(ui_centroid.flatten(), enc.flatten())
            else:
                ers = 0.0

            with torch.no_grad():
                enc_norm = enc / np.linalg.norm(enc)
                t_tensor = torch.tensor([enc_norm], dtype=torch.float32).to(device)
                prob = torch.sigmoid(model(t_tensor)).item()

            stress_ers_scores.append(ers)
            stress_prob_scores.append(prob)

stress_ers_scores = np.array(stress_ers_scores)
stress_prob_scores = np.array(stress_prob_scores)

if len(stress_ers_scores) > 2:
    corr_stress, _ = pearsonr(stress_ers_scores, stress_prob_scores)
    print(f"\nSTRESS TEST RESULTS:")
    print(f"Correlation on Degraded Data: {corr_stress:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(stress_ers_scores, stress_prob_scores, c=stress_prob_scores, cmap='viridis', alpha=0.6)
    plt.colorbar(label="Classifier Probability")
    plt.title(f"Stress Test: ERS vs Classifier Probability\nCorrelation: {corr_stress:.2f}")
    plt.xlabel("ERS Score")
    plt.ylabel("Classifier Probability")
    plt.grid(True, alpha=0.3)
    plt.savefig("stress_test_correlation.png")
    print(f"Stress test graph saved to: stress_test_correlation.png")
else:
    print("Not enough data collected for stress test.")

print("\nALL PHASES COMPLETED.")