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
verification_threshold = 0.25  # Sabit threshold referansı
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
# Not: Train Mismatch pairs eğitimde kullanılmıyor, sadece matchler degrade ediliyor.

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
    # Cache key mantığını koruyoruz
    orig_enc, _ = get_encoding_from_image(image1, face_detection_method, embedding_cache, f"train_orig_{i}", detector, embedder, device)
    verif_enc, _ = get_encoding_from_image(image2, face_detection_method, embedding_cache, f"train_verif_{i}", detector, embedder, device)

    if orig_enc is None or verif_enc is None:
        continue

    # 2. Apply Degradations
    for deg_fn in degradation_pool:
        for strength in range(min_degradation_strength, max_degradation_strength):
            # Degrade Image 1
            deg_img = deg_fn(image1.copy(), strength=strength)
            deg_key = f"train_deg_{i}_{deg_fn.__name__}_s{strength}"

            deg_enc, _ = get_encoding_from_image(deg_img, face_detection_method, embedding_cache, deg_key, detector, embedder, device)

            if deg_enc is None:
                training_stats['no_face'] += 1
                continue

            # 3. Labeling (Based on Cosine Similarity)
            sim = cosine_similarity([verif_enc], [deg_enc])[0][0]

            if sim > verification_threshold:
                # Recognizable
                X_train_list.append(deg_enc)
                y_train_list.append(1.0)
                training_stats['recognizable'] += 1
            else:
                # Unrecognizable
                X_train_list.append(deg_enc)
                y_train_list.append(0.0)
                unrecognizable_embeddings_for_centroid.append(deg_enc)
                training_stats['unrecognizable'] += 1

print(f"Generation Complete. Stats: {dict(training_stats)}")
# ==========================================
# 4. PHASE B: CLASSIFIER TRAINING
# ==========================================
print("\n--- PHASE B: Training Recognizability Classifier ---")

# Data Prep
X_train = np.array(X_train_list)
y_train = np.array(y_train_list)

# Normalize Training Data (CRITICAL)
if len(X_train) > 0:
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)

# Split for Synthetic Validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, shuffle=True)


# Define Model
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


# Convert to Tensor
X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

# Training Configuration
model = Classifier().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = FocalLoss(alpha=0.7)

batch_size = 256
dataset = TensorDataset(X_tr_t, y_tr_t)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- BEST MODEL TRACKING ---
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

    # Validation
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_val_t))
        val_acc = ((val_probs > 0.5).float() == y_val_t).float().mean().item()

    # Save Best Model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        # Optional: Print new best
        # print(f"Epoch {epoch+1}: New Best Acc: {best_val_acc:.4f}")

    # Log Progress
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
# 6. PHASE D: TESTING (CLEAN DATA - MATCH & MISMATCH)
# ==========================================
print("\n--- PHASE D: Testing & Analysis Data Collection ---")

# Bu liste her şeyi tutacak: ERS, Prob, Sim, Label
full_analysis_data = []


def process_test_pairs(pair_list, label_val, desc):
    count = 0
    for i in tqdm(range(len(pair_list)), desc=desc):
        img1, img2 = pair_list[i]

        # 1. Get Embeddings (Clean)
        enc1, _ = get_encoding_from_image(img1, face_detection_method, embedding_cache, f"test_{desc}_{i}_1", detector, embedder, device)
        enc2, _ = get_encoding_from_image(img2, face_detection_method, embedding_cache, f"test_{desc}_{i}_2", detector, embedder, device)

        if enc1 is None or enc2 is None:
            continue

        count += 1

        # 2. Calculate Similarity (Standard)
        sim = cosine_similarity([enc1], [enc2])[0][0]

        # 3. Calculate ERS (Distance to UI Centroid)
        # UI Centroid raw embeddingler üzerinden hesaplandıysa raw kullanıyoruz.
        if ui_centroid is not None:
            ers_score = euclidean(ui_centroid.flatten(), enc1.flatten())
        else:
            ers_score = 0.0

        # 4. Calculate Classifier Probability
        # Classifier Normalize edilmiş input bekler!
        with torch.no_grad():
            # Enc1 (Probe) için skor üretiyoruz
            enc_norm = enc1 / np.linalg.norm(enc1)
            enc_tensor = torch.tensor([enc_norm], dtype=torch.float32).to(device)
            prob_score = torch.sigmoid(model(enc_tensor)).item()

        # 5. Store Data
        full_analysis_data.append({
            'ers': ers_score,
            'prob': prob_score,
            'similarity': sim,
            'label': label_val
        })
    return count


# Run Loops
count_match = process_test_pairs(test_match_pairs, 1, "Match Pairs")
count_mismatch = process_test_pairs(test_mismatch_pairs, 0, "Mismatch Pairs")

print(f"Testing Complete. Collected {len(full_analysis_data)} data points.")

# Save Cache Finally
with open(CACHE_PATH, "wb") as f:
    pickle.dump(embedding_cache, f)

# ==========================================
# 7. PHASE E: FINAL COMPARATIVE ANALYSIS
# ==========================================
print("\n--- PHASE E: Method Comparison (ERS vs Classifier) ---")


def analyze_adaptive_thresholds(data_dicts, score_key, score_name, higher_is_better=True, num_bins=5):
    """
    Quality Score ile F1 Score arasındaki ilişkiyi analiz eder.
    """
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(data_dicts)

    # Score sütununu seç
    df['score'] = df[score_key]

    # Binning
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

        # FAR Calculation
        mismatches = subset[subset['label'] == 0]
        far = 0.0
        if len(mismatches) > 0:
            far = len(mismatches[mismatches['similarity'] >= best_thresh]) / len(mismatches)

        min_s, max_s = subset['score'].min(), subset['score'].max()
        print(f"{b:<5} | {min_s:.2f}-{max_s:.2f}     | {best_thresh:.4f}     | {best_f1:.4f}   | {far:.4f}")

        results.append({'center': (min_s + max_s) / 2, 'thresh': best_thresh})

    # Plot
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


# 1. ERS Analysis (Lower is Better -> Distance)
analyze_adaptive_thresholds(full_analysis_data, 'ers', "ERS (UI Distance)", higher_is_better=False)

# 2. Classifier Probability Analysis (Higher is Better)
analyze_adaptive_thresholds(full_analysis_data, 'prob', "Classifier Probability", higher_is_better=True)

# 3. Correlation Check
ers_list = [d['ers'] for d in full_analysis_data]
prob_list = [d['prob'] for d in full_analysis_data]

if np.std(prob_list) > 1e-9:
    corr, _ = pearsonr(ers_list, prob_list)
    print(f"\nCorrelation (ERS vs Prob): {corr:.4f}")
else:
    print("\nNo variance in probabilities (Clean data ceiling effect).")

print("\nDONE.")

# ==========================================
# 8. PHASE F: STRESS TEST (FINAL PROOF)
# ==========================================
print("\n--- PHASE F: Stress Test (Visualizing Correlation on Degraded Data) ---")
print("Applying degradations to a subset of test data to visualize ERS vs Classifier alignment...")

# 1. Config for Stress Test
stress_samples = 200  # Number of pairs to test
stress_ers_scores = []
stress_prob_scores = []

# 2. Loop through a subset of test pairs
# We use 'test_match_pairs' because we know they are the same person
for i in tqdm(range(min(stress_samples, len(test_match_pairs))), desc="Running Stress Test"):
    img1, _ = test_match_pairs[i]

    # Apply a mix of degradations to ensure variance
    # We cycle through degradations and strengths
    for deg_fn in degradation_pool:
        for strength in range(min_degradation_strength, max_degradation_strength):

            bad_img = deg_fn(img1.copy(), strength=strength)

            # 3. Get Embedding
            # Note: We use a temp cache key so we don't pollute the main cache
            enc, _ = get_encoding_from_image(bad_img, face_detection_method, {}, "temp_stress", detector, embedder, device)

            if enc is None:
                continue

            # 4. Calculate ERS (Unsupervised)
            if ui_centroid is not None:
                ers = euclidean(ui_centroid.flatten(), enc.flatten())
            else:
                ers = 0.0

            # 5. Calculate Classifier Probability (Supervised)
            with torch.no_grad():
                enc_norm = enc / np.linalg.norm(enc)  # Normalize!
                t_tensor = torch.tensor([enc_norm], dtype=torch.float32).to(device)
                prob = torch.sigmoid(model(t_tensor)).item()

            stress_ers_scores.append(ers)
            stress_prob_scores.append(prob)

# 3. Analysis & Plotting
stress_ers_scores = np.array(stress_ers_scores)
stress_prob_scores = np.array(stress_prob_scores)

if len(stress_ers_scores) > 2:
    # Correlation
    corr_stress, _ = pearsonr(stress_ers_scores, stress_prob_scores)
    print(f"\nSTRESS TEST RESULTS:")
    print(f"Correlation on Degraded Data: {corr_stress:.4f}")

    # Plot
    plt.figure(figsize=(8, 6))
    # Color points by probability to make it look nicer
    plt.scatter(stress_ers_scores, stress_prob_scores, c=stress_prob_scores, cmap='viridis', alpha=0.6)
    plt.colorbar(label="Classifier Probability")

    plt.title(f"Stress Test: ERS vs Classifier Probability\nCorrelation: {corr_stress:.2f}")
    plt.xlabel("ERS Score (Distance to UI Centroid) [Higher=Better]")
    plt.ylabel("Classifier Probability [Higher=Better]")
    plt.grid(True, alpha=0.3)

    save_path = "stress_test_correlation.png"
    plt.savefig(save_path)
    print(f"Stress test graph saved to: {save_path}")

    # Interpretation Message
    if corr_stress > 0.6:
        print("\nSUCCESS: High correlation detected!")
        print("This proves that when images are actually degraded, your Classifier and ERS agree significantly.")
        print("This graph is the visual proof that your classifier learned the concept of quality.")
    else:
        print("\nNOTE: Correlation is moderate. Check if 'ui_centroid' was calculated correctly.")

else:
    print("Not enough data collected for stress test.")

print("\nALL PHASES COMPLETED.")