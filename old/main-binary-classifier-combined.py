import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from degradations import degradation_pool
from utility import get_encoding_from_image, save_image, ensure_rgb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = MTCNN(image_size=160, margin=0, device=device)
embedder = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

os.makedirs("../output/recognizable", exist_ok=True)
os.makedirs("../output/unrecognizable", exist_ok=True)
os.makedirs("../output/no_embedding", exist_ok=True)

verification_threshold = 0.25
ers_threshold = 1

min_degradation_strength = 1
max_degradation_strength = 6

face_detection_method = "facenet_pytorch"  # "MTCNN"  # "deepface" # "facenet_pytorch"

save_degraded_images = False
filter_with_ERS = False

print("Loading LFW dataset...")

lfw_pairs_train = fetch_lfw_pairs(subset='train', color=True, resize=1.0)
train_pairs = lfw_pairs_train.pairs
train_labels = lfw_pairs_train.target

lfw_pairs_test = fetch_lfw_pairs(subset='test', color=True, resize=1.0)
test_pairs = lfw_pairs_test.pairs
test_labels = lfw_pairs_test.target

train_pairs = [(ensure_rgb(img1), ensure_rgb(img2)) for (img1, img2) in train_pairs]
test_pairs = [(ensure_rgb(img1), ensure_rgb(img2)) for (img1, img2) in test_pairs]

train_match_pairs = [pair for pair, label in zip(train_pairs, train_labels) if label == 1]
train_mismatch_pairs = [pair for pair, label in zip(train_pairs, train_labels) if label == 0]

test_match_pairs = [pair for pair, label in zip(test_pairs, test_labels) if label == 1]
test_mismatch_pairs = [pair for pair, label in zip(test_pairs, test_labels) if label == 0]

training_match_identity_count = len(train_match_pairs)
training_mismatch_identity_count = len(train_mismatch_pairs)
test_match_identity_count = len(test_match_pairs)
test_mismatch_identity_count = len(test_mismatch_pairs)

print("\nLFW dataset loaded.")

CACHE_PATH = "embedding_cache.pkl"

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        embedding_cache = pickle.load(f)
else:
    embedding_cache = {}

separated_caches = {fn.__name__: {} for fn in degradation_pool}

for key, value in embedding_cache.items():
    for fn in degradation_pool:
        name = fn.__name__
        if f"_{name}_" in key:
            separated_caches[name][key] = value
            break

for name, sub_cache in separated_caches.items():
    path = f"embedding_cache_{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(sub_cache, f)
    print(f"Saved {len(sub_cache)} items to {path}")

training_match_no_face_images_statistics = {fn.__name__: 0 for fn in degradation_pool}
training_match_recognizable_images_statistics = {fn.__name__: 0 for fn in degradation_pool}
training_match_unrecognizable_images_statistics = {fn.__name__: 0 for fn in degradation_pool}

training_match_total_count = 0
training_match_with_face_total_count = 0
training_match_no_face_count = 0
training_match_recognizable_count = 0
training_match_unrecognizable_count = 0

recognizable_training_match_images = []
unrecognizable_training_match_images = []

all_original_training_match_embeddings = []
all_training_match_embeddings = []

training_match_group_index = 0

all_original_match_training_labels = []
all_training_match_labels = []

for i in tqdm(range(training_match_identity_count)):
    image1, image2 = train_match_pairs[i]
    original_embedding, original_embedding_detection_prob = get_encoding_from_image(image1, face_detection_method, embedding_cache, f"training_match_original_{i}", detector, embedder, device)
    verification_embedding, verification_embedding_detection_prob = get_encoding_from_image(image2, face_detection_method, embedding_cache, f"training_match_verification_{i}", detector, embedder, device)
    if original_embedding is None or verification_embedding is None:
        continue

    training_match_group_id = f"group_{training_match_group_index}"
    training_match_group_index += 1

    all_training_match_embeddings.append(original_embedding)
    all_training_match_labels.append(training_match_group_id)

    all_original_training_match_embeddings.append(original_embedding)

    for degradation_fn in degradation_pool:
        for strength in range(min_degradation_strength, max_degradation_strength):
            # Degrade image1 and compare with image2
            degraded_img_1 = degradation_fn(image1.copy(), strength=strength)
            degraded_enc_1, degraded_enc_1_detection_prob = get_encoding_from_image(degraded_img_1, face_detection_method, embedding_cache, f"training_match_degraded_{i}_1_{degradation_fn.__name__}_s{strength}", detector, embedder, device)
            training_match_total_count += 1
            if degraded_enc_1 is None:
                training_match_no_face_images_statistics[degradation_fn.__name__] += 1
                training_match_no_face_count += 1
                if save_degraded_images:
                    save_image(degraded_img_1, "../output/no_embedding", f"img_{i}_1_{degradation_fn.__name__}_s{strength}.png")
                continue

            all_training_match_embeddings.append(degraded_enc_1)
            all_training_match_labels.append(training_match_group_id)

            similarity_1 = cosine_similarity([verification_embedding], [degraded_enc_1])[0][0]

            if similarity_1 > verification_threshold:
                recognizable_training_match_images.append((degraded_img_1, degraded_enc_1, image1, original_embedding))
                training_match_recognizable_images_statistics[degradation_fn.__name__] += 1
                training_match_recognizable_count += 1
                if save_degraded_images:
                    save_image(degraded_img_1, "../output/recognizable", f"img_{i}_1_{degradation_fn.__name__}_s{strength}_sim{similarity_1:.2f}.png")
            else:
                unrecognizable_training_match_images.append((degraded_img_1, degraded_enc_1, image1, original_embedding))
                training_match_unrecognizable_images_statistics[degradation_fn.__name__] += 1
                training_match_unrecognizable_count += 1
                if save_degraded_images:
                    save_image(degraded_img_1, "../output/unrecognizable", f"img_{i}_1_{degradation_fn.__name__}_s{strength}_sim{similarity_1:.2f}.png")

            training_match_with_face_total_count += 1

            # Degrade image2 and compare with image1
            degraded_img_2 = degradation_fn(image2.copy(), strength=strength)
            degraded_enc_2, degraded_end_2_detection_prob = get_encoding_from_image(degraded_img_2, face_detection_method, embedding_cache, f"training_match_degraded_{i}_2_{degradation_fn.__name__}_s{strength}", detector, embedder, device)
            training_match_total_count += 1
            if degraded_enc_2 is None:
                training_match_no_face_images_statistics[degradation_fn.__name__] += 1
                training_match_no_face_count += 1
                if save_degraded_images:
                    save_image(degraded_img_2, "../output/no_embedding", f"img_{i}_2_{degradation_fn.__name__}_s{strength}.png")
                continue

            all_training_match_embeddings.append(degraded_enc_2)
            all_training_match_labels.append(training_match_group_id)

            similarity_2 = cosine_similarity([original_embedding], [degraded_enc_2])[0][0]

            if similarity_2 > verification_threshold:
                recognizable_training_match_images.append((degraded_img_2, degraded_enc_2, image2, verification_embedding))
                training_match_recognizable_images_statistics[degradation_fn.__name__] += 1
                training_match_recognizable_count += 1
                if save_degraded_images:
                    save_image(degraded_img_2, "../output/recognizable", f"img_{i}_2_{degradation_fn.__name__}_s{strength}_sim{similarity_2:.2f}.png")
            else:
                unrecognizable_training_match_images.append((degraded_img_2, degraded_enc_2, image2, verification_embedding))
                training_match_unrecognizable_images_statistics[degradation_fn.__name__] += 1
                training_match_unrecognizable_count += 1
                if save_degraded_images:
                    save_image(degraded_img_2, "../output/unrecognizable", f"img_{i}_2_{degradation_fn.__name__}_s{strength}_sim{similarity_2:.2f}.png")

            training_match_with_face_total_count += 1

print("\nTraining match set statistics:")
print("Total images:", training_match_total_count)
print("Total images with faces processed:", training_match_with_face_total_count)
print("No face detections:", training_match_no_face_count)
print("Recognizable face detections:", training_match_recognizable_count)
print("Unrecognizable face detections:", training_match_unrecognizable_count)
print("Recognizable to total with faces ratio:", training_match_recognizable_count / training_match_with_face_total_count)
print("Recognizable to total ratio:", training_match_recognizable_count / training_match_total_count)
print("Unrecognizable to total with faces ratio:", training_match_unrecognizable_count / training_match_with_face_total_count)
print("Unrecognizable to total ratio:", training_match_unrecognizable_count / training_match_total_count)
print("No face to total ratio:", training_match_no_face_count / training_match_total_count)

print("\nDegradations causing 'no face' detections:")
for fn_name, count in training_match_no_face_images_statistics.items():
    print(f"{fn_name}: {count} times")
print("\nDegradations causing 'recognizable face' detections:")
for fn_name, count in training_match_recognizable_images_statistics.items():
    print(f"{fn_name}: {count} times")
print("\nDegradations causing 'unrecognizable face' detections:")
for fn_name, count in training_match_unrecognizable_images_statistics.items():
    print(f"{fn_name}: {count} times")

print("\nStarting UI centroid calculation...")

# Use list comprehension and filter out None
embeddings_array = np.array([enc for _, enc, _, _ in unrecognizable_training_match_images if enc is not None])

if embeddings_array.size > 0:
    ui_centroid = np.mean(embeddings_array, axis=0)
    print("UI centroid calculation completed.")
else:
    print("No unrecognizable embeddings available to calculate centroid.")
    ui_centroid = None

print("\nStarting ERS-filtered verification on match test set with mean UI Centroid...")

match_ers_filter_total_count = 0
match_ers_filter_success_count = 0
match_ers_filter_fail_count = 0
match_ers_filtered_out_count = 0

match_test_with_face_total_count = 0
match_test_total_count = 0
match_test_no_face_count = 0
match_test_recognizable_count = 0
match_test_unrecognizable_count = 0

total_degraded_img_count = 0

all_original_match_test_match_embeddings = []
degraded_test_match_images = []
recognizable_test_match_images = []
unrecognizable_test_match_images = []
ers_removed_test_match_images = []
ers_filtered_test_match_images = []

all_test_match_ers_and_cosine_similarities = []

match_ers_scores = []
match_detection_probs = []

if ui_centroid is not None:
    print("\nStarting ERS-filtered verification on CLEAN match test set...")
    for i in tqdm(range(test_match_identity_count)):
        image1, image2 = test_match_pairs[i]

        # We already loaded original_embedding and verification_embedding at the top of the loop
        # But in your script, you load them inside the loop. Let's stick to your structure:

        original_embedding, original_embedding_detection_prob = get_encoding_from_image(image1, face_detection_method, embedding_cache, f"test_match_original_{i}", detector, embedder, device)
        verification_embedding, verification_embedding_detection_prob = get_encoding_from_image(image2, face_detection_method, embedding_cache, f"test_match_verification_{i}", detector, embedder, device)

        if original_embedding is None or verification_embedding is None:
            continue

        match_test_total_count += 1
        match_test_with_face_total_count += 1

        # Use the CLEAN original embedding as the "Probe"
        encoding = original_embedding

        # Calculate scores
        distance = euclidean(ui_centroid.flatten(), encoding.flatten())
        similarity = cosine_similarity([verification_embedding], [encoding])[0][0]

        match_ers_scores.append(distance)
        match_detection_probs.append(original_embedding_detection_prob)
        all_test_match_ers_and_cosine_similarities.append((distance, similarity, 'S'))

        # Standard Evaluation Logic
        if filter_with_ERS and distance < ers_threshold:
            match_ers_filtered_out_count += 1
            continue  # Filtered out

        # If we pass filter (or filter is off):
        match_ers_filter_total_count += 1

        if similarity > verification_threshold:
            match_ers_filter_success_count += 1
            # SAVE FOR VALIDATION, NOT TRAIN
            recognizable_test_match_images.append((image1, encoding, image1, original_embedding))
        else:
            match_ers_filter_fail_count += 1
            unrecognizable_test_match_images.append((image1, encoding, image1, original_embedding))

print("\nMatch test set statistics:")
print("Total images:", match_test_total_count)
print("Total images with faces processed:", match_test_with_face_total_count)
print("No face detections:", match_test_no_face_count)
print("No face to total ratio:", match_test_no_face_count / match_test_total_count)
print("Successful verifications with ERS filtered images:", match_ers_filter_success_count)
print("Unsuccessful verifications with ERS filtered images:", match_ers_filter_fail_count)
print("Successful verification ratio with ERS filtered images:", match_ers_filter_success_count / match_ers_filter_total_count)
print("Total images that were filtered out by ERS:", match_ers_filtered_out_count)
print("Min ERS score:", min(match_ers_scores))
print("Max ERS score:", max(match_ers_scores))
print("Average ERS score:", sum(match_ers_scores) / len(match_ers_scores))

mismatch_ers_filter_total_count = 0
mismatch_ers_filter_success_count = 0
mismatch_ers_filter_fail_count = 0
mismatch_ers_filtered_out_count = 0

mismatch_test_with_face_total_count = 0
mismatch_test_total_count = 0
mismatch_test_no_face_count = 0
mismatch_test_recognizable_count = 0
mismatch_test_unrecognizable_count = 0

degraded_test_mismatch_images = []
recognizable_test_mismatch_images = []
unrecognizable_test_mismatch_images = []
ers_removed_test_mismatch_images = []
ers_filtered_test_mismatch_images = []

all_test_mismatch_ers_and_cosine_similarities = []

all_test_mismatch_embeddings_for_tsne = []
mismatch_labels_for_tsne = []
mismatch_group_index_for_tsne = 0

mismatch_ers_scores = []
mismatch_detection_probs = []

print("\nStarting ERS-filtered verification on mismatch test set")

if ui_centroid is not None:
    print("\nStarting ERS-filtered verification on CLEAN match test set...")
    for i in tqdm(range(test_mismatch_identity_count)):
        image1, image2 = test_mismatch_pairs[i]

        # We already loaded original_embedding and verification_embedding at the top of the loop
        # But in your script, you load them inside the loop. Let's stick to your structure:

        original_embedding, original_embedding_detection_prob = get_encoding_from_image(image1, face_detection_method, embedding_cache, f"test_mismatch_original_{i}", detector, embedder, device)
        verification_embedding, verification_embedding_detection_prob = get_encoding_from_image(image2, face_detection_method, embedding_cache, f"test_mismatch_verification_{i}", detector, embedder, device)

        if original_embedding is None or verification_embedding is None:
            continue

        mismatch_test_total_count += 1
        mismatch_test_with_face_total_count += 1

        # Use the CLEAN original embedding as the "Probe"
        encoding = original_embedding

        # Calculate scores
        distance = euclidean(ui_centroid.flatten(), encoding.flatten())
        similarity = cosine_similarity([verification_embedding], [encoding])[0][0]

        mismatch_ers_scores.append(distance)
        mismatch_detection_probs.append(original_embedding_detection_prob)
        all_test_mismatch_ers_and_cosine_similarities.append((distance, similarity, 'F'))

        # Standard Evaluation Logic
        if filter_with_ERS and distance < ers_threshold:
            mismatch_ers_filtered_out_count += 1
            continue  # Filtered out

        # If we pass filter (or filter is off):
        mismatch_ers_filter_total_count += 1

        if similarity > verification_threshold:
            mismatch_ers_filter_success_count += 1
            # SAVE FOR VALIDATION, NOT TRAIN
            recognizable_test_mismatch_images.append((image1, encoding, image1, original_embedding))
        else:
            mismatch_ers_filter_fail_count += 1
            unrecognizable_test_mismatch_images.append((image1, encoding, image1, original_embedding))

# Go through main cache and route entries to the appropriate degradation file
for key, value in embedding_cache.items():
    for fn in degradation_pool:
        name = fn.__name__
        if f"_{name}_" in key:
            separated_caches[name][key] = value
            break  # Skip once matched

# Save each separated cache
for name, sub_cache in separated_caches.items():
    path = f"embedding_cache_{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(sub_cache, f)
    print(f"Saved {len(sub_cache)} items to {path}")

# ==========================================
# 1. DATA CREATION (LEAKAGE-FREE & NORMALIZED)
# ==========================================
print("\nBuilding Binary Classification Data...")

# A) TRAINING DATA (Strictly from Training Split - Degraded)
# Model "kötü görüntü" kavramını sadece buradan öğrenecek.
X_train = []
y_train = []

# Recognizable (from Training Loop)
for _, encoding, _, _ in recognizable_training_match_images:
    X_train.append(encoding)
    y_train.append(1.0)  # Label 1.0 = Recognizable

# Unrecognizable (from Training Loop)
for _, encoding, _, _ in unrecognizable_training_match_images:
    X_train.append(encoding)
    y_train.append(0.0)  # Label 0.0 = Unrecognizable

X_train = np.array(X_train)
y_train = np.array(y_train)

# B) VALIDATION/TEST DATA (Strictly from Test Split - Clean/Real)
# Modelin başarısını ölçmek ve Olasılık (Probability) üretmek için.
X_val_real = []
y_val_real = []

# Test setindeki başarılı/başarısız tüm embeddingleri sırasıyla topluyoruz.
# Bu liste, ileride ERS ile Classifier Score'u karşılaştırmak için kritik.
all_test_embeddings_ordered = []

# Match Test Loop'tan gelenler
for _, encoding, _, _ in recognizable_test_match_images:
    X_val_real.append(encoding)
    y_val_real.append(1.0)
    all_test_embeddings_ordered.append(encoding)

for _, encoding, _, _ in unrecognizable_test_match_images:
    X_val_real.append(encoding)
    # Test verisi temiz olduğu için aslında "Recognizable" olmalıydı (Label 1.0)
    # Ama similarity düşük çıktığı için eşleşemedi. Yine de görüntü kalitesi 1.0.
    y_val_real.append(1.0)
    all_test_embeddings_ordered.append(encoding)

# Mismatch Test Loop'tan gelenler (Opsiyonel, veri setini büyütür)
for _, encoding, _, _ in recognizable_test_mismatch_images:
    X_val_real.append(encoding)
    y_val_real.append(1.0)
    all_test_embeddings_ordered.append(encoding)

for _, encoding, _, _ in unrecognizable_test_mismatch_images:
    X_val_real.append(encoding)
    y_val_real.append(1.0)
    all_test_embeddings_ordered.append(encoding)

X_val_real = np.array(X_val_real)
y_val_real = np.array(y_val_real)

# --- CRITICAL FIX: NORMALIZATION ---
# L2 Normalization (Cosine Similarity Space)
if len(X_train) > 0:
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
if len(X_val_real) > 0:
    X_val_real = X_val_real / np.linalg.norm(X_val_real, axis=1, keepdims=True)

print(f"Training Data Shape: {X_train.shape}")
print(f"Test/Val Data Shape: {X_val_real.shape}")

# Save Data (Fixing the pickle error by saving/loading consistently)
with open("binary_classifier_data.pkl", "wb") as f:
    pickle.dump((X_train, y_train, X_val_real, y_val_real), f)

with open(CACHE_PATH, "wb") as f:
    pickle.dump(embedding_cache, f)


# ==========================================
# 2. CLASSIFIER MODEL DEFINITION
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets_smooth)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class Classifier(nn.Module):
    def __init__(self, dropout, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 1)  # Output: Logits
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. TRAINING THE CLASSIFIER
# ==========================================
# Load Data (Fixing the unpacking error)
with open("binary_classifier_data.pkl", "rb") as f:
    X_train, y_train, X_val_real, y_val_real = pickle.load(f)

# Split Training Data for Synthetic Validation (80/20)
X_train_split, X_val_syn, y_train_split, y_val_syn = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, shuffle=True
)

# Convert to Tensors
X_tensor = torch.tensor(X_train_split, dtype=torch.float32)
y_tensor = torch.tensor(y_train_split, dtype=torch.float32).unsqueeze(1)

val_syn_X_tensor = torch.tensor(X_val_syn, dtype=torch.float32).to(device)
val_syn_y_tensor = torch.tensor(y_val_syn, dtype=torch.float32).unsqueeze(1).to(device)

val_real_X_tensor = torch.tensor(X_val_real, dtype=torch.float32).to(device)
val_real_y_tensor = torch.tensor(y_val_real, dtype=torch.float32).unsqueeze(1).to(device)

# Training Config
dataset = TensorDataset(X_tensor, y_tensor)
batch_size = 256
lr = 1e-4
epochs = 100

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
model = Classifier(dropout=0.4).to(device)

if len(y_train_split) > 0 and sum(y_train_split) > 0:
    pos_weight_val = (len(y_train_split) - sum(y_train_split)) / sum(y_train_split)
else:
    pos_weight_val = 1.0

pos_weight = torch.tensor([pos_weight_val]).to(device)
criterion = FocalLoss(alpha=pos_weight.item(), gamma=1, smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

best_val_acc = 0.0
best_model_state = None

print(f"\nStarting Classifier Training on {device}...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    # Validation (Synthetic)
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        val_logits = model(val_syn_X_tensor)
        val_preds = (torch.sigmoid(val_logits) > 0.5).float()
        val_correct += (val_preds == val_syn_y_tensor).sum().item()
        val_total += val_syn_y_tensor.size(0)

    val_acc = val_correct / val_total if val_total > 0 else 0

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{epochs} | Val Acc (Syn): {val_acc:.4f}")

print(f"Training Finished. Best Synthetic Val Acc: {best_val_acc:.4f}")

# ==========================================
# 4. GENERATING RECOGNIZABILITY SCORES (PROBABILITIES)
# ==========================================
print("\n--- Generating 'Recognizability Probabilities' for Test Set ---")

# Load best model
if best_model_state:
    model.load_state_dict(best_model_state)
model.eval()

# Generate Probabilities for the REAL test set
with torch.no_grad():
    test_logits = model(val_real_X_tensor)
    # Apply Sigmoid to get 0.0 - 1.0 Score
    test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()

print(f"Generated {len(test_probs)} probability scores.")
print(f"Avg Score: {np.mean(test_probs):.4f} (Should be high for clean data)")

# ==========================================
# 5. METHODOLOGY COMPARISON (ERS vs CLASSIFIER SCORE)
# ==========================================
if ui_centroid is not None and len(all_test_embeddings_ordered) > 0:
    print("\nCalculating Correlation: Classifier Probability vs ERS Score...")

    ers_scores_list = []
    # Using raw embeddings for Euclidean Distance (Standard ERS definition)
    # If you prefer normalized, use X_val_real here.
    temp_embeddings = np.array(all_test_embeddings_ordered)

    for emb in temp_embeddings:
        dist = euclidean(ui_centroid.flatten(), emb.flatten())
        ers_scores_list.append(dist)

    ers_scores_list = np.array(ers_scores_list)

    # Check Variance
    if np.std(test_probs) > 1e-9:
        corr_method, p_val_method = pearsonr(ers_scores_list, test_probs)
        print(f"Correlation (ERS vs Classifier Prob): {corr_method:.4f}")

        plt.figure(figsize=(8, 6))
        plt.scatter(ers_scores_list, test_probs, alpha=0.3, s=10)
        plt.title(f"Method Comparison: Unsupervised ERS vs Supervised Classifier\nCorrelation: {corr_method:.2f}")
        plt.xlabel("ERS Score (Distance to UI Centroid)")
        plt.ylabel("Classifier Probability Score")
        plt.grid(True)
        plt.savefig("comparison_ers_vs_classifier_prob.png")
        plt.close()
        print("Comparison plot saved.")
    else:
        print("Warning: Classifier scores have 0 variance (all 1.0). Correlation skipped.")

