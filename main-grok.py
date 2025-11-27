import csv
import os
import pickle
import numpy as np
import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from degradations import degradation_pool
from plotter import tsne_training_recognizable_unrecognizable_original_images_having_different_colors, tsne_test_images_originated_from_same_original_image_having_same_colors, plot_binned_bar_chart, plot_ers_similarity_scatter, plot_roc_curve, plot_ers_similarity_binned, tsne_with_clustered_ui_centroids, tsne_with_clustered_ui_centroids_hdbscan
from utility import get_encoding_from_image, save_image, ensure_rgb
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch import optim
from sklearn.metrics import precision_score, recall_score, f1_score

# Device and Model Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = MTCNN(image_size=160, margin=0, device=device)
embedder = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

os.makedirs("output/recognizable", exist_ok=True)
os.makedirs("output/unrecognizable", exist_ok=True)
os.makedirs("output/no_embedding", exist_ok=True)

verification_threshold = 0.3
ers_threshold = 1
training_match_identity_count = 100
training_mismatch_identity_count = 100
test_match_identity_count = 100
test_mismatch_identity_count = 100
min_degradation_strength = 1
max_degradation_strength = 6
face_detection_method = "facenet_pytorch"
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

# Training Match Set Processing
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
    original_embedding = get_encoding_from_image(image1, face_detection_method, embedding_cache, f"training_match_original_{i}", detector, embedder, device)
    verification_embedding = get_encoding_from_image(image2, face_detection_method, embedding_cache, f"training_match_verification_{i}", detector, embedder, device)
    if original_embedding is None or verification_embedding is None:
        continue
    training_match_group_id = f"group_{training_match_group_index}"
    training_match_group_index += 1
    all_training_match_embeddings.append(original_embedding)
    all_training_match_labels.append(training_match_group_id)
    all_original_training_match_embeddings.append(original_embedding)
    for degradation_fn in degradation_pool:
        for strength in range(min_degradation_strength, max_degradation_strength):
            degraded_img = degradation_fn(image1.copy(), strength=strength)
            degraded_enc = get_encoding_from_image(degraded_img, face_detection_method, embedding_cache, f"training_match_degraded_{i}_{degradation_fn.__name__}_s{strength}", detector, embedder, device)
            training_match_total_count += 1
            if degraded_enc is None:
                training_match_no_face_images_statistics[degradation_fn.__name__] += 1
                training_match_no_face_count += 1
                if save_degraded_images:
                    save_image(degraded_img, "output/no_embedding", f"img_{i}_{degradation_fn.__name__}_s{strength}.png")
                continue
            all_training_match_embeddings.append(degraded_enc)
            all_training_match_labels.append(training_match_group_id)
            similarity = cosine_similarity([verification_embedding], [degraded_enc])[0][0]
            if similarity > verification_threshold:
                recognizable_training_match_images.append((degraded_img, degraded_enc, image1, original_embedding))
                training_match_recognizable_images_statistics[degradation_fn.__name__] += 1
                training_match_recognizable_count += 1
                if save_degraded_images:
                    save_image(degraded_img, "output/recognizable", f"img_{i}_{degradation_fn.__name__}_s{strength}_sim{similarity:.2f}.png")
            else:
                unrecognizable_training_match_images.append((degraded_img, degraded_enc, image1, original_embedding))
                training_match_unrecognizable_images_statistics[degradation_fn.__name__] += 1
                training_match_unrecognizable_count += 1
                if save_degraded_images:
                    save_image(degraded_img, "output/unrecognizable", f"img_{i}_{degradation_fn.__name__}_s{strength}_sim{similarity:.2f}.png")
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
all_test_match_embeddings_for_tsne = []
match_labels_for_tsne = []
match_group_index_for_tsne = 0
match_ers_scores = []

if ui_centroid is not None:
    for i in tqdm(range(test_match_identity_count)):
        image1, image2 = test_match_pairs[i]
        original_embedding = get_encoding_from_image(image1, face_detection_method, embedding_cache, f"test_match_original_{i}", detector, embedder, device)
        verification_embedding = get_encoding_from_image(image2, face_detection_method, embedding_cache, f"test_match_verification_{i}", detector, embedder, device)
        if original_embedding is None or verification_embedding is None:
            continue
        group_id = f"group_{match_group_index_for_tsne}"
        match_group_index_for_tsne += 1
        all_test_match_embeddings_for_tsne.append(original_embedding)
        match_labels_for_tsne.append(group_id)
        for degradation_fn in degradation_pool:
            for strength in range(min_degradation_strength, max_degradation_strength):
                degraded_img = degradation_fn(image1.copy(), strength=strength)
                degraded_enc = get_encoding_from_image(degraded_img, face_detection_method, embedding_cache, f"test_match_degraded_{i}_{degradation_fn.__name__}_s{strength}", detector, embedder, device)
                match_test_total_count += 1
                total_degraded_img_count += 1
                if degraded_enc is None:
                    match_test_no_face_count += 1
                    if save_degraded_images:
                        save_image(degraded_img, "output/no_embedding", f"img_{i}_{degradation_fn.__name__}_s{strength}.png")
                    continue
                match_test_with_face_total_count += 1
                all_test_match_embeddings_for_tsne.append(degraded_enc)
                match_labels_for_tsne.append(group_id)
                distance = euclidean(ui_centroid.flatten(), degraded_enc.flatten())
                match_ers_scores.append(distance)
                similarity = cosine_similarity([verification_embedding], [degraded_enc])[0][0]
                all_test_match_ers_and_cosine_similarities.append((distance, similarity, 'S'))
                if filter_with_ERS:
                    if distance > ers_threshold:
                        if similarity > verification_threshold:
                            if save_degraded_images:
                                save_image(degraded_img, "output/recognizable", f"img_{i}_{degradation_fn.__name__}_s{strength}_sim{similarity:.2f}.png")
                            match_ers_filter_success_count += 1
                            recognizable_test_match_images.append((degraded_img, degraded_enc, image1, original_embedding))
                        else:
                            match_ers_filter_fail_count += 1
                            unrecognizable_test_match_images.append((degraded_img, degraded_enc, image1, original_embedding))
                        match_ers_filter_total_count += 1
                    else:
                        match_ers_filtered_out_count += 1
                else:
                    if similarity > verification_threshold:
                        if save_degraded_images:
                            save_image(degraded_img, "output/recognizable", f"img_{i}_{degradation_fn.__name__}_s{strength}_sim{similarity:.2f}.png")
                        match_ers_filter_success_count += 1
                        recognizable_test_match_images.append((degraded_img, degraded_enc, image1, original_embedding))
                    else:
                        match_ers_filter_fail_count += 1
                        unrecognizable_test_match_images.append((degraded_img, degraded_enc, image1, original_embedding))
                    match_ers_filter_total_count += 1
                    match_ers_filtered_out_count += 1

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

# Mismatch Test Set Processing (Truncated for brevity, assume similar structure)
# ... (Add mismatch processing if needed, currently incomplete in your code)

# Binary Classification Dataset
X_train = []
y_train = []
all_recognizable_embeddings = recognizable_training_match_images + recognizable_test_match_images
all_unrecognizable_embeddings = unrecognizable_training_match_images + unrecognizable_test_match_images
for _, degraded_enc, _, _ in all_recognizable_embeddings:
    X_train.append(degraded_enc)
    y_train.append(1)
for _, degraded_enc, _, _ in all_unrecognizable_embeddings:
    X_train.append(degraded_enc)
    y_train.append(0)
X_train = np.array(X_train)
y_train = np.array(y_train)
print("\nBinary classification training dataset created:")
print(f"Total samples: {len(X_train)}")
print(f"Recognizable: {np.sum(y_train == 1)}")
print(f"Unrecognizable: {np.sum(y_train == 0)}")
with open("binary_classifier_data.pkl", "wb") as f:
    pickle.dump((X_train, y_train), f)
with open(CACHE_PATH, "wb") as f:
    pickle.dump(embedding_cache, f)

per_degradation_data = defaultdict(lambda: {"X": [], "y": []})
for (_, degraded_enc, _, _), cache_key in zip(all_recognizable_embeddings, embedding_cache.keys()):
    for fn in degradation_pool:
        if f"_{fn.__name__}_" in cache_key:
            per_degradation_data[fn.__name__]["X"].append(degraded_enc)
            per_degradation_data[fn.__name__]["y"].append(1)
            break
for (_, degraded_enc, _, _), cache_key in zip(all_unrecognizable_embeddings, embedding_cache.keys()):
    for fn in degradation_pool:
        if f"_{fn.__name__}_" in cache_key:
            per_degradation_data[fn.__name__]["X"].append(degraded_enc)
            per_degradation_data[fn.__name__]["y"].append(0)
            break
for degradation_name, data in per_degradation_data.items():
    X = np.array(data["X"])
    y = np.array(data["y"])
    file_path = f"binary_classifier_data_{degradation_name}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump((X, y), f)
    print(f"Saved {len(X)} samples for '{degradation_name}' to {file_path}")

X_train = []
y_train = []
for data in per_degradation_data.values():
    X_train.extend(data["X"])
    y_train.extend(data["y"])
X_train = np.array(X_train)
y_train = np.array(y_train)
with open("binary_classifier_data.pkl", "wb") as f:
    pickle.dump((X_train, y_train), f)

# Visualization (Truncated for brevity)
# ... (Keep your similarity distribution and F1 score calculations)

# Training Setup
X_train_per_deg = {name: [] for name in per_degradation_data.keys()}
y_train_per_deg = {name: [] for name in per_degradation_data.keys()}
for name, data in per_degradation_data.items():
    X_train_per_deg[name] = np.array(data["X"])
    y_train_per_deg[name] = np.array(data["y"])
X_train_split, X_val, y_train_split, y_val = [], [], [], []
for name in X_train_per_deg:
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_train_per_deg[name], y_train_per_deg[name], test_size=0.2, stratify=y_train_per_deg[name], random_state=42
    )
    X_train_split.append(X_tr)
    X_val.append(X_v)
    y_train_split.append(y_tr)
    y_val.append(y_v)
X_train_split = np.concatenate(X_train_split)
X_val = np.concatenate(X_val)
X_train_split = X_train_split / np.linalg.norm(X_train_split, axis=1, keepdims=True)
X_val = X_val / np.linalg.norm(X_val, axis=1, keepdims=True)
y_train_split = np.concatenate(y_train_split)
y_val = np.concatenate(y_val)

train_deg_counts = Counter()
val_deg_counts = Counter()
for name in X_train_per_deg:
    n_samples = len(X_train_per_deg[name])
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_train_per_deg[name], y_train_per_deg[name], test_size=0.2, stratify=y_train_per_deg[name], random_state=42
    )
    train_deg_counts[name] += len(X_tr)
    val_deg_counts[name] += len(X_v)
print("Training perturbation counts:", train_deg_counts)
print("Validation perturbation counts:", val_deg_counts)

label_counts = Counter(y_train)
print(f"Label distribution in the full dataset: {label_counts}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
best_accuracies = []
train_label_counts = Counter(y_train_split)
val_label_counts = Counter(y_val)
print(f"Label distribution in the training set: {train_label_counts}")
print(f"Label distribution in the validation set: {val_label_counts}")

X_tensor = torch.tensor(X_train_split, dtype=torch.float32)
y_tensor = torch.tensor(y_train_split, dtype=torch.float32).unsqueeze(1)
val_X_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
val_y_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
dataset = TensorDataset(X_tensor, y_tensor)
learning_rates = [1e-4]
weight_decays = [1e-4]
dropout_values = [0.5]
batch_sizes = [128]


class Classifier(nn.Module):
    def __init__(self, dropout, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

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

for lr in learning_rates:
    for weight_decay in weight_decays:
        for batch_size in batch_sizes:
            for dropout in dropout_values:
                run_id = f"lr{lr}_wd{weight_decay}_bs{batch_size}_dropout{dropout}"
                print(f"\nStarting run {run_id}")

                epochs = 500
                train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                model = Classifier(dropout).to(device)
                pos_weight = torch.tensor([(len(y_train_split) - sum(y_train_split)) / sum(y_train_split)]).to(device)
                criterion = FocalLoss(alpha=pos_weight.item(), gamma=1.5, smoothing=0.1)
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

                best_val_acc = 0.0
                best_val_loss = float('inf')
                best_train_loss = None
                best_train_acc = None
                best_epoch = -1
                best_model_state = None
                patience = 20
                trigger_times = 0

                train_losses, train_accuracies = [], []
                val_losses, val_accuracies = [], []

                for epoch in range(epochs):
                    model.train()
                    running_loss, correct, total = 0.0, 0, 0
                    for X_batch, y_batch in train_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        noise = torch.normal(mean=0, std=0.05, size=X_batch.shape).to(device)
                        X_batch = X_batch + noise
                        optimizer.zero_grad()
                        logits = model(X_batch)
                        probs = torch.sigmoid(logits)
                        loss = criterion(logits, y_batch)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item() * X_batch.size(0)
                        preds = (probs > 0.5).float()
                        correct += (preds == y_batch).sum().item()
                        total += y_batch.size(0)
                    epoch_loss = running_loss / total
                    epoch_acc = correct / total
                    train_losses.append(epoch_loss)
                    train_accuracies.append(epoch_acc)

                    model.eval()
                    val_running_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    all_val_preds = []
                    all_val_labels = []
                    with torch.no_grad():
                        for val_X_batch, val_y_batch in DataLoader(TensorDataset(val_X_tensor, val_y_tensor), batch_size=batch_size):
                            val_X_batch, val_y_batch = val_X_batch.to(device), val_y_batch.to(device)
                            val_output = model(val_X_batch)
                            val_losses_batch = criterion(val_output, val_y_batch)
                            val_loss_mean = val_losses_batch.mean()
                            val_running_loss += val_loss_mean.item() * val_X_batch.size(0)
                            val_probs = torch.sigmoid(val_output)
                            val_preds = (val_probs > 0.5).float()
                            val_correct += (val_preds == val_y_batch).sum().item()
                            val_total += val_y_batch.size(0)
                            all_val_preds.append(val_preds.cpu())
                            all_val_labels.append(val_y_batch.cpu())
                    val_loss = val_running_loss / val_total
                    val_acc = val_correct / val_total
                    val_losses.append(val_loss)
                    val_accuracies.append(val_acc)
                    all_val_preds = torch.cat(all_val_preds).numpy().astype(int)
                    all_val_labels = torch.cat(all_val_labels).numpy().astype(int)
                    precision = precision_score(all_val_labels, all_val_preds, zero_division=0)
                    recall = recall_score(all_val_labels, all_val_preds, zero_division=0)
                    f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - "
                          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - "
                          f"F1: {f1:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f}")

                    # Early Stopping
                    if val_acc < best_val_acc:
                        best_val_loss = val_loss
                        best_val_acc = val_acc
                        best_train_loss = epoch_loss
                        best_train_acc = epoch_acc
                        best_epoch = epoch + 1
                        best_model_state = model.state_dict()
                        trigger_times = 0
                        print(f"New best model found at epoch {best_epoch} with val loss: {val_acc:.4f}")
                    else:
                        trigger_times += 1
                        if trigger_times >= patience:
                            print(f"Early stopping triggered at epoch {epoch + 1}")

                best_accuracies.append(best_val_acc)
                os.makedirs("results", exist_ok=True)
                if best_model_state is not None:
                    torch.save(best_model_state, f"results/{run_id}_best_model.pt")
                    with open(f"results/{run_id}_best_metrics.txt", "w") as f:
                        f.write(f"Best Epoch: {best_epoch}\n")
                        f.write(f"Train Loss: {best_train_loss:.4f}\n")
                        f.write(f"Train Accuracy: {best_train_acc:.4f}\n")
                        f.write(f"Validation Loss: {best_val_loss:.4f}\n")
                        f.write(f"Validation Accuracy: {best_val_acc:.4f}\n")
                    csv_path = "results/results_summary.csv"
                    file_exists = os.path.isfile(csv_path)
                    with open(csv_path, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        if not file_exists:
                            writer.writerow(["run_id", "best_epoch", "train_accuracy", "train_loss", "val_accuracy", "val_loss"])
                        writer.writerow([
                            run_id,
                            best_epoch,
                            round(best_train_acc, 4),
                            round(best_train_loss, 4),
                            round(best_val_acc, 4),
                            round(best_val_loss, 4)
                        ])
                    print(f"Best model saved with validation accuracy: {best_val_acc:.4f}")

                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(train_losses, label='Train Loss')
                plt.plot(val_losses, label='Validation Loss')
                plt.title("Loss over Epochs")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.subplot(1, 2, 2)
                plt.plot(train_accuracies, label='Train Accuracy')
                plt.plot(val_accuracies, label='Validation Accuracy')
                plt.title("Accuracy over Epochs")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"results/{run_id}_training_curves.png")
                plt.close()
print(best_accuracies)
