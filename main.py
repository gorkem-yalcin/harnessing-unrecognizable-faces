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
from plotter import tsne_training_recognizable_unrecognizable_original_images_having_different_colors, tsne_test_images_originated_from_same_original_image_having_same_colors, plot_binned_bar_chart, plot_ers_similarity_scatter, \
    plot_roc_curve, plot_ers_similarity_binned, tsne_with_clustered_ui_centroids, tsne_with_clustered_ui_centroids_hdbscan
from utility import get_encoding_from_image, save_image, ensure_rgb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = MTCNN(image_size=160, margin=0, device=device)
embedder = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

os.makedirs("output/recognizable", exist_ok=True)
os.makedirs("output/unrecognizable", exist_ok=True)
os.makedirs("output/no_embedding", exist_ok=True)

verification_threshold = 0.75
ers_threshold = 1

training_match_identity_count = 100
training_mismatch_identity_count = 100
test_match_identity_count = 100
test_mismatch_identity_count = 100

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
                        degraded_img.save(f"output/no_embedding/img_{i}_{degradation_fn.__name__}_s{strength}.png", detector, embedder, device)
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

print("\nStarting ERS-filtered verification on mismatch test set")

if ui_centroid is not None:
    for i in tqdm(range(test_mismatch_identity_count)):
        image1, image2 = test_mismatch_pairs[i]

        original_embedding = get_encoding_from_image(image1, face_detection_method, embedding_cache, f"test_mismatch_original_{i}", detector, embedder, device)
        verification_embedding = get_encoding_from_image(image2, face_detection_method, embedding_cache, f"test_mismatch_verification_{i}", detector, embedder, device)

        if original_embedding is None or verification_embedding is None:
            continue

        group_id = f"group_{mismatch_group_index_for_tsne}"
        mismatch_group_index_for_tsne += 1
        all_test_mismatch_embeddings_for_tsne.append(original_embedding)
        mismatch_labels_for_tsne.append(group_id)

        for degradation_fn in degradation_pool:
            for strength in range(min_degradation_strength, max_degradation_strength):
                degraded_img = degradation_fn(image1.copy(), strength=strength)
                degraded_enc = get_encoding_from_image(degraded_img, face_detection_method, embedding_cache, f"test_mismatch_degraded_{i}_{degradation_fn.__name__}_s{strength}", detector, embedder, device)
                degraded_test_mismatch_images.append(degraded_img)
                mismatch_test_total_count += 1

                if degraded_enc is None:
                    mismatch_test_no_face_count += 1
                    if save_degraded_images:
                        degraded_img.save(f"output/no_embedding/img_{i}_{degradation_fn.__name__}_s{strength}.png")
                    continue

                mismatch_test_with_face_total_count += 1

                all_test_mismatch_embeddings_for_tsne.append(degraded_enc)
                mismatch_labels_for_tsne.append(group_id)

                distance = euclidean(ui_centroid.flatten(), degraded_enc.flatten())
                mismatch_ers_scores.append(distance)

                similarity = cosine_similarity([verification_embedding], [degraded_enc])[0][0]

                all_test_mismatch_ers_and_cosine_similarities.append((distance, similarity, 'F'))

                if filter_with_ERS:
                    if distance > ers_threshold:
                        ers_filtered_test_mismatch_images.append(degraded_img)
                        if similarity < verification_threshold:
                            if save_degraded_images:
                                save_image(degraded_img, "output/recognizable", f"img_{i}_{degradation_fn.__name__}_s{strength}_sim{similarity:.2f}.png")
                            mismatch_ers_filter_success_count += 1
                            recognizable_test_mismatch_images.append((degraded_img, degraded_enc, image1, original_embedding))
                        else:
                            mismatch_ers_filter_fail_count += 1
                            unrecognizable_test_mismatch_images.append((degraded_img, degraded_enc, image1, original_embedding))
                        mismatch_ers_filter_total_count += 1
                    else:
                        mismatch_ers_filtered_out_count += 1
                        ers_removed_test_mismatch_images.append(degraded_img)
                else:
                    if similarity < verification_threshold:
                        if save_degraded_images:
                            save_image(degraded_img, "output/recognizable", f"img_{i}_{degradation_fn.__name__}_s{strength}_sim{similarity:.2f}.png")
                        mismatch_ers_filter_success_count += 1
                        recognizable_test_mismatch_images.append((degraded_img, degraded_enc, image1, original_embedding))
                    else:
                        mismatch_ers_filter_fail_count += 1
                        unrecognizable_test_mismatch_images.append((degraded_img, degraded_enc, image1, original_embedding))
                    mismatch_ers_filter_total_count += 1
                    mismatch_ers_filtered_out_count += 1
                    ers_removed_test_mismatch_images.append(degraded_img)

print("\nMismatch test set statistics:")
print("Total images:", mismatch_test_total_count)
print("Total images with faces processed:", mismatch_test_with_face_total_count)
print("No face detections:", mismatch_test_no_face_count)
print("No face to total ratio:", mismatch_test_no_face_count / mismatch_test_total_count)
print("Successful verifications with ERS filtered images:", mismatch_ers_filter_success_count)
print("Unsuccessful verifications with ERS filtered images:", mismatch_ers_filter_fail_count)
print("Successful verification ratio with ERS filtered images:", mismatch_ers_filter_success_count / mismatch_ers_filter_total_count)
print("Total images that were filtered out by ERS:", mismatch_ers_filtered_out_count)
print("Min ERS score:", min(mismatch_ers_scores))
print("Max ERS score:", max(mismatch_ers_scores))
print("Average ERS score:", sum(mismatch_ers_scores) / len(mismatch_ers_scores))

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

# Build binary classification dataset
X_train = []
y_train = []

all_recognizable_embeddings = recognizable_training_match_images + recognizable_test_match_images + recognizable_test_mismatch_images
all_unrecognizable_embeddings = unrecognizable_training_match_images + unrecognizable_test_match_images + unrecognizable_test_mismatch_images

# Label: 1 for recognizable
for _, degraded_enc, _, _ in all_recognizable_embeddings:
    X_train.append(degraded_enc)
    y_train.append(1)

# Label: 0 for unrecognizable
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

# Build dictionary: degradation_name → (X, y)
from collections import defaultdict

per_degradation_data = defaultdict(lambda: {"X": [], "y": []})

# Recognizable
for (_, degraded_enc, _, _), cache_key in zip(all_recognizable_embeddings, embedding_cache.keys()):
    for fn in degradation_pool:
        if f"_{fn.__name__}_" in cache_key:
            per_degradation_data[fn.__name__]["X"].append(degraded_enc)
            per_degradation_data[fn.__name__]["y"].append(1)
            break

# Unrecognizable
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

combined_data = all_test_match_ers_and_cosine_similarities + all_test_mismatch_ers_and_cosine_similarities

all_original_match_embeddings = all_original_training_match_embeddings + all_test_match_embeddings_for_tsne

recognizable_similarities = [cosine_similarity([enc], [verif_enc])[0][0] for _, enc, _, verif_enc in all_recognizable_embeddings]
unrecognizable_similarities = [cosine_similarity([enc], [verif_enc])[0][0] for _, enc, _, verif_enc in all_unrecognizable_embeddings]
plt.hist(recognizable_similarities, bins=50, alpha=0.5, label="Recognizable")
plt.hist(unrecognizable_similarities, bins=50, alpha=0.5, label="Unrecognizable")
plt.axvline(0.225, color='red', linestyle='--', label='Threshold')
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.legend()
plt.show()
plt.savefig("similarity_distribution.png")
plt.close()

from sklearn.metrics import f1_score

thresholds = [0.2, 0.225, 0.25, 0.3]
for thresh in thresholds:
    y_pred = [1 if sim > thresh else 0 for sim in recognizable_similarities + unrecognizable_similarities]
    y_true = [1] * len(recognizable_similarities) + [0] * len(unrecognizable_similarities)
    f1 = f1_score(y_true, y_pred)
    print(f"Threshold {thresh}: F1 Score = {f1:.4f}")

# First plot for unrecognizable, recognizable and original images having different colors
tsne_training_recognizable_unrecognizable_original_images_having_different_colors(all_original_training_match_embeddings, recognizable_training_match_images, unrecognizable_training_match_images, ui_centroid,
                                                                                  face_detection_method + " t-SNE - Training Set - Recognizable - Unrecognizable - Original Images")

tsne_training_recognizable_unrecognizable_original_images_having_different_colors(all_original_match_embeddings, all_recognizable_embeddings, all_unrecognizable_embeddings, ui_centroid,
                                                                                  face_detection_method + " t-SNE - All - Recognizable - Unrecognizable - Original Images")
tsne_training_recognizable_unrecognizable_original_images_having_different_colors([], [], unrecognizable_training_match_images, ui_centroid, face_detection_method + " t-SNE - Training Set - Unrecognizable  Images")
tsne_with_clustered_ui_centroids(all_original_training_match_embeddings, recognizable_training_match_images, unrecognizable_training_match_images, 0.5, 10,
                                 face_detection_method + " t-SNE - Training Set - Recognizable - Unrecognizable - Original Images Auto Clustering")
tsne_with_clustered_ui_centroids_hdbscan(all_original_training_match_embeddings, unrecognizable_training_match_images, 2,
                                         face_detection_method + " t-SNE - Training Set - Unrecognizable - Original Images Auto Clustering")

# Second plot for all images that are originated from the same original image having the same color
tsne_test_images_originated_from_same_original_image_having_same_colors(all_training_match_embeddings, all_training_match_labels, ui_centroid, face_detection_method + " t-SNE - Training Set - Images Originated from Same Original Image")
tsne_test_images_originated_from_same_original_image_having_same_colors(all_test_match_embeddings_for_tsne, match_labels_for_tsne, ui_centroid, face_detection_method + " t-SNE - Test Set - Images Originated from Same Original Image")

plot_binned_bar_chart(all_test_match_ers_and_cosine_similarities, face_detection_method + " Match set Distribution of ERS and Similarity Values in 0.01 Intervals")
plot_binned_bar_chart(all_test_mismatch_ers_and_cosine_similarities, face_detection_method + " Mismatch set Distribution of ERS and Similarity Values in 0.01 Intervals")
plot_binned_bar_chart(combined_data, face_detection_method + " Combined match and mismatch set Distribution of ERS and Similarity Values in 0.01 Intervals")

plot_ers_similarity_scatter(all_test_match_ers_and_cosine_similarities, face_detection_method + " Match dataset - ERS vs Similarity: Match (S) vs Mismatch (F)")
plot_ers_similarity_scatter(all_test_mismatch_ers_and_cosine_similarities, face_detection_method + " Mismatch dataset - ERS vs Similarity: Match (S) vs Mismatch (F)")
plot_ers_similarity_scatter(combined_data, face_detection_method + " Match and mismatch combined dataset - ERS vs Similarity: Match (S) vs Mismatch (F)")

plot_ers_similarity_binned(combined_data, face_detection_method + " Match and mismatch combined dataset - ERS vs Similarity: Match (S) vs Mismatch (F) With Line", 0.01)

plot_roc_curve(combined_data, False, face_detection_method + " Combined match and mismatch dataset - ROC Curve - Similarity", face_detection_method + " Combined match and mismatch dataset - ROC Curve - ERS")
