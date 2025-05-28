import os

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from degradations import degradation_pool
from plotter import tsne_training_recognizable_unrecognizable_original_images_having_different_colors, tsne_test_images_originated_from_same_original_image_having_same_colors, plot_binned_bar_chart, plot_ers_similarity_scatter, \
    plot_roc_curve, plot_ers_similarity_binned, tsne_with_clustered_ui_centroids, tsne_with_clustered_ui_centroids_hdbscan
from utility import get_encoding_from_image, save_image, ensure_rgb, get_ui_clusters_hdbscan

os.makedirs("output/recognizable", exist_ok=True)
os.makedirs("output/unrecognizable", exist_ok=True)
os.makedirs("output/no_embedding", exist_ok=True)

verification_threshold = 0.225
ers_threshold = 1

training_match_identity_count = 10
training_mismatch_identity_count = 10
test_match_identity_count = 10
test_mismatch_identity_count = 10

min_degradation_strength = 1
max_degradation_strength = 6

face_detection_method = "MTCNN"  # "MTCNN"  # "deepface"

save_degraded_images = False

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

training_no_face_images_statistics = {fn.__name__: 0 for fn in degradation_pool}
training_recognizable_images_statistics = {fn.__name__: 0 for fn in degradation_pool}
training_unrecognizable_images_statistics = {fn.__name__: 0 for fn in degradation_pool}

test_no_face_images_statistics = {fn.__name__: 0 for fn in degradation_pool}
test_recognizable_images_statistics = {fn.__name__: 0 for fn in degradation_pool}
test_unrecognizable_images_statistics = {fn.__name__: 0 for fn in degradation_pool}

training_total_count = 0
training_no_face_count = 0
training_recognizable_count = 0
training_unrecognizable_count = 0

original_training_images = []
recognizable_training_images = []
unrecognizable_training_images = []

all_original_training_embeddings = []
all_training_embeddings = []

training_group_index = 0

all_original_training_labels = []
all_training_labels = []

for i in tqdm(range(training_match_identity_count)):
    image1, image2 = train_match_pairs[i]
    # original_training_images.append(image1)
    original_embedding = get_encoding_from_image(image1, face_detection_method)
    verification_embedding = get_encoding_from_image(image2, face_detection_method)
    if original_embedding is None or verification_embedding is None:
        continue

    training_group_id = f"group_{training_group_index}"
    training_group_index += 1

    all_training_embeddings.append(original_embedding)
    all_training_labels.append(training_group_id)

    all_original_training_embeddings.append(original_embedding)
    # all_original_training_labels.append(training_group_id)

    for degradation_fn in degradation_pool:
        for strength in range(min_degradation_strength, max_degradation_strength):
            degraded_img = degradation_fn(image1.copy(), strength=strength)
            degraded_enc = get_encoding_from_image(degraded_img, face_detection_method)

            if degraded_enc is None:
                training_no_face_images_statistics[degradation_fn.__name__] += 1
                training_no_face_count += 1
                if save_degraded_images:
                    degraded_img.save(f"output/no_embedding/img_{i}_{degradation_fn.__name__}_s{strength}.png")
                continue

            all_training_embeddings.append(degraded_enc)
            all_training_labels.append(training_group_id)

            similarity = cosine_similarity([verification_embedding], [degraded_enc])[0][0]

            if similarity > verification_threshold:
                recognizable_training_images.append((degraded_img, degraded_enc, image1, original_embedding))
                training_recognizable_images_statistics[degradation_fn.__name__] += 1
                training_recognizable_count += 1
                if save_degraded_images:
                    save_image(degraded_img, "output/recognizable", f"img_{i}_{degradation_fn.__name__}_s{strength}_sim{similarity:.2f}.png")
            else:
                unrecognizable_training_images.append((degraded_img, degraded_enc, image1, original_embedding))
                training_unrecognizable_images_statistics[degradation_fn.__name__] += 1
                training_unrecognizable_count += 1
                if save_degraded_images:
                    save_image(degraded_img, "output/unrecognizable", f"img_{i}_{degradation_fn.__name__}_s{strength}_sim{similarity:.2f}.png")

            training_total_count += 1

print("\nTraining set statistics:")
print("Total images processed:", training_total_count)
print("No face detections:", training_no_face_count)
print("Recognizable face detections:", training_recognizable_count)
print("Unrecognizable face detections:", training_unrecognizable_count)
print("Recognizable to total ratio:", training_recognizable_count / training_total_count)
print("Unrecognizable to total ratio:", training_unrecognizable_count / training_total_count)
print("No face to total ratio:", training_no_face_count / training_total_count)

# Print out how many no face, recognizable face and unrecognizable face occurrences each degradation model caused
print("\nDegradations causing 'no face' detections:")
for fn_name, count in training_no_face_images_statistics.items():
    print(f"{fn_name}: {count} times")
print("\nDegradations causing 'recognizable face' detections:")
for fn_name, count in training_recognizable_images_statistics.items():
    print(f"{fn_name}: {count} times")
print("\nDegradations causing 'unrecognizable face' detections:")
for fn_name, count in training_unrecognizable_images_statistics.items():
    print(f"{fn_name}: {count} times")

print("\nStarting UI centroid calculation...")
unrecognizable_training_embeddings = []
for img, enc, _, _ in unrecognizable_training_images:
    if enc is not None:
        unrecognizable_training_embeddings.append(enc)

# Step 2: Calculate the centroid (mean of embeddings)
if len(unrecognizable_training_embeddings) > 0:
    embeddings_array = np.array(unrecognizable_training_embeddings)

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

match_test_total_count = 0
match_test_no_face_count = 0
match_test_recognizable_count = 0
match_test_unrecognizable_count = 0

total_degraded_img_count = 0

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

        original_embedding = get_encoding_from_image(image1, face_detection_method)
        verification_embedding = get_encoding_from_image(image2, face_detection_method)

        if original_embedding is None or verification_embedding is None:
            continue

        group_id = f"group_{match_group_index_for_tsne}"
        match_group_index_for_tsne += 1
        all_test_match_embeddings_for_tsne.append(original_embedding)
        match_labels_for_tsne.append(group_id)

        for degradation_fn in degradation_pool:
            for strength in range(min_degradation_strength, max_degradation_strength):
                degraded_img = degradation_fn(image1.copy(), strength=strength)
                degraded_enc = get_encoding_from_image(degraded_img, face_detection_method)
                # degraded_test_match_images.append(degraded_img)

                total_degraded_img_count += 1

                if degraded_enc is None:
                    test_no_face_images_statistics[degradation_fn.__name__] += 1
                    match_test_no_face_count += 1
                    if save_degraded_images:
                        degraded_img.save(f"output/no_embedding/img_{i}_{degradation_fn.__name__}_s{strength}.png")
                    continue

                all_test_match_embeddings_for_tsne.append(degraded_enc)
                match_labels_for_tsne.append(group_id)

                distance = euclidean(ui_centroid.flatten(), degraded_enc.flatten())
                match_ers_scores.append(distance)

                similarity = cosine_similarity([verification_embedding], [degraded_enc])[0][0]
                all_test_match_ers_and_cosine_similarities.append((distance, similarity, 'S'))

                if distance > ers_threshold:
                    # ers_filtered_test_match_images.append(degraded_img)
                    if similarity > verification_threshold:
                        if save_degraded_images:
                            save_image(degraded_img, "output/recognizable", f"img_{i}_{degradation_fn.__name__}_s{strength}_sim{similarity:.2f}.png")
                        match_ers_filter_success_count += 1
                        # recognizable_test_match_images.append(degraded_img)
                    else:
                        test_unrecognizable_images_statistics[degradation_fn.__name__] += 1
                        match_ers_filter_fail_count += 1
                        # unrecognizable_test_match_images.append(degraded_img)
                    match_ers_filter_total_count += 1
                else:
                    match_ers_filtered_out_count += 1
                    # ers_removed_test_match_images.append(degraded_img)
                match_test_total_count += 1

print("\nMatch test set statistics:")
print("Total images processed:", match_test_total_count)
print("No face detections:", match_test_no_face_count)
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

        original_embedding = get_encoding_from_image(image1, face_detection_method)
        verification_embedding = get_encoding_from_image(image2, face_detection_method)

        if original_embedding is None or verification_embedding is None:
            continue

        group_id = f"group_{mismatch_group_index_for_tsne}"
        mismatch_group_index_for_tsne += 1
        all_test_mismatch_embeddings_for_tsne.append(original_embedding)
        mismatch_labels_for_tsne.append(group_id)

        for degradation_fn in degradation_pool:
            for strength in range(min_degradation_strength, max_degradation_strength):
                degraded_img = degradation_fn(image1.copy(), strength=strength)
                degraded_enc = get_encoding_from_image(degraded_img, face_detection_method)
                degraded_test_mismatch_images.append(degraded_img)

                if degraded_enc is None:
                    test_no_face_images_statistics[degradation_fn.__name__] += 1
                    mismatch_test_no_face_count += 1
                    if save_degraded_images:
                        degraded_img.save(f"output/no_embedding/img_{i}_{degradation_fn.__name__}_s{strength}.png")
                    continue

                all_test_mismatch_embeddings_for_tsne.append(degraded_enc)
                mismatch_labels_for_tsne.append(group_id)

                distance = euclidean(ui_centroid.flatten(), degraded_enc.flatten())
                mismatch_ers_scores.append(distance)

                similarity = cosine_similarity([verification_embedding], [degraded_enc])[0][0]

                all_test_mismatch_ers_and_cosine_similarities.append((distance, similarity, 'F'))

                if distance > ers_threshold:
                    ers_filtered_test_mismatch_images.append(degraded_img)
                    if similarity > verification_threshold:
                        if save_degraded_images:
                            save_image(degraded_img, "output/recognizable", f"img_{i}_{degradation_fn.__name__}_s{strength}_sim{similarity:.2f}.png")
                        mismatch_ers_filter_success_count += 1
                        recognizable_test_mismatch_images.append(degraded_img)
                    else:
                        test_unrecognizable_images_statistics[degradation_fn.__name__] += 1
                        mismatch_ers_filter_fail_count += 1
                        unrecognizable_test_mismatch_images.append(degraded_img)
                    mismatch_ers_filter_total_count += 1
                else:
                    mismatch_ers_filtered_out_count += 1
                    ers_removed_test_mismatch_images.append(degraded_img)
                mismatch_test_total_count += 1

print("\nMismatch test set statistics:")
print("Total images processed:", mismatch_test_total_count)
print("No face detections:", mismatch_test_no_face_count)
print("Successful verifications with ERS filtered images:", mismatch_ers_filter_success_count)
print("Unsuccessful verifications with ERS filtered images:", mismatch_ers_filter_fail_count)
print("Successful verification ratio with ERS filtered images:", mismatch_ers_filter_success_count / mismatch_ers_filter_total_count)
print("Total images that were filtered out by ERS:", mismatch_ers_filtered_out_count)
print("Min ERS score:", min(mismatch_ers_scores))
print("Max ERS score:", max(mismatch_ers_scores))
print("Average ERS score:", sum(mismatch_ers_scores) / len(mismatch_ers_scores))

combined_data = all_test_match_ers_and_cosine_similarities + all_test_mismatch_ers_and_cosine_similarities

# First plot for unrecognizable, recognizable and original images having different colors
tsne_training_recognizable_unrecognizable_original_images_having_different_colors(all_original_training_embeddings, recognizable_training_images, unrecognizable_training_images, ui_centroid,
                                                                                  face_detection_method + " t-SNE - Training Set - Recognizable - Unrecognizable - Original Images")
tsne_training_recognizable_unrecognizable_original_images_having_different_colors([], [], unrecognizable_training_images, ui_centroid, face_detection_method + " t-SNE - Training Set - Unrecognizable  Images")
tsne_with_clustered_ui_centroids(all_original_training_embeddings, recognizable_training_images, unrecognizable_training_images, 0.5, 10,
                                 face_detection_method + " t-SNE - Training Set - Recognizable - Unrecognizable - Original Images Auto Clustering")
tsne_with_clustered_ui_centroids_hdbscan(all_original_training_embeddings, unrecognizable_training_images, 2,
                                         face_detection_method + " t-SNE - Training Set - Unrecognizable - Original Images Auto Clustering")

# Second plot for all images that are originated from the same original image having the same color
tsne_test_images_originated_from_same_original_image_having_same_colors(all_training_embeddings, all_training_labels, ui_centroid, face_detection_method + " t-SNE - Training Set - Images Originated from Same Original Image")
tsne_test_images_originated_from_same_original_image_having_same_colors(all_test_match_embeddings_for_tsne, match_labels_for_tsne, ui_centroid, face_detection_method + " t-SNE - Test Set - Images Originated from Same Original Image")

plot_binned_bar_chart(all_test_match_ers_and_cosine_similarities, face_detection_method + " Match set Distribution of ERS and Similarity Values in 0.01 Intervals")
plot_binned_bar_chart(all_test_mismatch_ers_and_cosine_similarities, face_detection_method + " Mismatch set Distribution of ERS and Similarity Values in 0.01 Intervals")
plot_binned_bar_chart(combined_data, face_detection_method + " Combined match and mismatch set Distribution of ERS and Similarity Values in 0.01 Intervals")

plot_ers_similarity_scatter(all_test_match_ers_and_cosine_similarities, face_detection_method + " Match dataset - ERS vs Similarity: Match (S) vs Mismatch (F)")
plot_ers_similarity_scatter(all_test_mismatch_ers_and_cosine_similarities, face_detection_method + " Mismatch dataset - ERS vs Similarity: Match (S) vs Mismatch (F)")
plot_ers_similarity_scatter(combined_data, face_detection_method + " Match and mismatch combined dataset - ERS vs Similarity: Match (S) vs Mismatch (F)")

plot_ers_similarity_binned(combined_data, face_detection_method + " Match and mismatch combined dataset - ERS vs Similarity: Match (S) vs Mismatch (F) With Line", 0.01)

plot_roc_curve(combined_data, face_detection_method + " Combined match and mismatch dataset - ROC Curve - Similarity", face_detection_method + " Combined match and mismatch dataset - ROC Curve - ERS")
