import os
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.spatial.distance import euclidean
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from degradations import degradation_pool
from plotter import tsne_training_recognizable_unrecognizable_original_images_having_different_colors, tsne_test_images_originated_from_same_original_image_having_same_colors
from utility import get_encoding_from_image, save_image

os.makedirs("output/recognizable", exist_ok=True)
os.makedirs("output/unrecognizable", exist_ok=True)
os.makedirs("output/no_embedding", exist_ok=True)

min_images_per_person_lfw = 10
used_training_identity_count = 10
used_training_images_per_identity = 5
used_test_identity_count = 10
used_test_images_per_identity = 5

test_to_training_ratio = 0.2

verification_threshold = 0.4
ers_threshold = 1

min_degradation_strength = 1
max_degradation_strength = 10

save_degraded_images = True

print("Loading LFW dataset...")
lfw = fetch_lfw_people(min_faces_per_person=min_images_per_person_lfw, resize=1.0)
images = lfw.images
targets = lfw.target
target_names = lfw.target_names

unique_labels = np.unique(targets)
train_labels, test_labels = train_test_split(unique_labels, test_size=test_to_training_ratio, random_state=42)

train_indices = [i for i in range(len(targets)) if targets[i] in train_labels]
train_images = [Image.fromarray((images[i] * 255).astype(np.uint8)) for i in train_indices]
train_labels_mapped = [targets[i] for i in train_indices]

test_indices = [i for i in range(len(targets)) if targets[i] in test_labels]
test_images = [Image.fromarray((images[i] * 255).astype(np.uint8)) for i in test_indices]
test_labels_mapped = [targets[i] for i in test_indices]

identity_to_images_train = defaultdict(list)
for img, label in zip(train_images, train_labels_mapped):
    identity_to_images_train[label].append(img)

identity_to_images_test = defaultdict(list)
for img, label in zip(test_images, test_labels_mapped):
    identity_to_images_test[label].append(img)
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

all_training_embeddings = []

training_group_index = 0
training_labels = []

for label in tqdm(list(identity_to_images_train.keys())[:used_training_identity_count]):
    img_list = identity_to_images_train[label]
    for original_img in img_list[:used_training_images_per_identity]:
        original_training_images.append(original_img)
        original_enc = get_encoding_from_image(original_img)
        if original_enc is None:
            continue

        training_group_id = f"group_{training_group_index}"
        training_group_index += 1

        all_training_embeddings.append(original_enc)
        training_labels.append(training_group_id)

        for degradation_fn in degradation_pool:
            for strength in range(min_degradation_strength, max_degradation_strength):
                degraded_img = degradation_fn(original_img.copy(), strength=strength)
                degraded_enc = get_encoding_from_image(degraded_img)

                all_training_embeddings.append(degraded_enc)
                training_labels.append(training_group_id)

                if degraded_enc is None:
                    training_no_face_images_statistics[degradation_fn.__name__] += 1
                    training_no_face_count += 1
                    if save_degraded_images:
                        degraded_img.save(f"output/no_embedding/img_{label}_{degradation_fn.__name__}_s{strength}.png")
                    continue

                similarity = cosine_similarity([original_enc], [degraded_enc])[0][0]

                if similarity > verification_threshold:
                    recognizable_training_images.append((degraded_img, degraded_enc, original_img, original_enc))
                    training_recognizable_images_statistics[degradation_fn.__name__] += 1
                    training_recognizable_count += 1
                    if save_degraded_images:
                        save_image(degraded_img, "output/recognizable", f"img_{label}_{degradation_fn.__name__}_s{strength}_sim{similarity:.2f}.png")
                else:
                    unrecognizable_training_images.append((degraded_img, degraded_enc, original_img, original_enc))
                    training_unrecognizable_images_statistics[degradation_fn.__name__] += 1
                    training_unrecognizable_count += 1
                    if save_degraded_images:
                        save_image(degraded_img, "output/unrecognizable", f"img_{label}_{degradation_fn.__name__}_s{strength}_sim{similarity:.2f}.png")

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

print("\nStarting ERS-filtered verification on test set...")

ers_filter_total_count = 0
ers_filter_success_count = 0
ers_filter_fail_count = 0
ers_filtered_out_count = 0

test_total_count = 0
test_no_face_count = 0
test_recognizable_count = 0
test_unrecognizable_count = 0

original_test_images = []
degraded_test_images = []
recognizable_test_images = []
unrecognizable_test_images = []
ers_removed_test_images = []
ers_filtered_test_images = []

if ui_centroid is not None:
    for label in tqdm(list(identity_to_images_test.keys())[:used_test_identity_count]):
        img_list = identity_to_images_test[label]

        for original_img in img_list[:used_test_images_per_identity]:
            original_enc = get_encoding_from_image(original_img)
            original_test_images.append(original_img)
            if original_enc is None:
                continue

            for degradation_fn in degradation_pool:
                for strength in range(min_degradation_strength, max_degradation_strength):
                    degraded_img = degradation_fn(original_img.copy(), strength=strength)
                    degraded_enc = get_encoding_from_image(degraded_img)
                    degraded_test_images.append(degraded_img)
                    if degraded_enc is None:
                        test_no_face_images_statistics[degradation_fn.__name__] += 1
                        test_no_face_count += 1
                        if save_degraded_images:
                            degraded_img.save(f"output/no_embedding/img_{label}_{degradation_fn.__name__}_s{strength}.png")
                        continue

                    distance = euclidean(ui_centroid.flatten(), degraded_enc.flatten())

                    if distance > ers_threshold:
                        ers_filtered_test_images.append(degraded_img)
                        similarity = cosine_similarity([original_enc], [degraded_enc])[0][0]
                        if similarity > verification_threshold:
                            if save_degraded_images:
                                save_image(degraded_img, "output/recognizable", f"img_{label}_{degradation_fn.__name__}_s{strength}_sim{similarity:.2f}.png")
                            ers_filter_success_count += 1
                            recognizable_test_images.append(degraded_img)
                        else:
                            test_unrecognizable_images_statistics[degradation_fn.__name__] += 1
                            ers_filter_fail_count += 1
                            unrecognizable_test_images.append(degraded_img)
                        ers_filter_total_count += 1
                    else:
                        ers_filtered_out_count += 1
                        ers_removed_test_images.append(degraded_img)
                    test_total_count += 1

print("\nTest set statistics:")
print("Total images processed:", test_total_count)
print("No face detections:", test_no_face_count)
print("Successful verifications with ERS filtered images:", ers_filter_success_count)
print("Unsuccessful verifications with ERS filtered images:", ers_filter_fail_count)
print("Successful verification ratio with ERS filtered images:", ers_filter_success_count / ers_filter_total_count)
print("Total images that were filtered out by ERS:", ers_filtered_out_count)

# Test image t-SNE for images originated from same original images
all_test_embeddings = []
labels = []
group_index = 0
for label in tqdm(list(identity_to_images_test.keys())[:5]):
    img_list = identity_to_images_test[label]

    for original_img in img_list[:5]:
        original_enc = get_encoding_from_image(original_img)
        if original_enc is None:
            continue

        group_id = f"group_{group_index}"
        group_index += 1

        all_test_embeddings.append(original_enc)
        labels.append(group_id)

        for degradation_fn in degradation_pool:
            for strength in range(min_degradation_strength, max_degradation_strength):
                degraded_img = degradation_fn(original_img.copy(), strength=strength)
                degraded_enc = get_encoding_from_image(degraded_img)
                if degraded_enc is None:
                    continue

                all_test_embeddings.append(degraded_enc)
                labels.append(group_id)

# First plot for unrecognizable, recognizable and original images having different colors
tsne_training_recognizable_unrecognizable_original_images_having_different_colors(original_training_images, recognizable_training_images, unrecognizable_training_images, ui_centroid,
                                                                                  "t-SNE - Training Set - Recognizable - Unrecognizable - Original Images")
# Second plot for all images that are originated from the same original image having the same color
tsne_test_images_originated_from_same_original_image_having_same_colors(all_training_embeddings, training_labels, ui_centroid, "t-SNE - Training Set - Images Originated from Same Original Image")
tsne_test_images_originated_from_same_original_image_having_same_colors(all_test_embeddings, labels, ui_centroid, "t-SNE - Test Set - Images Originated from Same Original Image")
