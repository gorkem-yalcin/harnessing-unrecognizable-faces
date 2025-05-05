import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from deepface import DeepFace
from scipy.spatial.distance import euclidean
from sklearn.datasets import fetch_lfw_people
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


def gaussian_blur(img, strength=1):
    radius = 1 + strength * 0.5  # more intense blur
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def motion_blur(img, strength=1):
    kernel_size = int(5 + strength * 2)
    angle = random.choice([0, 45, 90])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    if angle == 0:
        kernel[int(kernel_size / 2), :] = np.ones(kernel_size)
    elif angle == 90:
        kernel[:, int(kernel_size / 2)] = np.ones(kernel_size)
    else:
        np.fill_diagonal(kernel, 1)

    kernel /= kernel.sum()
    img_np = np.array(img)
    blurred = cv2.filter2D(img_np, -1, kernel)
    return Image.fromarray(np.uint8(blurred))


def low_resolution(img, strength=1):
    w, h = img.size
    factor = int(2 + strength * 2)  # more downsampling
    downsample = img.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR)
    return downsample.resize((w, h), Image.BILINEAR)


def rotate_image(img, strength=1):
    angle = strength * 5 * random.choice([-1, 1])  # rotate up to ±75°
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)


def affine_transform(img, strength=1):
    width, height = img.size
    scale = 0.05 * strength  # stronger distortion
    return img.transform(
        (width, height),
        Image.AFFINE,
        (1, random.uniform(-scale, scale), 0,
         random.uniform(-scale, scale), 1, 0),
        resample=Image.BILINEAR,
        fillcolor=0
    )


def occlude_image(img, strength=1):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    box_size = int(min(w, h) * (0.10 + 0.02 * strength))  # larger occlusion
    x1 = random.randint(0, w - box_size)
    y1 = random.randint(0, h - box_size)
    draw.rectangle([x1, y1, x1 + box_size, y1 + box_size], fill=0)  # black box
    return img


# Function to apply random degradation
def apply_random_degradation(img, strength=1):
    fn = random.choice(degradation_pool)
    degraded_img = fn(img.copy(), strength=strength)
    return degraded_img, fn.__name__, strength


def get_encoding_from_array(img):
    """Convert grayscale to RGB and extract face encoding using DeepFace."""
    # Convert grayscale image to RGB (DeepFace requires RGB images)
    rgb_img = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2RGB)

    # Use DeepFace to extract embeddings from the image
    try:
        result = DeepFace.represent(rgb_img, model_name="VGG-Face", enforce_detection=False)
        # DeepFace returns a list of embeddings, we take the first (and only) element in the list
        embedding = result[0]["embedding"]
        return np.array(embedding)  # Convert embedding to numpy array for consistency
    except Exception as e:
        print(f"Error extracting face encoding: {e}")
        return None


os.makedirs("output/recognizable", exist_ok=True)
os.makedirs("output/unrecognizable", exist_ok=True)
os.makedirs("output/no_embedding", exist_ok=True)

print("Loading LFW dataset...")
lfw = fetch_lfw_people(min_faces_per_person=10, resize=1.0)
images = lfw.images
targets = lfw.target
target_names = lfw.target_names

# Split identities (labels) into train (80%) and test (20%)
unique_labels = np.unique(targets)
train_labels, _ = train_test_split(unique_labels, test_size=0.2, random_state=42)

# Filter images belonging to train_labels
train_indices = [i for i in range(len(targets)) if targets[i] in train_labels]
train_images = [Image.fromarray((images[i] * 255).astype(np.uint8)) for i in train_indices]
train_labels_mapped = [targets[i] for i in train_indices]

# Convert NumPy arrays to PIL images for compatibility with PIL operations
original_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in original_images]
# Initialize list to store recognizable, unrecognizable, and no face images
recognizable_images = []
unrecognizable_images = []

degradation_pool = [
    gaussian_blur,
    motion_blur,
    low_resolution,
    rotate_image,
    affine_transform,
    occlude_image,
]

no_face_count = {fn.__name__: 0 for fn in degradation_pool}  # Track counts of "no face" for each degradation function
recognizable_images_count = {fn.__name__: 0 for fn in degradation_pool}
unrecognizable_images_count = {fn.__name__: 0 for fn in degradation_pool}

verification_threshold = 0.5  # cosine similarity threshold
ers_threshold = 0.9  # ERS threshold

no_filter_total_count = 0
no_filter_success_count = 0

for i, original_img in enumerate(original_images):
    original_enc = get_encoding_from_array(original_img)
    if original_enc is None:
        continue  # skip if face not found in original

    for degradation_fn in degradation_pool:
        for strength in range(1, 5):
            degraded_img = degradation_fn(original_img.copy(), strength=strength)
            degraded_enc = get_encoding_from_array(degraded_img)

            if degraded_enc is None:
                # No face detected, increment the count for the corresponding degradation function
                no_face_count[degradation_fn.__name__] += 1

                # Save the image in the "no_embedding" folder
                no_face_filename = f"img{i}_{degradation_fn.__name__}_s{strength}.png"
                degraded_img.save(os.path.join("output/no_embedding", no_face_filename))
                continue  # Skip if face not found in degraded image

            similarity = cosine_similarity([original_enc], [degraded_enc])[0][0]
            # print(similarity)

            filename = f"img{i}_{degradation_fn.__name__}_s{strength}_sim{similarity:.2f}.png"

            if similarity > verification_threshold:
                recognizable_images.append((degraded_img, degradation_fn.__name__, strength, similarity, original_enc, degraded_enc))
                recognizable_images_count[degradation_fn.__name__] += 1
                degraded_img.save(os.path.join("output/recognizable", filename))
                no_filter_success_count += 1
            else:
                unrecognizable_images.append((degraded_img, degradation_fn.__name__, strength, similarity, original_enc, degraded_enc))
                unrecognizable_images_count[degradation_fn.__name__] += 1
                degraded_img.save(os.path.join("output/unrecognizable", filename))

            no_filter_total_count += 1

print(no_filter_success_count / no_filter_total_count, "no filter success rate", no_filter_success_count, "/", no_filter_total_count)
all_degraded_embeddings = recognizable_images + unrecognizable_images
# Print out how many "no face" occurrences each degradation model caused
print("\nDegradation models causing 'no face' detections:")
for fn_name, count in no_face_count.items():
    print(f"{fn_name}: {count} times")
print("\nDegradation models causing 'recognizable image' detections:")
for fn_name, count in recognizable_images_count.items():
    print(f"{fn_name}: {count} times")
print("\nDegradation models causing 'unrecognizable image' detections:")
for fn_name, count in unrecognizable_images_count.items():
    print(f"{fn_name}: {count} times")
# Initialize list to store embeddings of unrecognizable images
# Step 1: Collect embeddings for all unrecognizable images
unrecognizable_embeddings = []
for img, _, _, _, _, _ in unrecognizable_images:
    enc = get_encoding_from_array(img)
    if enc is not None:
        unrecognizable_embeddings.append(enc)

# Step 2: Calculate the centroid (mean of embeddings)
if len(unrecognizable_embeddings) > 0:
    # Convert the list of embeddings to a NumPy array (2D)
    embeddings_array = np.array(unrecognizable_embeddings)

    # Calculate the mean across all embeddings to get the centroid
    ui_centroid = np.mean(embeddings_array, axis=0)
    print("UI Centroid (mean of embeddings):")
    print(ui_centroid)
else:
    print("No unrecognizable embeddings available to calculate centroid.")
    ui_centroid = None

ers_scores = []
# Compute ERS only if ui_centroid exists

ers_filter_total_count = 0
ers_filter_success_count = 0

if ui_centroid is not None:
    for embedding in all_degraded_embeddings:
        # Calculate the Euclidean distance between the centroid and each embedding
        distance = euclidean(ui_centroid.flatten(), embedding[5].flatten())
        ers_scores.append(distance)
        if distance > ers_threshold:
            similarity = cosine_similarity([embedding[4]], [embedding[5]])[0][0]

            if similarity > verification_threshold:
                ers_filter_success_count += 1
            else:
                ers_filter_success_count += 0
            ers_filter_total_count += 1

print(ers_filter_success_count / ers_filter_total_count, "ers filter success rate", ers_filter_success_count, "/", ers_filter_total_count)
# Collect all embeddings for t-SNE (original, recognizable, unrecognizable)
all_embeddings = []
labels = []

# Original embeddings (label: 'original')
for original_img in original_images:
    enc = get_encoding_from_array(original_img)
    if enc is not None:
        all_embeddings.append(enc)
        labels.append('original')

# Recognizable embeddings (label: 'recognizable')
for img, _, _, _, _, _ in recognizable_images:
    enc = get_encoding_from_array(img)
    if enc is not None:
        all_embeddings.append(enc)
        labels.append('recognizable')

# Unrecognizable embeddings (label: 'unrecognizable')
for img, _, _, _, _, _ in unrecognizable_images:
    enc = get_encoding_from_array(img)
    if enc is not None:
        all_embeddings.append(enc)
        labels.append('unrecognizable')

# Add the centroid embedding (if exists) to the embeddings list
if ui_centroid is not None:
    all_embeddings.append(ui_centroid)
    labels.append('ui_centroid')

# Convert to NumPy array
all_embeddings = np.array(all_embeddings)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(all_embeddings)

# Plot
plt.figure(figsize=(10, 7))
colors = {'original': 'black', 'recognizable': 'green', 'unrecognizable': 'red', 'ui_centroid': 'purple'}

for label in set(labels):
    idxs = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], label=label, alpha=0.6, c=colors[label])

# Highlight the UI centroid
if ui_centroid is not None:
    centroid_idx = len(all_embeddings) - 1  # The last point is the centroid
    plt.scatter(embeddings_2d[centroid_idx, 0], embeddings_2d[centroid_idx, 1], color='purple', s=100, marker='X', label='UI Centroid')

plt.title("t-SNE Visualization of Face Embeddings with UI Centroid")
plt.legend()
plt.grid(True)
plt.show()
