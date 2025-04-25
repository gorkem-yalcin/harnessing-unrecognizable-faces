from PIL import Image, ImageFilter, ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import os
import random
import numpy as np
import cv2
from sklearn.manifold import TSNE


# to install these libraries, run the command below:
# pip install pillow facenet-pytorch torch torchvision matplotlib tqdm numpy opencv-python scikit-learn

# ----------------------------
# Degradation Functions
# ----------------------------

def gaussian_blur(img, strength=1):
    radius = 1 + strength * 2
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def motion_blur(img, strength=1):
    kernel_size = 5 + strength * 5
    angle = random.choice([0, 45, 90])
    kernel = np.zeros((kernel_size, kernel_size))
    if angle == 0:
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    elif angle == 90:
        kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    else:
        np.fill_diagonal(kernel, 1)
    kernel /= kernel_size
    img_np = np.array(img)
    blurred = cv2.filter2D(img_np, -1, kernel)
    return Image.fromarray(blurred)


def low_resolution(img, strength=1):
    w, h = img.size
    factor = 2 + strength * 2
    downsample = img.resize((w // factor, h // factor), Image.BILINEAR)
    return downsample.resize((w, h), Image.BILINEAR)


def rotate_image(img, strength=1):
    angle = strength * 5 * random.choice([-1, 1])
    return img.rotate(angle)


def affine_transform(img, strength=1):
    width, height = img.size
    scale = 0.05 * strength
    return img.transform(
        (width, height),
        Image.AFFINE,
        (1, random.uniform(-scale, scale), 0,
         random.uniform(-scale, scale), 1, 0),
        resample=Image.BILINEAR
    )


def occlude_image(img, strength=1):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    box_size = int(min(w, h) * (0.1 * strength))
    x1 = random.randint(0, w - box_size)
    y1 = random.randint(0, h - box_size)
    draw.rectangle([x1, y1, x1 + box_size, y1 + box_size], fill=(0, 0, 0))
    return img


degradation_pool = [
    gaussian_blur,
    motion_blur,
    low_resolution,
    rotate_image,
    affine_transform,
    occlude_image,
]


def apply_random_degradation(img, strength=1):
    fn = random.choice(degradation_pool)
    degraded_img = fn(img.copy(), strength=strength)
    return degraded_img, fn.__name__, strength


def get_embedding(img_tensor):
    face = mtcnn(to_pil_image(img_tensor))
    if face is not None:
        # print("Face detected.")
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = resnet(face)
            return F.normalize(emb, p=2, dim=1).cpu().squeeze(0)
    # print("Face not detected.")
    return None


# ----------------------------
# Setup
# ----------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

lfw_dataset = ImageFolder(root='./data/lfw-deepfunneled', transform=transform)  # local dataset
save_dir = './unrecognizable_faces_parametric'  # to save the unrecognizable images
os.makedirs(save_dir, exist_ok=True)

# ----------------------------
# Process dataset
# ----------------------------
verification_threshold = 0.5  # This is the threshold for similarity probability
ers_threshold = 0.75  # This is the threshold for ERS, degraded images that have a higher threshold than this will be tested
test_image_count = 13233  # This is the count for how many images we are using from the dataset, increase this to test with more images, max is 13233

save_degraded_images_in_project_directory = False

original_embeddings = []
degraded_recognizable_embeddings = []
unrecognizable_embeddings = []  # List of tuples: (original_embedding, degraded_embedding, degradation_name, strength)
degraded_embeddings = []

for i in tqdm(range(test_image_count)):
    img, label = lfw_dataset[i]
    class_name = lfw_dataset.classes[label]
    emb_orig = get_embedding(img)  # original embedding of the dataset image

    if emb_orig is not None:
        original_embeddings.append(emb_orig)

        for degradation_fn in degradation_pool:  # for each degradation method, could also do this in a parametric - randomized way but this is to show all types of degradations
            degradation_name = degradation_fn.__name__
            for strength in range(1, 6):  # degradation strengths from 1 to 5
                degraded_img = degradation_fn(to_pil_image(img).copy(), strength=strength)  # degraded image
                emb_degraded = get_embedding(transforms.ToTensor()(degraded_img))  # degraded image's embedding

                if emb_degraded is not None:  # if the model finds a face
                    sim = F.cosine_similarity(emb_orig.unsqueeze(0), emb_degraded.unsqueeze(0)).item()  # similarity score with the original embedding
                    file_name = f"{class_name}_{i}_{degradation_name}_s{strength}_sim{sim:.2f}.jpg"
                    save_path = os.path.join(save_dir, file_name)

                    if sim < verification_threshold:  # if the similarity between degraded image and original image is below the verification threshold
                        unrecognizable_embeddings.append((emb_orig, emb_degraded, degradation_name, strength))  # degraded image is an unrecognizable entity, can not verify with the original image
                        if save_degraded_images_in_project_directory:
                            degraded_img.save(save_path)  # saves the image under 'unrecognizable_faces_parametric' or save_dir variable
                    else:
                        degraded_recognizable_embeddings.append((emb_orig, emb_degraded, degradation_name, strength))  # degraded image is still recognizable can verify with the original image

degraded_embeddings = degraded_recognizable_embeddings + unrecognizable_embeddings  # all degraded embeddings, both recognizable and unrecognizable

# ----------------------------
# Compute UI Centroid
# ----------------------------

if unrecognizable_embeddings:
    embeddings_tensor = torch.stack([e[1] for e in unrecognizable_embeddings])  # Only using degraded embeddings
    ui_centroid = torch.mean(embeddings_tensor, dim=0)
    ui_centroid = F.normalize(ui_centroid, p=2, dim=0)  # calculate the unrecognizable identity cluster point
    print("UI Centroid calculated")
else:
    print("No valid embeddings found in the unrecognizable set.")

print("\n")
# ----------------------------
# ERS Calculation
# ----------------------------

print_degradation_type_ers_verification = False


def calculate_verification_success_rate(degraded_embeddings, do_ers_threshold_verification=True):
    successful_verifications = 0
    total_tested = 0
    test_index = 0
    for emb_orig, emb_degraded, degradation_name, strength in degraded_embeddings:  # for all degraded embeddings, both containing recognizable and unrecognizable identities
        ers = torch.norm(emb_degraded - ui_centroid).item()  # Euclidean distance relative to the centroid

        # Apply ERS threshold and perform cosine similarity verification
        if (not do_ers_threshold_verification) or ers > ers_threshold:  # if the calculated ers is higher than our threshold, so we disregard the embeddings that have a lower score than our threshold to see the improvement of the study
            sim = F.cosine_similarity(emb_orig.unsqueeze(0), emb_degraded.unsqueeze(0)).item()  # calculate the similarity between degraded embedding and the original
            verified = sim >= verification_threshold  # if the similarity score between degraded and original image is higher than the threshold, the degraded image is verified with the original

            if verified:
                successful_verifications += 1
            total_tested += 1
            if print_degradation_type_ers_verification:
                print(f"[{test_index}] {degradation_name} s{strength} | ERS={ers:.2f} | SIM={sim:.2f} | VERIFIED={verified}")
        test_index += 1
    # Final result
    if total_tested > 0:
        if do_ers_threshold_verification:
            print(f"\nVerification success rate (ERS > {ers_threshold}): {successful_verifications}/{total_tested} "
                  f"({(successful_verifications / total_tested * 100):.2f}%)")
        else:
            print(f"\nVerification success rate with no ERS filtering: {successful_verifications}/{total_tested} "
                  f"({(successful_verifications / total_tested * 100):.2f}%)")

    else:
        print("\nNo degraded images to verify.")


print("Start of verification with ERS filtering")
# Calculate ERS for all degraded images (including recognizable ones)
calculate_verification_success_rate(degraded_embeddings, True)
# this is the same thing as the block above, but here, we dont care about the ERS threshold, we use all of the degraded embeddings, so that we can find the difference between using ERS filtering and not
print("\n======================================\n")
print("Start of verification with no ERS filtering")
calculate_verification_success_rate(degraded_embeddings, False)

# ----------------------------
# t-SNE Visualization
# ----------------------------


# Append ui_centroid to the embeddings (if available)
if unrecognizable_embeddings:
    all_embeddings = torch.stack(original_embeddings + [e[1] for e in degraded_recognizable_embeddings] + [e[1] for e in unrecognizable_embeddings] + [ui_centroid])
    labels = (
            ['Original'] * len(original_embeddings) +
            ['Degraded'] * len(degraded_recognizable_embeddings) +
            ['Unrecognizable'] * len(unrecognizable_embeddings) +
            ['UI Centroid']
    )
else:
    all_embeddings = torch.stack(original_embeddings + degraded_recognizable_embeddings)
    labels = (
            ['Original'] * len(original_embeddings) +
            ['Degraded'] * len(degraded_recognizable_embeddings)
    )

# Run t-SNE
X = all_embeddings.numpy()
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X_reduced = tsne.fit_transform(X)

# Plot
plt.figure(figsize=(10, 8))
for label in set(labels):
    idx = [i for i, l in enumerate(labels) if l == label]
    if label == "UI Centroid":
        plt.scatter(X_reduced[idx, 0], X_reduced[idx, 1], label=label, c='red', marker='X', s=100, edgecolors='k',
                    linewidths=1.5)
    else:
        plt.scatter(X_reduced[idx, 0], X_reduced[idx, 1], label=label, alpha=0.7, s=10)

plt.legend()
plt.title('t-SNE of Face Embeddings with UI Centroid')
plt.xlabel('t-SNE Dim 1')
plt.ylabel('t-SNE Dim 2')
plt.grid(True)
plt.tight_layout()
plt.show()
