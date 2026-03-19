import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
import random
from degradations import degradation_pool
from utility import get_encoding_from_image, ensure_rgb
import torch.nn as nn

# --- CONFIG ---
MODEL_PATH = "results/lr0.0001_wd0.0001_bs256_dropout0.4_best_model.pt"  # Update this path!
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- MODEL DEFINITION (Must match your training script) ---
class QualityPredictor(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# --- LOAD ---
print("Loading Model...")
model = QualityPredictor().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except:
    print("Warning: Could not load model weights. Using random weights (Just for code testing).")
model.eval()

print("Loading LFW Data...")
lfw = fetch_lfw_pairs(subset='test', color=True, resize=1.0)
pairs = lfw.pairs
labels = lfw.target  # 1 = Match, 0 = Mismatch

# Separate into Match and Mismatch
match_pairs = [p for p, l in zip(pairs, labels) if l == 1]
mismatch_pairs = [p for p, l in zip(pairs, labels) if l == 0]

# Limit size for speed if needed
match_pairs = match_pairs[:500]
mismatch_pairs = mismatch_pairs[:500]

detector = MTCNN(image_size=160, margin=0, device=DEVICE)
embedder = InceptionResnetV1(pretrained='casia-webface').eval().to(DEVICE)
cache = {}

# --- EXPERIMENT ---
print("Generating 'Degraded vs Degraded' Scores...")


def process_pairs(pair_list, label_name):
    qualities = []
    similarities = []

    for img1, img2 in tqdm(pair_list, desc=label_name):
        img1 = ensure_rgb(img1)
        img2 = ensure_rgb(img2)

        # Apply Random Degradations to BOTH images (The Professor's scenario)
        deg_fn = random.choice(degradation_pool)
        strength = random.randint(1, 5)

        deg_img1 = deg_fn(img1.copy(), strength=strength)
        deg_img2 = deg_fn(img2.copy(), strength=strength)  # Same degradation, or random? Usually random is more realistic.

        # Get Embeddings
        # Note: In real script, handle 'None' if face not found
        try:
            emb1 = get_encoding_from_image(deg_img1, "facenet_pytorch", cache, f"d1", detector, embedder, DEVICE)
            emb2 = get_encoding_from_image(deg_img2, "facenet_pytorch", cache, f"d2", detector, embedder, DEVICE)
        except:
            continue

        if emb1 is None or emb2 is None:
            continue

        # Predict Quality
        t1 = torch.tensor(emb1).float().unsqueeze(0).to(DEVICE)
        t1 = torch.nn.functional.normalize(t1, p=2, dim=1)
        t2 = torch.tensor(emb2).float().unsqueeze(0).to(DEVICE)
        t2 = torch.nn.functional.normalize(t2, p=2, dim=1)

        with torch.no_grad():
            q1 = model(t1).item()
            q2 = model(t2).item()

        # Similarity
        sim = cosine_similarity([emb1], [emb2])[0][0]

        # Metric: Use the LOWER quality of the two (The bottleneck)
        qualities.append(min(q1, q2))
        similarities.append(sim)

    return qualities, similarities


# Run processing
match_q, match_sim = process_pairs(match_pairs, "Matches")
mismatch_q, mismatch_sim = process_pairs(mismatch_pairs, "Mismatches")

# --- PLOTTING ---
plt.figure(figsize=(10, 8))

# Scatter plot
plt.scatter(mismatch_q, mismatch_sim, alpha=0.5, color='red', label='Mismatches (Different Id)', s=10)
plt.scatter(match_q, match_sim, alpha=0.5, color='blue', label='Matches (Same Id)', s=10)

# Trend Lines (Polynomial fit)
if len(match_q) > 0 and len(mismatch_q) > 0:
    z_match = np.polyfit(match_q, match_sim, 2)
    p_match = np.poly1d(z_match)

    z_mismatch = np.polyfit(mismatch_q, mismatch_sim, 2)
    p_mismatch = np.poly1d(z_mismatch)

    x_range = np.linspace(0, 1, 100)
    plt.plot(x_range, p_match(x_range), "b--", linewidth=2, label="Match Trend")
    plt.plot(x_range, p_mismatch(x_range), "r--", linewidth=2, label="Mismatch Trend")

plt.xlabel("Predicted Quality Score (0=Bad, 1=Good)")
plt.ylabel("Cosine Similarity")
plt.title("Effect of Degrading BOTH Images on Similarity")
plt.xlim(0, 1)
plt.ylim(-0.2, 1.0)
plt.axhline(0.7, color='gray', linestyle=':', label='Standard Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("results/professor_hypothesis_check.png")
print("Plot saved to results/professor_hypothesis_check.png")