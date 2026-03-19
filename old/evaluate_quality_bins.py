import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
from utility import get_encoding_from_image, ensure_rgb

# --- CONFIG ---
MODEL_PATH = "results/lr0.0001_wd0.0001_bs256_dropout0.4_best_model.pt"  # Update this path!
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- DEFINITIONS ---
class Classifier(nn.Module):
    def __init__(self, dropout=0.4, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
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
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# --- LOAD DATA ---
print("Loading LFW Test Data...")
lfw_pairs_test = fetch_lfw_pairs(subset='test', color=True, resize=1.0)
test_pairs = lfw_pairs_test.pairs
test_labels = lfw_pairs_test.target
test_pairs = [(ensure_rgb(img1), ensure_rgb(img2)) for (img1, img2) in test_pairs]

# --- LOAD MODEL ---
model = Classifier().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except:
    print("Warning: Could not load model weights. Using random weights (Just for code testing).")
model.eval()
print("Model loaded.")

# --- PREPARE EXTRACTOR ---
detector = MTCNN(image_size=160, margin=0, device=DEVICE)
embedder = InceptionResnetV1(pretrained='casia-webface').eval().to(DEVICE)
cache = {}  # Temporary cache for this run

# --- INFERENCE LOOP ---
print("Running Inference on Test Set...")
results = []

for i in tqdm(range(len(test_pairs))):
    img1, img2 = test_pairs[i]
    is_same = test_labels[i]  # 1 if match, 0 if mismatch

    # Get Embeddings
    # Note: We are mocking the cache/name system for simplicity here
    emb1 = get_encoding_from_image(img1, "facenet_pytorch", cache, f"t1_{i}", detector, embedder, DEVICE)
    emb2 = get_encoding_from_image(img2, "facenet_pytorch", cache, f"t2_{i}", detector, embedder, DEVICE)

    if emb1 is None or emb2 is None:
        continue

    # Get Quality Scores (The "Novelty")
    # 1. Normalize input (Crucial: Training was on normalized data)
    inp1 = torch.tensor(emb1).float().unsqueeze(0).to(DEVICE)
    inp1 = torch.nn.functional.normalize(inp1, p=2, dim=1)

    inp2 = torch.tensor(emb2).float().unsqueeze(0).to(DEVICE)
    inp2 = torch.nn.functional.normalize(inp2, p=2, dim=1)

    with torch.no_grad():
        q1_logit = model(inp1)
        q2_logit = model(inp2)

        # Sigmoid to get 0-1 probability
        q1_score = torch.sigmoid(q1_logit).item()
        q2_score = torch.sigmoid(q2_logit).item()

    # Calculate Face Verification Score
    sim = cosine_similarity([emb1], [emb2])[0][0]

    # Store Data: (Quality_Min, Similarity, True_Label)
    # We use the minimum quality of the pair because the "chain is only as strong as the weakest link"
    pair_quality = min(q1_score, q2_score)
    results.append({'q': pair_quality, 'sim': sim, 'label': is_same})

# --- ANALYSIS: ERROR vs REJECT CURVE ---
print(f"\nAnalyzing {len(results)} pairs...")

# Sort by Quality (Low to High)
results.sort(key=lambda x: x['q'])

ratios = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
accuracies = []
retained_samples = []

threshold = 0.7  # Verification Threshold

print("\n--- Results ---")
print(f"{'Reject Ratio':<15} | {'Quality Thresh':<15} | {'Accuracy':<10} | {'Samples Left':<10}")
print("-" * 60)

for ratio in ratios:
    # Cut off the bottom X%
    cut_index = int(len(results) * ratio)
    kept_data = results[cut_index:]

    if len(kept_data) == 0:
        break

    # Calculate Accuracy on the kept data
    correct = 0
    for res in kept_data:
        pred_same = 1 if res['sim'] > threshold else 0
        if pred_same == res['label']:
            correct += 1

    acc = correct / len(kept_data)
    accuracies.append(acc)
    retained_samples.append(len(kept_data))

    q_thresh = results[cut_index]['q'] if cut_index < len(results) else 0
    print(f"{ratio * 100:<14.0f}% | {q_thresh:<15.4f} | {acc:.4f}     | {len(kept_data)}")

# --- PLOT ---
plt.figure(figsize=(10, 6))
plt.plot([r * 100 for r in ratios], accuracies, marker='o', linewidth=2)
plt.title("Effect of Quality Filtering on Verification Accuracy")
plt.xlabel("Percentage of Images Rejected (Lowest Quality First)")
plt.ylabel("Face Verification Accuracy")
plt.grid(True)
plt.savefig("results/error_reject_curve.png")
print("\nPlot saved to results/error_reject_curve.png")