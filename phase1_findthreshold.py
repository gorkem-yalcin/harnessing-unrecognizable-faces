import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, f1_score
from tqdm import tqdm

# ==========================================
# CONFIG
# ==========================================
MODEL_NAME = 'buffalo_l'
CACHE_FILE = f"lfw_base_embeddings_{MODEL_NAME}.pkl"

# ==========================================
# LOAD
# ==========================================
print(f"Loading {CACHE_FILE}...")
with open(CACHE_FILE, 'rb') as f:
    df = pickle.load(f)

print(f"Loaded {len(df)} records.")

# ==========================================
# COMPUTE COSINE SIMILARITIES
# ==========================================
cos_sims = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing cosines"):
    emb_clean = row['clean_embedding']
    emb_deg = row['degraded_embedding']
    norm_c = emb_clean / np.linalg.norm(emb_clean)
    norm_d = emb_deg / np.linalg.norm(emb_deg)
    cos_sims.append(np.dot(norm_c, norm_d))

cos_sims = np.array(cos_sims)

# ==========================================
# PLOT 1: Distribution of cosine similarities
# ==========================================
plt.figure(figsize=(10, 5))
plt.hist(cos_sims, bins=100, color='steelblue', edgecolor='none', alpha=0.8)
plt.axvline(0.25, color='red', linestyle='--', label='buffalo_l threshold (0.25)')
plt.axvline(np.median(cos_sims), color='orange', linestyle='--',
            label=f'Median ({np.median(cos_sims):.3f})')
plt.axvline(np.percentile(cos_sims, 20), color='green', linestyle='--',
            label=f'20th percentile ({np.percentile(cos_sims, 20):.3f})')
plt.xlabel('Cosine similarity (clean vs degraded)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title(f'Clean vs Degraded Cosine Distribution — {MODEL_NAME}', fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig(f'threshold_distribution_{MODEL_NAME}.png', dpi=200)
plt.close()
print(f"Saved: threshold_distribution_{MODEL_NAME}.png")

# ==========================================
# PRINT KEY STATISTICS
# ==========================================
print("\n--- Key statistics ---")
print(f"  Mean:            {np.mean(cos_sims):.4f}")
print(f"  Median:          {np.median(cos_sims):.4f}")
print(f"  Std:             {np.std(cos_sims):.4f}")
print(f"  10th percentile: {np.percentile(cos_sims, 10):.4f}")
print(f"  20th percentile: {np.percentile(cos_sims, 20):.4f}")
print(f"  30th percentile: {np.percentile(cos_sims, 30):.4f}")

# ==========================================
# PLOT 2: Class balance at each threshold
# ==========================================
thresholds = np.arange(0.05, 0.50, 0.025)
class1_ratios = []
for t in thresholds:
    ratio = np.mean(cos_sims >= t)
    class1_ratios.append(ratio)

plt.figure(figsize=(10, 5))
plt.plot(thresholds, class1_ratios, marker='o', color='steelblue')
plt.axhline(0.5, color='gray', linestyle='--', label='50/50 balance')
plt.axvline(0.25, color='red', linestyle='--', label='buffalo_l threshold (0.25)')
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Fraction labeled as recognizable (class=1)', fontsize=12)
plt.title(f'Class Balance vs Threshold — {MODEL_NAME}', fontsize=13)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'threshold_balance_{MODEL_NAME}.png', dpi=200)
plt.close()
print(f"Saved: threshold_balance_{MODEL_NAME}.png")

# ==========================================
# FIND BALANCED THRESHOLD (closest to 50/50)
# ==========================================
balanced_idx = np.argmin(np.abs(np.array(class1_ratios) - 0.5))
print(f"\n  Most balanced threshold: {thresholds[balanced_idx]:.3f} "
      f"(gives {class1_ratios[balanced_idx] * 100:.1f}% class=1)")
print(f"\nSuggested thresholds to try in phase1_prepare_lfw.py:")
for t in [thresholds[balanced_idx] - 0.05,
          thresholds[balanced_idx],
          thresholds[balanced_idx] + 0.05]:
    print(f"  DECISION_THRESHOLD = {t:.2f}")