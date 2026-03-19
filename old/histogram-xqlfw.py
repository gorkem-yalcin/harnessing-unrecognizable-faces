import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV'yi yükle
df = pd.read_csv("../results_xqlfw/xqlfw_raw_data.csv")

# Match ve Mismatch similarity skorlarını ayır
sim_match = df[df['label'] == 1]['sim']
sim_mismatch = df[df['label'] == 0]['sim']

print(f"Match Mean Sim: {sim_match.mean():.4f} (Std: {sim_match.std():.4f})")
print(f"Mismatch Mean Sim: {sim_mismatch.mean():.4f} (Std: {sim_mismatch.std():.4f})")

# Histogram Çiz
plt.figure(figsize=(10, 6))
plt.hist(sim_match, bins=50, alpha=0.5, label='Matches', color='green', density=True)
plt.hist(sim_mismatch, bins=50, alpha=0.5, label='Mismatches', color='red', density=True)
plt.title("XQLFW Similarity Distribution")
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("xqlfw_distribution_check.png")
print("Grafik kaydedildi: xqlfw_distribution_check.png")