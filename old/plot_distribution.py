import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from analyze_final_results import get_hybrid_scores, model, TINY_PAIRS, TINY_IMG # Önceki scriptten import

# Verileri Al
print("📊 Dağılım grafiği için veriler çekiliyor...")
y_true, y_scores = get_hybrid_scores(model, TINY_PAIRS, TINY_IMG, "TinyFace")

# Pozitif ve Negatifleri Ayır
pos_scores = y_scores[y_true == 1]
neg_scores = y_scores[y_true == 0]

# Grafiği Çiz
plt.figure(figsize=(10, 6))
sns.histplot(pos_scores, color='green', label='Matches (Aynı Kişi)', kde=True, stat="density", bins=50, alpha=0.5)
sns.histplot(neg_scores, color='red', label='Mismatches (Farklı Kişi)', kde=True, stat="density", bins=50, alpha=0.5)

# Threshold Çizgisi (64.62)
plt.axvline(x=64.62, color='blue', linestyle='--', linewidth=2, label='Optimal Threshold (64.62)')

plt.title('Hybrid Classifier Score Distribution (TinyFace)')
plt.xlabel('Similarity Score')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("score_distribution.png", dpi=300)
print("✅ Grafik kaydedildi: score_distribution.png")