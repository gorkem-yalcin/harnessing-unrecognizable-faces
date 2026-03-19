import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV'leri yükle
df_tiny = pd.read_csv("results_tinyface/tinyface_raw_data.csv")
df_xqlfw = pd.read_csv("../results_xqlfw/xqlfw_raw_data.csv") # Bunu önceki adımda kaydetmiştik

plt.figure(figsize=(10, 6))

# TinyFace Dağılımı (Mavi)
sns.kdeplot(df_tiny['prob'], label='TinyFace Probabilities', fill=True, color='blue', alpha=0.3)

# XQLFW Dağılımı (Kırmızı)
sns.kdeplot(df_xqlfw['prob'], label='XQLFW Probabilities', fill=True, color='red', alpha=0.3)

plt.title("Domain Gap Analysis: Probability Distributions")
plt.xlabel("Classifier Probability Score")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("results_xqlfw/domain_gap_analysis.png")
print("Grafik kaydedildi: results_xqlfw/domain_gap_analysis.png")