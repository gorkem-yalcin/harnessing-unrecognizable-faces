import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.datasets import fetch_lfw_pairs
from tqdm import tqdm
import optuna
from arcfaceutility import ensure_rgb, get_encoding_from_image
from degradations import degradation_pool

# ==========================================
# 1. AYARLAR & VERİ HAZIRLIĞI
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Running Optuna Optimization on: {device}")

# Embedding Cache Yükle (Hız için)
if os.path.exists("embedding_cache_lfw.pkl"):
    with open("embedding_cache_lfw.pkl", "rb") as f:
        embedding_cache = pickle.load(f)
else:
    embedding_cache = {}

print("📊 Veri Seti Hazırlanıyor...")
lfw_train = fetch_lfw_pairs(subset='train', color=True, resize=1.0)
train_pairs = [(ensure_rgb(p[0]), ensure_rgb(p[1])) for p in lfw_train.pairs]
train_labels = lfw_train.target
match_pairs = [p for p, l in zip(train_pairs, train_labels) if l == 1]

# Veri Üretimi
X_data, y_data = [], []
AUGMENT_FACTOR = 5  # Optimizasyon hızlı olsun diye 5, finalde 10 yaparız

print(f"   Generating augmented data (x{AUGMENT_FACTOR})...")
for i in tqdm(range(len(match_pairs))):
    img1, img2 = match_pairs[i]

    # Cache keylerini önceki scriptle uyumlu kullanıyoruz
    v_enc, _ = get_encoding_from_image(img2, "", embedding_cache, f"train_verif_{i}")
    o_enc, _ = get_encoding_from_image(img1, "", embedding_cache, f"train_orig_{i}")

    if v_enc is None or o_enc is None: continue

    # 1. Pozitif Örnek (Orijinal)
    diff = np.abs(o_enc - v_enc)
    mult = o_enc * v_enc
    sim = np.dot(o_enc, v_enc)
    feat = np.concatenate([o_enc, v_enc, diff, mult, [sim]])  # 2049 Dim
    X_data.append(feat)
    y_data.append(1.0)

    # 2. Negatif Örnekler (Augmentation)
    for k in range(AUGMENT_FACTOR):
        deg_fn = degradation_pool[(i + k) % len(degradation_pool)]
        # Rastgele bozulma şiddeti
        strength = np.random.randint(2, 7)
        d_img = deg_fn(img1.copy(), strength=strength)

        # Bunu cache'lemeye gerek yok, çok rastgele
        d_enc, _ = get_encoding_from_image(d_img, "", {}, "temp")
        if d_enc is None: continue

        sim_d = np.dot(v_enc, d_enc)
        # Threshold: 0.35 altı negatif sayılsın (Hard Negative mantığı)
        lbl = 1.0 if sim_d > 0.35 else 0.0

        diff = np.abs(d_enc - v_enc)
        mult = d_enc * v_enc
        feat = np.concatenate([d_enc, v_enc, diff, mult, [sim_d]])
        X_data.append(feat)
        y_data.append(lbl)

# Tensorlara Çevir
X_tensor = torch.tensor(np.array(X_data, dtype=np.float32))
y_tensor = torch.tensor(np.array(y_data, dtype=np.float32)).unsqueeze(1)

# Train - Val Split (%80 Train, %20 Val)
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

print(f"✅ Veri Hazır: Train {len(train_ds)}, Val {len(val_ds)}")


# ==========================================
# 2. OPTUNA OBJECTIVE FUNCTION
# ==========================================
def objective(trial):
    # --- A. Hiperparametre Uzayı (Search Space) ---
    n_layers = trial.suggest_int("n_layers", 2, 4)  # Kaç katmanlı olsun?
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    hidden_dim_start = trial.suggest_categorical("hidden_start", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

    # --- B. Dinamik Model Oluşturma ---
    layers = []
    input_dim = 2049  # Sabit (512+512+512+512+1)
    current_dim = hidden_dim_start

    for i in range(n_layers):
        layers.append(nn.Linear(input_dim, current_dim))
        layers.append(nn.BatchNorm1d(current_dim))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Dropout(dropout_rate))

        input_dim = current_dim
        current_dim = current_dim // 2  # Her katmanda boyutu yarıya indir
        if current_dim < 64: current_dim = 64

    layers.append(nn.Linear(input_dim, 1))  # Çıkış katmanı

    model = nn.Sequential(*layers).to(device)

    # --- C. Eğitim ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    # Hızlı Eğitim (Trial başına 10 epoch yeterli)
    for epoch in range(10):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Pruning (Kötü giden eğitimi erken bitir)
        # model.eval() ... (Basitlik için şimdilik kapalı)

    # --- D. Validasyon (Sonuç) ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    return avg_val_loss  # Optuna bunu minimize etmeye çalışacak


# ==========================================
# 3. OPTİMİZASYONU BAŞLAT
# ==========================================
print("\n🔍 Optuna Hiperparametre Araması Başlıyor...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)  # 30 farklı kombinasyon dene

print("\n🏆 EN İYİ PARAMETRELER BULUNDU:")
print(study.best_params)

# Kaydet
with open("best_hyperparams.pkl", "wb") as f:
    pickle.dump(study.best_params, f)
print("✅ Parametreler 'best_hyperparams.pkl' dosyasına kaydedildi.")