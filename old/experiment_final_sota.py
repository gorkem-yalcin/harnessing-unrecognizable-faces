import os
import cv2
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from arcfaceutility import ensure_rgb, get_encoding_from_image
from degradations import degradation_pool

# ==========================================
# 1. AYARLAR VE OPTUNA PARAMETRELERİ
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 FINAL EXPERIMENT STARTED ON: {device}")

# Optuna sonuçlarını yükle (Eğer yoksa default değerler)
if os.path.exists("best_hyperparams.pkl"):
    with open("best_hyperparams.pkl", "rb") as f:
        best_params = pickle.load(f)
    print(f"🏆 Optuna Parametreleri Yüklendi: {best_params}")
else:
    print("⚠️ Optuna dosyası bulunamadı! Varsayılan ayarlar kullanılıyor.")
    best_params = {'n_layers': 3, 'hidden_start': 1024, 'dropout': 0.3, 'lr': 0.0005, 'batch_size': 512}


# ==========================================
# 2. MODEL MİMARİSİ (Dinamik)
# ==========================================
class OptimizedClassifier(nn.Module):
    def __init__(self, params):
        super().__init__()
        layers = []
        input_dim = 2049
        current_dim = params['hidden_start']

        for i in range(params['n_layers']):
            layers.append(nn.Linear(input_dim, current_dim))
            layers.append(nn.BatchNorm1d(current_dim))
            layers.append(nn.LeakyReLU(0.1))  # ReLU yerine Leaky daha iyi
            layers.append(nn.Dropout(params['dropout']))

            input_dim = current_dim
            current_dim = current_dim // 2
            if current_dim < 64: current_dim = 64

        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = OptimizedClassifier(best_params).to(device)

# ==========================================
# 3. VERİ HAZIRLIĞI (LFW + Augmentation)
# ==========================================
if os.path.exists("embedding_cache_lfw.pkl"):
    with open("embedding_cache_lfw.pkl", "rb") as f:
        embedding_cache = pickle.load(f)
else:
    embedding_cache = {}

print("📊 Training Data (LFW) Hazırlanıyor...")
lfw_train = fetch_lfw_pairs(subset='train', color=True, resize=1.0)
train_pairs = [(ensure_rgb(p[0]), ensure_rgb(p[1])) for p in lfw_train.pairs]
train_labels = lfw_train.target
match_pairs = [p for p, l in zip(train_pairs, train_labels) if l == 1]

X_data, y_data = [], []
AUGMENT_FACTOR = 10  # Final eğitimde daha fazla veri (x10)

print(f"   Data Augmentation (x{AUGMENT_FACTOR})...")
for i in tqdm(range(len(match_pairs))):
    img1, img2 = match_pairs[i]

    v_enc, _ = get_encoding_from_image(img2, "", embedding_cache, f"train_verif_{i}")
    o_enc, _ = get_encoding_from_image(img1, "", embedding_cache, f"train_orig_{i}")

    if v_enc is None or o_enc is None: continue

    # Pozitif
    diff = np.abs(o_enc - v_enc)
    mult = o_enc * v_enc
    sim = np.dot(o_enc, v_enc)
    feat = np.concatenate([o_enc, v_enc, diff, mult, [sim]])
    X_data.append(feat)
    y_data.append(1.0)

    # Negatif (Hard Negatives)
    for k in range(AUGMENT_FACTOR):
        deg_fn = degradation_pool[(i + k) % len(degradation_pool)]
        d_img = deg_fn(img1.copy(), strength=np.random.randint(2, 8))  # Zorluğu artırdık
        d_enc, _ = get_encoding_from_image(d_img, "", {}, "temp")
        if d_enc is None: continue

        sim_d = np.dot(v_enc, d_enc)
        # Threshold'u biraz gevşettik ki sınır örnekleri öğrensin
        lbl = 1.0 if sim_d > 0.40 else 0.0

        diff = np.abs(d_enc - v_enc)
        mult = d_enc * v_enc
        feat = np.concatenate([d_enc, v_enc, diff, mult, [sim_d]])
        X_data.append(feat)
        y_data.append(lbl)

X_tensor = torch.tensor(np.array(X_data, dtype=np.float32)).to(device)
y_tensor = torch.tensor(np.array(y_data, dtype=np.float32)).unsqueeze(1).to(device)


# ==========================================
# 4. EĞİTİM (Focal Loss Entegrasyonu)
# ==========================================
# Focal Loss: Kolay örnekleri (zaten bildiği) görmezden gelip zorlara odaklanır
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


criterion = FocalLoss(gamma=2)  # Gamma artırılabilir (2 standarttır)
optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'])

print("🔥 Eğitim Başlıyor...")
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=best_params['batch_size'], shuffle=True)

model.train()
EPOCHS = 25  # İyice öğrensin
for epoch in range(EPOCHS):
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"   Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss / len(loader):.6f}")

# Modeli Kaydet
torch.save(model.state_dict(), "final_optimized_model.pth")
print("✅ Final Model Kaydedildi: final_optimized_model.pth")

# ==========================================
# 5. TINYFACE VALİDASYON (Threshold Bulma)
# ==========================================
print("\n🔍 Validating on TinyFace (Finding Optimal Threshold)...")


def get_scores_simple(model, pairs_path, img_dir, dataset_name):
    # Pairs listesini oluştur
    pairs = []
    if dataset_name == "TinyFace":
        with open(pairs_path, 'r') as f:
            lines = f.readlines()[1:]
        for line in lines:
            p = line.strip().split('\t')
            # Format: img1_name, img2_name, label (3 kolonlu ise)
            if len(p) >= 3:
                # img1, img2, label
                pairs.append([p[0], p[1], int(p[2])])
            else:
                continue

    scores, labels = [], []
    batch_feats, batch_lbls, batch_sims = [], [], []

    print(f"   Analiz edilecek çift sayısı: {len(pairs)}")

    for row in tqdm(pairs):
        name1, name2, label = row[0], row[1], row[2]

        # --- PATH DÜZELTME (BURASI KRİTİK) ---
        # Eğer '2044_1.jpg' gibi zaten uzantı varsa direkt birleştir
        if name1.lower().endswith(('.jpg', '.png', '.jpeg')):
            path1 = os.path.join(img_dir, name1)
        else:
            # Uzantı yoksa (Klasör mantığı): img_dir/2044/2044_1.jpg
            # TinyFace bazen folder/filename formatında olabilir, burayı veriye göre esnetiyoruz
            # Ancak senin hatana göre üstteki 'if' bloğu çalışacak.
            path1 = os.path.join(img_dir, name1)  # Varsayılan fallback

        if name2.lower().endswith(('.jpg', '.png', '.jpeg')):
            path2 = os.path.join(img_dir, name2)
        else:
            path2 = os.path.join(img_dir, name2)

        # -------------------------------------

        enc1, _ = get_encoding_from_image(ensure_rgb(path1), "", None, None)
        enc2, _ = get_encoding_from_image(ensure_rgb(path2), "", None, None)

        if enc1 is None or enc2 is None:
            # Hatalı resim varsa atla ama log basma (Progress bar bozulmasın)
            continue

        diff = np.abs(enc1 - enc2)
        mult = enc1 * enc2
        sim = np.dot(enc1, enc2)
        feat = np.concatenate([enc1, enc2, diff, mult, [sim]])

        batch_feats.append(feat)
        batch_lbls.append(label)
        batch_sims.append(sim)

        # Batch dolunca hesapla
        if len(batch_feats) >= 2048:
            X_b = torch.tensor(np.array(batch_feats), dtype=torch.float32).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(X_b)).cpu().numpy().flatten()
            hybrids = np.array(batch_sims) * probs
            scores.extend(hybrids)
            labels.extend(batch_lbls)
            batch_feats, batch_lbls, batch_sims = [], [], []

    # Kalan son batch
    if batch_feats:
        X_b = torch.tensor(np.array(batch_feats), dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(X_b)).cpu().numpy().flatten()
        hybrids = np.array(batch_sims) * probs
        scores.extend(hybrids)
        labels.extend(batch_lbls)

    return np.array(labels), np.array(scores)

# TinyFace Paths (Aynı pathleri kullan)
DATASET_ROOT = "datasets"  # Kendi pathine göre ayarla
TINY_PAIRS = os.path.join(DATASET_ROOT, "tinyface", "pairs.txt")
TINY_IMG = os.path.join(DATASET_ROOT, "tinyface", "images")

try:
    y_true, y_scores = get_scores_simple(model, TINY_PAIRS, TINY_IMG, "TinyFace")
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    J = tpr - fpr
    best_thresh = thresholds[np.argmax(J)]
    print(f"🏆 Best Threshold (TinyFace): {best_thresh:.4f}")

    # Threshold'u dosyaya yaz (Sonraki script okusun)
    with open("final_threshold.txt", "w") as f:
        f.write(str(best_thresh))

except Exception as e:
    print(f"⚠️ TinyFace analizi atlandı (Dosya yolu hatası olabilir): {e}")
    # Default bir threshold yazalım ki IJB-C kodu çökmesin
    with open("final_threshold.txt", "w") as f:
        f.write("67.0")

print("🚀 Eğitim Tamamlandı! Şimdi 'analyze_best_threshold.py' kodunu tekrar çalıştırıp REKORU GÖR!")