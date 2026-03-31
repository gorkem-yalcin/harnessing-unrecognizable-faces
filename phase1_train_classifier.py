import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pickle

# 🌟 KONTROL MERKEZİNDEN AYARLARI ÇEK
from config import *

# ==========================================
# ⚙️ PARAMETRELER (ARTIK CONFIG'DEN GELİYOR)
# ==========================================
# Dosya yolları config.py'dan otomatik olarak çekiliyor!
INPUT_DATA_FILE = LFW_TRAIN_DATA
MODEL_SAVE_PATH = CLASSIFIER_PATH

# Eğitim Hiperparametreleri
BATCH_SIZE = 128
EPOCHS = 50  # Veri seti çok karmaşık olmadığı için 15-20 epoch yeterli olacaktır
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. VERİ HAZIRLIĞI
# ==========================================
print(f"🚀 Eğitim Başlıyor... Cihaz: {DEVICE}")
print(f"🧠 Eğitilen Kalite (Hakem) Modeli: {QUALITY_MODEL}")
print(f"📦 Veri yükleniyor: {INPUT_DATA_FILE}")

with open(INPUT_DATA_FILE, 'rb') as f:
    df = pickle.load(f)

# X: Girdi (Sadece bozuk yüzün embedding'i)
X = np.vstack(df['degraded_embedding'].values)
# y: Hedef (0 veya 1 Hard Label)
y = df['hard_label'].values.astype(np.float32).reshape(-1, 1)

print(f"📊 Toplam Veri: {X.shape[0]} satır. Sınıf Dağılımı: 0 ({int(len(y) - y.sum())}), 1 ({int(y.sum())})")

# Train/Test ayrımı (%80 Eğitim, %20 Validasyon)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# PyTorch Tensorlerine Çevirme
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ==========================================
# 2. MODEL MİMARİSİ (Recognizability Model)
# ==========================================
# 512 boyutlu ArcFace vektörünü alıp, 0 ile 1 arası olasılık üreten ağ
class RecognizabilityClassifier(nn.Module):
    def __init__(self, input_dim=512):
        super(RecognizabilityClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 1)
            # Binary Cross Entropy with Logits kullanacağımız için Sigmoid eklemiyoruz
        )

    def forward(self, x):
        return self.net(x)


model = RecognizabilityClassifier().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 3. EĞİTİM DÖNGÜSÜ
# ==========================================
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Başarı hesaplama (Sigmoid > 0.5 ise 1'dir, yani output > 0)
        predicted = (outputs > 0.0).float()
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    train_acc = 100 * correct / total

    # Validasyon
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

            predicted = (outputs > 0.0).float()
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()

    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(test_loader)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {total_loss / len(train_loader):.4f} Acc: %{train_acc:.2f} | Val Loss: {avg_val_loss:.4f} Acc: %{val_acc:.2f}")

    # En iyi modeli kaydetme
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"\n✅ Eğitim Tamamlandı! En iyi model '{MODEL_SAVE_PATH}' olarak kaydedildi.")