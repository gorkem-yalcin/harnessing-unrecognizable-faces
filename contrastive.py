import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# ----- Model Definitions -----
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dist = torch.norm(out1 - out2, dim=1)
        loss = label * dist**2 + (1 - label) * torch.clamp(self.margin - dist, min=0)**2
        return loss.mean()


class Classifier(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def generate_pairs(X, y, max_pairs=5000):
    np.random.seed(42)
    pos, neg = [], []
    for i in range(len(X)):
        j = np.random.randint(0, len(X))
        if y[i] == y[j]:
            pos.append((X[i], X[j], 1))
        else:
            neg.append((X[i], X[j], 0))
        if len(pos) >= max_pairs and len(neg) >= max_pairs:
            break
    return pos[:max_pairs] + neg[:max_pairs]


# ----- Load Data -----
with open("binary_classifier_data.pkl", "rb") as f:
    X, y = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Split -----
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val = np.array(X_train), np.array(X_val)
y_train, y_val = np.array(y_train), np.array(y_val)

# ----- Contrastive Pretraining -----
embedding_net = EmbeddingNet(input_dim=512).to(device)
contrastive_loss_fn = ContrastiveLoss()
contrastive_opt = optim.Adam(embedding_net.parameters(), lr=1e-3)

pairs = generate_pairs(X_train, y_train, max_pairs=3000)
X1 = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
X2 = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
Y = torch.tensor([p[2] for p in pairs], dtype=torch.float32)
contrastive_loader = DataLoader(TensorDataset(X1, X2, Y), batch_size=256, shuffle=True)

print("Pretraining contrastive encoder...")
for epoch in range(15):
    embedding_net.train()
    total_loss = 0
    for x1, x2, label in contrastive_loader:
        x1, x2, label = x1.to(device), x2.to(device), label.to(device)
        emb1 = embedding_net(x1)
        emb2 = embedding_net(x2)
        loss = contrastive_loss_fn(emb1, emb2, label)
        contrastive_opt.zero_grad()
        loss.backward()
        contrastive_opt.step()
        total_loss += loss.item()
    print(f"[Contrastive Epoch {epoch+1}] Loss: {total_loss / len(contrastive_loader):.4f}")

# ----- Embed -----
with torch.no_grad():
    train_emb = embedding_net(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu()
    val_emb = embedding_net(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu()

# ----- Classification Training -----
train_dataset = TensorDataset(train_emb, torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

val_X = val_emb.to(device)
val_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

model = Classifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
epochs = 200
best_val_acc = 0

train_accs, val_accs = [], []

print("Training classifier...")
for epoch in range(epochs):
    model.train()
    total_loss, correct = 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        probs = torch.sigmoid(logits)
        weights = 1.0 - torch.abs(probs - y_batch)
        loss = (criterion(logits, y_batch) * weights).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (probs > 0.5).float()
        correct += (preds == y_batch).sum().item()

    train_acc = correct / len(train_loader.dataset)
    train_accs.append(train_acc)

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(val_X)
        val_probs = torch.sigmoid(val_logits)
        val_preds = (val_probs > 0.5).float()
        val_acc = (val_preds == val_y).float().mean().item()
        val_accs.append(val_acc)

        precision = precision_score(y_val, val_preds.cpu())
        recall = recall_score(y_val, val_preds.cpu())
        f1 = f1_score(y_val, val_preds.cpu())

        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, F1: {f1:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_classifier.pt")

# ----- Save Plot -----
plt.figure(figsize=(10, 5))
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("training_accuracy_plot.png")
