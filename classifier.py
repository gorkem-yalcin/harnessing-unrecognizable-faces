import pickle
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from collections import Counter
import os
import csv


class Classifier(nn.Module):
    def __init__(self, dropout, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# Load data
with open("binary_classifier_data.pkl", "rb") as f:
    X_train, y_train = pickle.load(f)

# Label distribution
label_counts = Counter(y_train)
print(f"Label distribution in the full dataset: {label_counts}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

best_accuracies = []
# Split
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=True
)

train_label_counts = Counter(y_train_split)
val_label_counts = Counter(y_val)
print(f"Label distribution in the training set: {train_label_counts}")
print(f"Label distribution in the validation set: {val_label_counts}")

# Tensor conversion
X_tensor = torch.tensor(X_train_split, dtype=torch.float32)
y_tensor = torch.tensor(y_train_split, dtype=torch.float32).unsqueeze(1)
val_X_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
val_y_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

# Dataset and config
dataset = TensorDataset(X_tensor, y_tensor)
learning_rates = [1e-4]
weight_decays = [1e-6]
dropout_values = [0.1]
batch_sizes = [128]

for lr in learning_rates:
    for weight_decay in weight_decays:
        for batch_size in batch_sizes:
            for dropout in dropout_values:
                run_id = f"lr{lr}_wd{weight_decay}_bs{batch_size}_dropout{dropout}"
                print(f"\nStarting run {run_id}")

                epochs = 500
                train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                model = Classifier(dropout).to(device)

                pos_weight = torch.tensor([(len(y_train_split) - sum(y_train_split)) / sum(y_train_split)]).to(device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

                # Best metrics
                best_val_acc = 0.0
                best_train_loss = None
                best_train_acc = None
                best_val_loss = None
                best_epoch = -1
                best_model_state = None

                # Tracking
                train_losses, train_accuracies = [], []
                val_losses, val_accuracies = [], []

                # Training loop
                for epoch in range(epochs):
                    model.train()
                    running_loss, correct, total = 0.0, 0, 0

                    for X_batch, y_batch in train_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        optimizer.zero_grad()
                        output = model(X_batch)
                        loss = criterion(output, y_batch)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item() * X_batch.size(0)
                        probs = torch.sigmoid(output)
                        preds = (probs > 0.5).float()
                        correct += (preds == y_batch).sum().item()
                        total += y_batch.size(0)

                    epoch_loss = running_loss / total
                    epoch_acc = correct / total
                    train_losses.append(epoch_loss)
                    train_accuracies.append(epoch_acc)

                    # Validation
                    model.eval()
                    val_running_loss = 0.0
                    val_correct = 0
                    val_total = 0

                    with torch.no_grad():
                        for val_X_batch, val_y_batch in DataLoader(TensorDataset(val_X_tensor, val_y_tensor), batch_size=batch_size):
                            val_X_batch, val_y_batch = val_X_batch.to(device), val_y_batch.to(device)
                            val_output = model(val_X_batch)
                            val_loss = criterion(val_output, val_y_batch)
                            val_running_loss += val_loss.item() * val_X_batch.size(0)

                            val_probs = torch.sigmoid(val_output)
                            val_preds = (val_probs > 0.5).float()
                            val_correct += (val_preds == val_y_batch).sum().item()
                            val_total += val_y_batch.size(0)

                    val_loss = val_running_loss / val_total
                    val_acc = val_correct / val_total
                    scheduler.step(val_loss)

                    val_losses.append(val_loss)
                    val_accuracies.append(val_acc)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_val_loss = val_loss
                        best_train_loss = epoch_loss
                        best_train_acc = epoch_acc
                        best_epoch = epoch + 1
                        best_model_state = model.state_dict()
                        print(f"New best model found at epoch {best_epoch} with val acc: {val_acc:.4f}")

                    print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - "
                          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

                # Save best model and metrics
                best_accuracies.append(best_val_acc)
                os.makedirs("results", exist_ok=True)
                if best_model_state is not None:
                    torch.save(best_model_state, f"results/{run_id}_best_model.pt")

                    with open(f"results/{run_id}_best_metrics.txt", "w") as f:
                        f.write(f"Best Epoch: {best_epoch}\n")
                        f.write(f"Train Loss: {best_train_loss:.4f}\n")
                        f.write(f"Train Accuracy: {best_train_acc:.4f}\n")
                        f.write(f"Validation Loss: {best_val_loss:.4f}\n")
                        f.write(f"Validation Accuracy: {best_val_acc:.4f}\n")

                    # Append to summary CSV
                    csv_path = "results/results_summary.csv"
                    file_exists = os.path.isfile(csv_path)
                    with open(csv_path, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        if not file_exists:
                            writer.writerow(["run_id", "best_epoch", "train_accuracy", "train_loss", "val_accuracy", "val_loss"])
                        writer.writerow([
                            run_id,
                            best_epoch,
                            round(best_train_acc, 4),
                            round(best_train_loss, 4),
                            round(best_val_acc, 4),
                            round(best_val_loss, 4)
                        ])

                    print(f"Best model saved with validation accuracy: {best_val_acc:.4f}")

                # Plotting
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(train_losses, label='Train Loss')
                plt.plot(val_losses, label='Validation Loss')
                plt.title("Loss over Epochs")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(train_accuracies, label='Train Accuracy')
                plt.plot(val_accuracies, label='Validation Accuracy')
                plt.title("Accuracy over Epochs")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()

                plt.tight_layout()
                plt.savefig(f"results/{run_id}_training_curves.png")
                plt.close()
print(best_accuracies)