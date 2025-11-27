if __name__ == "__main__":
    from collections import Counter
    from sklearn.model_selection import StratifiedShuffleSplit
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    from sklearn.metrics import precision_score, recall_score, f1_score
    from tqdm import tqdm
    import numpy as np

    # ✅ Paths
    data_dir = 'output'  # Folder with 'recognizable' and 'unrecognizable'

    # ✅ Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ✅ Load weights and transforms
    weights = MobileNet_V3_Large_Weights.DEFAULT
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # ✅ Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ✅ Dataset and split
    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names = full_dataset.classes
    targets = np.array(full_dataset.targets)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    train_set.dataset.transform = train_transform
    val_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # ✅ Load MobileNetV3 and modify
    model = mobilenet_v3_large(weights=weights)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only classifier
    model.classifier = nn.Sequential(
        nn.Linear(960, 1280),  # same as before
        nn.Hardswish(),  # same
        nn.Dropout(0.3),  # optional higher dropout
        nn.Linear(1280, 1)  # binary classification
    )
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.features[-1:].parameters():
        param.requires_grad = True
    model = model.to(device)

    # ✅ Loss & Optimizer
    counts = Counter(full_dataset.targets)
    pos_weight = torch.tensor([counts[0] / counts[1]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # ✅ Training loop
    epochs = 50
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # ✅ Validation
        model.eval()
        val_correct, val_total = 0, 0
        val_preds_all, val_labels_all = [], []

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.float().unsqueeze(1).to(device)
                val_outputs = model(val_images)
                val_probs = torch.sigmoid(val_outputs)
                val_preds = (val_probs > 0.5).float()

                val_preds_all.extend(val_preds.cpu().numpy())
                val_labels_all.extend(val_labels.cpu().numpy())
                val_correct += (val_preds == val_labels).sum().item()
                val_total += val_labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        precision = precision_score(val_labels_all, val_preds_all)
        recall = recall_score(val_labels_all, val_preds_all)
        f1 = f1_score(val_labels_all, val_preds_all)

        print(f"Epoch {epoch + 1}/{epochs} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f} - "
              f"F1: {f1:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f}")
