import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# ==== CONFIG ====
DATASET_DIR = "dataset"
MODEL_OUTPUT = "resnet50_face_mask.pth"
BATCH_SIZE = 32
EPOCHS = 10
VAL_SPLIT = 0.2
LEARNING_RATE = 1e-4
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==== LOAD DATASET ====
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
class_names = full_dataset.classes
print(f"📂 Classes: {class_names}")

val_size = int(VAL_SPLIT * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== MODEL ====
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==== TRAIN LOOP ====
print("🚀 Training started...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

# ==== SAVE MODEL ====
torch.save(model.state_dict(), MODEL_OUTPUT)
print(f"✅ Model saved to: {MODEL_OUTPUT}")

# ==== EVALUATE ====
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

acc = 100 * correct / total
print(f"🎯 Validation Accuracy: {acc:.2f}%")
