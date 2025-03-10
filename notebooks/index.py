import torch
import torch.nn as tnn
import torch.optim as opt
import timm
import cv2
import albumentations as album
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader

print(torch.cuda.is_available())
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
PROCESSOR = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
TRAIN_DIR = "datasets/train"
VAL_DIR = "datasets/val"

class FolderDataset(Dataset):
    def __init__(self, asal_jaga, tabdeeli=None):
        self.asal_jaga = asal_jaga
        self.tabdeeli = tabdeeli
        self.samples = []
        for label in sorted(os.listdir(asal_jaga)):
            label_jaga = os.path.join(asal_jaga, label)
            if not os.path.isdir(label_jaga):
                continue
            for file in os.listdir(label_jaga):
                if file.lower().endswith(('.jpeg')):
                    self.samples.append((os.path.join(label_jaga, file), int(label)))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_rasta, label = self.samples[idx]
        tasveer = cv2.imread(img_rasta)
        if tasveer is None:
            raise FileNotFoundError(f"Tasveer nahi mili: {img_rasta}")
        tasveer = cv2.cvtColor(tasveer, cv2.COLOR_BGR2RGB)
        if self.tabdeeli:
            tasveer = self.tabdeeli(image=tasveer)["image"]
        return tasveer, torch.tensor(label, dtype=torch.long)

tabdeeli = album.Compose([
    album.Resize(IMG_SIZE, IMG_SIZE),
    album.RandomBrightnessContrast(p=0.3),
    album.HorizontalFlip(p=0.5),
    album.Rotate(limit=20, p=0.5),
    album.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

class HybridModel(tnn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.cnn = timm.create_model("resnet50", pretrained=True, num_classes=1000)
        self.transformer = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1000)
        self.fc = tnn.Linear(2000, 5)

    def forward(self, x):
        cnn_out = self.cnn(x)
        trans_out = self.transformer(x)
        out = torch.cat([cnn_out, trans_out], dim=1)
        return self.fc(out)

def model_ka_jayza(model, val_loader):
    model.eval()
    sahi, total = 0, 0
    with torch.no_grad():
        for tasveerain, labels in tqdm(val_loader, desc="Validation kr raha hn hosla rkho"):
            tasveerain, labels = tasveerain.to(PROCESSOR), labels.to(PROCESSOR)
            outputs = model(tasveerain)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            sahi += (predicted == labels).sum().item()
    val_acc = sahi / total
    print(f"validation Accuracy: {val_acc:.4f}")
    return val_acc

def model_ka_training(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss, sahi, total = 0.0, 0, 0
        for tasveerain, labels in tqdm(train_loader, desc=f"Sikharha hn isko  {epoch+1}/{epochs} itna hogaya hy"):
            tasveerain, labels = tasveerain.to(PROCESSOR), labels.to(PROCESSOR)
            optimizer.zero_grad()
            outputs = model(tasveerain)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            sahi += (predicted == labels).sum().item()
        train_acc = sahi / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}, Training Acc: {train_acc:.4f}")
        val_acc = model_ka_jayza(model, val_loader)
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/224_img_size_model.pth")
            print(f"ye wala best model save krliya hy iski Validation Acc: {val_acc:.4f}")

if __name__ == '__main__':
    train_dataset = FolderDataset(TRAIN_DIR, tabdeeli=tabdeeli)
    val_dataset = FolderDataset(VAL_DIR, tabdeeli=tabdeeli)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = HybridModel().to(PROCESSOR)
    criterion = tnn.CrossEntropyLoss()
    optimizer = opt.AdamW(model.parameters(), lr=LEARNING_RATE)

    model_ka_training(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
    print("Jani mokammal sikhadia hy model ko aur bol bhai hazir hy")

    model.load_state_dict(torch.load("models/khtarnak.pth", map_location=PROCESSOR))
    model.eval()


    fake_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(PROCESSOR)
    scripted_model = torch.jit.trace(model, fake_input)
    torch.jit.save(scripted_model, "models/224_img_size_model.pt")
    print("Model successfully converted to TorchScript and saved as models/224_img_size_model.pt")