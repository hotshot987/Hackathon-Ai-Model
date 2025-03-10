import sys
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import os
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog

IMAGE_SIZE = 224
PROCESSOR = "cuda" if torch.cuda.is_available() else "cpu"

class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.cnn = timm.create_model("resnet50", pretrained=False, num_classes=1000)
        self.transformer = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1000)
        self.fc = nn.Linear(2000, 5)
    
    def forward(self, x):
        cnn_out = self.cnn(x)
        trans_out = self.transformer(x)
        out = torch.cat([cnn_out, trans_out], dim=1)
        return self.fc(out)

model = torch.jit.load("224_img_size_model.pt", map_location=PROCESSOR)
model.eval()
print("Model load hogaya!")

def tasveer_tayyar_karo(image_path):
    tasveer = cv2.imread(image_path)
    if tasveer is None:
        raise FileNotFoundError(f"Tasveer nahi mili: {image_path}")
    tasveer = cv2.cvtColor(tasveer, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    tasveer = transform(image=tasveer)["image"]
    return tasveer.unsqueeze(0).to(PROCESSOR)

class_mapping = {
    0: "No_Dr",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

def process_tasveer(file_path):
    print("Selected file:", file_path)
    tasveer_tensor = tasveer_tayyar_karo(file_path)
    with torch.no_grad():
        output = model(tasveer_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    predicted_label = class_mapping.get(predicted_class, "Unknown")
    print(f"Predicted Class: {predicted_label}")

while True:
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        sys.argv = sys.argv[:1]
    else:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        process_tasveer(file_path)
    else:
        print("Koi file select nahi hui. Exiting...")
        break
