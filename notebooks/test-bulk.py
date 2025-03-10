import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import pandas as pd

IMAGE_SIZE = 224
PROCESSOR = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.jit.load("224_img_size_model.pt", map_location=PROCESSOR)
model.eval()
print("model load hogaya!")

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

TEST_FOLDER = "datasets/test"

class_mapping = {
    0: "No_Dr",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

results = []
total_tasveerain = 0
sahi_predictions = 0

for asal_label in sorted(os.listdir(TEST_FOLDER)):
    subfolder_path = os.path.join(TEST_FOLDER, asal_label)
    if not os.path.isdir(subfolder_path):
        continue
    
    for filename in os.listdir(subfolder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(subfolder_path, filename)
            try:
                tasveer_tensor = tasveer_tayyar_karo(image_path)
                with torch.no_grad():
                    output = model(tasveer_tensor)
                    predicted_class = torch.argmax(output, dim=1).item()
                
                asal_label_int = int(asal_label)
                total_tasveerain += 1
                if predicted_class == asal_label_int:
                    sahi_predictions += 1
                
                predicted_label = class_mapping.get(predicted_class, "Unknown")
                results.append((image_path, asal_label_int, predicted_class, predicted_label))
                print(f"Tasveer: {image_path}, Asal: {asal_label_int}, Predicted: {predicted_class} ({predicted_label})")
            except Exception as e:
                print(f"Tasveer process karte hue error: {image_path}: {e}")

accuracy = sahi_predictions / total_tasveerain if total_tasveerain > 0 else 0
print(f"\nTest Accuracy: {accuracy:.4f} ({sahi_predictions}/{total_tasveerain})")

df_results = pd.DataFrame(results, columns=["image_path", "ground_truth", "predicted_class", "predicted_label"])
df_results.to_csv("test_predictions.csv", index=False)
print("Inference complete. Predictions save kardi hain test_predictions.csv mein")
