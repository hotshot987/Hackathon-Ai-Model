# Hackathon-Ai-Model
 Infymna Hackathon Ai Model
# Diabetic Retinopathy Detection using Hybrid CNN-Transformer

This project implements a hybrid model that combines a ResNet-50 and a Vision Transformer (ViT) for classifying diabetic retinopathy stages. The model is trained using a dataset organized into folder structures for training, validation, and testing.

## Folder Structure

Your dataset should be organized as follows:


- Each numbered folder corresponds to a class label:
  - 0: No_Dr
  - 1: Mild
  - 2: Moderate
  - 3: Severe
  - 4: Proliferative DR

## Requirements

- Python 3.7+
- PyTorch
- timm
- OpenCV
- Albumentations
- tqdm

You can install the necessary packages with:

pip install -r requirements.txt

*Prepare the Dataset:
Organize your dataset as follows:
datasets/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
├── val/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
└── test/
    ├── 0/
    ├── 1/
    ├── 2/
    ├── 3/
    └── 4/*

Train the Model:
Run the training script:

python index.py

if you just want to test the model 

Run Inference:
To test a single image, run the test script :
with model as same as path folder
python test-single.py
