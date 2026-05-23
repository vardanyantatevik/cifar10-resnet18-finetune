```markdown
# 🚀 CIFAR-10 Image Classification with ResNet18 Fine-Tuning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Overview

This project implements **fine-tuning of ResNet18** for CIFAR-10 image classification with **95.16% test accuracy**, exceeding the typical benchmark (93-95%). The model is trained with hyperparameter optimization, data augmentation, and comprehensive evaluation metrics.

### Key Features
- ✅ **ResNet18 fine-tuning** with pretrained ImageNet weights
- ✅ **Hyperparameter grid search** (learning rate, dropout, optimizer)
- ✅ **Data augmentation** (random flip, rotation, color jitter)
- ✅ **Learning rate scheduling** (ReduceLROnPlateau)
- ✅ **Comprehensive evaluation** (accuracy, precision, recall, F1-score)
- ✅ **Confusion matrix & per-class accuracy visualization**
- ✅ **Inference pipeline** for custom images

## 📊 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **95.16%** |
| Precision | 0.9522 |
| Recall | 0.9516 |
| F1-Score | 0.9516 |
| Benchmark | 93-95% ✅ |

### Best Hyperparameters
```yaml
learning_rate: 0.0001
optimizer: adam
unfreeze_layers: 2
dropout_rate: 0.3
batch_size: 32
num_epochs: 15
```

## 🏗️ Project Structure

```
CIFAR10-ResNet18-Finetune/
├── train.py                 # Main training script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── sample_images/          # Sample images for inference
│   └── example.jpg
├── outputs/                # Generated files (after running)
│   ├── best_model.pth      # Best model checkpoint
│   ├── final_model.pth     # Final saved model
│   ├── training_curves.png # Loss & accuracy plots
│   ├── evaluation_results.png # Confusion matrix
│   └── inference_result.png # Inference visualization
└── data/                   # CIFAR-10 dataset (auto-downloaded)
```

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/vardanyantatevik/CIFAR10-ResNet18-Finetune.git
cd CIFAR10-ResNet18-Finetune
```

### 2. Create virtual environment (optional but recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify CUDA (optional, for GPU training)
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## 🚀 Usage

### Training
```bash
python train.py
```

This will:
1. Download CIFAR-10 dataset automatically
2. Perform hyperparameter search (4 combinations, ~35 minutes)
3. Train the best model for 15 epochs
4. Generate evaluation plots
5. Save model checkpoints

### Inference on Custom Images

```python
from train import load_model, inference
from torchvision import transforms
import torch

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('./final_model.pth', device)

# Define transform (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Run inference
predicted_class, confidence, probs, labels = inference(
    model, 'path/to/your/image.jpg', transform, device, class_names
)
print(f"Predicted: {class_names[predicted_class]} (confidence: {confidence:.3f})")
```

## 📈 Results Visualization

### Training Curves
![Training Curves](training_curves.png)

### Confusion Matrix & Per-Class Accuracy
![Evaluation Results](evaluation_results.png)

### Inference Example
![Inference Result](inference_result.png)

## 🧪 Hyperparameter Search Results

| Rank | Parameters | Validation Accuracy |
|------|------------|---------------------|
| 1 | lr=0.0001, dropout=0.3 | **92.68%** |
| 2 | lr=0.0001, dropout=0.5 | 92.48% |
| 3 | lr=0.0005, dropout=0.3 | 89.44% |
| 4 | lr=0.0005, dropout=0.5 | 89.42% |

## 🎯 Benchmark Comparison

| Model | Expected Accuracy | Our Result |
|-------|------------------|------------|
| ResNet18 (paper) | 93-95% | **95.16%** ✅ |
| ResNet34 | 94-96% | - |

## 📝 Model Architecture Details

```python
ResNet18 (pretrained on ImageNet)
├── Conv1 + BatchNorm + ReLU + MaxPool
├── Layer1 (2 blocks, 64 channels)      # Frozen
├── Layer2 (2 blocks, 128 channels)     # Frozen
├── Layer3 (2 blocks, 256 channels)     # Fine-tuned
├── Layer4 (2 blocks, 512 channels)     # Fine-tuned
└── Custom Classifier Head
    ├── Dropout(0.3)
    ├── Linear(512 → 512)
    ├── ReLU
    ├── Dropout(0.3)
    └── Linear(512 → 10)
```

## 🔄 Training Configuration

```python
CONFIG = {
    "batch_size": 32,
    "num_epochs": 15,
    "input_size": 224,
    "val_fraction": 0.1,
    "learning_rate": 1e-4,
    "optimizer": "adam",
    "unfreeze_layers": 2,
    "dropout_rate": 0.3
}
```

## 📊 Data Augmentation

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
```

## 💻 System Requirements

- **Python**: 3.8+
- **RAM**: 8GB+ (16GB recommended)
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional, CPU works but slower)
- **Storage**: 2GB free space

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` in CONFIG
2. **Slow training**: Use GPU or reduce `num_epochs`
3. **Download stuck**: Set `download=True` manually in load_data()

## 📚 References

- [ResNet Paper (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch ResNet Documentation](https://pytorch.org/vision/stable/models/resnet.html)
