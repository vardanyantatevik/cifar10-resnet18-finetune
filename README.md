## 📈 Results Visualization

### Training Curves
![Training Curves](training_curves.png)

### Confusion Matrix & Per-Class Accuracy
![Evaluation Results](evaluation_results.png)

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

## 🛠️ Technologies & Hyperparameters

- **Architecture**: ResNet18 (ImageNet pretrained)
- **Optimizer**: Adam (lr=1e-4)
- **Scheduler**: ReduceLROnPlateau
- **Data Augmentation**: RandomHorizontalFlip, RandomRotation(15), ColorJitter
- **Best Params**: lr=0.0001, dropout=0.3, unfreeze_layers=2