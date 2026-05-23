import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# 1. Configuration
CONFIG = {
    "seed": 42,
    "data_dir": "./data",
    "num_classes": 10,
    "val_fraction": 0.1,
    "num_workers": 0,
    "batch_size": 32,
    "num_epochs": 15,
    "input_size": 224,
    
    # Model architecture variants to try
    "architectures": ["resnet18", "resnet34"],
    
    # Hyperparameters to search
    "hyperparams": {
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "optimizer": ["adam", "sgd"],
        "unfreeze_layers": [1, 2, 3],
        "dropout_rate": [0.0, 0.3, 0.5]
    },
    
    # Best model config
    "best_checkpoint": "./best_model.pth",
    "final_checkpoint": "./final_model.pth"
}

# 2. Seed and device Setup
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Data loading and augmentation
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((CONFIG["input_size"], CONFIG["input_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# Validation/Test transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((CONFIG["input_size"], CONFIG["input_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

def load_data():
    """Load CIFAR-10 with train/val/test split"""
    full_train = datasets.CIFAR10(root=CONFIG["data_dir"], train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=CONFIG["data_dir"], train=False, download=True, transform=val_transform)
    
    # Split train into train/val
    val_size = int(len(full_train) * CONFIG["val_fraction"])
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG["seed"])
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    
    print(f"Data splits:")
    print(f"   Train: {train_size:,} | Val: {val_size:,} | Test: {len(test_dataset):,}")
    return train_loader, val_loader, test_loader

# 4. Model creation
def create_model(architecture="resnet18", num_classes=10, unfreeze_layers=2, dropout_rate=0.3):
    """Create ResNet model with customizable architecture"""
    # Load pretrained model
    if architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = 512
    elif architecture == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        feat_dim = 512
    elif architecture == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        feat_dim = 2048
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last 'unfreeze_layers' layers
    if unfreeze_layers >= 1:
        for param in model.layer4.parameters():
            param.requires_grad = True
    if unfreeze_layers >= 2:
        for param in model.layer3.parameters():
            param.requires_grad = True
    if unfreeze_layers >= 3:
        for param in model.layer2.parameters():
            param.requires_grad = True
    
    # Replace classifier with custom head
    if hasattr(model, 'fc'):  # ResNet18,34,50
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    else:  # Other architectures
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, num_classes)
        )
    
    return model

# 5. Training and Evaluation Functions
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / total, correct / total, all_preds, all_labels

def hyperparameter_search(train_loader, val_loader):
    """Grid search for best hyperparameters"""
    print("\n" + "="*70)
    print("Hyperparameter search")
    print("="*70)
    
    param_grid = list(ParameterGrid({
        "architecture": ["resnet18"],
        "learning_rate": [1e-4, 5e-4],
        "optimizer": ["adam"],
        "unfreeze_layers": [2],
        "dropout_rate": [0.3, 0.5]
    }))
    
    best_score = 0
    best_params = {}
    results = []
    
    total_combos = len(param_grid)
    print(f"Total combinations: {total_combos}\n")
    print("─" * 70)
    
    from datetime import datetime
    import time
    
    start_time = time.time()
    
    for i, params in enumerate(param_grid, 1):
        # Calculate progress
        progress = (i / total_combos) * 100
        
        print(f"\nExperiment {i}/{total_combos} ({progress:.0f}%)" + " " * (45 - len(str(i)) - len(str(total_combos))))
        print(f"\n   Architecture:     {params['architecture']:<48}")
        print(f"   Learning Rate:    {params['learning_rate']:<48}")
        print(f"   Optimizer:        {params['optimizer']:<48}")
        print(f"   Unfreeze Layers:  {params['unfreeze_layers']:<46}")
        print(f"   Dropout Rate:     {params['dropout_rate']:<48}")
        
        # Create model
        model = create_model(
            architecture=params["architecture"],
            num_classes=CONFIG["num_classes"],
            unfreeze_layers=params["unfreeze_layers"],
            dropout_rate=params["dropout_rate"]
        ).to(device)
        
        # Setup optimizer
        if params["optimizer"] == "adam":
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params["learning_rate"])
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=params["learning_rate"], momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training with progress
        print(f"\nTraining (3 epochs)")
        
        best_val_acc = 0
        for epoch in range(3):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            
            # Epoch progress
            bar_len = 40
            epoch_progress = ((epoch + 1) / 3) * 100
            filled = int(bar_len * (epoch + 1) // 3)
            bar = "█" * filled + "░" * (bar_len - filled)
            
            print(f"  Epoch {epoch+1}/3 [{bar}] {epoch_progress:.0f}%")
            print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"    Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
        # Best validation accuracy for this combo
        if best_val_acc >= 0.90:
            result_icon = "✅"
        elif best_val_acc >= 0.85:
            result_icon = "⚠️"
        else:
            result_icon = "❌"
        
        print(f"{result_icon} Best Val Acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        
        results.append({
            'params': params,
            'val_acc': best_val_acc
        })
        
        if best_val_acc > best_score:
            best_score = best_val_acc
            best_params = params
            print(f"New best (Previous: {best_score:.4f})" if len(results) > 1 else "First best found")
    
    # Final summary
    print("\n" + "="*70)
    print("SEARCH SUMMARY")
    print("="*70)
    
    # Sort results by accuracy
    results.sort(key=lambda x: x['val_acc'], reverse=True)
    
    print("\n┌────┬────────────────────────────────────────────────┬──────────────────┐")
    print("│ #  │ PARAMETERS                                     │ VAL ACC          │")
    print("├────┼────────────────────────────────────────────────┼──────────────────┤")
    
    for rank, res in enumerate(results[:5], 1):
        params = res['params']
        param_str = f"lr={params['learning_rate']}, drop={params['dropout_rate']}"
        print(f"│ {rank:<2} │ {param_str:<46} │ {res['val_acc']:.4f} ({res['val_acc']*100:.2f}%) │")
    
    print("└────┴────────────────────────────────────────────────┴──────────────────┘")
    
    # Elapsed time
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print(f"\nSearch completed in: {minutes}m {seconds}s")
    print("="*70)
    print(f"Best Hyperparameters:")
    print(f"   • Architecture:      {best_params.get('architecture', 'resnet18')}")
    print(f"   • Learning Rate:     {best_params['learning_rate']}")
    print(f"   • Optimizer:         {best_params['optimizer']}")
    print(f"   • Unfreeze Layers:   {best_params['unfreeze_layers']}")
    print(f"   • Dropout Rate:      {best_params['dropout_rate']}")
    print(f"\nBest Validation Accuracy: {best_score:.4f} ({best_score*100:.2f}%)")
    print("="*70)
    
    return best_params

# 7. Full training with best hyperparameters
def train_full_model(train_loader, val_loader, best_params):
    print("\n" + "="*60)
    print("Training Best Model")
    print("="*60)
    
    # Create model with best hyperparameters
    model = create_model(
        architecture=best_params.get("architecture", "resnet18"),
        num_classes=CONFIG["num_classes"],
        unfreeze_layers=best_params["unfreeze_layers"],
        dropout_rate=best_params["dropout_rate"]
    ).to(device)
    
    # Setup optimizer
    if best_params["optimizer"] == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=best_params["learning_rate"])
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=best_params["learning_rate"], momentum=0.9, weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0
    best_epoch = 0
    
    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9} | {'LR':>8}")
    print("-" * 70)
    
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'best_params': best_params
            }, CONFIG["best_checkpoint"])
            note = "✓ saved"
        else:
            note = ""
        
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | "
              f"{val_loss:>10.4f} | {val_acc:>9.4f} | {current_lr:>8.6f} | {note}")
    
    print(f"\nTraining complete!")
    print(f"   Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    
    # Load best model
    checkpoint = torch.load(CONFIG["best_checkpoint"], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history, best_val_acc

# 8. Evaluation with metrics and visualization
def evaluate_model(model, test_loader, device):
    print("\n" + "="*60)
    print("Final Model Evaluation")
    print("="*60)
    
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
    
    # Calculate additional metrics
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\nTest Results:")
    print(f"   Loss:       {test_loss:.4f}")
    print(f"   Accuracy:   {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Precision:  {precision:.4f}")
    print(f"   Recall:     {recall:.4f}")
    print(f"   F1-Score:   {f1:.4f}")
    
    # Confusion matrix
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    cm = confusion_matrix(all_labels, all_preds)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=axes[0], xticks_rotation=45, cmap='Blues')
    axes[0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Class-wise accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    bars = axes[1].bar(range(len(classes)), class_acc, color='steelblue', alpha=0.7)
    axes[1].set_xticks(range(len(classes)))
    axes[1].set_xticklabels(classes, rotation=45, ha='right')
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_acc):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: evaluation_results.png")
    
    return test_acc, precision, recall, f1

# 9. Training Curves
def plot_training_curves(history, best_val_acc):
    """Plot loss and accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 's-', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'o-', label='Train Accuracy', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_acc'], 's-', label='Val Accuracy', linewidth=2, markersize=4)
    ax2.axhline(y=best_val_acc, color='green', linestyle='--', linewidth=1, label=f'Best Val: {best_val_acc:.3f}')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'ResNet18 Training on CIFAR-10 (Best Val Acc: {best_val_acc:.3f})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    print("Saved: training_curves.png")

# 10. Inference function
def inference(model, image_path, transform, device, class_names):
    """Run inference on a single image and visualize results"""
    model.eval()
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Display result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Show image
    ax1.imshow(image)
    ax1.set_title(f'Input Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Show top predictions
    top_probs, top_classes = torch.topk(probabilities[0], k=5)
    top_probs = top_probs.cpu().numpy()
    top_classes = top_classes.cpu().numpy()
    top_labels = [class_names[c] for c in top_classes]
    
    bars = ax2.barh(range(5), top_probs, color='steelblue', alpha=0.7)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(top_labels)
    ax2.set_xlabel('Probability')
    ax2.set_title('Top-5 Predictions', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    
    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        ax2.text(prob + 0.02, bar.get_y() + bar.get_height()/2, f'{prob:.3f}', va='center', fontsize=9)
    
    plt.suptitle(f'Prediction: {class_names[predicted_class]} (Confidence: {confidence:.3f})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('inference_result.png', dpi=150)
    plt.show()
    
    return predicted_class, confidence, top_probs, top_labels

# 11. Save and load model with metadata
def save_model(model, filepath, metadata=None):
    """Save model with metadata"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'architecture': 'resnet18',
            'num_classes': CONFIG['num_classes'],
            'input_size': CONFIG['input_size']
        }
    }
    if metadata:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, device):
    """Load model from file"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create model with saved config
    model = create_model(
        architecture='resnet18',
        num_classes=checkpoint['model_config']['num_classes'],
        unfreeze_layers=2,  # Default, will be overridden by saved weights
        dropout_rate=0.3
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {filepath}")
    
    return model

# 12. Benchmark comparison
def benchmark_comparison(test_acc):
    """Compare with known ResNet18 benchmarks"""
    print("\n" + "="*60)
    print("Benchmark comparison")
    print("="*60)
    
    benchmarks = {
        'ResNet18': (93.0, 95.0),  # (min, max) benchmark range
        'ResNet34': (94.0, 96.0)
    }
    
    print(f"Your Test Accuracy: {test_acc*100:.2f}%")
    print(f"ResNet18 Benchmark Range: {benchmarks['ResNet18'][0]}-{benchmarks['ResNet18'][1]}%")
    print(f"Source: PapersWithCode / Original ResNet paper")
    
    if test_acc*100 >= benchmarks['ResNet18'][0]:
        print("SUCCESS: Your model meets or exceeds the benchmark!")
    else:
        print(f"Below benchmark by {benchmarks['ResNet18'][0] - test_acc*100:.2f}%")

# 13. Main Execution
def main():
    print(f"Using device: {device}")
    print("\n" + "="*60)
    
    # Load data
    train_loader, val_loader, test_loader = load_data()
    
    # Hyperparameter search
    do_hyperparameter_search = True
    
    if do_hyperparameter_search:
        best_params = hyperparameter_search(train_loader, val_loader)
    else:
        best_params = {
            'learning_rate': 1e-4,
            'optimizer': 'adam',
            'unfreeze_layers': 2,
            'dropout_rate': 0.3
        }
        print(f"\nUsing default hyperparameters: {best_params}")
    
    # Train full model
    model, history, best_val_acc = train_full_model(train_loader, val_loader, best_params)
    
    # Plot training curves
    plot_training_curves(history, best_val_acc)
    
    # Final evaluation
    test_acc, precision, recall, f1 = evaluate_model(model, test_loader, device)
    
    # Add benchmark comparison
    benchmark_comparison(test_acc)
    
    # Save final model
    save_model(model, CONFIG['final_checkpoint'], metadata={
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'best_params': best_params
    })
    
    print("\n" + "="*60)
    print("✅Done")
    
    return model, history, test_acc

# Run the main function
if __name__ == "__main__":
    model, history, test_acc = main()