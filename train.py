import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
from model import MultimodalBERTModel
from dataset import MultimodalDataset
import pandas as pd
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

def set_seed(seed=42):
    """Sets random seeds for reproducibility across different platforms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    # Compatibility for Mac MPS devices
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # Ensure deterministic behavior for convolutional layers
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.0001): 
    # Label smoothing helps prevent the model from becoming overconfident
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Learning rate scheduler to gradually reduce LR for better convergence
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    model.to(device)
    train_loss_track = []
    val_loss_track = []
    val_f1_track = []

    best_val_f1 = 0.0  
    save_path = 'best_multimodal_model.pth'  

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            audio = batch['audio'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(audio, input_ids,attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluation phase on validation set
        val_f1, val_loss = evaluate(model, val_loader, criterion, device)
        train_loss_track.append(train_loss/len(train_loader))
        val_loss_track.append(val_loss)
        val_f1_track.append(val_f1)
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Model Checkpointing: Save the best wights based on F1-Score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"--> Best model saved with Val F1: {best_val_f1:.4f}")
        
        scheduler.step()

    print(f"\nTraining complete. Best Validation F1: {best_val_f1:.4f}")
    return train_loss_track, val_loss_track, val_f1_track

def evaluate(model, loader, criterion, device):
    """Evaluates the model on a given dataset loader."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            audio = batch['audio'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(audio, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Macro F1 is used to account for class imbalance in MELD dataset
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return macro_f1, total_loss / len(loader)


def main():
    set_seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
    base_path = '..'
    train_csv = os.path.join(base_path, 'train', 'data.csv')
    
    # Initialize separate datasets for training, validation, and testing
    train_dataset = MultimodalDataset(
        csv_file=train_csv,
        audio_dir=os.path.join(base_path, 'train', 'audio'),
        
    )
    val_dataset = MultimodalDataset(
        csv_file=os.path.join(base_path, 'val', 'data.csv'),
        audio_dir=os.path.join(base_path, 'val', 'audio'),
        
    )
    test_dataset = MultimodalDataset(
        csv_file=os.path.join(base_path, 'test', 'data.csv'),
        audio_dir=os.path.join(base_path, 'test', 'audio'),
       
    )
   # DataLoaders for batch processing
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model and move to target device
    model = MultimodalBERTModel().to(device)

    # Start training process
    train_loss_track, val_loss_track, val_f1_track = train_model(model, train_loader, val_loader, device=device,epochs=50, lr=0.0005)

    # Plotting Loss Curves for the final report
    history = {'train_loss': train_loss_track, 'val_loss': val_loss_track, 'val_f1': val_f1_track}
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss (BERT + AudioCNN)', fontsize=15, pad=20) # 标题
    plt.xlabel('Epochs', fontsize=12)         
    plt.ylabel('Loss Value', fontsize=12)     
    plt.legend(fontsize=10)
    plt.legend()
    plt.savefig('loss_curve.png')  

    # Final inference on the test set
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            audio = batch['audio'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(audio, input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())

    with open('test.txt', 'w') as f:
        for pred in all_preds:
            f.write(f"{pred}\n")

    print(f"Successfully saved {len(all_preds)} predictions to test.txt")


if __name__ == '__main__':
    main()
