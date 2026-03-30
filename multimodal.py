import os
import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import random
from torch.optim.lr_scheduler import CosineAnnealingLR


class MultimodalDataset(Dataset):
    def __init__(self, csv_file, audio_dir, max_text_len=17, max_audio_dur=6.63, augment=False):
        """
        Initializes the dataset.
        Args:
            csv_file: Path to the CSV file containing labels and text metadata.
            audio_dir: Directory containing the corresponding audio files.
            max_text_len: Maximum sequence length for BERT tokenization.
            max_audio_dur: Fixed duration in seconds for audio processing.
        """
        self.df = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.max_text_len = max_text_len
        self.max_audio_dur = max_audio_dur
        self.sr = 22050  # sample rate
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _preprocess_audio(self, audio_path):
        # 1. Load audio file with a fixed sampling rate
        y, sr = librosa.load(audio_path, sr=self.sr)

        # 2. Standardize audio duration through Padding or Truncation
        target_length = int(self.max_audio_dur * self.sr)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]


         # Data augmentation
        if self.augment:
            # 1. random noise
            if np.random.rand() < 0.5:
                noise = np.random.randn(len(y)) * 0.005
                y = y + noise

        # 3. Extract Mel-spectrogram features and convert to Decibel (dB) scale
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        # 4. Perform Min-Max normalization to scale features between 0 and 1
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
        return np.expand_dims(S_norm, axis=0)

    def _preprocess_text(self, text):

        # Text augmentationn
        if self.augment:
            words = str(text).split()
            
            # 1. delete（possibility: 10%）
            if np.random.rand() < 0.5 and len(words) > 3:
                words = [w for w in words if np.random.rand() > 0.1]
            
            # 2. swap neighbor
            if np.random.rand() < 0.5 and len(words) > 2:
                i = np.random.randint(0, len(words) - 1)
                words[i], words[i+1] = words[i+1], words[i]
            
            text = ' '.join(words)
        
        # Tokenize text and generate BERT-specific inputs
        encoding = self.tokenizer.encode_plus(
            str(text),
            add_special_tokens=True,
            max_length = self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Flatten to remove the extra batch dimension (1, L) -> (L,)
        return encoding['input_ids'].flatten(),encoding['attention_mask'].flatten()

    def __getitem__(self, idx):
        # Retrieve metadata for the current sample
        row = self.df.iloc[idx]
        file_id = row['FileID']
        text = row['Text']

        # Handle labels for training vs. inference (e.g., test set without labels)
        if 'Label' in self.df.columns:
            label = row['Label']
        else:
            label = -1

        # Execute multimodal preprocessing pipeline
        audio_feature = self._preprocess_audio(os.path.join(self.audio_dir, file_id))
        input_ids, attention_mask = self._preprocess_text(text)

        return {
            'audio': torch.tensor(audio_feature, dtype=torch.float32),
            'input_ids': input_ids,
            'attention_mask': attention_mask,  # BERT mask to ignore padding tokens
            'label': torch.tensor(label, dtype=torch.long)
        }


class MultimodalBERTModel(nn.Module):
    def __init__(self):
        super(MultimodalBERTModel, self).__init__()
        # 1. Audio Branch: Extracts features from Mel-spectrograms
        self.audio_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),  # add a second conv layer
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32)  # project to 32-d
        )

        # 2. Text Branch: Pre-trained BERT (Frozen to preserve knowledge)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        for param in self.bert.pooler.parameters():
            param.requires_grad = True




        # 3. Fusion & Classifier: Integrates modalities and predicts sentiment
        # Audio (32-d) + BERT pooler_output (768-d) = 800-d
        self.classifier = nn.Sequential(
            nn.Linear(32 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(64, 3)  #3 categories：Negative, Neutral, Positive
        )

    def forward(self, audio_data, input_ids, attention_mask):
        # Forward pass for audio branch (extracts 32-d features)
        audio_features = self.audio_branch(audio_data)  

        # Ablation Test 1: shut down audio branch
        #audio_features = torch.zeros_like(audio_features).detach()

        # Forward pass for text branch (extracts 768-d global summary)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.pooler_output # (Batch, 768)

        # Ablation Test 2: shut down text branch
        #text_features = torch.zeros_like(text_features).detach()
        

        # Multimodal Fusion: Concatenate features along the dimension 1 (features)
        combined_features = torch.cat((audio_features, text_features), dim=1)

        # Final prediction (sentiment classification)
        output = self.classifier(combined_features)
        return output


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
    optimizer = optim.Adam(
        [
            {'params': model.bert.encoder.layer[-2:].parameters(),'lr': 2e-5},
            {'params': model.bert.pooler.parameters(), 'lr': 2e-5},
            {'params': model.audio_branch.parameters(), 'lr': lr},
            {'params': model.classifier.parameters(), 'lr': lr},
        ]
    )
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
    base_path = ''
    train_csv = os.path.join(base_path, 'train', 'data.csv')
    
    # Initialize separate datasets for training, validation, and testing
    train_dataset = MultimodalDataset(
        csv_file=train_csv,
        audio_dir=os.path.join(base_path, 'train', 'audio'),
        augment=True,
        
    )
    val_dataset = MultimodalDataset(
        csv_file=os.path.join(base_path, 'val', 'data.csv'),
        audio_dir=os.path.join(base_path, 'val', 'audio'),
        augment=False,
        
    )
    test_dataset = MultimodalDataset(
        csv_file=os.path.join(base_path, 'test', 'data.csv'),
        audio_dir=os.path.join(base_path, 'test', 'audio'),
        augment=False,
       
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
     # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_loss_track, label='Train Loss')
    ax1.plot(val_loss_track, label='Val Loss')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(val_f1_track, label='Val F1', color='green')
    ax2.set_title('Validation F1 Score')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1 Score')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    #load the best model

    model.load_state_dict(torch.load('best_multimodal_model.pth', map_location=device))
    model.eval()

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
