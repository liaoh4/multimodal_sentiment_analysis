import os
import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class MultimodalDataset(Dataset):
    def __init__(self, csv_file, audio_dir, max_text_len=17, max_audio_dur=6.63):
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

        # 3. Extract Mel-spectrogram features and convert to Decibel (dB) scale
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        # 4. Perform Min-Max normalization to scale features between 0 and 1
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
        return np.expand_dims(S_norm, axis=0)

    def _preprocess_text(self, text):
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

