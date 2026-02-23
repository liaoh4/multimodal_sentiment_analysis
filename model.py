import torch
import torch.nn as nn
from transformers import BertModel

class MultimodalBERTModel(nn.Module):
    def __init__(self):
        super(MultimodalBERTModel, self).__init__()
        # 1. Audio Branch: Extracts features from Mel-spectrograms
        self.audio_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  
            nn.Flatten()  
        )

        # 2. Text Branch: Pre-trained BERT (Frozen to preserve knowledge)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False


        # 3. Fusion & Classifier: Integrates modalities and predicts sentiment
        # Audio (32-d) + BERT pooler_output (768-d) = 800-d
        self.classifier = nn.Sequential(
            nn.Linear(32 + 768, 256),
            nn.ReLU(),
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Linear(64, 3)  #3 categories：Negative, Neutral, Positive
        )

    def forward(self, audio_data, input_ids, attention_mask):
        # Forward pass for audio branch (extracts 32-d features)
        audio_features = self.audio_branch(audio_data)  

        # Ablation Test 1: shut down audio branch
        #audio_features = torch.zeros_like(audio_features) 

        # Forward pass for text branch (extracts 768-d global summary)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.pooler_output # (Batch, 768)

        # Ablation Test 2: shut down text branch
        #text_features = torch.zeros_like(text_features) 
        

        # Multimodal Fusion: Concatenate features along the dimension 1 (features)
        combined_features = torch.cat((audio_features, text_features), dim=1)

        # Final prediction (sentiment classification)
        output = self.classifier(combined_features)
        return output

