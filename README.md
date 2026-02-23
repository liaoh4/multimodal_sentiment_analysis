# Audio Branch
A standard sample rate of 22,050 Hz was utilized to balance spectral detail and computational efficiency. 
The Variable Lengths are handled through a combination of truncation and zero-padding to produce uniform input tensors. 
Mel-Spectrograms were chosen to preserve richer spatial-frequency textures.
# Text Branch
The text branch utilizes BERT-base-uncased to extract high-level semantic features from dialogue. It employs WordPiece Tokenization,
which decomposes informal or rare words into sub-units. The pooler_output—a 768-dimensional vector- was utilized as a fixed-length
summary of the entire utterance. This global representation is then concatenated with the audio features for final sentiment classification.
# Fusion Architecture
The 768-dimensional pooler_output from BERT was concatenated with the 32-dimensional output from the Audio branch, resulting in an
800-dimensional multimodal feature vector. The fused vector was passed through a multi-layer perceptron (MLP) consisting of 
Linear(800, 256) -> ReLU -> Linear(800, 256) -> ReLU -> Linear(64, 3).

