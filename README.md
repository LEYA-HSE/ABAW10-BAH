# VideoMAE for Behavioral Uncertainty Detection

This repository contains the implementation of a video classification pipeline designed to detect behavioral uncertainty using the VideoMAE architecture. The model is built on the Vision Transformer (ViT) framework and optimized for spatio-temporal feature extraction.

---

## Pipeline Architecture

The solution is organized into three primary stages to ensure a modular and efficient workflow.

### 1. Preprocessing and Data Acquisition
The first stage handles the transformation of raw video data into a normalized format.
* Uniform sampling of T=16 frames from the input video.
* Spatial resizing of each frame to 224 x 224 pixels.
* Normalization using ImageNet mean and standard deviation.
* Optional face-centered cropping based on a YOLO detector.

### 2. Feature Extraction and Aggregation
This stage extracts high-level representations using the VideoMAE-base encoder.
* Tubelet Embedding: The video volume is partitioned into 2 x 16 x 16 spatio-temporal cubes to capture motion and appearance.
* Transformer Layers: A stack of 12 encoder layers processes the tokens.
* Pretrained Weights: The model is initialized with weights pre-trained on the Kinetics-400 dataset.
* Global Average Pooling (GAP): The output sequence is aggregated into a 768-dimensional video embedding.

### 3. Classification and Decision
The final stage performs the mapping from latent features to class labels.
* Linear Head: A fully connected layer projects the embedding to the target classes.
* Softmax: Logits are converted into probability distributions to determine model confidence.
* Output: The system generates both the final classification label and the raw embedding for further analysis.

---

## Technical Foundations

The core of the model utilizes the spatio-temporal self-attention mechanism:

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

To derive a compact video representation $h$, we apply global average pooling to the final output tokens:

$$\mathbf{h} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{z}_i$$

---

## Experimental Setup

The model was trained and evaluated under the following technical conditions:

* Optimizer: AdamW with a learning rate of 2e-5 and weight decay of 0.05.
* Scheduler: Cosine annealing learning rate scheduler.
* Training: 15 epochs with a batch size of 4.
* Regularization: Label smoothing with a factor of 0.1.
* Data Split: Training, validation, and testing sets partitioned in a 70:15:15 ratio.

---

## Usage

To run a prediction, ensure you have the required dependencies installed and use the following snippet:

```python
from model import VideoPredictor

predictor = VideoPredictor(model_path='best_model.pth')
result = predictor.predict("sample_video.mp4")

print(f"Prediction: {result['label']}")
print(f"Confidence: {result['probs']}")
