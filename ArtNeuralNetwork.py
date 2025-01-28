# ArtNeuralNetwork.py
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Kaggle dataset for art images (paintings, sculptures)
# Link: https://www.kaggle.com/datasets/thedownhill/art-images-drawings-paintings-sculptures-engravings

# Constants for image preprocessing
RESIZE_WIDTH = 256  # Width for resized images
RESIZE_HEIGHT = 256  # Height for resized images
CHANNELS = 3  # Number of color channels (RGB)

# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def preprocess(f):
    """
    Preprocesses input images by resizing, normalizing, and reshaping.
    Args:
        f (str): File path of the image.
    Returns:
        np.ndarray: Preprocessed image as a 4D array (1, C, H, W).
    """
    image = Image.open(f).convert("RGB")
    image = image.resize((RESIZE_WIDTH, RESIZE_HEIGHT))  # Resize to 256x256
    a = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    a = a.transpose((2, 0, 1))  # Rearrange dimensions (C, H, W)
    return a.reshape(1, CHANNELS, RESIZE_HEIGHT, RESIZE_WIDTH)  # Add batch dimension

class ArtNeuralNetwork(nn.Module):
    """
    A Convolutional Neural Network for binary classification (sculpture vs. painting).
    """
    def __init__(self):
        super(ArtNeuralNetwork, self).__init__()
        self.device = device

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, kernel_size=3, stride=1, padding=1),  # Convolutional Layer 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 128x128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Convolutional Layer 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to 64x64
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the feature map
            nn.Linear(128 * (RESIZE_HEIGHT // 4) * (RESIZE_WIDTH // 4), 1),  # Fully connected layer
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

        # Loss function and optimizer
        self.loss_function = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)  # Adam optimizer

        # Move the model to the appropriate device
        self.to(self.device)

    def forward(self, inputs):
        """
        Forward pass through the network.
        Args:
            inputs (torch.Tensor): Input tensor (batch, C, H, W).
        Returns:
            torch.Tensor: Output probabilities for the binary classification.
        """
        features = self.feature_extractor(inputs)  # Extract features
        return self.classifier(features)  # Classify the extracted features

    def train_model(self, inputs, target):
        """
        Performs one step of training: forward pass, loss computation, and weight update.
        Args:
            inputs (torch.Tensor): Input tensor (batch, C, H, W).
            target (torch.Tensor): Target labels (batch, 1).
        """
        self.train()  # Set the model to training mode
        self.optimizer.zero_grad()  # Reset gradients
        outputs = self.forward(inputs)  # Forward pass
        loss = self.loss_function(outputs, target)  # Compute loss
        loss.backward()  # Backpropagation
        self.optimizer.step()  # Update model weights
