#ArtNeuralNetwork.py
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Kaggle dataset for art images (paintings, sculptures, etc.)
# Link: https://www.kaggle.com/datasets/thedownhill/art-images-drawings-paintings-sculptures-engravings

# Constants for image preprocessing
CROP_WIDTH = 256
CROP_HEIGHT = 256
CHANNELS = 3

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess input images by resizing, normalizing, and reshaping to dimensions
def preprocess(f):
    image = Image.open(f).convert("RGB") 
    image = image.resize((CROP_WIDTH, CROP_HEIGHT))  # Resize image to 256x256 pixels
    a = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    a = a.transpose((2, 0, 1))  # Rearrange dimensions from (Height, Width, Channels) to (Channels, Height, Width)
    return a.reshape(1, CHANNELS, CROP_HEIGHT, CROP_WIDTH)  # Add batch dimension

# Define neural network 
class ArtNeuralNetwork(nn.Module):
    def __init__(self):
        super(ArtNeuralNetwork, self).__init__()
        self.device = device

        # Feature extraction layers (two convolutional layers)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, kernel_size=3, stride=1, padding=1),  # Layer 1
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsampling grid size to 128x128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Layer 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling grid size to 64x64
        )

        # Classification layers (fully connected layer with sigmoid activation)
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten feature map into a single vector
            nn.Linear(128 * (CROP_HEIGHT // 4) * (CROP_WIDTH // 4), 1),  # Fully connected layer
            nn.Sigmoid()  # Output probabilities
        )

        # Define binary cross-entropy loss and Adam optimizer
        self.loss_function = nn.BCELoss()  # Suitable for binary classification
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001) # learning rate

        # Move model to the appropriate device (GPU/CPU)
        self.to(self.device)

    # Forward pass through the network
    def forward(self, inputs):
        """
        Passes input through the feature extractor and classifier 
        to produce the output probabilities.
        """
        features = self.feature_extractor(inputs)  # Extract features
        return self.classifier(features)  # Classify features

    # Training function for one step of gradient descent
    def train_model(self, inputs, target):
        """
        Performs a single training step: forward pass, loss computation,
        backpropagation, and weight update.
        """
        self.train()  # Set model to training mode
        self.optimizer.zero_grad()  # Clear previous gradients
        outputs = self.forward(inputs)  # Forward pass
        loss = self.loss_function(outputs, target)  # Compute loss
        loss.backward()  # Backpropagation
        self.optimizer.step()  # Update weights

