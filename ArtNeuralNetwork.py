#ArtNeuralNetwork.py
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps

# https://www.kaggle.com/datasets/thedownhill/art-images-drawings-painting-sculpture-engraving

# Constants for image preprocessing
CROP_WIDTH = 128
CROP_HEIGHT = 128
CHANNELS = 3

# Set device to CPU
device = torch.device('cpu')

def center_crop(image):
    """
    Center crop the image to the specified width and height.
    """
    width, height = image.size
    left = (width - CROP_WIDTH) / 2
    top = (height - CROP_HEIGHT) / 2
    right = left + CROP_WIDTH
    bottom = top + CROP_HEIGHT
    return image.crop((left, top, right, bottom))

def preprocess(f):
    """
    Preprocess the image by resizing, center cropping, normalizing, and reshaping.
    
    Arguments:
    f -- Path to the image file.

    Returns:
    A numpy array formatted for model input.
    """
    image = Image.open(f)
    
    # Convert to grayscale if needed
    if CHANNELS == 1:
        image = ImageOps.grayscale(image)
    
    # Resize and center crop the image
    image = image.resize((CROP_WIDTH, CROP_HEIGHT))
    image = center_crop(image)

    # Normalize the pixel values to [0, 1] range
    a = np.array(image) / 255.0

    # Rearrange the axes for PyTorch and return
    a = a.transpose((2, 0, 1)).reshape(1, 3, CROP_HEIGHT, CROP_WIDTH)
    
    return a

class View(nn.Module):
    """
    Custom PyTorch layer to reshape the input tensor.
    """
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)

class ArtNeuralNetwork(nn.Module):
    """
    Convolutional Neural Network (CNN) for classifying images as either sculptures or paintings.
    """
    def __init__(self):
        super(ArtNeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            # First convolutional layer: 256 filters, 4x4 kernel, stride 2
            nn.Conv2d(3, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),  # Batch normalization for faster training
            nn.ReLU(),  # ReLU activation function
            # Second convolutional layer: 3 filters, 4x4 kernel, stride 2
            nn.Conv2d(256, 3, kernel_size=4, stride=2),
            nn.ReLU(),  # ReLU activation function
            # Flatten the output tensor to 2700 elements
            View(2700),
            # Fully connected layer to reduce to 1 output (binary classification)
            nn.Linear(2700, 1),
            # Sigmoid activation to output a probability for binary classification
            nn.Sigmoid()
        )
        self.loss_function = nn.BCELoss()  # Binary Cross-Entropy Loss for classification
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)  # Adam optimizer
        self.to(device)  # Move the model to the appropriate device (CPU)

    def forward(self, inputs):
        """
        Forward pass through the network.
        
        Arguments:
        inputs -- Preprocessed image input
        
        Returns:
        Output of the network (probability of being a painting)
        """
        inputs = torch.Tensor(inputs).to(device)
        return self.model(inputs)
    
    def train_model(self, inputs, target):
        """
        Train the model by performing a forward pass, computing the loss, and updating the weights.
        
        Arguments:
        inputs -- Preprocessed image input
        target -- The target label (1 for painting, 0 for sculpture)
        """
        target = torch.Tensor(target).to(device)
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, target)
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagate to compute gradients
        self.optimizer.step()  # Update weights using the optimizer
