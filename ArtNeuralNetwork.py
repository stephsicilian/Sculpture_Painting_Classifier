#ArtNeuralNetwork.py
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps

# Kaggle dataset for art images (paintings, sculptures, etc.)
# Link: https://www.kaggle.com/datasets/thedownhill/art-images-drawings-painting-sculpture-engraving

# Fixed sizes for image preprocessing
CROP_WIDTH = 128
CROP_HEIGHT = 128
CHANNELS = 3

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to center crop images to fixed size
def center_crop(image):
    width, height = image.size
    left = (width - CROP_WIDTH) / 2
    top = (height - CROP_HEIGHT) / 2
    right = left + CROP_WIDTH
    bottom = top + CROP_HEIGHT
    return image.crop((left, top, right, bottom))

#  Preprocess the image by resizing, center cropping, normalizing, and reshaping.
def preprocess(f):
    """
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

    # Normalize pixel values to [0, 1] range
    a = np.array(image) / 255.0

    # Rearrange the axes for PyTorch (Channels x Height x Width)
    a = a.transpose((2, 0, 1)).reshape(1, 3, CROP_HEIGHT, CROP_WIDTH)
    return a

# Custom PyTorch layer to reshape the input tensor
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)

# Define the neural network for art classification (painting vs sculpture)
class ArtNeuralNetwork(nn.Module):
    def __init__(self):
        super(ArtNeuralNetwork, self).__init__()
        self.model = nn.Sequential(
           
            # First convolutional layer: 256 filters, 4x4 kernel
            nn.Conv2d(3, 256, 4),
            nn.MaxPool2d(2),  # Downsample the image by half
            nn.ReLU(),
            
            # Second convolutional layer: 256 filters, 4x4 kernel
            nn.Conv2d(256, 256, 4),
            nn.MaxPool2d(2),
            nn.ReLU(),
            
            # Fully connected layer
            View((-1, 256 * 28 * 28)),
            nn.Linear(256 * 28 * 28, 1),
            nn.Sigmoid()  # Output value between 0 and 1
        )

    def forward(self, x):
        return self.model(torch.tensor(x).float().to(device))

    # Training function for backpropagation: minimize the error by adjusting the weights.
    '''
        1. Forward pass: Calculate the output of the model.
        2. Loss calculation: Measure the difference between the output and the target value.
        3. Backward pass: Compute gradients of the loss with respect to model parameters.
        4. Weight update: Adjust the weights to reduce the loss.
    '''
    def train_model(self, x, target):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        optimizer.zero_grad()
        output = self.forward(x)
        loss = criterion(output, torch.tensor(target).float().to(device))
        loss.backward()
        optimizer.step()
