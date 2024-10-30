# Painting or Sculpture Classification with Convolutional Neural Networks

This project is a binary image classifier built with PyTorch, designed to distinguish between two types of art: **sculptures** and **paintings**. The model uses a Convolutional Neural Network architecture to classify images and is trained on a dataset of artwork images. <br>

Highest Achieved Accuracy: 89 % <br>

## Table of Contents
- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Requirements](#requirements)  
- [Setup and Installation](#setup-and-installation)  
- [Training the Model](#training-the-model)  
- [Testing the Model](#testing-the-model)  
- [Results](#results)  
- [File Structure](#file-structure)  
<br>

## Project Overview
The ArtNeuralNetwork project is a deep learning-based system designed to classify artwork into two primary categories: sculptures or paintings. The project is organized around three key Python scripts:

1. **ArtNeuralNetwork.py** - Defines the neural network architecture, image preprocessing procedures, and training functionalities.
2. **train.py** - Manages the training of the model using a labeled dataset of art images.
3. **test.py** - Evaluates the trained model's performance on a separate testing dataset.

## Dataset
The dataset comprises images of sculptures and paintings organized into training and testing sets. The images are stored in respective directories to aid in supervised learning.

## Model Architecture
The network consists of two convolutional layers, each followed by max pooling and ReLU activation. A fully connected layer is used to classify images as either sculptures or paintings, with a sigmoid activation function to produce output probabilities.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Pillow (PIL)

Install the required packages using:
```sh
pip install torch numpy pillow
```

## Setup and Installation
1. Clone the repository to your local machine.
2. Install the necessary dependencies as outlined in the Requirements section.
3. Set up the dataset structure as described in the [File Structure](#file-structure) section.

## Training the Model
1. Place your training images in the appropriate `datasets/train/` subdirectories (`sculpture` and `painting`).
2. Run `train.py` to train the model:
   ```sh
   python train.py
   ```

## Testing the Model
1. Place your test images in the `datasets/test/` subdirectories.
2. Run `test.py` to evaluate the model:
   ```sh
   python test.py
   ```

## Results
- **Highest Achieved Accuracy**: 89%
- The model's performance is evaluated on a separate test dataset, and both overall accuracy and class-wise accuracy metrics are calculated.

## File Structure
- **datasets/train/**: Contains training images organized in subdirectories `sculpture` and `painting`.
- **datasets/test/**: Contains test images organized similarly in subdirectories `sculpture` and `painting`.

The directory structure should follow this pattern:
```
datasets/
  train/
    sculpture/
      image1.jpg
      image2.jpg
      ...
    painting/
      image1.jpg
      image2.jpg
      ...
  test/
    sculpture/
      image1.jpg
      image2.jpg
      ...
    painting/
      image1.jpg
      image2.jpg
      ...
```

