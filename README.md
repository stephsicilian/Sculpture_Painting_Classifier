# Painting or Sculpture Classification with Convolutional Neural Networks

Highest Achieved Accuracy: 93 % <br>

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
The dataset comprises images of sculptures and paintings organized into training and testing sets. The images are stored in respective directories to aid in supervised learning. Recent updates include:
- Cleaning the dataset to remove invalid files.
- Balancing the test set to ensure equal representation of sculptures and paintings.
- Implementing a dynamic 80/20 split for training and testing data, which creates new directories (`train/` and `test/`) to store the split data.

**Dataset Source**: [Art Images - Drawings, Paintings, Sculptures, Engravings (Kaggle)](https://www.kaggle.com/datasets/thedownhill/art-images-drawings-painting-sculpture-engraving)

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
2. Run `train.py` to dynamically split the dataset and train the model:
   ```sh
   python train.py
   ```
   - This step creates new directories (`datasets/train/` and `datasets/test/`) based on an 80/20 split.

## Testing the Model
1. Place your test images in the `datasets/test/` subdirectories.
2. Run `test.py` to evaluate the model:
   ```sh
   python test.py
   ```
   - The script balances the test set during evaluation to ensure equal representation of sculptures and paintings.

## Results
- **Highest Achieved Accuracy**: 93%
- The model was evaluated on a cleaned and balanced test dataset.
- Class-wise accuracy:
  - Sculptures: Correctly classified 271
  - Paintings: Correctly classified 232

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


