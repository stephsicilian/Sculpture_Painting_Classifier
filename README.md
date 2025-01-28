# Sculptures vs. Paintings Classifier

A convolutional neural network designed to classify images as either sculptures or paintings with high accuracy. The project processes images, trains a binary classifier, and achieves an accuracy of **99%** after applying  contrast enhancements to the scuplture class.

---

## **Project Overview**

This project leverages deep learning techniques to classify art images (sculptures vs. paintings). The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/thedownhill/art-images-drawings-paintings-sculptures-engravings), and the model processes the data with the following pipeline:
- Image preprocessing: resizing, normalization, and balancing datasets.
- Training a CNN with feature extraction (convolutional layers) and classification.
- Contrast enhancement applied selectively to sculptures for improved accuracy.

---

## **Key Features**
- **Image Preprocessing**:  
  - Resizes images to `256x256`.  
  - Filters invalid/corrupted images.  
  - Balances classes for training and testing (80/20 split).  
- **Model Architecture**:  
  - Two convolutional layers with ReLU activation and max-pooling.  
  - Fully connected layer for binary classification.  
  - Sigmoid activation for probabilistic outputs.  
- **Results**:  
  - Accuracy improved from **93%** to **99%** with contrast-enhanced sculpture images.  

---

## **Setup Instructions**

### Prerequisites
- Python 3.8 or higher
- Libraries:
  - `torch` (PyTorch)
  - `numpy`
  - `Pillow`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sculpture-painting-classifier.git
   cd sculpture-painting-classifier
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

### 1. Preprocess Dataset
Ensure your dataset is structured into `balanced_datasets` with subfolders for `sculpture` and `painting`. To preprocess and split the data:
```bash
python train.py
```

### 2. Train the Model
Run the training script to train the model:
```bash
python train.py
```
The trained model will be saved as `Art.pth`.

### 3. Evaluate the Model
Test the trained model on the validation dataset:
```bash
python test.py
```

---

## **Model Versions**

### Current Version: **1.5**
- **Accuracy:** 99%
- **Key Changes:**  
  - Contrast enhancement for sculpture images (`contrast_factor=1.5`).  
  - Training for 8 epochs at `0.0001` learning rate.  
  - Testing on a balanced dataset of 269 images per class.

### Experimentation History:
- **Base Model (Baseline)**: Achieved 93% accuracy with 2 convolutional layers, no contrast adjustment.  
- **Experiment Highlights**:
  - Added image rotations: Minimal improvements.  
  - Increased convolutional layers: No significant accuracy gains.  
  - Contrast enhancement: Breakthrough improvement to **99% accuracy**.  

---

## **File Descriptions**

### `train.py`
- Handles preprocessing, data splitting, and training the CNN.
- Saves the trained model as `Art.pth`.

### `test.py`
- Evaluates the trained model on the test dataset.
- Outputs accuracy and per-class results in a tabular format.

### `ArtNeuralNetwork.py`
- Defines the CNN architecture.
- Includes preprocessing and training functions.

### Dataset Structure:
```
balanced_datasets/
├── sculpture/
├── painting/

split_datasets/
├── train/
│   ├── sculpture/
│   ├── painting/
├── test/
│   ├── sculpture/
│   ├── painting/
```

---

## **Results**
- **Balanced Test Set Size**: 269 images per class.
- **Accuracy**:  
  - **Overall:** 99%  
  - **Sculptures:** 98.9%  
  - **Paintings:** 98.1%  

---

## **Future Work**
- Further optimize model architecture for speed and scalability.

---

## **Credits**
- Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/thedownhill/art-images-drawings-paintings-sculptures-engravings).

---
