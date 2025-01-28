# train.py
import os
import random
from datetime import datetime
import torch
from ArtNeuralNetwork import preprocess, ArtNeuralNetwork
import shutil
from PIL import Image, ImageEnhance  

def is_bad_file(f): #Checks imaage, corrupted images are excluded.
    try:
        preprocess(f)
        return False
    except Exception as e:
        print(f"Error in preprocessing file {f}: {e}")
        return True

def enhance_contrast(image_path, output_path, contrast_factor=1.5):
    """
    Enhances the contrast of an image.
    - image_path: Path to the input image.
    - output_path: Path where the enhanced image will be saved.
    - contrast_factor: Factor to increase contrast (1.5 means 50% more contrast).
    """
    try:
        image = Image.open(image_path).convert("RGB")  # Open image in RGB
        enhancer = ImageEnhance.Contrast(image)  # Initialize contrast enhancer
        enhanced_image = enhancer.enhance(contrast_factor)  # Increase contrast
        enhanced_image.save(output_path)  # Save enhanced image
    except Exception as e:
        print(f"Error enhancing contrast for {image_path}: {e}")

def split_dataset(input_folder, output_folder, classes, split_ratio=0.8):
    """
    Splits the dataset into train and test sets with an 80/20 split.
    Ensures equal numbers of images for each class and applies contrast enhancement to sculptures.
    - input_folder: Folder containing original images for each class.
    - output_folder: Folder where split datasets will be stored.
    - classes: List of class names ["sculpture", "painting"].
    - split_ratio: Percentage of images used for training (80%).
    """
    # Create output directories for train and test sets
    for folder in ["train", "test"]:
        for class_name in classes:
            os.makedirs(os.path.join(output_folder, folder, class_name), exist_ok=True)

    # Find the smallest class size to balance the dataset
    class_sizes = {class_name: len(os.listdir(os.path.join(input_folder, class_name))) for class_name in classes}
    min_size = min(class_sizes.values())

    # Process each class
    for class_name in classes:
        class_path = os.path.join(input_folder, class_name)
        images = os.listdir(class_path)

        # Shuffle images and truncate to match the smallest class size
        random.shuffle(images)
        images = images[:min_size]

        # Split into train and test sets
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        # Copy or enhance contrast for train and test images
        for img in train_images:
            src = os.path.join(class_path, img)
            dest = os.path.join(output_folder, "train", class_name, img)
            if class_name == "sculpture":  # Enhance contrast 
                enhance_contrast(src, dest)
            else:
                shutil.copy(src, dest)

        for img in test_images:
            src = os.path.join(class_path, img)
            dest = os.path.join(output_folder, "test", class_name, img)
            if class_name == "sculpture":  # Enhance contrast
                enhance_contrast(src, dest)
            else:
                shutil.copy(src, dest)

# Parameters for dataset split and model training
input_folder = "balanced_datasets"  # Source folder containing original datasets
output_folder = "split_datasets"  # Destination folder for split datasets
classes = ["sculpture", "painting"]  # Class labels

# Check if the dataset has already been split, otherwise split it
if not os.path.exists(output_folder):
    split_dataset(input_folder, output_folder, classes)
else:
    print("Split datasets already exist. Skipping dataset splitting.")

# Training parameters
epochs = 8
learning_rate = 0.0001

# Initialize the neural network and optimizer
n = ArtNeuralNetwork()
n.optimizer = torch.optim.Adam(n.parameters(), lr=learning_rate)

print("Start:", datetime.now())  # Log the start time of training

# Load training data
directory = os.path.join(output_folder, "train")
labels = classes
num_classes = len(labels)

# Validate and prepare image file lists for each class
file_lists = []
for label in range(num_classes):
    dir_path = os.path.join(directory, labels[label])
    files = os.listdir(dir_path)
    valid_files = [f for f in files if not is_bad_file(os.path.join(dir_path, f))]
    file_lists.append(valid_files)

# Determine the minimum number of valid images per class (w/ check statement)
min_files = min([len(files) for files in file_lists])
if min_files == 0:
    raise ValueError("One of the classes does not have any valid images to train on.")

# Start training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    correct = 0
    total = 0

    # Iterate over all images for the current epoch
    for i in range(min_files):
        for label in range(num_classes):
            dir_path = os.path.join(directory, labels[label])
            file_name = file_lists[label][i]
            f = os.path.join(dir_path, file_name)

            # Preprocess and prepare the image tensor and target label
            img = preprocess(f)
            img_tensor = torch.Tensor(img).to(n.device)  # Send tensor to device (GPU/CPU)
            target = torch.tensor([[1.0]] if label == 1 else [[0.0]], device=n.device)  # Binary target

            # Forward pass and prediction
            output = n.forward(img_tensor).detach().cpu().numpy()
            guess = 1 if output[0, 0] > 0.5 else 0  # Classify based on threshold

            total += 1
            if guess == label:
                correct += 1

            # Perform a training step
            n.train_model(img_tensor, target)

        # Print accuracy at regular intervals
        if i % 100 == 0:
            accuracy = correct / total if total > 0 else 0
            print(f"Step {i}, Accuracy: {accuracy:.4f}")

# Save the trained model
torch.save(n.state_dict(), 'Art.pth')
print("End:", datetime.now())  # Log the end time of training
