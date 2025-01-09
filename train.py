#train.py
import os
import random
from datetime import datetime
import torch
from ArtNeuralNetwork import preprocess, ArtNeuralNetwork
import shutil

def is_bad_file(f):
    try:
        preprocess(f)
        return False
    except Exception as e:
        print(f"Error in preprocessing file {f}: {e}")
        return True

def split_dataset(input_folder, output_folder, classes, split_ratio=0.8):
    """
    Splits the dataset into train and test sets with an 80/20 split
    and ensures the same number of images for both classes.
    """
    # Create output directories
    for folder in ["train", "test"]:
        for class_name in classes:
            os.makedirs(os.path.join(output_folder, folder, class_name), exist_ok=True)

    # Find the smallest class size
    class_sizes = {class_name: len(os.listdir(os.path.join(input_folder, class_name))) for class_name in classes}
    min_size = min(class_sizes.values())

    # Process each class
    for class_name in classes:
        class_path = os.path.join(input_folder, class_name)
        images = os.listdir(class_path)

        # Shuffle and truncate to match the smallest class size
        random.shuffle(images)
        images = images[:min_size]

        # Split into train and test
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        # Copy images to train and test folders
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(output_folder, "train", class_name))
        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(output_folder, "test", class_name))

# Parameters
input_folder = "balanced_datasets"
output_folder = "split_datasets"
classes = ["sculpture", "painting"]

if not os.path.exists(output_folder):
    split_dataset(input_folder, output_folder, classes)
else:
    print("Split datasets already exist. Skipping dataset splitting.")

epochs = 3
learning_rate = 0.0001

n = ArtNeuralNetwork()

n.optimizer = torch.optim.Adam(n.parameters(), lr=learning_rate)

print("Start:", datetime.now())

directory = os.path.join(output_folder, "train")
labels = classes
num_classes = len(labels)

file_lists = []
for label in range(num_classes):
    dir_path = os.path.join(directory, labels[label])
    files = os.listdir(dir_path)
    valid_files = [f for f in files if not is_bad_file(os.path.join(dir_path, f))]
    file_lists.append(valid_files)

min_files = min([len(files) for files in file_lists])
if min_files == 0:
    raise ValueError("One of the classes does not have any valid images to train on.")

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    correct = 0
    total = 0

    for i in range(min_files):
        for label in range(num_classes):
            dir_path = os.path.join(directory, labels[label])
            file_name = file_lists[label][i]
            f = os.path.join(dir_path, file_name)

            img = preprocess(f)
            img_tensor = torch.Tensor(img).to(n.device)  # Convert to tensor and send to device
            target = torch.tensor([[1.0]] if label == 1 else [[0.0]], device=n.device)  # Fix target size

            output = n.forward(img_tensor).detach().cpu().numpy()

            guess = 1 if output[0, 0] > 0.5 else 0  # Access the correct output index

            total += 1
            if guess == label:
                correct += 1

            n.train_model(img_tensor, target)

        if i % 100 == 0:
            accuracy = correct / total if total > 0 else 0
            print(f"Step {i}, Accuracy: {accuracy:.4f}")

torch.save(n.state_dict(), 'Art.pth')
print("End:", datetime.now())
