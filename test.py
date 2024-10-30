#test.py
import os
import torch
from ArtNeuralNetwork import preprocess, ArtNeuralNetwork

# Load the trained model
n = ArtNeuralNetwork()
n.load_state_dict(torch.load('Art.pth'))  # Load model weights from saved file

# Set the test directory and class labels
directory = "datasets/test/"
labels = ["sculpture", "painting"]
num_classes = len(labels)

# Variables to track accuracy
correct = 0
total = 0
label_count = [0, 0]  # Count of images per label
label_correct = [0, 0]  # Correct predictions per label

# Loop through each label (sculpture and painting)
for label in range(num_classes):
    dir_path = os.path.join(directory, labels[label])
    files = os.listdir(dir_path)

    # Test each file in the directory
    for file_name in files:
        f = os.path.join(dir_path, file_name)
        try:
            img = preprocess(f)  # Preprocess the image
        except:
            continue

        output = n.forward(img).detach().cpu().numpy()  # Get model output
        guess = 1 if output[0] > 0.5 else 0  # Binary classification threshold

        label_count[label] += 1
        total += 1

        if guess == label:
            correct += 1
            label_correct[label] += 1

# Print overall accuracy and per-label statistics
print(f"Accuracy: {correct / total:.2f}")
print(f"Label Count: {label_count}")
print(f"Label Correct: {label_correct}")
