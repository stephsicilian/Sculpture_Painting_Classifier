# test.py
import os
import random
import torch
from ArtNeuralNetwork import preprocess, ArtNeuralNetwork

# Load the trained model
n = ArtNeuralNetwork()
n.load_state_dict(torch.load('Art.pth'))

# Test dataset setup
directory = "split_datasets/test/"  # Path to test dataset
labels = ["sculpture", "painting"]  # Class labels
num_classes = len(labels)  # Number of classes

# Initialize evaluation metrics
correct = 0  # Total correct predictions
total = 0  # Total predictions
label_count = [0, 0]  # Number of samples per class
label_correct = [0, 0]  # Correct predictions per class

# Validate and load files for each class
class_files = []
for label in range(num_classes):
    dir_path = os.path.join(directory, labels[label])
    files = os.listdir(dir_path)
    valid_files = []
    for file_name in files:
        file_path = os.path.join(dir_path, file_name)
        try:
            preprocess(file_path)  # Verify the file is valid
            valid_files.append(file_path)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    class_files.append(valid_files)

# Balance test sets by sampling the smaller class size
min_test_size = min(len(class_files[0]), len(class_files[1]))
print(f"Balanced test set size: {min_test_size} per class")
for label in range(num_classes):
    class_files[label] = random.sample(class_files[label], min_test_size)

# Run the test and collect predictions
for label in range(num_classes):
    for file_path in class_files[label]:
        try:
            img = preprocess(file_path)
            img_tensor = torch.Tensor(img).to(n.device)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

        output = n.forward(img_tensor).detach().cpu().numpy()
        guess = 1 if output[0, 0] > 0.5 else 0

        label_count[label] += 1
        total += 1

        if guess == label:
            correct += 1
            label_correct[label] += 1

# Display results in tabular format with accuracy per class
print("---------------------------------------------")
print("| Class       | Total | Correct | Accuracy   |")
print("---------------------------------------------")
for i, label in enumerate(labels):
    class_accuracy = (label_correct[i] / label_count[i]) * 100  # Accuracy per class
    print(f"| {label.capitalize():<11} | {label_count[i]:>5} | {label_correct[i]:>7} | {class_accuracy:>8.1f}% |")
print("---------------------------------------------")
print(f"Overall Accuracy: {(correct / total) * 100:.1f}%")
