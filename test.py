#test.py
import os
import random
import torch
from ArtNeuralNetwork import preprocess, ArtNeuralNetwork

# Load the model
n = ArtNeuralNetwork()
n.load_state_dict(torch.load('Art.pth'))

# Directories and labels
directory = "split_datasets/test/"
labels = ["sculpture", "painting"]
num_classes = len(labels)

# Initialize variables
correct = 0
total = 0
label_count = [0, 0]
label_correct = [0, 0]

# Load files and balance test set
class_files = []
for label in range(num_classes):
    dir_path = os.path.join(directory, labels[label])
    files = os.listdir(dir_path)
    valid_files = []
    for file_name in files:
        file_path = os.path.join(dir_path, file_name)
        try:
            preprocess(file_path)  # Verify if file is valid
            valid_files.append(file_path)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    class_files.append(valid_files)

# Balance test sets based on the lesser number of valid images
min_test_size = min(len(class_files[0]), len(class_files[1]))
print(f"Balanced test set size: {min_test_size} per class")
for label in range(num_classes):
    class_files[label] = random.sample(class_files[label], min_test_size)

# Run the test
for label in range(num_classes):
    for file_path in class_files[label]:
        try:
            img = preprocess(file_path)
            img_tensor = torch.Tensor(img).to(n.device)  # Convert to tensor
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

# Print results
print(f"Accuracy: {correct / total:.2f}")
print(f"Label Count: {label_count}")
print(f"Label Correct: {label_correct}")
