#test.py
import os
import torch
from ArtNeuralNetwork import preprocess, ArtNeuralNetwork

# Loads the trained model
n = ArtNeuralNetwork()
n.load_state_dict(torch.load('Art.pth'))  # Load model weights from saved file

# Sets the test directory and class labels
directory = "datasets/test/"
labels = ["sculpture", "painting"]
num_classes = len(labels)

# Defining variables to track accuracy
correct = 0
total = 0
label_count = [0, 0]  # Number of images tested per label
label_correct = [0, 0]  # Correct predictions per label

# Loops through each label (sculpture and painting)
for label in range(num_classes):
    dir_path = os.path.join(directory, labels[label])
    files = os.listdir(dir_path)

    # Tests each img in the directories
    for file_name in files:
        f = os.path.join(dir_path, file_name)
        try:
            img = preprocess(f)  # Preprocess image
        except:
            continue  # Skips files bad files

        # Gets model output and make prediction
        output = n(img).detach().cpu().numpy()
        guess = 1 if output[0] > 0.5 else 0  # Classifies based on threshold

        # Tracks total and per-label counts
        label_count[label] += 1
        total += 1

        if guess == label:
            correct += 1
            label_correct[label] += 1

# Overall accuracy and per-label statistics
print(f"Overall Accuracy: {correct} / {total} = {correct / total:.2%}")
for i, label in enumerate(labels):
    print(f"{label.capitalize()} - Correct: {label_correct[i]} / {label_count[i]} = {label_correct[i] / label_count[i]:.2%} Accuracy")
