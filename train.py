#train.py
import os
from datetime import datetime
import torch
from ArtNeuralNetwork import preprocess, ArtNeuralNetwork

    
# Checks if the file can be preprocessed.
def is_bad_file(f):
    try:
        preprocess(f)
        return False
    except:
        return True
    
# Training parameters
epochs = 3  
learning_rate = 0.0001 

# Initialize neural network
n = ArtNeuralNetwork()

print("Start:", datetime.now())  # Start time for tracking

# Training directory and labels
directory = "datasets/train/"
labels = ["sculpture", "painting"]
num_classes = len(labels)

# Prepares file lists for each label
file_lists = []
for label in range(num_classes):
    dir_path = os.path.join(directory, labels[label])
    files = os.listdir(dir_path)
    valid_files = [f for f in files if not is_bad_file(os.path.join(dir_path, f))]
    file_lists.append(valid_files)

#Checks all classes have valid images to train on
min_files = min([len(files) for files in file_lists])
if min_files == 0:
    raise ValueError("One of the classes does not have any valid images to train on.")

# Train the model over the specified number of epochs
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    correct = 0
    total = 0

    # Loop over the dataset and train on each image
    for i in range(min_files):  # (all imgs) 
        for label in range(num_classes):
            dir_path = os.path.join(directory, labels[label])
            file_name = file_lists[label][i]
            f = os.path.join(dir_path, file_name)

            img = preprocess(f)  # Preprocess images
            target = [1.0] if label == 1 else [0.0]  # Set target label

            output = n(img).detach().cpu().numpy()

            # Make predictions based on the output
            guess = 1 if output[0] > 0.5 else 0

            # Track accuracy
            total += 1
            if guess == label:
                correct += 1

            # Train the model on the current image
            n.train_model(img, target)

        if i % 600 == 0:
            # Print accuracy every 500 steps
            accuracy = correct / total if total > 0 else 0
            print(f"Step {i}, Accuracy: {accuracy:.4f}")

# Save the trained model for future use
torch.save(n.state_dict(), 'Art.pth')
print("End:", datetime.now())  # End time for tracking
