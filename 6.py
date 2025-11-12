import os
import shutil
import torch
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy
import matplotlib.pyplot as plt




import zipfile
import tarfile
import os
import shutil

zip_path = '/content/caltech-101.zip'
extract_path = '/content/'

# Step 1: Unzip the main zip file
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Successfully unzipped '{os.path.basename(zip_path)}'.")
else:
    raise FileNotFoundError(f"The zip file was not found at '{zip_path}'. Please make sure it has been uploaded correctly.")

# Step 2: Find and extract the inner .tar.gz file
tar_gz_path = '/content/caltech-101/101_ObjectCategories.tar.gz'

if os.path.exists(tar_gz_path):
    with tarfile.open(tar_gz_path, 'r:gz') as tar_ref:
        tar_ref.extractall(path=extract_path)
    print(f"Successfully extracted '{os.path.basename(tar_gz_path)}'.")
else:
    raise FileNotFoundError(f"The inner archive '{tar_gz_path}' was not found. The contents of the zip file may be different than expected.")

# Step 3: Set the correct data directory path
data_dir = os.path.join(extract_path, '101_ObjectCategories')

# Step 4: Verify the path and remove the background class
if os.path.isdir(data_dir):
    print(f"Dataset directory is correctly set to: {data_dir}")
    background_dir = os.path.join(data_dir, 'BACKGROUND_Google')
    if os.path.exists(background_dir):
        shutil.rmtree(background_dir)
        print(f"'{os.path.basename(background_dir)}' directory has been removed.")
    else:
        print(f"'{os.path.basename(background_dir)}' directory not found, no removal needed.")
else:
    raise FileNotFoundError(f"The directory '{data_dir}' was not found after extraction. Please check the contents of the archive.")



import numpy as np
import shutil

# Define the paths for the new directories
base_dir = 'datadir'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# Clean up any previous splits to ensure a fresh start
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

# Create the main directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# The source_dir now correctly points to the data_dir variable defined in Cell 2
source_dir = data_dir
class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

for class_name in class_names:
    # Create class subdirectories in train, valid, and test
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Get all the images for the class and shuffle them
    class_images = os.listdir(os.path.join(source_dir, class_name))
    np.random.shuffle(class_images)

    # Define split points
    train_split = int(0.5 * len(class_images))
    valid_split = int(0.25 * len(class_images))

    # Get the image lists for each set
    train_images = class_images[:train_split]
    valid_images = class_images[train_split : train_split + valid_split]
    test_images = class_images[train_split + valid_split:]

    # Copy the images to their new directories
    for image in train_images:
        shutil.copy(os.path.join(source_dir, class_name, image), os.path.join(train_dir, class_name))
    for image in valid_images:
        shutil.copy(os.path.join(source_dir, class_name, image), os.path.join(valid_dir, class_name))
    for image in test_images:
        shutil.copy(os.path.join(source_dir, class_name, image), os.path.join(test_dir, class_name))

print("Dataset successfully split into train, validation, and test sets.")
print(f"Total classes: {len(class_names)}")


from torchvision import transforms

image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}



from torch.utils.data import DataLoader
from torchvision import datasets

batch_size = 64

# Load the data from folders
data = {
    'train': datasets.ImageFolder(root=train_dir, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_dir, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_dir, transform=image_transforms['test'])
}

# Create DataLoaders
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'valid': DataLoader(data['valid'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=False)
}

print("DataLoaders created successfully.")
print(f"Training samples: {len(data['train'])}")
print(f"Validation samples: {len(data['valid'])}")
print(f"Test samples: {len(data['test'])}")



from torchvision import models

model = models.vgg16(pretrained=True)

print(model)



for param in model.parameters():
    param.requires_grad = False




import torch.nn as nn

n_inputs = model.classifier[6].in_features
n_classes = len(data['train'].classes)

# Add our custom classifier
model.classifier[6] = nn.Sequential(
    nn.Linear(n_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, n_classes),
    nn.LogSoftmax(dim=1)
)

# Unfreeze the parameters of the new classifier layers so they can be trained
for param in model.classifier[6].parameters():
    param.requires_grad = True

print("Custom classifier added and its parameters are unfrozen for training.")
print(model.classifier[6])




import torch.optim as optim

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())




import time
import copy
import torch
import numpy as np # Make sure numpy is imported

def train_model(model, criterion, optimizer, num_epochs=25, patience=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # CORRECTED LINE: Changed np.Inf to np.inf
    val_loss_min = np.inf
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Check for GPU and move the model to the device
    # This line automatically detects and selects the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on device: {device}")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data[phase])
            epoch_acc = running_corrects.double() / len(data[phase])

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else: # Validation phase
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                # Check for early stopping
                if epoch_loss < val_loss_min:
                    print(f'Validation loss decreased ({val_loss_min:.6f} --> {epoch_loss:.6f}). Saving model...')
                    val_loss_min = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if epochs_no_improve == patience:
            print(f'Early stopping triggered after {patience} epochs with no improvement.')
            break
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, train_accs, val_accs

# Start training
model, train_losses, val_losses, train_accs, val_accs = train_model(model, criterion, optimizer, num_epochs=20, patience=5)





import matplotlib.pyplot as plt

# Ensure the lists from the training output are available
try:
    # Check if the variables exist, otherwise, show an error.
    train_losses
    val_losses
    train_accs
    val_accs
except NameError:
    print("Training data not found. Please run the training cell (Cell 10) before plotting.")
else:
    # Create a figure and a set of subplots
    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss (NLLLoss)")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy Curve")
    plt.legend()
    plt.grid(True)

    # Display the plots
    plt.show()



import torch

def test_model(model):
    # Set the model to evaluation mode
    model.eval()

    running_corrects = 0

    # Set the device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Evaluating on device: {device}")

    # Disable gradient calculations for inference
    with torch.no_grad():
        # Iterate over the test data
        for inputs, labels in dataloaders['test']:
            # Move inputs and labels to the selected device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass to get outputs
            outputs = model(inputs)
            # Get the predicted classes
            _, preds = torch.max(outputs, 1)

            # Update the count of correctly classified images
            running_corrects += torch.sum(preds == labels.data)

    # Calculate the final accuracy
    accuracy = running_corrects.double() / len(data['test'])
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total correct predictions: {running_corrects.item()} out of {len(data['test'])} images.")

# Run the test evaluation on the trained model
test_model(model)