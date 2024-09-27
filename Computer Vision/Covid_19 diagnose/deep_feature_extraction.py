# importing libraries

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image transformations (including augmentation)
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset (e.g., CIFAR-10 or any custom dataset)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) # Replace with your image dataset
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) 

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Load pre-trained ResNet-50v2 model
resnet50v2 = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
resnet50v2 = resnet50v2.to(device)

# Modify the model to extract 1024 features by changing the fully connected layer
resnet50v2.fc = nn.Sequential(
    nn.Linear(resnet50v2.fc.in_features, 1024),  # Extract 1024 deep features
    nn.ReLU(),
    nn.Dropout(0.5),  # Optional dropout
).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet50v2.parameters(), lr=0.001)

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Function to extract 1024 deep features
def extract_deep_features(model, data_loader):
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in tqdm(data_loader):
            images = images.to(device)
            outputs = model(images)  # Extract the 1024 features
            features.append(outputs.cpu())

    features = torch.cat(features, dim=0)
    return features

# Train the model
train_model(resnet50v2, train_loader, criterion, optimizer, num_epochs=5)

# Extract deep features from the test set
deep_features = extract_deep_features(resnet50v2, test_loader)
print(f"Deep features shape: {deep_features.shape}")
