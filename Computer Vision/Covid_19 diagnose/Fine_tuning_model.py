# Importing libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image transformations (including augmentation)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset (replace with your custom dataset)
full_dataset = datasets.ImageFolder(root='./data', transform=transform)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

# Load pre-trained ResNet-50v2 model
resnet50v2 = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')

# Freeze all layers except the last one
for param in resnet50v2.parameters():
    param.requires_grad = False

# Modify the model to extract 1024 features and add a 2-neuron output layer
resnet50v2.fc = nn.Sequential(
    nn.Linear(resnet50v2.fc.in_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 2)  # 2-neuron output layer for binary classification
)

# Unfreeze the last layer
for param in resnet50v2.fc.parameters():
    param.requires_grad = True

resnet50v2 = resnet50v2.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet50v2.fc.parameters(), lr=0.001)

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save the best model (excluding the output layer)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.fc[:-1].state_dict(), 'best_feature_extractor.pth')

# Function to extract 1024 deep features
def extract_deep_features(model, data_loader):
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            outputs = model.fc[:-1](model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(images))))))))))
            features.append(outputs.squeeze().cpu())
    features = torch.cat(features, dim=0)
    return features

# Train the model
train_model(resnet50v2, train_loader, val_loader, criterion, optimizer, num_epochs=5)

# Save the feature extractor weights (up to 1024 hidden units)
torch.save(resnet50v2.fc[:-1].state_dict(), 'feature_extractor_weights.pth')

print("Training completed. Feature extractor weights saved.")
