import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load test dataset
test_dataset = ImageFolder(root='./test_data', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Load pre-trained ResNet model
feature_extractor = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
feature_extractor.fc = nn.Sequential(
    nn.Linear(feature_extractor.fc.in_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.5)
)
feature_extractor = feature_extractor.to(device)

# Load pre-trained weights
feature_extractor.fc.load_state_dict(torch.load('best_feature_extractor.pth'))
feature_extractor.eval()

# Function to extract 1024 deep features
def extract_deep_features(model, data_loader):
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            outputs = model.fc(model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(images))))))))))
            features.append(outputs.squeeze().cpu().numpy())
            labels.extend(targets.numpy())
    return np.vstack(features), np.array(labels)

# Extract features
X, y = extract_deep_features(feature_extractor, test_loader)

# Feature reduction techniques
def anova_selection(X, y, k=20):
    f_scores, _ = f_classif(X, y)
    return np.argsort(f_scores)[-k:]

def mifs_selection(X, y, k=20):
    mi_scores = mutual_info_classif(X, y)
    return np.argsort(mi_scores)[-k:]

def chi_square_selection(X, y, k=20):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    chi_scores, _ = chi2(X_scaled, y)
    return np.argsort(chi_scores)[-k:]

# User input for feature selection technique
print("Select feature reduction technique:")
print("1. ANOVA")
print("2. MIFS")
print("3. Chi-square")
technique = input("Enter the number of your choice (1-3): ")

# Apply selected feature reduction technique
if technique == '1':
    selected_features = anova_selection(X, y)
elif technique == '2':
    selected_features = mifs_selection(X, y)
elif technique == '3':
    selected_features = chi_square_selection(X, y)
else:
    raise ValueError("Invalid choice. Please select 1, 2, or 3.")

# Reduce features
X_reduced = X[:, selected_features]

# Fine-KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_reduced, y)

# Predict and evaluate
y_pred = knn.predict(X_reduced)
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y, y_pred))
