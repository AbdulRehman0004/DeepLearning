import torch
import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import argparse

from model.model import get_model, load_feature_extractor
from augmentation.augmentation import get_val_transform
from dataloader.dataloader import get_test_dataset, get_dataloader

def select_top_k_features(scores, k=20):
    top_k_indices = np.argsort(scores)[-k:]
    return sorted(top_k_indices)  # Return sorted indices to maintain original feature order

def anova_selection(X, y, k=20):
    f_scores, _ = f_classif(X, y)
    return select_top_k_features(f_scores, k)

def mifs_selection(X, y, k=20):
    mi_scores = mutual_info_classif(X, y)
    return select_top_k_features(mi_scores, k)

def chi_square_selection(X, y, k=20):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    chi_scores, _ = chi2(X_scaled, y)
    return select_top_k_features(chi_scores, k)

def extract_features(model, data_loader, device):
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            outputs = model.extract_features(images)
            features.append(outputs.cpu().numpy())
            labels.extend(targets.numpy())
    return np.vstack(features), np.array(labels)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and data
    model = get_model(num_classes=args.num_classes)
    model = load_feature_extractor(model, args.weights_path)
    model = model.to(device)
    
    test_transform = get_val_transform()
    test_dataset = get_test_dataset(args.data_dir, test_transform)
    test_loader = get_dataloader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Extract features
    X, y = extract_features(model, test_loader, device)
    
    # Apply selected feature reduction technique
    if args.technique == 'anova':
        selected_features = anova_selection(X, y, k=args.num_features)
    elif args.technique == 'mifs':
        selected_features = mifs_selection(X, y, k=args.num_features)
    elif args.technique == 'chi2':
        selected_features = chi_square_selection(X, y, k=args.num_features)
    else:
        raise ValueError("Invalid technique. Choose 'anova', 'mifs', or 'chi2'.")
    
    # Reduce features
    X_reduced = X[:, selected_features]
    
    # Train and evaluate KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_reduced, y)
    y_pred = knn.predict(X_reduced)
    
    # Print results
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature extraction and classification")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to test data directory")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to pre-trained weights")
    parser.add_argument("--technique", type=str, choices=['anova', 'mifs', 'chi2'], required=True, help="Feature selection technique")
    parser.add_argument("--num_features", type=int, default=20, help="Number of top features to select")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes in the dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading")
    
    args = parser.parse_args()
    main(args)