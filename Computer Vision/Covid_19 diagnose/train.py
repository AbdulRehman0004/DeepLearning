import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

from model import  get_model, load_model
from augmentation import get_train_transform, get_val_transform
from dataloader import get_datasets, get_dataloader

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
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
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            # Save feature extractor (all layers except the last one)
            feature_extractor_state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith('resnet.fc.3')}
            torch.save(feature_extractor_state_dict, 'best_feature_extractor.pth')

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get data transforms
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    # Get datasets and dataloaders
    train_dataset, val_dataset = get_datasets(args.data_dir, train_transform, val_transform, args.val_split)
    train_loader = get_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = get_dataloader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Get model
    model = get_model(num_classes=args.num_classes)
    if args.resume:
        model = load_model(model, args.resume)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, args.num_epochs)
    
    print("Training completed. Best model and feature extractor weights saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a ResNet50v2 model for feature extraction")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes in the dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading")
    args = parser.parse_args()
    main(args)