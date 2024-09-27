
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
import matplotlib.pyplot as plt

# Function to load datasets with augmentations
def load_datasets(dataset_name, batch_size=64):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError('Invalid dataset name. Choose CIFAR10 or CIFAR100.')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Function to load a pre-trained model from timm
def load_model(model_name, num_classes):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

# Function to train and evaluate the model
def train_and_evaluate(model, train_loader, test_loader, epochs=10, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_acc_list, test_acc_list = [], []
    train_loss_list, test_loss_list = [], []

    for epoch in range(epochs):
        model.train()
        correct_train = 0
        total_train = 0
        running_loss_train = 0.0

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            running_loss_train += loss.item()

        train_acc = 100 * correct_train / total_train
        train_acc_list.append(train_acc)
        train_loss_list.append(running_loss_train / len(train_loader))

        # Evaluation on test set
        model.eval()
        correct_test = 0
        total_test = 0
        running_loss_test = 0.0

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                running_loss_test += loss.item()

        test_acc = 100 * correct_test / total_test
        test_acc_list.append(test_acc)
        test_loss_list.append(running_loss_test / len(test_loader))

        print(f'Epoch {epoch+1}/{epochs} => Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

    return train_acc_list, test_acc_list, train_loss_list, test_loss_list

# Function to plot accuracy and loss
def plot_curves(train_acc, test_acc, train_loss, test_loss):
    epochs = range(1, len(train_acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'g-', label='Train Accuracy')
    plt.plot(epochs, test_acc, 'b-', label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epochs')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'g-', label='Train Loss')
    plt.plot(epochs, test_loss, 'b-', label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Main function
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_name = input('Enter dataset name (CIFAR10/CIFAR100): ').strip()
    model_name = input('Enter model name (e.g., vit_small_patch16_224, swin_base_patch4_window7_224, twins_svt_base): ').strip()

    train_loader, test_loader = load_datasets(dataset_name)

    if dataset_name == 'CIFAR10':
        num_classes = 10
    elif dataset_name == 'CIFAR100':
        num_classes = 100

    model = load_model(model_name, num_classes)
    model = model.to(device)

    # Train and evaluate the model
    train_acc_list, test_acc_list, train_loss_list, test_loss_list = train_and_evaluate(
        model, train_loader, test_loader, epochs=10, device=device)

    # Plot the accuracy and loss curves
    plot_curves(train_acc_list, test_acc_list, train_loss_list, test_loss_list)

if __name__ == '__main__':
    main()
