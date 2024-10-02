import torch
from torchvision import datasets
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np

class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        return x, y

def get_datasets(data_dir, train_transform, val_transform, val_split=0.2, seed=42):
    # Load the full dataset without any transforms
    full_dataset = datasets.ImageFolder(root=data_dir)
    
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Split the indices
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    # Create train and validation subsets with their respective transforms
    train_dataset = TransformSubset(full_dataset, train_indices, transform=train_transform)
    val_dataset = TransformSubset(full_dataset, val_indices, transform=val_transform)

    return train_dataset, val_dataset

def get_test_dataset(data_dir, transform):
    return datasets.ImageFolder(root=data_dir, transform=transform)

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)