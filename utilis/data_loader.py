import torch
from typing import List, Tuple
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.model_selection import KFold

from config import config


transform = transforms.Compose([
    transforms.Resize(config['transform_resize']),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['transform_mean'], std=config['transform_std'])
])

class My_twoPhoton_Dataset(Dataset):
    def __init__(self, image_paths: List[str], vector_labels: np.ndarray, transform=transform, config=config):
        self.image_paths = image_paths
        self.vector_labels = vector_labels
        self.transform = transform
        self.config = config

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = self.vector_labels[index, :]
        label = torch.from_numpy(label).float()

        if self.config['transform'] == True:
            image = self.transform(image)
        
        return image, label
    

def train_val_split(dataset, n_splits, seed=0, cv=False):
    """
    Splits the dataset into training and validation sets.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be split.
        n_splits (int): The number of splits to create.
        seed (int, optional): The random seed for reproducibility. Defaults to 0.
        cv (bool, optional): If True, returns the indices for cross-validation. Defaults to False.

    Returns:
        If cv is True, returns the indices for cross-validation.
         and then use for train_index, val_index in train_val_split() in the main.py file to do cv.
        Otherwise, returns the training and validation datasets.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if cv == True:
        return kf.split(dataset)
    else:
        train_index, val_index = next(kf.split(dataset))
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)
        return train_dataset, val_dataset