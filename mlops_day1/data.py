import torch
import numpy as np
from torch.utils.data import Dataset

train_path = "D:/dtu_mlops/data/corruptmnist/train_0.npz"
train_data = np.load(train_path)
test_path = "D:/dtu_mlops/data/corruptmnist/test.npz"
test_data = np.load(test_path)

class TrainDataset(Dataset):
    def __init__(self):
        self.data = train_data.f.images
        self.target = train_data.f.labels
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        data_dict={'image': x, 'label': y}
        return data_dict
    
    def __len__(self):
        return len(self.data)

class TestDataset(Dataset):
    def __init__(self):
        self.data = test_data.f.images
        self.target = test_data.f.labels
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        data_dict={'image': x, 'label': y}
        return data_dict
    
    def __len__(self):
        return len(self.data)
    
def mnist():
    train = TrainDataset()
    test = TestDataset()
    
    return train, test
