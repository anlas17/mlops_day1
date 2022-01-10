import torch
from torch.utils.data import Dataset
from collections import Counter
import sys
import numpy as np
import os.path
sys.path.append("C:/Users/zebra/.cookiecutters/mlops_day1/")

def test():
    test_path = "C:/Users/zebra/.cookiecutters/mlops_day1/data/raw/test.npz"
    test_data = np.load(test_path)	

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
    
    def test_mnist():
        """
        Return a Pytorch Dataset for testing.
            
            Returns:
                    test (Dataset): Testing dataset to be used in a dataloader
        """
        test = TestDataset()
    
        return test

    train_path = "C:/Users/zebra/.cookiecutters/mlops_day1/data/raw/train_0.npz"
    train_data = np.load(train_path)

    class TrainDataset(Dataset):
        def __init__(self):
            self.data = train_data.f.images
            self.target = train_data.f.labels

        def __getitem__(self, index):
            x = self.data[index]
            y = self.target[index]
            data_dict = {"image": x, "label": y}
            return data_dict

        def __len__(self):
            return len(self.data)

    def train_mnist():
        """
        Returns a Pytorch Dataset created with training data.

            Returns:
                    train (Dataset): Training dataset to be used through a dataloader
        """
        train = TrainDataset()
        return train

    train_dataset = train_mnist()
    assert len(train_dataset) == 5000, 'Train Dataset length was wrong'

    test_dataset = test_mnist()
    assert len(test_dataset) == 5000, 'Test Dataset length was wrong'

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    x = next(iter(trainloader))['image']
    assert x.shape == (1,28,28), 'train shape was wrong'
    y = next(iter(testloader))['image']
    assert y.shape == (1,28,28), 'test shape was wrong'

    train_labels = [int(i['label']) for i in train_dataset]
    assert set(train_labels) == set(range(0,10)), 'not all labels are present in train data'
    
    test_labels = [int(i['label']) for i in test_dataset]
    assert set(test_labels) == set(range(0,10)), 'not all labels are present in test data'
