import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import Dataset

from pytorch_lightning import Trainer

def evaluate():
    test_path = "data/raw/test.npz"
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
    
    def mnist():
        """
        Return a Pytorch Dataset for testing.
            
            Returns:
                    test (Dataset): Testing dataset to be used in a dataloader
        """
        test = TestDataset()
    
        return test
    
        
    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    state_dict = torch.load('models/checkpoint.pth')
    model.load_state_dict(state_dict)
    test_set = mnist()
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    
    trainer = Trainer()
    trainer.test(model, testloader)
    #with torch.no_grad():
    #   for (i, batch) in enumerate(testloader):
    #      images, labels = batch["image"], batch["label"]
    #      ps = torch.exp(model(images.float()))
    #      top_p, top_class = ps.topk(1, dim=1)
    #      equals = top_class == labels.view(*top_class.shape)
    #      accuracy = torch.mean(equals.type(torch.FloatTensor))
    #print(f'Accuracy: {accuracy.item()*100}%')
if __name__ == '__main__':
    evaluate()
