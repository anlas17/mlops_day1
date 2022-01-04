import argparse
import sys

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from data import mnist
from model import MyAwesomeModel
import numpy as np
import matplotlib.pyplot as plt

def evaluate(self):
    test_path = "C:/Users/zebra/.cookiecutters/cookiecutter-data-science/{{ cookiecutter.repo_name }}/data/raw/test.npz"
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
    
    print("Evaluating until hitting the ceiling")
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('load_model_from', default="")
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)
        
    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    state_dict = torch.load('C:/Users/zebra/.cookiecutters/cookiecutter-data-science/{{ cookiecutter.repo_name }}/models/checkpoint.pth')
    model.load_state_dict(state_dict)
    test_set = mnist()
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
        
    with torch.no_grad():
       for (i, batch) in enumerate(testloader):
          images, labels = batch["image"], batch["label"]
          ps = torch.exp(model(images.float()))
          top_p, top_class = ps.topk(1, dim=1)
          equals = top_class == labels.view(*top_class.shape)
          accuracy = torch.mean(equals.type(torch.FloatTensor))
     print(f'Accuracy: {accuracy.item()*100}%')
if __name__ == '__main__':
    evaluate()
