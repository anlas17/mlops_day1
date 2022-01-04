import argparse
import sys

import torch
from torch import nn, optim
from torch.utils.data import Dataset

from data import mnist
from model import MyAwesomeModel
import numpy as np
import matplotlib.pyplot as plt

def train(self):
        # Loading training data
        train_path = "C:/Users/zebra/.cookiecutters/cookiecutter-data-science/{{ cookiecutter.repo_name }}/data/raw/train_0.npz"
        train_data = np.load(train_path)
        
        # Creating training dataset for torch dataloader
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
        
        def mnist():
            """
            Returns a Pytorch Dataset created with training data.
            
                Returns:
                        train (Dataset): Training dataset to be used through a dataloader
            """
            train = TrainDataset()
            return train
       
        # Initializing model, loss and optimizer
        model = MyAwesomeModel()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        # Calling dataset function and creating dataloader
        train_set = mnist()
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
        
        epochs = 100
        steps = 0

        train_losses, test_losses = [], []
        # Training model
        for e in range(epochs):
            running_loss = 0
            for (i, batch) in enumerate(trainloader):
                images, labels = batch["image"], batch["label"]
                optimizer.zero_grad()
        
                log_ps = model(images.float()) # Data to float explicitly
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
            train_loss = running_loss/len(trainloader)
            train_losses.append(train_loss)
            print(train_loss)
        # Plotting the training loss and saving the plot
        plt.plot(train_losses)
        plt.show()
        plt.savefig('C:/Users/zebra/.cookiecutters/cookiecutter-data-science/{{ cookiecutter.repo_name }}/reports/figures/train_plot.png')
        torch.save(model.state_dict(), 'C:/Users/zebra/.cookiecutters/cookiecutter-data-science/{{ cookiecutter.repo_name }}/models/checkpoint.pth')

if __name__ == '__main__':
        train()
