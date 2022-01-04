import argparse
import sys

import torch
from torch import nn, optim

from data import mnist
from model import MyAwesomeModel

import matplotlib.pyplot as plt

def train(self):
        train_path = "C:/Users/zebra/.cookiecutters/cookiecutter-data-science/{{ cookiecutter.repo_name }}/data/raw/train_0.npz"
        train_data = np.load(train_path)

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
            train = TrainDataset()
    
            return train
        
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        train_set = mnist()
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
        
        epochs = 100
        steps = 0

        train_losses, test_losses = [], []
        for e in range(epochs):
            running_loss = 0
            for (i, batch) in enumerate(trainloader):
                images, labels = batch["image"], batch["label"]
                optimizer.zero_grad()
        
                log_ps = model(images.float())
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
            train_loss = running_loss/len(trainloader)
            train_losses.append(train_loss)
            print(train_loss)
        plt.plot(train_losses)
        plt.show()
        plt.savefig('C:/Users/zebra/.cookiecutters/cookiecutter-data-science/{{ cookiecutter.repo_name }}/reports/figures/train_plot.png')
        torch.save(model.state_dict(), 'C:/Users/zebra/.cookiecutters/cookiecutter-data-science/{{ cookiecutter.repo_name }}/models/checkpoint.pth')

if __name__ == '__main__':
        train()
