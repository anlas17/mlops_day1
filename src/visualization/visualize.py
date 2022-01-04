import argparse
import sys

import torch
from torch import nn, optim

from data import mnist
from model import MyAwesomeModel

import matplotlib.pyplot as plt

import sklearn.manifold.TSNE as TSNE

def visualize(self):
    model = MyAwesomeModel()
    state_dict = torch.load('C:/Users/zebra/.cookiecutters/cookiecutter-data-science/{{ cookiecutter.repo_name }}/models/checkpoint.pth')
    model.load_state_dict(state_dict)
    
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
    
    train_set = mnist()
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)

    for (i, batch) in enumerate(trainloader):
        images, labels = batch["image"], batch["label"]
        intermid = model.conv3(images)
        vis = TSNE(n_components=2, learning_rate='auto',
                   init='random').fit_transform(intermid)
        plt.plot(vis)
        plt.show()
        plt.savefig('C:/Users/zebra/.cookiecutters/cookiecutter-data-science/{{ cookiecutter.repo_name }}/reports/figures/intermid_data')
        

if __name__ == '__main__':
    visualize()
