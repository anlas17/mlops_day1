import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from model import MyAwesomeModel
from sklearn.manifold import TSNE
from torch import nn, optim
from torch.utils.data import Dataset


def visualize():
    model = MyAwesomeModel()
    state_dict = torch.load('models/checkpoint.pth')
    model.load_state_dict(state_dict)
    
    train_path = "data/raw/train_0.npz"
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
        images = images.float()
        intermid = F.relu(model.conv2(F.relu(model.conv1(images.view(images.shape[0], -1)))))
        intermid = intermid.detach()
        #vis = TSNE(n_components=2, learning_rate='auto',
        #           init='random').fit_transform(intermid)
        plt.plot(intermid)
        plt.show()
        plt.savefig('reports/figures/intermid_data.png')
        

if __name__ == '__main__':
    visualize()
