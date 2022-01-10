import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import Dataset

import hydra

from pytorch_lightning import Trainer

#line for .flake8 testing
string = "0000000000000000000000000000000000000000000000000000000000000000000000000"

@hydra.main(config_name="training_conf.yaml")
def train(config):
    os.makedirs("models/", exist_ok=True)
    # Loading training data
    train_path = "data/raw/train_0.npz"
    train_data = np.load(train_path)

    # Creating training dataset for torch dataloader
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
    optimizer = optim.Adam(model.parameters(), lr=config.hyperparameters.lr)

    # Calling dataset function and creating dataloader
    train_set = mnist()
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=config.hyperparameters.batch_size, shuffle=True)

    epochs = config.hyperparameters.epochs

    trainer = Trainer(max_epochs=20, logger=pl.loggers.WandbLogger(project="dtu_mlops"))
    trainer.fit(model, trainloader)
    #train_losses, test_losses = [], []
    # Training model
    #for e in range(epochs):
    #    running_loss = 0
    #    for (i, batch) in enumerate(trainloader):
    #        images, labels = batch["image"], batch["label"]
    #        optimizer.zero_grad()

    #        log_ps = model(images.float())  # Data to float explicitly
    #        loss = criterion(log_ps, labels)
    #        loss.backward()
    #        optimizer.step()

    #        running_loss += loss.item()
    #    train_loss = running_loss / len(trainloader)
    #    train_losses.append(train_loss)
    #    print(train_loss)
    # Plotting the training loss and saving the plot
    #plt.plot(train_losses)
    #plt.show()
    #os.makedirs("reports/figures/", exist_ok=True)
    #plt.savefig("reports/figures/train_plot.png")
    torch.save(model.state_dict(), "models/checkpoint.pth")

    for p in optimizer.param_groups:
        lr = p['lr']

    assert lr == config.hyperparameters.lr


if __name__ == "__main__":
    train()
