import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
import wandb

class MyAwesomeModel(LightningModule):
    def __init__(self):
        super().__init__()
        config = OmegaConf.load('model_conf.yaml')
        input_size = config.hyperparameters.input_size
        output_size = config.hyperparameters.output_size
        dropout_prob = config.hyperparameters.dropout_prob

        self.conv1 = nn.Linear(input_size, 256)
        self.conv2 = nn.Linear(256, 128)
        self.conv3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        
        self.dropout = nn.Dropout(p=dropout_prob)

        wandb.init()

        self.criterium = nn.NLLLoss()
        
    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
        
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

    def training_step(step, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.logger.experiment.log({'logits':wandb.Histogram(preds)})
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-2)

    def test_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        self.log('test_loss', loss)
        
