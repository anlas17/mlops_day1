from torch import nn
import torch.nn.functional as F
import numpy as np

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Linear(784, 256)
        self.conv2 = nn.Linear(256, 128)
        self.conv3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x
