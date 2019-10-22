import torch
import torch.nn as nn

from utils.BinarizedLinear import BinarizedLinear

class Binarized_MLP(nn.Module):
    def __init__(self, mode):
        super(Binarized_MLP, self).__init__()
        self.fc1 = BinarizedLinear(28 * 28, 512, bias=False, mode=mode)
        self.batch1 = nn.BatchNorm1d(512)
        self.fc2 = BinarizedLinear(512, 512, bias=False, mode=mode)
        self.batch2 = nn.BatchNorm1d(512)
        self.fc3 = BinarizedLinear(512, 10, bias=False, mode=mode)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.sigmoid(self.fc1(x))
        x = self.batch1(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.batch2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
