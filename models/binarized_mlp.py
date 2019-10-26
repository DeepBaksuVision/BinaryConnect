import torch
import torch.nn as nn

from utils.BinaryLinear import BinaryLinear

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = BinaryLinear(28 * 28, 512, bias=False, mode="Stocastic")
        self.batch1 = nn.BatchNorm1d(512)
        self.fc2 = BinaryLinear(512, 512, bias=False, mode="Stocastic")
        self.batch2 = nn.BatchNorm1d(512)
        self.fc3 = BinaryLinear(512, 10, bias=False, mode="Stocastic")
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
