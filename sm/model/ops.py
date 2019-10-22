import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *

# args
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
COLOR_CHANNELS = 3
BATCH_SIZE = 32


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.activation = nn.ReLU()
        self.fc_1 = nn.Linear(IMAGE_WIDTH * IMAGE_HEIGHT * COLOR_CHANNELS, 120)
        self.dropout = nn.Dropout()
        self.fc_2 = nn.Linear(120, 120)
        self.dropout = nn.Dropout()
        self.out = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, IMAGE_WIDTH * IMAGE_HEIGHT * COLOR_CHANNELS)
        x = self.activation(self.fc_1(x))
        x = self.dropout(x)
        x = self.activation(self.fc_2(x))
        x = self.dropout(x)
        return F.log_softmax(self.out(x))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
