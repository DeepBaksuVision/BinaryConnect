import torch.nn as nn
import torch.nn.functional as F

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
