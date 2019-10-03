import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, shape):
        """

        Args:
             shape (tuple)
             : structed as (image width, image height, image channels)
        """
        super(MLP, self).__init__()
        self.activation = nn.ReLU()
        self.fc_1 = nn.Linear(shape[0] * shape[1] * shape[2], 120)
        self.dropout = nn.Dropout()
        self.fc_2 = nn.Linear(120, 120)
        self.dropout = nn.Dropout()
        self.out = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, x.shape[0] * x.shape[1] * x.shape[2])
        x = self.activation(self.fc_1(x))
        x = self.dropout(x)
        x = self.activation(self.fc_2(x))
        x = self.dropout(x)
        return F.log_softmax(self.out(x))