import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from layers.binarized_linear import BinarizedLinear


class Binarized_MLP(nn.Module):
    def __init__(self,
                 mode: str = "determistic",
                 is_dropout: bool = False,
                 dropout_prob: float = 0.5,
                 optimizer: optim = optim.SGD,
                 learning_rate: float = 0.01,
                 momentum: float = 0,
                 weight_decay: float = 1e-5,
                 scheduler: optim.lr_scheduler = None,
                 scheduler_gamma: float = 0.1):
        super(Binarized_MLP, self).__init__()

        # Layers
        self.fc1 = BinarizedLinear(28 * 28, 1024, bias=False, mode=mode)
        self.batch1 = nn.BatchNorm1d(1024)
        self.fc2 = BinarizedLinear(1024, 1024, bias=False, mode=mode)
        self.batch2 = nn.BatchNorm1d(1024)
        self.fc3 = BinarizedLinear(1024, 1024, bias=False, mode=mode)
        self.batch3 = nn.BatchNorm1d(1024)
        self.fc4 = BinarizedLinear(1024, 10, bias=False, mode=mode)

        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.is_dropout = is_dropout
        if self.is_dropout:
            self.dropout1 = nn.Dropout(dropout_prob)
            self.dropout2 = nn.Dropout(dropout_prob)
            self.dropout3 = nn.Dropout(dropout_prob)

        # Optimizer
        self.optim = optimizer(self.parameters(),
                               lr=learning_rate,
                               momentum=momentum,
                               weight_decay=weight_decay)

        # Optimizer Scheduler
        if scheduler:
            self.scheduler = scheduler(self.optim, scheduler_gamma)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.batch1(x)
        x = self.relu(x)
        if self.is_dropout:
            x = self.dropout1(x)

        x = self.fc2(x)
        x = self.batch2(x)
        x = self.relu(x)
        if self.is_dropout:
            x = self.dropout2(x)

        x = self.fc3(x)
        x = self.batch3(x)
        x = self.relu(x)
        if self.is_dropout:
            x = self.dropout3(x)

        x = self.fc4(x)
        x = self.sigmoid(x)

        return x

    def weight_clipping(self):
        with torch.no_grad():
            for key in self.state_dict():
                if "weight" in key:
                    value = self.state_dict().get(key)
                    value.clamp_(-1, 1)

    def summary(self):
        summary(self, (1, 28, 28))

    def train_step(self, data: torch.tensor, target: torch.tensor) -> torch.tensor:
        outputs = self.forward(data)
        return self.loss_fn(outputs, target)

    def optim_step(self):
        if self.scheduler:
            self.scheduler.step()
        else:
            self.optim.step()

    def count_tp(self, outputs: torch.tensor, target: torch.tensor) -> int:
        self.eval()
        pred: torch.tensor = outputs.data.max(1, keepdim=True)[1]
        tp: int = pred.eq(target.data.view_as(pred)).sum()

        return tp


if __name__ == "__main__":
    model = Binarized_MLP()
    model.weight_clipping()
