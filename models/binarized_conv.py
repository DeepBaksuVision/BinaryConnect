import torch
import torch.nn as nn
import torch.optim as optim

from torchsummary import summary
from layers.binarized_conv import BinarizedConv2d
from layers.binarized_linear import BinarizedLinear


class Binarized_CONV(nn.Module):
    def __init__(self,
                 mode="determistic",
                 is_dropout: bool = False,
                 dropout_prob: float = 0.5,
                 optimizer: optim = optim.SGD,
                 learning_rate: float = 0.01,
                 momentum: float = 0,
                 weight_decay: float = 1e-5,
                 scheduler: optim.lr_scheduler = None,
                 scheduler_gamma: float = 0.1):
        super(Binarized_CONV, self).__init__()

        self.conv1 = BinarizedConv2d(in_channels=1,
                                     out_channels=128,
                                     kernel_size=3,
                                     bias=False,
                                     mode=mode,
                                     padding=0,
                                     dilation=1)
        self.batch1 = nn.BatchNorm2d(128)

        self.conv2 = BinarizedConv2d(in_channels=128,
                                     out_channels=128,
                                     kernel_size=3,
                                     bias=False,
                                     mode=mode,
                                     padding=0,
                                     dilation=1)
        self.batch2 = nn.BatchNorm2d(128)

        self.conv3 = BinarizedConv2d(in_channels=128,
                                     out_channels=256,
                                     kernel_size=3,
                                     bias=False,
                                     mode=mode,
                                     padding=0,
                                     dilation=1)
        self.batch3 = nn.BatchNorm2d(256)

        self.conv4 = BinarizedConv2d(in_channels=256,
                                     out_channels=256,
                                     kernel_size=3,
                                     bias=False,
                                     mode=mode,
                                     padding=0,
                                     dilation=1)
        self.batch4 = nn.BatchNorm2d(256)

        self.conv5 = BinarizedConv2d(in_channels=256,
                                     out_channels=512,
                                     kernel_size=3,
                                     bias=False,
                                     mode=mode,
                                     padding=0,
                                     dilation=1)
        self.batch5 = nn.BatchNorm2d(512)

        self.conv6 = BinarizedConv2d(in_channels=512,
                                     out_channels=512,
                                     kernel_size=3,
                                     bias=False,
                                     mode=mode,
                                     padding=0,
                                     dilation=1)
        self.batch6 = nn.BatchNorm2d(512)

        self.fc1 = BinarizedLinear(512 * 3 * 3, 1024, bias=False, mode=mode)
        self.fc2 = BinarizedLinear(1024, 1024, bias=False, mode=mode)
        self.final = BinarizedLinear(1024, 10, bias=False, mode=mode)

        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=2,
                                    stride=2)

        # Dropout
        self.is_dropout = is_dropout
        if self.is_dropout:
            self.dropout1 = nn.Dropout(dropout_prob)
            self.dropout2 = nn.Dropout(dropout_prob)
            self.dropout4 = nn.Dropout(dropout_prob)
            self.dropout5 = nn.Dropout(dropout_prob)
            self.dropout6 = nn.Dropout(dropout_prob)
            self.dropout7 = nn.Dropout(dropout_prob)
            self.dropout8 = nn.Dropout(dropout_prob)

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

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        if self.is_dropout:
            x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)
        if self.is_dropout:
            x = self.dropout2(x)

        #x = self.maxpool(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu(x)
        if self.is_dropout:
            x = self.dropout3(x)

        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu(x)
        if self.is_dropout:
            x = self.dropout4(x)

        x = self.maxpool(x)

        x = self.conv5(x)
        x = self.batch5(x)
        x = self.relu(x)
        if self.is_dropout:
            x = self.dropout5(x)

        x = self.conv6(x)
        x = self.batch6(x)
        x = self.relu(x)
        if self.is_dropout:
            x = self.dropout6(x)

        x = self.maxpool(x)

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.relu(x)
        if self.is_dropout:
            x = self.dropout7(x)

        x = self.fc2(x)
        x = self.relu(x)
        if self.is_dropout:
            x = self.dropout8(x)

        x = self.final(x)
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
