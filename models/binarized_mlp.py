import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchsummary import summary
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from utils.BinarizedLinear import BinarizedLinear

class Binarized_MLP(pl.LightningModule):
    def __init__(self, mode="Stochastic"):
        super(Binarized_MLP, self).__init__()
        self.fc1 = BinarizedLinear(28 * 28, 512, bias=False, mode=mode)
        self.dropout = nn.Dropout(0.2)
        self.batch1 = nn.BatchNorm1d(512)
        self.fc2 = BinarizedLinear(512, 512, bias=False, mode=mode)
        self.dropout = nn.Dropout(0.2)
        self.batch2 = nn.BatchNorm1d(512)
        self.fc3 = BinarizedLinear(512, 10, bias=False, mode=mode)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.sigmoid(self.fc1(x))
        x = self.batch1(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.batch2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def summary(self):
        summary(self, (1, 28, 28))

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32)


if __name__ == "__main__":
    from pytorch_lightning import Trainer
    model = Binarized_MLP()
    model.summary()
    trainer = Trainer()
    trainer.fit(model)