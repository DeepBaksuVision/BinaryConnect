import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchsummary import summary
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from layers.binarized_linear import BinarizedLinear


class Binarized_MLP(pl.LightningModule):
    def __init__(self, device, mode="Stochastic"):
        super(Binarized_MLP, self).__init__()

        self.lr = 0.001
        self.fc1 = BinarizedLinear(28 * 28, 1024, bias=True, mode=mode, device=device)
        self.batch1 = nn.BatchNorm1d(1024)
        self.fc2 = BinarizedLinear(1024, 1024, bias=True, mode=mode, device=device)
        self.batch2 = nn.BatchNorm1d(1024)
        self.fc3 = BinarizedLinear(1024, 1024, bias=True, mode=mode, device=device)
        self.batch3 = nn.BatchNorm1d(1024)
        self.fc4 = BinarizedLinear(1024, 10, bias=True, mode=mode, device=device)

        self.relu = nn.ReLU()

        self.loss = F.cross_entropy

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.batch1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.batch2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.batch3(x)
        x = self.relu(x)

        x = self.fc4(x)

        return x

    def summary(self):
        summary(self, (1, 28, 28))

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': self.loss(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': self.loss(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': self.loss(y_hat, y)}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.lr,
                                    momentum=0.95)
        optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [optimizer_scheduler]

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(MNIST(os.getcwd(),
                                train=True,
                                download=True,
                                transform=transforms.ToTensor()),
                          batch_size=200)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(MNIST(os.getcwd(),
                                train=True,
                                download=True,
                                transform=transforms.ToTensor()),
                          batch_size=200)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(MNIST(os.getcwd(),
                                train=False,
                                download=True,
                                transform=transforms.ToTensor()),
                          batch_size=10000)


if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    weight_path = os.path.join(os.getcwd(), 'checkpoint')
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)

    checkpoint_callback = ModelCheckpoint(
        filepath=weight_path,
        save_best_only=False,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='',
        save_weights_only=True
    )

    gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Binarized_MLP(device=device, mode="Stochastic")
    model.to(device)
    model.summary()

    trainer = Trainer(checkpoint_callback=checkpoint_callback,
                      max_nb_epochs=5, train_percent_check=0.1)
    trainer.fit(model)
    trainer.test(model)