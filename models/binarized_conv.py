import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchsummary import summary
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from layers.binarized_conv import BinarizedConv2d
from layers.binarized_linear import BinarizedLinear


class Binarized_CONV(pl.LightningModule):
    def __init__(self, device, mode="Stochastic"):
        super(Binarized_CONV, self).__init__()

        self.conv1 = BinarizedConv2d(in_channels=3,
                                     out_channels=128,
                                     kernel_size=3,
                                     bias=True,
                                     mode=mode,
                                     padding=1,
                                     device=device)
        self.batch1 = nn.BatchNorm2d(128)

        self.conv2 = BinarizedConv2d(in_channels=128,
                                     out_channels=128,
                                     kernel_size=3,
                                     bias=True,
                                     mode=mode,
                                     padding=1,
                                     device=device)
        self.batch2 = nn.BatchNorm2d(128)

        self.conv3 = BinarizedConv2d(in_channels=128,
                                     out_channels=256,
                                     kernel_size=3,
                                     bias=True,
                                     mode=mode,
                                     padding=1,
                                     device=device)
        self.batch3 = nn.BatchNorm2d(256)

        self.conv4 = BinarizedConv2d(in_channels=256,
                                     out_channels=256,
                                     kernel_size=3,
                                     bias=True,
                                     mode=mode,
                                     padding=1,
                                     device=device)
        self.batch4 = nn.BatchNorm2d(256)

        self.conv5 = BinarizedConv2d(in_channels=256,
                                     out_channels=512,
                                     kernel_size=3,
                                     bias=True,
                                     mode=mode,
                                     padding=1,
                                     device=device)
        self.batch5 = nn.BatchNorm2d(512)

        self.conv6 = BinarizedConv2d(in_channels=512,
                                     out_channels=512,
                                     kernel_size=3,
                                     bias=True,
                                     mode=mode,
                                     padding=1,
                                     device=device)
        self.batch6 = nn.BatchNorm2d(512)

        # ??

        self.fc1 = BinarizedLinear(512 * 4 * 4, 1024, bias=True, mode=mode, device=device)
        self.fc2 = BinarizedLinear(1024, 1024, bias=True, mode=mode, device=device)

        self.final = BinarizedLinear(1024, 10, bias=True, mode=mode, device=device)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=2,
                                    stride=2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.conv5(x)
        x = self.batch5(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.batch6(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.tanh(x)

        x = self.fc2(x)
        x = self.tanh(x)

        x = self.final(x)

        return x

    def summary(self):
        summary(self, (3, 32, 32))

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
        return DataLoader(CIFAR10(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=50)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(CIFAR10(os.getcwd(),
                                  train=True,
                                  download=True,
                                  transform=transforms.ToTensor()),
                          batch_size=32)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(CIFAR10(os.getcwd(),
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
        save_weights_only= True
    )

    gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Binarized_CONV(device=device, mode="Stochastic")
    model.to(device)
    model.summary()
    trainer = Trainer(checkpoint_callback=checkpoint_callback,
                      max_nb_epochs=1, train_percent_check=0.1)
    trainer.fit(model)
    trainer.test(model)

