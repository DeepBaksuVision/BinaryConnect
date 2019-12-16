import torch

from utils.custom_op import binary_conv2d


class BinarizedConv2d(torch.nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode='zeros',
                 mode="determistic"):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         groups,
                         bias,
                         padding_mode)
        self.mode = mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return binary_conv2d(input,
                                 self.weight,
                                 self.bias,
                                 self.mode,
                                 self.stride,
                                 self.padding,
                                 self.dilation,
                                 self.groups)

        return binary_conv2d(input,
                             self.weight,
                             self.bias,
                             self.mode,
                             self.stride,
                             self.padding,
                             self.dilation,
                             self.groups)


if __name__ == "__main__":
    import os
    import torch.optim as optim
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinarizedConv2d(in_channels=1,
                            out_channels=10,
                            kernel_size=28,
                            bias=False,
                            mode="stochastic")
    model.to(device)
    dataloader = DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
                            batch_size=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for i, (data, target) in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        output = output.view(dataloader.batch_size, -1)
        loss = criterion(output, target)
        print("Output : {}".format(output))
        print("Model Weight Before")
        print(model.weight)
        loss.backward()
        optimizer.step()
        print("Grad Func")
        print(output.grad_fn)
        print("---------")
        print("Grad")
        print(model.weight.grad)
        print("---------")
        print("Model Weight After")
        print(model.weight)
        print("---------")
        print("data")
        print(data)
        print("---------")
        print(data.grad)
        break
