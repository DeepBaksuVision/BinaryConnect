
import torch
from utils.custom_op import binary_linear


class BinarizedLinear(torch.nn.Linear):

    def __init__(self, in_features, out_features, bias=False, mode="determistic"):
        super().__init__(in_features, out_features, bias)
        self.mode = mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return binary_linear(input, self.weight, self.bias)
        return binary_linear(input, self.weight)

    def reset_parameters(self):
        # xavier initialization
        torch.nn.init.xavier_normal(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.01)


if __name__ == "__main__":
    import os
    import torch.optim as optim
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinarizedLinear(28*28, 10, bias=False, mode="determistic")
    model.train()
    model.to(device)
    dataloader = DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for i, (data, target) in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
        optimizer.zero_grad()
        data = data.view(dataloader.batch_size, -1)
        data = data.to(device)
        target = target.to(device)
        output = model(data)
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
        break


