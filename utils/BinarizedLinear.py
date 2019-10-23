import copy
import torch
import torch.nn.functional as F

class BinarizedLinear(torch.nn.Linear):

    def __init__(self, in_features, out_features, device: torch.device, bias=True, mode="Stochastic"):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.device = device
        self.bin_weight = self.weight_binarization(self.weight, self.mode)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.clipping_weight(self.weight)
        self.bin_weight = self.weight_binarization(self.weight, self.mode)
        return F.linear(input, self.bin_weight, self.bias)

    def weight_binarization(self, weight: torch.tensor, mode:str):
        with torch.set_grad_enabled(False):
            if mode == "Stochastic":
                bin_weight = self.stocastic(weight)
            elif mode == "Deterministic":
                bin_weight = self.deterministic(weight)
            else:
                raise RuntimeError("{} is does not exist or not supported".format(mode))
        bin_weight.requires_grad = True
        bin_weight.register_hook(self.cp_bin_grad_to_real_grad_hook)
        return bin_weight

    @staticmethod
    def deterministic(weight: torch.tensor) -> torch.tensor:
        return weight.sign()

    @staticmethod
    def stocastic(weight: torch.tensor) -> torch.tensor:
        p = torch.sigmoid(weight)
        uniform_matrix = torch.empty(p.shape).uniform_(0,1)
        bin_weight = (p >= uniform_matrix).type(torch.float32)
        bin_weight[bin_weight==0] = -1
        return bin_weight

    def cp_bin_grad_to_real_grad_hook(self, grad):
        # it not using deepcopy
        # you will be meet Error as
        # `Can't detach views in-place. Use detach() instead`
        self.weight.grad = copy.deepcopy(grad)

    def clipping_weight(self, weight:torch.tensor) -> torch.tensor:
        with torch.set_grad_enabled(False):
            weight = torch.clamp(weight, -1, 1)
        weight.requires_grad = True

        return weight


if __name__ == "__main__":
    import os
    import torch.optim as optim
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinarizedLinear(5, 1, bias=False, mode="Stochastic", device=device)
    model.to(device)
    dataloader = DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for i, (data, target) in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print("Loss : {}".format(loss))


