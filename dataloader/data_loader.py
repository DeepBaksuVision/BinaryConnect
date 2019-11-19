import torch
import torchvision
import torchvision.transforms as transforms

def data_loader(dataset="CIFAR-10", batch_size = 16):

    if dataset == "CIFAR-10":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif dataset == "MNIST":
        transform = transforms.Compose(
            [transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
        classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

    return train_loader, test_loader, classes
