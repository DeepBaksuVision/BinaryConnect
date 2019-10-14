import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import Sampler


class CustomSampler(Sampler):
    """
    Samples elements sequentially from some offset.
    Arguments:
        num_samples: number of desired datapoints
        start: offset where we shuld start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


num_train = 49000
num_valid = 1000


def train_valid_loader(valid_size=0.1,
                       shuffle=True,
                       pin_memory=True,
                       random_seed=42):
    """
    train_data loader and validation_data loader
    :param num_workers: number of subprocesses
    :param valid_size: valid_size(type=fload, in [0, 1]).
    :param augment: whether to apply the data augment
    :param shuffle: whether to shuffle data
    :param pin_memory: copy tensors into CUDA pinned memory
    :param random_seed:
    :return: train_loader valid_loader
    """
    # define transforms
    train_transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    valid_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                             transform=train_transform)
    valid_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,
                                             transform=valid_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, sampler=CustomSampler(num_train),
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=4, sampler=CustomSampler(num_valid, num_train),
                                               num_workers=0)

    return train_loader, valid_loader


def test_loader():
    transform = transforms.Compose(
         [transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_data_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(test_data_set, batch_size=4, shuffle=False, num_workers=0)

    return test_data_loader

