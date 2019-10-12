import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
# from utils import plot_images


def train_valid_loader(batch_size,
                       augment,
                       num_workers,
                       valid_size=0.1,
                       shuffle=True,
                       show_sample=False,
                       pin_memory=True,
                       random_seed=42):
    """
    train_data loader and validation_data loader

    :param batch_size: samples per batch to load.
    :param augment: whether to apply the data augment.
    :param valid_size: valid_size(type=float, in [0, 1]).
    :param shuffle: whether to shuffle the train_validation indices.
    :param show_sample: plot sample grid of the dataset.
    :param num_workers: number of subprocesses to use when loading the dataset.
    :param pin_memory: whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
    :param random_seed:
    :return: train_loader, valid_loader
    """

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    valid_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=valid_transform)

    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)

    # 시각화
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_set, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)

    return train_loader, valid_loader


def test_loader(batch_size, shuffle=False, num_workers=4, pin_memory=False):
    """
    test_data loader

    :param batch_size: samples per batch to load.
    :param shuffle: whether to shuffle the train_validation indices.
    :param num_workers: number of subprocesses to use when loading the dataset.
    :param pin_memory: whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
    :return: test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )

    return test_loader


