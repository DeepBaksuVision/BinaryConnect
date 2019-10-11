import torch
import torch.nn as nn
import torch.optim as optim
from dataloader.data_loader import data_loader
from tqdm import tqdm
import argparse
from models.mlp import MLP
from models.conv import CNN
from models.resnet import ResNet50
from torchvision.models import ResNet


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # calculate accuracy of predictions in the current batch
    correct, total = 0, 0
    for i, (data, target) in tqdm(enumerate(train_loader, 0)):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        criterion = nn.CrossEntropyLoss()

        _, predicted = (torch.max(outputs.data, 1))

        total += target.size(0)
        correct += (predicted == target).sum().item()
        train_acc = 100. * correct / total

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAccuracy: {:.6f}\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(train_loader.dataset),
                       100. * i / len(train_loader), train_acc, loss.item()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ResNet50')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    args = parser.parse_args()

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if args.cuda else "cpu")
    train_loader = data_loader()[0]

    # if args.model == 'CNN':
    #     model = CNN()
    # if args.model == 'MLP':
    #     model = MLP()
    model = ResNet50()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in tqdm(range(1, args.epochs + 1)):
        train(args, model, device, train_loader, optimizer, epoch)

    if (args.save_model):
        torch.save(model.state_dict(), "CIFAR-10_ResNet50.pt")


if __name__ == "__main__":
    main()

