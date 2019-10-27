import torch
import wandb
import time
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from dataset import MNIST
from model import LeNet5, CustomMLP

# It makes
wandb.init(project="individual-mlp")


def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    model.to(device)
    trn_loss = 0
    acc = 0

    for i, (data, target) in enumerate(trn_loader):                 # i means how many
                                                                    # times trained model
                                                                    # data, target type is tensor
        optimizer.zero_grad()                                       # pytorch has gradient before nodes
        data, target = data.to(device), target.to(device)           # gramma caution
        output = model(data)                                        # input data in model
        # w = model.named_parameters()
        trn_loss = criterion(output, target)  # cost fcn is Cross_entropy
        trn_loss.backward()  # backpropagation
        optimizer.step()  # training model

        # accuracy
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).sum()
        length_of_data = data.shape[0]
        acc = (float(correct) / length_of_data) * 100

        wandb.log({"Test Accuracy": acc, "Test Loss": trn_loss})    #log to wandb at live-stream
        if i % 10 == 0 and i != 0:  # interval setting
            print('Train : [{} / {} ({:.0f}%)]\tLoss: {:.6f}'.format(
                i * len(data), len(trn_loader.dataset),
                100. * i / len(trn_loader), trn_loss))

    print('Finished Training Trainset')

    return trn_loss, acc


def test(model, tst_loader, device, criterion):
    model.eval()
    model.to(device)
    tst_loss = 0
    correct = 0

    for data, target in tst_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        tst_loss += criterion(output, target)

        # get the index of the max log-probability
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    tst_loss /= len(tst_loader.dataset)
    acc = 100. * correct / len(tst_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(tst_loss, correct, len(tst_loader.dataset), acc))
    print('Finished Testing Test set')

    return tst_loss, acc


def main():

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    batch_size = 128
    train_dataset_dir = 'train/'
    test_dataset_dir = "test/"

    train_dataset = MNIST(data_dir=train_dataset_dir,
                          transform=transform)

    test_dataset = MNIST(data_dir=test_dataset_dir,
                         transform=transform)

    trn_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    tst_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)

    # using cpu
    device = torch.device('cpu' if torch.cuda.is_available() else 'cuda:0')

    # using gpu
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    epoch = 5

    # LeNet5
    model = LeNet5()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    wandb.watch(model)                                                      # visualization Gradients

    print("model = LeNet5")

    for j in range(epoch):
        print("epoch = {} \n".format(j + 1))
        train(model, trn_loader, device, criterion, optimizer)
        test(model, tst_loader, device, criterion)
    test(model, tst_loader, device, criterion)
    # torch.save(model.state_dict(), wandb.run.dir)





if __name__ == '__main__':
    main()
