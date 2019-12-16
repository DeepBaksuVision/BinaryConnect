import torch
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm
from models.binarized_mlp import Binarized_MLP

def train(model, device, train_loader, epoch):
    model.train()
    avg_loss = []
    for i, (data, target) in enumerate(tqdm(train_loader)):
        model.train()
        data = data.to(device)
        target = target.to(device)
        loss = model.train_step(data, target)
        loss.backward()
        model.optim.step()
        model.weight_clipping()
        model.optim.zero_grad()
        avg_loss.append(loss.item())

    return sum(avg_loss) / len(avg_loss)


def accuracy(output: torch.tensor,
             target: torch.tensor) -> float:
    pred: torch.tensor = output.data.max(1, keepdim=True)[1]
    correct: int = pred.eq(target.data.view_as(pred)).sum()
    len_data: int = output.shape[0]

    return float(correct) / len_data


def valid(model, device, test_loader, epoch):
    avg_loss = []
    avg_acc = []
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader)):
            data = data.to(device)
            target = target.to(device)
            outputs = model.forward(data)
            loss = model.loss_fn(outputs, target)
            acc = accuracy(outputs, target)

            avg_loss.append(loss)
            avg_acc.append(acc)

    print('Valid Epoch: {}\tLoss: {:.6f}\tAcc: {:.6f}'.format(
        epoch,
        loss.item(),
        acc))

    return sum(avg_loss) / len(avg_loss), sum(avg_acc) / len(avg_acc)


def main():
    # Classes
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # Set Common Parameters
    batch_size: int = 256
    epochs: int = 1000
    model_name = "./Vanila_Deterministic_BinaryConnect.pth"

    # Transform
    transform = transforms.Compose([transforms.ToTensor()])

    # Dataset directory
    root = "./data"
    MNIST_train_datasets = torchvision.datasets.MNIST(root,
                                                      train=True,
                                                      transform=transform,
                                                      target_transform=None,
                                                      download=True)

    MNIST_test_datasets = torchvision.datasets.MNIST(root,
                                                     train=False,
                                                     transform=transform,
                                                     target_transform=None,
                                                     download=True)

    train_loader = torch.utils.data.DataLoader(dataset=MNIST_train_datasets,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=MNIST_test_datasets,
                                              batch_size=len(MNIST_test_datasets),
                                              shuffle=False)

    # TensorBoard
    writer = SummaryWriter()

    # device
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Optimizer
    optimizer = optim.SGD

    # Exponential Learning Rate Decay
    scheduler = torch.optim.lr_scheduler.ExponentialLR

    scheduler_gamma = 0.1
    learning_rate = 0.01
    momentum = 0.
    weight_decay = 0.

    model = Binarized_MLP(mode="Deterministic",
                          is_dropout=False,
                          dropout_prob=1.0,
                          optimizer=optimizer,
                          learning_rate=learning_rate,
                          momentum=momentum,
                          weight_decay=weight_decay,
                          scheduler=None,
                          scheduler_gamma=scheduler_gamma)

    # Train
    model.to(device)

    train_loss_per_epoch = [0]
    valid_loss_per_epoch = [0]
    valid_acc_per_epoch = [0]

    not_improved_count = 0

    for epoch in range(epochs):

        train_loss = train(model, device, train_loader, epoch)
        valid_loss, valid_acc = valid(model, device, test_loader, epoch)

        if valid_acc > max(valid_acc_per_epoch):
            not_improved_count = 0
        else:
            not_improved_count += 1
            print("ACC not improved. count : {}".format(not_improved_count))
            print("Present Acc : {}, Max Acc : {}".format(valid_acc, max(valid_acc_per_epoch)))

        if not_improved_count > 5:
            print("Early Stopping")
            break

        writer.add_scalar("Vanila BinaryConnect/Loss/train", train_loss, epoch)
        writer.add_scalar("Vanila BinaryConnect/Loss/valid", valid_loss, epoch)
        writer.add_scalar("Vanila BinaryConnect/Acc/valid", valid_acc, epoch)
        train_loss_per_epoch.append(train_loss)
        valid_loss_per_epoch.append(valid_loss)
        valid_acc_per_epoch.append(valid_acc)

    torch.save(model.state_dict(), model_name)


if __name__ == "__main__":
    main()
