
import torch.optim as optim
import argparse
from models.binarized_mlp import Binarized_MLP
from ray import tune
from ray.tune.examples.mnist_pytorch import get_data_loaders, test, train

def train_bnn(config):
    model = config["model"]
    epoch = config["epoch"]

    train_loader, test_loader = get_data_loaders()

    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    for epoch in range(1, epoch + 1):
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)
        tune.track.log(mean_accuracy=acc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bnn_type', type=str, default='Stochastic')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()

    if args.bnn_type == "Stochastic" or args.bnn_type == "Deterministic":
        model = Binarized_MLP(args.bnn_type)
    else:
        raise RuntimeError("not supported quantization method")

    analysis = tune.run(
        train, config={"model": model,
                       "epoch": args.epochs,
                       "lr": tune.grid_search([0.0001, 0.001, 0.01, 0.1]),
                       "momentum": tune.grid_search([0.3, 0.5, 0.9, 0.99])
                       }

    )
    print("Best Config : {}".format(analysis.get_best_config(metric="mean_accuracy")))
    df = analysis.dataframe()

if __name__ == "__main__":
    main()