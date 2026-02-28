import argparse
import torch
from torchvision import datasets, transforms
import math

import stage

import mnist_utils

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)

        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def test_model(model, test_loader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    return correct / len(test_loader.dataset)

def main():

    parser = argparse.ArgumentParser(description="serch HPRAMS lr and tau")
    # parser.add_argument(
    #     '',
    #     type=,
    #     default=,
    #     metavar=,
    #     help=,
    # )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for train (def=64)',
    )
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1000,
        metavar='N',
        help='input batch size for test (def=1000)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        metavar='N',
        help='epochs (def=15)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        metavar='LR',
        help='learning rate (def=0.001)',
    )
    parser.add_argument(
        '--tau',
        type=float,
        default=1.0,
        metavar='TAU',
        help='sech hparam (def=1.0)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='S',
        help='random seed (def=1)'
    )

    args = parser.parse_args()

    mnist_utils.init_results_file()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size}
    if device.type == "cuda":
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True,
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,)
        ),
    ])
    train_data = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.MNIST(
        './data',
        train=False,
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        **train_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        **test_kwargs,
    )

    optimizers = {
        "torch-SGD": torch.optim.SGD,
        "stage-SGD": stage.STAGE_SGD,
    }

    for name, func_class in optimizers.items():
        
        torch.manual_seed(args.seed)
        model = Net().to(device)
        
        optm_kwargs = {'lr': args.lr}
        if name == "stage-SGD":
            stage_kwargs = {'tau': args.tau}
            optm_kwargs.update(stage_kwargs)
        
        optimizer = func_class(model.parameters(), **optm_kwargs)

        for epoch in range(args.epochs):
            loss = train_model(
                model,
                train_loader,
                optimizer,
                device,
            )

            acc = test_model(
                model,
                test_loader,
                device
            )

            print(f'{name} | epoch | train loss | {loss} | test acc | {acc}')
        
            mnist_utils.append_result({
                "optimizer": name,
                "lr": args.lr,
                "tau": args.tau,
                "seed": args.seed,
                "epoch": epoch,
                "epoch_loss": loss,
                "epoch_acc": acc,
            })

if __name__ == "__main__":
    main()