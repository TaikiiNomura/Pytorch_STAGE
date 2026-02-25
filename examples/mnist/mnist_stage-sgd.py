import argparse
import torch
from torchvision import datasets, transforms

import stage

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

def train_model(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    parser = argparse.ArgumentParser(description="STAGE MNIST Example")
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for train (def=64)'
    )
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1000,
        metavar='N',
        help='input batch size for test (def=1000)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        metavar='N',
        help='number of epochs (def=10)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0025,
        metavar='LR',
        help='learning rate (def=0.0025)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='S',
        help='random seed (def=1)'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=300,
        help='number of wait logging'
    )

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
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
        '../data',
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.MNIST(
        '../data',
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

    model = Net().to(device)
    optimizer = stage.STAGE_SGD(
        model.parameters(),
        lr=args.lr,
    )

    for epoch in range(1, args.epochs+1):
        train_model(
            args,
            model,
            device,
            train_loader, 
            optimizer,
            epoch,
        )
        test_model(
            model,
            device,
            test_loader,
        )
    
if __name__ == '__main__':
    main()