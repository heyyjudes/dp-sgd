import time
import math
import torch
import argparse
import numpy as np
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

# datasets
import torchvision
from torchvision import datasets, transforms
from gradcnn import make_optimizer, replicate_model

# local libs
import models 
import models_dp  
import train 

def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def run_mnist_model(args, device, kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.dp_mode == 'no-dp': 
        print("Runing MNIST with no differential privacy")
        model = models.MNIST_Net()
        Optimizer = optim.SGD
        train_F = train.train

    elif args.dp_mode == 'naive':
        print("Runing MNIST with differential privacy using naive gradient computations")
        model = models_dp.MNIST_Net()
        Optimizer = make_optimizer(cls=optim.SGD, noise_multiplier=1.1, l2_norm_clip=1.0)
        train_F = train.train_naive

    elif args.dp_mode == 'oprod': 
        print("Runing MNIST with differential privacy using outer product")
        model = models_dp.MNIST_Net() 
        model.get_detail(True)
        Optimizer = make_optimizer(cls=optim.SGD, noise_multiplier=1.1, l2_norm_clip=1.0)
        train_F = train.train_outer_product
        
    elif args.dp_mode == 'multi':
        print("Runing MNIST with differential privacy using multiple models")
        MNet = replicate_model(net_class=models_dp.MNIST_Net, batch_size=args.batch_size)
        model = MNet()
        model.get_detail(True)
        Optimizer = make_optimizer(cls=optim.SGD, noise_multiplier=1.1, l2_norm_clip=1.0)
        train_F = train.train_multi

    elif args.dp_mode == 'single-fwd-lg':
        print("Runing MNIST with differential privacy using naive gradient computations")
        model = models_dp.MNIST_Net()
        Optimizer = make_optimizer(cls=optim.SGD, noise_multiplier=1.1, l2_norm_clip=1.0)
        train_F = train.train_single_fwd_lg

    else: 
        print("Runing MNIST with differential privacy using single forward")
        model = models.MNIST_Net()
        Optimizer = optim.SGD
        train_F = train.train_single_fwd_sm
    
    model.to(device)
    optimizer_ = Optimizer(model.parameters(), lr=args.lr)
    criterion = F.cross_entropy

    for epoch in range(1, args.epochs + 1):
        train_F(args, model, device, train_loader, optimizer_, epoch, criterion)
        test(args, model, device, test_loader, F.cross_entropy)

def run_cifar_model(args, device, kwargs):
    # load CIFAR dataset
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2,  drop_last=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                         shuffle=False, num_workers=2)
    
    if args.dp_mode == 'no-dp': 
        print("Runing CIFAR-10 with no differential privacy")
        model = models.CIFAR_Net()
        Optimizer = optim.SGD
        train_F = train.train

    elif args.dp_mode == 'naive':
        print("Runing CIFAR-10 with differential privacy using naive gradient computations")
        model = models_dp.CIFAR_Net()
        Optimizer = make_optimizer(cls=optim.SGD, noise_multiplier=1.1, l2_norm_clip=1.0)
        train_F = train.train_naive

    elif args.dp_mode == 'oprod': 
        print("Runing CIFAR-10 with differential privacy using outer product")
        model = models_dp.CIFAR_Net() 
        model.get_detail(True)
        Optimizer = make_optimizer(cls=optim.SGD, noise_multiplier=1.1, l2_norm_clip=1.0)
        train_F = train.train_outer_product
        
    elif args.dp_mode == 'multi':
        print("Runing CIFAR-10 with differential privacy using multiple models")
        MNet = replicate_model(net_class=models_dp.CIFAR_Net, batch_size=args.batch_size)
        model = MNet()
        model.get_detail(True)
        Optimizer = make_optimizer(cls=optim.SGD, noise_multiplier=1.1, l2_norm_clip=1.0)
        train_F = train.train_multi

    else: 
        print("Runing CIFAR-10 with differential privacy using single forward")
        model = models.CIFAR_Net()
        Optimizer = optim.SGD
        train_F = train.train_single_fwd_lg
    
    model.to(device)
    optimizer_ = Optimizer(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = F.cross_entropy

    for epoch in range(1, args.epochs + 1):
        train_F(args, model, device, train_loader, optimizer_, epoch, criterion)
        test(args, model, device, test_loader, F.cross_entropy)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch DP-SGD Experiments')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dp-mode', type=str, default='no-dp',
                        help='specifiy dp mode: "no-dp", "naive", "oprod", "multi" or "single-fwd"' )
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='specifiy data task: "MNIST" or "CIFAR"')

    args = parser.parse_args()

    torch.manual_seed(1)
    device = torch.device("cuda:0" if args.use_cuda else "cpu")
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.use_cuda else {}

    assert(args.dp_mode in ['no-dp', 'naive', 'oprod', 'multi', 'single-fwd-sm', 'single-fwd-lg'])

    if args.use_cuda:
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    if args.dataset.upper() == 'MNIST': 
        run_mnist_model(args, device, kwargs)
    elif args.dataset.upper() == 'CIFAR': 
        args.lr = 0.01
        run_cifar_model(args, device, kwargs)
    else: 
        print('please specifiy either "MNIST" or "CIFAR"  as --dataset parameter')
        return

    if args.use_cuda:
        torch.cuda.synchronize()
    
    elapsed_time = time.perf_counter() - start_time   

    print('Elapsed time: {:.2f}s'.format(elapsed_time))

if __name__ == '__main__':
    main()
