import time
import math
import torch
import scipy
import scipy.stats
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
import pdb

def test(args, model, device, test_loader, criterion, dataset_str):
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

    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        dataset_str, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def calculate_eps(args, train_loader): 
    q = args.batch_size/len(train_loader.dataset)
    sigma = args.sigma
    delta = args.delta
    T = args.epochs/q
    SAMPLE_SIZE = 100000 # Increase for getting more accurate results
    alpha_list = np.arange(14,20,1) #Search with a wider and finer range for getting more accurate results.
    eps_list = {}
    for alpha in alpha_list:
        P = lambda x: scipy.exp(-x**2/2)/(scipy.sqrt(2*scipy.pi))
        Q = lambda x: ((1-q)*scipy.exp(-x**2/2)+q*scipy.exp(-(x-1/sigma)**2/2))/(scipy.sqrt(2*scipy.pi))
        Normal_Samples = scipy.stats.norm.rvs(size=SAMPLE_SIZE)

        D_alpha_P_Q = 1/(alpha-1) * scipy.log(np.mean([(P(x)/Q(x))**(alpha-1) for x in Normal_Samples]))
        D_alpha_Q_P = 1/(alpha-1) * scipy.log(np.mean([(Q(x)/P(x))**alpha for x in Normal_Samples]))
        #print(D_alpha_P_Q,D_alpha_Q_P)
        eps = scipy.log(1/delta)/(alpha-1) + T*max(D_alpha_P_Q,D_alpha_Q_P)
        eps_list[alpha]=eps
        
    eps = min(eps_list.values())
    print('Best eps: ',eps)


def calculate_eps_approx(args, train_loader): 
    q = args.batch_size/len(train_loader.dataset)
    sigma = args.sigma
    delta = args.delta
    T = args.epochs/q
    eps_1 = 2*(q/sigma)*scipy.sqrt(T*scipy.log(1/delta))
    eps_2 = 2*scipy.log(1/delta)/(sigma**2 * scipy.log(1/(q*sigma)))
    eps=max(eps_1,eps_2)
    print('Best eps: ',eps)


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

    if args.sigma != 0: 
        calculate_eps_approx(args, train_loader)

    if args.dp_mode == 'no-dp': 
        print("Runing MNIST with no differential privacy")
        model = models.MNIST_Net()
        Optimizer = optim.SGD
        train_F = train.train

    elif args.dp_mode == 'naive':
        print("Runing MNIST with differential privacy using naive gradient computations")
        model = models_dp.MNIST_Net()
        Optimizer = make_optimizer(cls=optim.SGD, noise_multiplier=args.sigma, l2_norm_clip=1.0)
        train_F = train.train_naive

    elif args.dp_mode == 'naive-sm':
        print("Runing MNIST with differential privacy using naive gradient computations")
        model = models.MNIST_Net()
        Optimizer = optim.SGD
        train_F = train.train_naive_sm

    elif args.dp_mode == 'oprod': 
        print("Runing MNIST with differential privacy using outer product")
        model = models_dp.MNIST_Net() 
        model.get_detail(True)
        Optimizer = make_optimizer(cls=optim.SGD, noise_multiplier=args.sigma, l2_norm_clip=1.0)
        train_F = train.train_outer_product
        
    elif args.dp_mode == 'multi':
        print("Runing MNIST with differential privacy using multiple models")
        MNet = replicate_model(net_class=models_dp.MNIST_Net, batch_size=args.batch_size)
        model = MNet()
        model.get_detail(True)
        Optimizer = make_optimizer(cls=optim.SGD, noise_multiplier=args.sigma, l2_norm_clip=1.0)
        train_F = train.train_multi

    elif args.dp_mode == 'single-fwd-lg':
        print("Runing MNIST with differential privacy using using large single forward model")
        model = models_dp.MNIST_Net()
        Optimizer = make_optimizer(cls=optim.SGD, noise_multiplier=args.sigma, l2_norm_clip=1.0)
        train_F = train.train_single_fwd_lg

    elif args.dp_mode == 'single-fwd-sm': 
        print("Runing MNIST with differential privacy using single forward")
        model = models.MNIST_Net()
        Optimizer = optim.SGD
        train_F = train.train_single_fwd_sm
    
    model.to(device)
    optimizer_ = Optimizer(model.parameters(), lr=args.lr)
    criterion = F.cross_entropy

    for epoch in range(1, args.epochs + 1):
        train_F(args, model, device, train_loader, optimizer_, epoch, criterion)
        print("Epoch: {}".format(epoch))
        test(args, model, device, train_loader, F.cross_entropy, 'Train Set')
        test(args, model, device, test_loader, F.cross_entropy, 'Test Set')

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
    
    calculate_eps_approx(args, train_loader)

    if args.dp_mode == 'no-dp': 
        print("Runing CIFAR-10 with no differential privacy")
        model = models.CIFAR_Net()
        Optimizer = optim.SGD
        train_F = train.train

    elif args.dp_mode == 'naive':
        print("Runing CIFAR-10 with differential privacy using naive gradient computations")
        model = models_dp.CIFAR_Net()
        Optimizer = make_optimizer(cls=optim.SGD, noise_multiplier=args.sigma, l2_norm_clip=1.0)
        train_F = train.train_naive

    elif args.dp_mode == 'oprod': 
        print("Runing CIFAR-10 with differential privacy using outer product")
        model = models_dp.CIFAR_Net() 
        model.get_detail(True)
        Optimizer = make_optimizer(cls=optim.SGD, noise_multiplier=args.sigma, l2_norm_clip=1.0)
        train_F = train.train_outer_product
        
    elif args.dp_mode == 'multi':
        print("Runing CIFAR-10 with differential privacy using multiple models")
        MNet = replicate_model(net_class=models_dp.CIFAR_Net, batch_size=args.batch_size)
        model = MNet()
        model.get_detail(True)
        Optimizer = make_optimizer(cls=optim.SGD, noise_multiplier=args.sigma, l2_norm_clip=1.0)
        train_F = train.train_multi

    else: 
        print("Runing CIFAR-10 with differential privacy using single forward")
        model = models.CIFAR_Net()
        Optimizer = optim.SGD
        train_F = train.train_single_fwd_lg
    
    model.to(device)
    optimizer_ = Optimizer(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_, 3.0, gamma=0.80)
    criterion = F.cross_entropy

    for epoch in range(1, args.epochs + 1):
        train_F(args, model, device, train_loader, optimizer_, epoch, criterion)
        print("Epoch: {}".format(epoch))
        test(args, model, device, train_loader, F.cross_entropy, 'Train Set')
        test(args, model, device, test_loader, F.cross_entropy, 'Test Set')
        scheduler.step() 

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Vision DP-SGD Experiments')

    ###############################################################################
    #                   Arguments without default values                          #
    #   different language tasks have different default values (specified later)  #
    ###############################################################################

    parser.add_argument('--batch_size', type=int, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--lr', type=float, metavar='LR',
                        help='learning rate')

    ###############################################################################
    #                   Arguments with default values                             #
    #                   common across language tasks                              #
    ###############################################################################
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dp-mode', type=str, default='no-dp',
                        help='specifiy dp mode: "no-dp", "naive", "oprod", "multi" or "single-fwd"' )
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='specifiy data task: "MNIST" or "CIFAR"')
    parser.add_argument('--delta', type=float, default=10**(-5),
                        help='privacy parameter')
    parser.add_argument('--sigma', type=float, default=1.9, 
                        help='noise multiplier')
    parser.add_argument('--l2_clip', type=float, default=1, help='l2 clip')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='prints sub-epoch progress')

    args = parser.parse_args()

    torch.manual_seed(1)
    device = torch.device("cuda:0" if args.use_cuda else "cpu")
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.use_cuda else {}

    assert(args.dp_mode in ['no-dp', 'naive', 'naive-sm', 'oprod', 'multi', 'single-fwd-sm', 'single-fwd-lg'])

    if args.use_cuda:
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    if args.dataset.upper() == 'MNIST':
        if not args.batch_size:
            args.batch_size = 64
        if not args.test_batch_size: 
            args.test_batch_size = 1000 
        if not args.lr: 
            args.lr = 0.05
        run_mnist_model(args, device, kwargs)

    elif args.dataset.upper() == 'CIFAR': 
        if not args.batch_size:
            args.batch_size = 32
        if not args.test_batch_size: 
            args.test_batch_size = 1000 
        if not args.lr: 
            args.lr = 0.01
        run_cifar_model(args, device, kwargs)
    else: 
        print('please specifiy either "MNIST" or "CIFAR"  as --dataset parameter')

    if args.use_cuda:
        torch.cuda.synchronize()
    
    elapsed_time = time.perf_counter() - start_time   

    print('Elapsed time: {:.2f}s'.format(elapsed_time))

if __name__ == '__main__':
    main()
