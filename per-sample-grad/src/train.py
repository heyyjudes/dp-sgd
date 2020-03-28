import time
import math
import torch
import torch.nn.functional as F
from pytorch_memlab import MemReporter

bptt = 35
import pdb

# helper methods 
def print_runtime(epoch, batch_idx, loss, batch_size, train_loader): 
    # print runtime
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss))

def clip_grads(args, grads, model, noise_multiplier, l2_norm_clip): 
    # clip gradients 
        squared_norms = torch.stack([(p.grad.view(1, -1) ** 2).sum(dim=1) for p in model.parameters()])
        grad_norm = torch.sqrt(squared_norms.sum(dim=0))

        factor = l2_norm_clip/grad_norm
        factor = torch.clamp(factor, max=1.0) 

        # # add to gradient vector 
        for g, p in zip(grads, model.parameters()):
            g += (factor/args.batch_size)*p.grad.clone()

def save_grads(grads, model, noise_multiplier, l2_norm_clip): 
    # save gradient vector in actual 
    for g, p in zip(grads, model.parameters()):
        p.grad = torch.add(g, alpha=noise_multiplier*l2_norm_clip, other=torch.randn_like(g))

# methods for training models 
def train(args, model, device, train_loader, optimizer,  epoch, criterion, text=False):
    '''
    Method for training without differential privacy (no-dp) for text (batch size dim 1) or image data 
    '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if text:
            x, text_lengths = data
            actual_batch_size = x.shape[1]
        else:
            x = data
            actual_batch_size = x.shape[0]

        x, target = x.to(device), target.to(device)
        optimizer.zero_grad()
        predictions = model(x, text_lengths).squeeze(1) if text else model(x)
                         target) if text else criterion(predictions, target)
        loss.backward()
        optimizer.step()

        freq = int(len(train_loader)/args.log_interval)
        if batch_idx % freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx *
                actual_batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return


def train_naive(args, model, device, train_loader, optimizer, epoch, criterion, text=False):
    '''
    Method for naive method for training with differential privacy (naive) for text (batch size dim 1) or image data 
    with O(BxP) memory
    '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if text:
            x, text_lengths = data
            actual_batch_size = x.shape[1]
        else:
            x = data
            actual_batch_size = x.shape[0]
        x, target = x.to(device), target.to(device)

        if actual_batch_size < args.batch_size:
            print("last batch sized {} too small".format(actual_batch_size))
        else:
            for i in range(args.batch_size):
                optimizer.zero_grad()
                if text:
                    output = model(x[:, i].unsqueeze(1), [text_lengths[i]])
                    loss = criterion(output, target[i].reshape(1, 1, 1))
                else:
                    output = model(x[i].unsqueeze(0))
                    loss = criterion(output, target[i].unsqueeze(0))

                loss.backward()
                for p in model.parameters():
                    if i == 0:
                        p.bgrad = torch.empty(
                            (args.batch_size, *p.grad.shape)).to(device)
                    p.bgrad[i] = p.grad.clone() / args.batch_size

            optimizer.step()
            freq = int(len(train_loader)/args.log_interval)
            if batch_idx % freq == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx *
                    actual_batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def train_naive_sm(args, model, device, train_loader, optimizer, epoch, criterion, text=False):
    '''
    Method for naive method for training with differential privacy (naive) for text (batch size dim 1) or image data 
    with O(P) memory
    '''
    start_time = time.perf_counter()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if text:
            x, text_lengths = data
            actual_batch_size = x.shape[1]
        else:
            x = data
            actual_batch_size = x.shape[0]
        x, target = x.to(device), target.to(device)

        if actual_batch_size < args.batch_size:
            print("last batch sized {} too small".format(actual_batch_size))
        else:
            #initialize gradient vector size O(P)
            grads = [torch.zeros((*p.shape)).to(device) for p in model.parameters()]
            
            for i in range(args.batch_size):
                optimizer.zero_grad()
                if text:
                    output = model(x[:, i].unsqueeze(1), [text_lengths[i]])
                    loss = criterion(output, target[i].reshape(1, 1, 1))
                else:
                    output = model(x[i].unsqueeze(0))
                    loss = criterion(output, target[i].unsqueeze(0))

                loss.backward()
                
                # compute gradient norm and clip gradient 
                noise_multiplier = 1.1/args.batch_size
                l2_norm_clip = 1.0
                clip_grads(args, grads, model, noise_multiplier, l2_norm_clip)

            save_grads(grads, model, noise_multiplier, l2_norm_clip)
            
            optimizer.step()

            freq = int(len(train_loader)/args.log_interval)
            if batch_idx % freq == 0:
                print_runtime(epoch, batch_idx, loss.item(), actual_batch_size, train_loader)


def train_outer_product(args, model, device, train_loader, optimizer, epoch, criterion):
    '''
    Method for outer product method for training with differential privacy (outer-prod) 
    for image data 
    '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        freq = int(len(train_loader)/args.log_interval)
        if batch_idx % freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def train_multi(args, model, device, train_loader, optimizer, epoch, criterion, text=False):
    '''
    Method for multiple model method for training with differential privacy (multi) 
    for image data 
    '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if text:
            x, text_lengths = data
            actual_batch_size = x.shape[1]
        else:
            x = data
            actual_batch_size = x.shape[0]

        if actual_batch_size < args.batch_size:
            print("last batch sized {} too small".format(actual_batch_size))
        else: 
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            predictions = model(x, text_lengths) if text else model(x)
    
            loss = criterion(predictions.reshape(target.shape),
                            target) if text else criterion(predictions, target)
            loss.backward()
            model.reduce_batch()
            optimizer.step()
            model.reassign_params()

            freq = int(len(train_loader)/args.log_interval)
            if batch_idx % freq == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def train_single_fwd_sm(args, model, device, train_loader, optimizer, epoch, criterion, text=False):
    '''
    Method for single forward method for training with differential privacy for image data
    with O(P) memory
    '''
    model.train()
    start_time = time.perf_counter()
    for batch_idx, (data, target) in enumerate(train_loader):
        if text:
            x, text_lengths = data
            actual_batch_size = x.shape[1]
        else:
            x = data
            actual_batch_size = x.shape[0]
        x, target = x.to(device), target.to(device)

        if actual_batch_size < args.batch_size:
            print("last batch sized {} too small".format(actual_batch_size))
        else:
            #initialize gradient vector size O(P)
            grads = [torch.zeros((*p.shape)).to(device) for p in model.parameters()]
            
            predictions = model(x, text_lengths).squeeze(1) if text else model(x)

            loss = criterion(predictions.reshape(target.shape),
                           target, reduction='none') if text else criterion(predictions, target, reduction='none')

            for i in range(args.batch_size):
                
                optimizer.zero_grad()
                loss[i].backward(retain_graph=True)

                # # compute gradient norm and clip gradient 
                noise_multiplier = 1.1/args.batch_size
                l2_norm_clip = 1.0
                clip_grads(args, grads, model, noise_multiplier, l2_norm_clip)
            
            save_grads(grads, model, noise_multiplier, l2_norm_clip)
            
            optimizer.step()

            freq = int(len(train_loader)/args.log_interval)
            if batch_idx % freq == 0:
                print_runtime(epoch, batch_idx, loss.mean().item(), actual_batch_size, train_loader)

def train_single_fwd_lg(args, model, device, train_loader, optimizer, epoch, criterion):
    '''
    Method for single forward method for training with differential privacy with O(BxP) memory
    '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target, reduction='none')

        for i in range(args.batch_size):
            if i == args.batch_size - 1:
                loss[i].backward()
            else:
                optimizer.zero_grad()
                loss[i].backward(retain_graph=True)

            for p in model.parameters():
                if i == 0:
                    p.bgrad = torch.empty(
                        (args.batch_size, *p.grad.shape)).to(device)
                p.bgrad[i] = p.grad.clone() / args.batch_size

        optimizer.step()

        freq = int(len(train_loader)/args.log_interval)
        if batch_idx % freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.sum().item()))


def get_batch(source, i, maintain_shape=False):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    if maintain_shape:
        target = source[i+1:i+1+seq_len]
    else:
        target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def train_transformer(args, model, device, TEXT, train_data, optimizer, criterion, epoch):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(train_data) // bptt, args.lr,
                      elapsed * 1000 / log_interval,
                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
