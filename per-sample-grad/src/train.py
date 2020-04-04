import time
import math
import torch
import torch.nn.functional as F
from pytorch_memlab import MemReporter

###############################################################################
#                              Helper Methods                                 #
#                                                                             #
###############################################################################
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

def get_batch(source, i, bptt, maintain_shape=False):
    # separate data and target into batches
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    if maintain_shape:
        target = source[i+1:i+1+seq_len]
    else:
        target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def print_lm_stats(args, cur_loss, epoch, batch, train_data, log_interval, elapsed, scheduler):
    # print language model training progress
    print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // args.bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))

###############################################################################
#                     Methods for Classification Tasks                        #
#                                                                             #
###############################################################################
def train(args, model, device, train_loader, optimizer,  epoch, criterion, text=False):
    '''
    Method for training without differential privacy (no-dp) for text (batch size dim 1) 
    or image classification
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
        loss = criterion(predictions.reshape(target.shape), target) if text else criterion(predictions, target)
        loss.backward()
        optimizer.step()


        freq = int(len(train_loader)/args.log_interval)
        if args.verbose and batch_idx % freq == 0:
            print_runtime(epoch, batch_idx, loss.item(), actual_batch_size, train_loader)
    return


def train_naive(args, model, device, train_loader, optimizer, epoch, criterion, text=False):
    '''
    Method for naive method for training with differential privacy (naive) for text 
    (batch size dim 1) or image data with O(BxP) memory
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
            if args.verbose and batch_idx % freq == 0:
               print_runtime(epoch, batch_idx, loss.item(), actual_batch_size, train_loader)

def train_naive_sm(args, model, device, train_loader, optimizer, epoch, criterion, text=False):
    '''
    Method for naive method for training with differential privacy (naive) for 
    text (batch size dim 1) or image classification with O(P) memory
    '''
    model.train()
    noise_multiplier = math.pow(args.sigma, 2)/args.batch_size
    l2_norm_clip = 1.0
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
                clip_grads(args, grads, model, noise_multiplier, l2_norm_clip)

            save_grads(grads, model, noise_multiplier, l2_norm_clip)
            
            optimizer.step()

            freq = int(len(train_loader)/args.log_interval)
            if args.verbose and batch_idx % freq == 0:
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
        if args.verbose and batch_idx % freq == 0:
            print_runtime(epoch, batch_idx, loss.item(), args.batch_size, train_loader)


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
            if args.verbose and batch_idx % freq == 0:
                print_runtime(epoch, batch_idx, loss.item(), actual_batch_size, train_loader)

def train_single_fwd_sm(args, model, device, train_loader, optimizer, epoch, criterion, text=False):
    '''
    Method for single forward method for training with differential privacy for image data
    with O(P) memory
    '''
    model.train()
    noise_multiplier = math.pow(args.sigma, 2)/args.batch_size
    l2_norm_clip = 1.0
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

            loss = criterion(predictions.reshape(target.shape), target, reduction='none') \
                if text else criterion(predictions, target, reduction='none')

            for i in range(args.batch_size):
                
                optimizer.zero_grad()
                loss[i].backward(retain_graph=True)

                # # compute gradient norm and clip gradient 
                clip_grads(args, grads, model, noise_multiplier, l2_norm_clip)
            
            save_grads(grads, model, noise_multiplier, l2_norm_clip)
            
            optimizer.step()

            freq = int(len(train_loader)/args.log_interval)
            if args.verbose and batch_idx % freq == 0:
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
        if args.verbose and batch_idx % freq == 0:
            print_runtime(epoch, batch_idx, loss.sum().item(), args.batch_size, train_loader)

###############################################################################
#                     Methods for Language Modeling                           #
#                                                                             #
###############################################################################

def train_transformer(args, model, device, TEXT, train_data, optimizer, criterion, scheduler, epoch):
    '''
    Method for training without differential privacy (no-dp) for language modeling
    '''
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args.bptt)
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
            print_lm_stats(args, cur_loss, epoch, batch, train_data, log_interval, elapsed, scheduler)
            total_loss = 0
            start_time = time.time()

def train_transformer_naive(args, model, device, TEXT, train_data, optimizer, criterion, scheduler, epoch):
    '''
    Method for naive method for training with differential privacy (naive) for 
    language modeling with O(P) memory
    '''
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    noise_multiplier = math.pow(args.sigma, 2)/args.batch_size
    l2_norm_clip = 1.0
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args.bptt)
        actual_batch_size = data.shape[1]
        if actual_batch_size < args.batch_size or data.shape[0] != args.bptt:
            print("last batch sized {} too small".format(actual_batch_size))
        else:
            #initialize gradient vector size O(P)
            grads = [torch.zeros((*p.shape)).to(device) for p in model.parameters()]

            for i in range(args.batch_size): 
                optimizer.zero_grad()
                output = model(data[:, i].unsqueeze(1))
                loss = criterion(output.view(-1, ntokens), targets[i*args.bptt:i*args.bptt + args.bptt])
                loss.backward()
                # compute gradient norm and clip gradient 
                clip_grads(args, grads, model, noise_multiplier, l2_norm_clip)
                                
            save_grads(grads, model, noise_multiplier, l2_norm_clip)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

            log_interval = 200
            if args.verbose and batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print_lm_stats(args, cur_loss, epoch, batch, train_data, log_interval, elapsed, scheduler)
                total_loss = 0
                start_time = time.time()

def train_transformer_single_fwd(args, model, device, TEXT, train_data, optimizer, criterion, scheduler, epoch):
    '''
    Method for single forward method for training with differential privacy for 
    language modeling with O(P) memory
    '''
    model.train()  # Turn on the train mode
    total_loss = 0.
    noise_multiplier = math.pow(args.sigma, 2)/args.batch_size
    start_time = time.time()
    l2_norm_clip = 1.0
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args.bptt, maintain_shape=True)
        actual_batch_size = data.shape[1]
        if actual_batch_size < args.batch_size or data.shape[0] != args.bptt:
            print("last batch sized {} too small".format(actual_batch_size))
        else:
            #initialize gradient vector size O(P)
            grads = [torch.zeros((*p.shape)).to(device) for p in model.parameters()]
            optimizer.zero_grad()
            output = model(data) 
            
            for i in range(args.batch_size): 
                loss = criterion(output[:, i, :], targets[:, i])
                loss.backward(retain_graph=True)
                # compute gradient norm and clip gradient 
                clip_grads(args, grads, model, noise_multiplier, l2_norm_clip)
                optimizer.zero_grad()
                                
            save_grads(grads, model, noise_multiplier, l2_norm_clip)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

            log_interval = 200
            if args.verbose and batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print_lm_stats(args, cur_loss, epoch, batch, train_data, log_interval, elapsed, scheduler)
                total_loss = 0
                start_time = time.time()