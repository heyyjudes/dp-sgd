import time
import math
import torch
import torch.nn.functional as F
import pdb
from pytorch_memlab import MemReporter

bptt = 35


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

        loss = criterion(predictions.reshape(target.shape),
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
        pdb.set_trace()


def train_multi(args, model, device, train_loader, optimizer, epoch, criterion):
    '''
    Method for multiple model method for training with differential privacy (multi) 
    for image data 
    '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        model.reduce_batch()
        optimizer.step()
        model.reassign_params()

        freq = int(len(train_loader)/args.log_interval)
        if batch_idx % freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def train_single_fwd_sm(args, model, device, train_loader, optimizer, epoch, criterion):
    '''
    Method for single forward method for training with differential privacy (single-fwd-sm) 
    for image data 
    '''
    noise_multiplier = 1.1/args.batch_size
    l2_norm_clip = 1.0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target, reduction='none')

        grad_weights_dict = {}
        grad_bias_dict = {}

        for key in model._modules:
            grad_bias_dict[key] = torch.zeros(
                (model._modules[key].bias.shape[0])).to(device)
            flattened = model._modules[key].weight.flatten().shape
            grad_weights_dict[key] = torch.zeros((flattened[0])).to(device)

        for i in range(args.batch_size):

            if i > 0:
                optimizer.zero_grad()

            if i == args.batch_size - 1:
                loss[i].backward()
            else:
                loss[i].backward(retain_graph=True)

            squared_norms = torch.stack(
                [(p.grad.view(1, -1) ** 2).sum(dim=1) for p in model.parameters()])
            grad_norm = torch.sqrt(squared_norms.sum(dim=0))

            factor = 1.0 if l2_norm_clip >= grad_norm else torch.div(
                l2_norm_clip, grad_norm)
            std_noise = noise_multiplier * l2_norm_clip

            for key in grad_weights_dict:
                grad_weights_dict[key] += model._modules[key].weight.grad.data.flatten(
                )*factor + std_noise * torch.randn_like(grad_weights_dict[key])
                grad_bias_dict[key] += model._modules[key].bias.grad.data * \
                    factor + std_noise * torch.randn_like(grad_bias_dict[key])

            optimizer.zero_grad()

        for key in grad_weights_dict:
            grad_weights_dict[key] /= args.batch_size
            grad_bias_dict[key] /= args.batch_size
            model._modules[key].weight.grad.data = grad_weights_dict[key].data.reshape(
                model._modules[key].weight.grad.data.shape)
            model._modules[key].bias.grad.data = grad_bias_dict[key].data

        optimizer.step()
        freq = int(len(train_loader)/args.log_interval)
        if batch_idx % freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.sum().item()))


def train_single_fwd_lg(args, model, device, train_loader, optimizer, epoch, criterion):
    '''
    Method for single forward method for training with differential privacy (single-fwd-lg) 
    for image data 
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


def train_transformer_naive(args, model, device, TEXT, train_data, optimizer, criterion, epoch):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, maintain_shape=True)

        pdb.set_trace()
        for j in range(args.batch_size):
            optimizer.zero_grad()
            output = model(data[:, j].unsqueeze(1))
            pdb.set_trace()
            loss = criterion(output.view(-1, ntokens), targets[:, j])
            loss.backward()
            for p in model.parameters():
                if j == 0:
                    p.bgrad = torch.empty(
                        (args.batch_size, p.grad.shape)).to(device)
                p.bgrad[j] = p.grad.clone() / args.batch_size

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


def evaluate_transformer(eval_model, TEXT, data_source, criterion):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def train_transformer_multi(args, model, device, TEXT, train_data, optimizer, criterion, epoch):
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
        model.reduce_batch()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        model.reassign_params()

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
