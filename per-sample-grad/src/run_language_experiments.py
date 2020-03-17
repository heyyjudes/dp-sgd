import time
import math
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn 
from random import shuffle

# datasets
import torchtext
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from torchtext.data.utils import get_tokenizer

from gradcnn import make_optimizer, replicate_model

import models 
import models_dp  
import train 

# DEBUG flag skips loading word vecs for faster debugging 
DEBUG = False

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions[0, :, 0], batch.label)
            acc = binary_accuracy(predictions[0, :, 0], batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def run_sentiment_model(args, device, kwargs): 
    if DEBUG: 
        TEXT = data.Field(include_lengths = True)
    else: 
        TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
    LABEL = data.LabelField(dtype = torch.float)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    MAX_VOCAB_SIZE = 25_000
    
    if DEBUG: 
        print("done tokenizing")
        TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        LABEL.build_vocab(train_data)
    else: 
        TEXT.build_vocab(train_data, 
                    max_size = MAX_VOCAB_SIZE, 
                    vectors = "glove.6B.100d", 
                    unk_init = torch.Tensor.normal_)
        LABEL.build_vocab(train_data)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data), batch_size = args.batch_size, sort_within_batch = True, 
    device = device)

    EMBEDDING_DIM = 100
    INPUT_DIM = len(TEXT.vocab)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    if args.dp_mode == 'no-dp': 
        print("Runing LSTM Sentiment Classification with no differential privacy")
        model = models.LSTM_net(INPUT_DIM, EMBEDDING_DIM, PAD_IDX)
        Optimizer = optim.Adam
        train_F = train.train

    elif args.dp_mode == 'naive': 
        print("Runing LSTM Sentiment Classification with differential privacy using naive gradient computations")
        model = models_dp.LSTM_net(INPUT_DIM, EMBEDDING_DIM, PAD_IDX)
        Optimizer = make_optimizer(cls=optim.Adam, noise_multiplier=1.1, l2_norm_clip=1.0)
        train_F = train.train_naive

    elif args.dp_mode == 'multi': 
        # TODO: add multi version 
        print("Runing LSTM Sentiment Classification with differential privacy using multi gradient computations")
        MNet = replicate_model(net_class=models_dp.LSTM_net, batch_size = args.batch_size)
        model = MNet(INPUT_DIM, EMBEDDING_DIM, PAD_IDX)
        model.get_detail(True)
        Optimizer = make_optimizer(cls=optim.SGD, noise_multiplier=1.1, l2_norm_clip=1.0)
        train_F = train.train_multi
    else: 
        print("error, not implemented yet ")

    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    if not DEBUG: 
        pretrained_embeddings = TEXT.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer_ = Optimizer(model.parameters())
    
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float('inf')

    for epoch in range(1, args.epochs + 1):

        start_time = time.time()
        
        train_F(args, model, device, train_iterator, optimizer_, epoch, criterion, text=True)
        test_loss, test_acc = evaluate(model, test_iterator, criterion)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

def batchify(text, data, bsz, device):
    data = text.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def run_language_model(args, device, kwargs): 
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)

    # language modeling params 
    batch_size = 20
    args.batch_size = batch_size
    eval_batch_size = 10
    args.test_batch_size = eval_batch_size
    ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value

    # divide into batches
    train_data = batchify(TEXT, train_txt, batch_size, device)
    val_data = batchify(TEXT, val_txt, eval_batch_size, device)
    test_data = batchify(TEXT, test_txt, eval_batch_size, device)

    criterion = nn.CrossEntropyLoss()
    lr = 5.0 # learning rate
    args.lr = lr 
    epochs = 3 # The number of epochs
    best_model = None
    best_val_loss = float("inf")


    if args.dp_mode == 'no-dp': 
        model = models.TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
        train_F = train.train_transformer
        Optimizer = optim.SGD

    elif args.dp_mode == 'naive': 
        model = models_dp.TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
        train_F = train.train_transformer_naive
        Optimizer = optim.SGD

    elif args.dp_mode == 'multi': 
        print("Runing MNIST with differential privacy using multiple models")
        MNet = replicate_model(net_class=models_dp.TransformerModel, batch_size=args.batch_size)
        model = MNet(ntoken=ntokens, ninp=emsize, nhead=nhead, nhid=nhid, nlayers=nlayers, dropout=dropout).to(device)
        model.get_detail(True)
        Optimizer = make_optimizer(cls=optim.SGD, noise_multiplier=1.1, l2_norm_clip=0.5)
        train_F = train.train_transformer_multi

    elif args.dp_mode == 'single-fwd': 
        pass 
    else: 
        print("only dp-mode settings allowed are: no-dp, naive, multi, single-fwd")

    optimizer = Optimizer(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_F(args, model, device, TEXT, train_data, optimizer, criterion, epoch)
        val_loss = train.evaluate_transformer(model, TEXT, val_data, criterion)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        # scheduler.step()

    test_loss = train.evaluate_transformer(best_model, TEXT, test_data, criterion)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

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
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dp-mode', type=str, default='no-dp',
                        help='specifiy dp mode: "no-dp", "naive", "multi" or "single-fwd"' )
    parser.add_argument('--task', type=str, default='sentiment',
                        help='specifiy data task: "sentiment" or "language-model"' )

    args = parser.parse_args()

    if not args.no_cuda and torch.cuda.is_available(): 
        use_cuda = True 
        num_devices = torch.cuda.device_count()
        print("number of devices found", num_devices)

    torch.manual_seed(1)

    device = torch.device("cuda:0" if use_cuda else "cpu")

    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    assert(args.dp_mode in ['no-dp', 'naive', 'multi', 'single-fwd'])

    if use_cuda:
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    if args.task == 'language-model': 
        assert(args.dp_mode != 'oprod')
        run_language_model(args, device, kwargs)
    elif args.task == 'sentiment': 
        assert(args.dp_mode != 'oprod')
        run_sentiment_model(args, device, kwargs)
    else: 
        print('please specifiy either sentiment or language model as --task parameter')
        return

    # compute overall training time
    if use_cuda:
        torch.cuda.synchronize()
    
    elapsed_time = time.perf_counter() - start_time   

    print('Elapsed time: {:.2f}s'.format(elapsed_time))

if __name__ == '__main__':
    main()
