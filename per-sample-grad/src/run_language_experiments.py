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

from gradcnn import make_optimizer, replicate_model, replicate_model_text

import models 
import models_dp  
import train 

# DEBUG flag skips loading word vecs for faster debugging 
DEBUG = True

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
        print("Runing LSTM Sentiment Classification with differential privacy using multi gradient computations")
        MNet = replicate_model_text(net_class=models_dp.LSTM_net, batch_size = args.batch_size)
        model = MNet(vocab_size=INPUT_DIM, embedding_dim=EMBEDDING_DIM, pad_idx=PAD_IDX)
        model.get_detail(True)
        Optimizer = make_optimizer(cls=optim.Adam, noise_multiplier=1.1, l2_norm_clip=1.0)
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
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enable CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dp-mode', type=str, default='no-dp',
                        help='specifiy dp mode: "no-dp", "naive" or "multi"' )

    args = parser.parse_args()

    torch.manual_seed(1)

    device = torch.device("cuda:0" if args.use_cuda else "cpu")

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.use_cuda else {}

    assert(args.dp_mode in ['no-dp', 'naive', 'multi'])

    if args.use_cuda:
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    run_sentiment_model(args, device, kwargs)

    if args.use_cuda:
        torch.cuda.synchronize()
    
    elapsed_time = time.perf_counter() - start_time   

    print('Elapsed time: {:.2f}s'.format(elapsed_time))

if __name__ == '__main__':
    main()
