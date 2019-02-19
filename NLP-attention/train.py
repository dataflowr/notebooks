from __future__ import print_function
from models import *

from util import Dictionary, get_args

import torch
import torch.nn as nn
import torch.optim as optim
#from torch.autograd import Variable

import json
import time
import random
import os


def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1), 1).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')


def package(data):
    """Package data for training / evaluation."""
    data = list(map(lambda x: json.loads(x), data))
    dat = list(map(lambda x: list(map(lambda y: dictionary.word2idx[y], x['text'])), data))
    maxlen = 0
    for item in dat:
        maxlen = max(maxlen, len(item))
    targets = list(map(lambda x: x['label'], data))
    maxlen = min(maxlen, 500)
    for i in range(len(data)):
        if maxlen < len(dat[i]):
            dat[i] = dat[i][:maxlen]
        else:
            for j in range(maxlen - len(dat[i])):
                dat[i].append(dictionary.word2idx['<pad>'])
    dat = torch.LongTensor(dat)
    targets = torch.LongTensor(targets)
    return dat.t(), targets

@torch.no_grad()
def evaluate():
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0
    for batch, i in enumerate(range(0, len(data_val), args.batch_size)):
        data, targets = package(data_val[i:min(len(data_val), i+args.batch_size)])
        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()
        hidden = model.init_hidden(data.size(1))
        output, attention, _ = model.forward(data, hidden)
        output_flat = output.view(data.size(1), -1)
        total_loss += criterion(output_flat, targets).data
        prediction = torch.max(output_flat, 1)[1]
        total_correct += torch.sum((prediction == targets).float())
    return total_loss.item() / (len(data_val) // args.batch_size), total_correct.data.item() / len(data_val)


def train(epoch_number):
    global best_val_loss, best_acc
    model.train()
    total_loss = 0
    total_pure_loss = 0  # without the penalization term
    start_time = time.time()
    for batch, i in enumerate(range(0, len(data_train), args.batch_size)):
        data, targets = package(data_train[i:i+args.batch_size])
        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()
        hidden = model.init_hidden(data.size(1))
        output, attention, out_bool = model.forward(data, hidden)
        loss = criterion(output.view(data.size(1), -1), targets)
        total_pure_loss += loss.data.item()
        if out_bool:  # add penalization term
            attentionT = torch.transpose(attention, 1, 2).contiguous()
            extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])
            loss += args.penalization_coeff * extra_loss
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data.item()

        if batch % args.log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f} | pure loss {:5.4f}'.format(
                  epoch_number, batch, len(data_train) // args.batch_size,
                  elapsed * 1000 / args.log_interval, total_loss / args.log_interval,
                  total_pure_loss / args.log_interval))
            total_loss = 0
            total_pure_loss = 0
            start_time = time.time()

#            for item in model.parameters():
#                print item.size(), torch.sum(item.data ** 2), torch.sum(item.grad ** 2).data[0]
#            print model.encoder.ws2.weight.grad.data
#            exit()
    evaluate_start_time = time.time()
    val_loss, acc = evaluate()
    print('-' * 89)
    fmt = '| evaluation | time: {:5.2f}s | valid loss (pure) {:5.4f} | Acc {:8.4f}'
    print(fmt.format((time.time() - evaluate_start_time), val_loss, acc))
    print('-' * 89)
    # Save the model, if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
        f.close()
        best_val_loss = val_loss
    else:  # if loss doesn't go down, divide the learning rate by 5.
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.2
    if not best_acc or acc > best_acc:
        with open(args.save[:-3]+'.best_acc.pt', 'wb') as f:
            torch.save(model, f)
        f.close()
        best_acc = acc
    with open(args.save[:-3]+'.epoch-{:02d}.pt'.format(epoch_number), 'wb') as f:
        torch.save(model, f)
    f.close()


if __name__ == '__main__':
    # parse the arguments
    args = get_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # Load Dictionary
    assert os.path.exists(args.train_data)
    assert os.path.exists(args.val_data)
    print('Begin to load the dictionary.')
    dictionary = Dictionary(path=args.dictionary)

    best_val_loss = None
    best_acc = None

    n_token = len(dictionary)
    
    model = Classifier({
        'dropout': args.dropout,
        'ntoken': n_token,
        'nlayers': args.nlayers,
        'nhid': args.nhid,
        'ninp': args.emsize,
        'pooling': 'all',
        'attention-unit': args.attention_unit,
        'attention-hops': args.attention_hops,
        'nfc': args.nfc,
        'dictionary': dictionary,
        'word-vector': args.word_vector,
        'class-number': args.class_number
    })
    if args.cuda:
        model = model.cuda()

    I = torch.zeros(args.batch_size, args.attention_hops, args.attention_hops)
    for i in range(args.batch_size):
        for j in range(args.attention_hops):
            I.data[i][j][j] = 1
    if args.cuda:
        I = I.cuda()

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
    else:
        raise Exception('For other optimizers, please add it yourself. '
                        'supported ones are: SGD and Adam.')
    print('Begin to load data.')
    data_train = open(args.train_data).readlines()
    data_val = open(args.val_data).readlines()
    try:
        for epoch in range(args.epochs):
            train(epoch)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exit from training early.')
        data_val = open(args.test_data).readlines()
        evaluate_start_time = time.time()
        test_loss, acc = evaluate()
        print('-' * 89)
        fmt = '| test | time: {:5.2f}s | test loss (pure) {:5.4f} | Acc {:8.4f}'
        print(fmt.format((time.time() - evaluate_start_time), test_loss, acc))
        print('-' * 89)
        exit(0)
