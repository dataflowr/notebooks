import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import math


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, BATCH_SIZE, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, BATCH_SIZE, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[-1])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, BATCH_SIZE, self.hidden_size, device=device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 128
MAX_LENGTH = 70
N_WORDS = 31
BATCH_SIZE = 10
encoder1 = EncoderRNN(N_WORDS, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, N_WORDS).to(device)


# Feeding random tensors
batch = torch.rand((MAX_LENGTH, BATCH_SIZE)).long()
hidden = torch.rand((1, BATCH_SIZE, hidden_size))
a, b = encoder1(batch, hidden)
print(a.shape)  # MAX_LENGTH x BATCH_SIZE x hidden_size = S x B x H
print(b.shape)  #          1 x BATCH_SIZE x hidden_size = 1 x B x H

# Load your dataset
X = torch.from_numpy(np.load('X_morse.npy')).long()
y = torch.from_numpy(np.load('y_morse.npy')).long()
indices = np.arange(len(X))
np.random.shuffle(indices)
TRUNCATE = 10
X_morse = X[indices[:TRUNCATE]]
y_morse = y[indices[:TRUNCATE]]
n_samples, _ = X_morse.shape
n_batches = n_samples // BATCH_SIZE

# Training
dataset = TensorDataset(X_morse, y_morse)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
SOS_token = 29
EOS_token = 30


'''
for i_batch, (input_tensor, target_tensor) in enumerate(dataloader):
    print(batch.shape)
    break
'''


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    #plt.show()


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_input = torch.tensor([[SOS_token for _ in range(BATCH_SIZE)]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True  #if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            pass
            # Your code here
    else:
        pass
        # Your code here

    # loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return 0
    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    n_epochs = n_iters // n_batches
    for i_epoch in tqdm(range(n_epochs)):
        for i_batch, (input_tensor, target_tensor) in enumerate(dataloader):
            iter_ = 1 + i_epoch * n_batches + i_batch

            input_tensor = input_tensor.T
            target_tensor = target_tensor.T
            
            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter_ % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter_ / n_iters),
                                             iter_, iter_ / n_iters * 100, print_loss_avg))

            if iter_ % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)


trainIters(encoder1, decoder1, 100 * n_batches, print_every=200, plot_every=200, learning_rate=0.01)
