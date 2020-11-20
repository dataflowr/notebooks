import torch
from torch import nn


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
N_WORDS = 31
BATCH_SIZE = 10
encoder1 = EncoderRNN(N_WORDS, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, N_WORDS).to(device)


# Feeding random tensors
batch = torch.rand((N_WORDS, BATCH_SIZE)).long()
hidden = torch.rand((1, BATCH_SIZE, hidden_size))
a, b = encoder1(batch, hidden)
print(a.shape)
print(b.shape)
