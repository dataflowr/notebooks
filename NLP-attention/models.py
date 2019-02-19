from __future__ import print_function
import torch
import torch.nn as nn
import os


class BiLSTM(nn.Module):

    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        self.encoder = nn.Embedding(config['ntoken'], config['ninp'])
        self.bilstm = nn.LSTM(config['ninp'], config['nhid'], config['nlayers'], dropout=config['dropout'],
                              bidirectional=True)
        self.nlayers = config['nlayers']
        self.nhid = config['nhid']
        self.pooling = config['pooling']
        self.dictionary = config['dictionary']
#        self.init_weights()
        self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0
        if os.path.exists(config['word-vector']):
            print('Loading word vectors from', config['word-vector'])
            vectors = torch.load(config['word-vector'])
            #print(vectors[3], config['ninp'])
            assert vectors[3] >= config['ninp']#2
            vocab = vectors[1]#0
            vectors = vectors[2]#1
            loaded_cnt = 0
            for word in self.dictionary.word2idx:
                if word not in vocab:
                    continue
                real_id = self.dictionary.word2idx[word]
                loaded_id = vocab[word]
                self.encoder.weight.data[real_id] = vectors[loaded_id][:config['ninp']]
                loaded_cnt += 1
            print('%d words from external word vectors loaded.' % loaded_cnt)

    # note: init_range constraints the value of initial weights
    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        emb = self.drop(self.encoder(inp))
        outp = self.bilstm(emb, hidden)[0]
        if self.pooling == 'mean':
            outp = torch.mean(outp, 0).squeeze()
        elif self.pooling == 'max':
            outp = torch.max(outp, 0)[0].squeeze()
        elif self.pooling == 'all' or self.pooling == 'all-word':
            outp = torch.transpose(outp, 0, 1).contiguous()
        return outp, emb

    def init_hidden( self, bsz ):
        # trick see https://discuss.pytorch.org/t/when-to-initialize-lstm-hidden-state/2323
        weight = next(self.parameters())
        # attention: hidden contains both hidden state and cell state
        return (weight.new_zeros(self.nlayers*2, bsz, self.nhid),
                weight.new_zeros(self.nlayers*2, bsz, self.nhid))
    

class SelfAttentiveEncoder(nn.Module):

    def __init__(self, config):
        super(SelfAttentiveEncoder, self).__init__()
        self.bilstm = BiLSTM(config)
        self.drop = nn.Dropout(config['dropout'])
        self.ws1 = nn.Linear(config['nhid'] * 2, config['attention-unit'], bias=False)
        self.ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dictionary = config['dictionary']
#        self.init_weights()
        self.attention_hops = config['attention-hops']

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        outp = self.bilstm.forward(inp, hidden)[0]
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]
        transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
        transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        penalized_alphas = alphas + (
            -10000 * (concatenated_inp == self.dictionary.word2idx['<pad>']).float())
            # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

    def init_hidden(self, bsz):
        return self.bilstm.init_hidden(bsz)


class Classifier(nn.Module):

    def __init__(self, config):
        super(Classifier, self).__init__()
        if config['pooling'] == 'mean' or config['pooling'] == 'max':
            self.encoder = BiLSTM(config)
            self.fc = nn.Linear(config['nhid'] * 2, config['nfc'])
        elif config['pooling'] == 'all':
            self.encoder = SelfAttentiveEncoder(config)
            self.fc = nn.Linear(config['nhid'] * 2 * config['attention-hops'], config['nfc'])
        else:
            raise Exception('Error when initializing Classifier')
        self.drop = nn.Dropout(config['dropout'])
        self.tanh = nn.Tanh()
        self.pred = nn.Linear(config['nfc'], config['class-number'])
        self.dictionary = config['dictionary']
#        self.init_weights()

    def init_weights(self, init_range=0.1):
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0)
        self.pred.weight.data.uniform_(-init_range, init_range)
        self.pred.bias.data.fill_(0)

    def forward(self, inp, hidden):
        outp, attention = self.encoder.forward(inp, hidden)
        outp = outp.view(outp.size(0), -1)
        fc = self.tanh(self.fc(self.drop(outp)))
        pred = self.pred(self.drop(fc))
        out_bool = True
        if type(self.encoder) == BiLSTM:
            attention = None
            out_bool = False
        return pred, attention, out_bool 

    def init_hidden(self, bsz):
        return self.encoder.init_hidden(bsz)

    def encode(self, inp, hidden):
        return self.encoder.forward(inp, hidden)[0]
