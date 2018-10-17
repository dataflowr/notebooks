import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_utils import ScaledEmbedding

class FirstModel(nn.Module):
    
    def __init__(self,
                 embedding_dim=30,vocab_size = 1,seq_len = 1):
        
        super(FirstModel, self).__init__()
        
        self._seq_len = seq_len
        self._embedding_dim = embedding_dim

        #self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings = ScaledEmbedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(seq_len*embedding_dim,100)
        self.fc2 = nn.Linear(100,1)
        
    def forward(self, words_id):
        
        words_embedding = self.embeddings(words_id).view(-1,self._seq_len*self._embedding_dim)
        words_embedding = words_embedding.squeeze()
        x = F.relu(self.fc1(words_embedding))
        x = self.fc2(F.dropout(x,0.7))
        return F.sigmoid(x)


class ConvModel(nn.Module):

    def __init__(self,embedding_dim = 30, vocab_size = 1, seq_len = 1):

        super(ConvModel,self).__init__()

        self._seq_len = seq_len
        self._embedding_dim = embedding_dim
        self.embeddings = ScaledEmbedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim,64,5,padding=2)
        self.fc1 = nn.Linear(32*seq_len,100)
        self.mp = nn.MaxPool1d(2)
        self.fc2 = nn.Linear(100,1)

    def forward(self,words_id):
        words_embedding = self.embeddings(words_id).permute(0,2,1)
        x = F.dropout(words_embedding,0.2)
        x = F.relu(self.conv(x))
        x = self.mp(F.dropout(x,0.2))
        x = x.view(-1,self._seq_len*32)
        x = F.dropout(self.fc1(x),0.7)
        return F.sigmoid(self.fc2(x))




        
