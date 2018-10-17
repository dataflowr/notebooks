import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim

import torch_utils; 
from torch_utils import gpu, minibatch_sentences, shuffle_sentences, accuracy_one
from nlp_models import FirstModel

class SentimentModel(object):

	def __init__(self, embedding_dim = 30, n_iter = 2, batch_size = 64,
		learning_rate = 1e-3, 
		net = None, loss = None, use_cuda=False,
		vocab_size = 1, seq_len = 1):
		self._embedding_dim = embedding_dim
		self._n_iter = n_iter
		self._batch_size = batch_size
		self._learning_rate = learning_rate
		self._use_cuda = use_cuda
		
		if net != None:
			self._net = gpu(net,use_cuda)
		else:
			self._net = None
		self._loss = loss
		self._optimizer = None
		self._vocab_size = vocab_size
		self._seq_len = seq_len


	def _initialize(self):
		if self._net is None:
			self._net = gpu(FirstModel(self._embedding_dim,self._vocab_size,self._seq_len),self._use_cuda)
			
		self._optimizer = optim.Adam(self._net.parameters(),lr=self._learning_rate, weight_decay=0)
				
		if self._loss is None:
			self._loss = torch.nn.BCELoss()

	@property
	def _initialized(self):

		return self._optimizer is not None


	def fit(self, word_ids, sentiment, word_ids_test, sentiment_test, verbose=True):

		word_ids = word_ids.astype(np.int64)
		word_ids_test = word_ids_test.astype(np.int64)
				
		if not self._initialized:
			self._initialize()

		self._net.train(True)

		for epoch_num in range(self._n_iter):

			words, sents = shuffle_sentences(word_ids,np.asarray(sentiment).astype(np.float32))
			word_ids_tensor = gpu(torch.from_numpy(words), self._use_cuda)
			sent_tensor = gpu(torch.from_numpy(sents), self._use_cuda)
			epoch_loss = 0.0
			epoch_acc = 0.0
			for (minibatch_num, (batch_word, batch_sent)) in enumerate(minibatch_sentences(self._batch_size, word_ids_tensor, sent_tensor)):
				word_var = Variable(batch_word)
				sent_var = Variable(batch_sent.unsqueeze(1),requires_grad=False)
				predictions = self._net(word_var)
				#print(sent_var.size())
				preds = accuracy_one(predictions.data).view(-1,1)
				
				self._optimizer.zero_grad()
				
				loss = self._loss(predictions, sent_var)
				
				epoch_loss += loss.data.item()
				epoch_acc += torch.sum(preds != sent_var.data.byte()).data.item()/float(sent_var.size(0))

				loss.backward()
				
				self._optimizer.step()
			#print(epoch_acc)
			#print(sent_var.size())
				
			epoch_loss = epoch_loss / (minibatch_num + 1)
			epoch_acc = epoch_acc/ float(minibatch_num + 1)

			if verbose:
				val_loss, val_acc = self.test(word_ids_test, sentiment_test)
				#val_loss = 0
				#val_acc = 0
				print('Epoch {}: train loss {}'.format(epoch_num, epoch_loss), 
					'train acc', epoch_acc,
					'validation loss', val_loss,
					'validation acc', val_acc)
				self._net.train(True)


	def test(self, word_ids, sentiment):
		self._net.train(False)
		word_ids = word_ids.astype(np.int64)
		
		word_ids_tensor = gpu(torch.from_numpy(word_ids), self._use_cuda)
		sent_tensor = gpu(torch.from_numpy(np.asarray(sentiment).astype(np.float32)), self._use_cuda)
		epoch_loss = 0.0
		epoch_acc = 0.0
		for (minibatch_num, (batch_word, batch_sent)) in enumerate(minibatch_sentences(self._batch_size, word_ids_tensor, sent_tensor)):
			word_var = Variable(batch_word)
			sent_var = Variable(batch_sent.unsqueeze(1),requires_grad=False)
			predictions = self._net(word_var)
			preds = accuracy_one(predictions.data).view(-1,1)
			loss = self._loss(predictions, sent_var)
			epoch_loss = epoch_loss + loss.data.item()
			epoch_acc += torch.sum(preds != sent_var.data.byte()).data.item()/float(sent_var.size(0))
		epoch_loss = epoch_loss / (minibatch_num + 1)
		epoch_acc = epoch_acc / float(minibatch_num + 1)

		return epoch_loss, epoch_acc 


												
