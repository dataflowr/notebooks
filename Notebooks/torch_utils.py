import numpy as np

import torch
import torch.nn as nn


def gpu(tensor, gpu=False):

    if gpu:
        return tensor.cuda()
    else:
        return tensor


def accuracy_one(x):
    
    return x[:,0] < 0.5


def shuffle_sentences(word,sent):

    random_state = np.random.RandomState()
    shuffle_indices = np.arange(len(sent))
    random_state.shuffle(shuffle_indices)
    
    return tuple([word[shuffle_indices,:], sent[shuffle_indices]])


def shuffle(*arrays):

    random_state = np.random.RandomState()
    shuffle_indices = np.arange(len(arrays[0]))
    random_state.shuffle(shuffle_indices)

    if len(arrays) == 1:
        return arrays[0][shuffle_indices]
    else:
        return tuple(x[shuffle_indices] for x in arrays)

def minibatch_sentences(batch_size, word, sent):
    for i in range(0, len(sent), batch_size):
        yield tuple([word[i:i+batch_size,:], sent[i:i+batch_size]])

def minibatch(batch_size, *tensors):

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def regression_loss(observed_ratings, predicted_ratings):
    return ((observed_ratings - predicted_ratings) ** 2).mean()


def l1_loss(observed_ratings, predicted_ratings):
    return (torch.abs(observed_ratings-predicted_ratings)).mean()


def hinge_loss(positive_predictions, negative_predictions):
    loss = torch.clamp(negative_predictions -
                       positive_predictions +
                       1.0, 0.0)
    return loss.mean()


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the emedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


def sample_items(num_items, shape, random_state=None):
    """
    Randomly sample a number of items.

    Parameters
    ----------

    num_items: int
        Total number of items from which we should sample:
        the maximum value of a sampled item id will be smaller
        than this.
    shape: int or tuple of ints
        Shape of the sampled array.
    random_state: np.random.RandomState instance, optional
        Random state to use for sampling.

    Returns
    -------

    items: np.array of shape [shape]
        Sampled item ids.
    """

    if random_state is None:
        random_state = np.random.RandomState()

    items = random_state.randint(0, num_items, shape)

    return items
