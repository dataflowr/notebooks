import torchvision
from torchvision import models,transforms,datasets
import torch
import os
import bcolz
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()
def load_array(fname):
    return bcolz.open(fname)[:]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

prep1 = transforms.Compose([
                transforms.CenterCrop(224),
                #transforms.RandomSizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

def imshow(inp, title=None):
#   Imshow for Tensor.
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# shuffle only to get nice pictures
def shuffle_valtrain(x):
    if x == 'train':
        return True
    else:
        return False

def var_cgpu(x,use_gpu):
    if use_gpu:
        return x.cuda()
    else:
        return x


