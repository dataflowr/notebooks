import os

import torch
from torchtext import data
from util_glove import get_args, makedirs

args = get_args()
torch.cuda.set_device(args.gpu)
device = torch.device('cuda:{}'.format(args.gpu))

inputs = data.Field(lower=args.lower, tokenize='spacy')
