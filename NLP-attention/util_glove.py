# from https://github.com/pytorch/examples/tree/master/snli
import os

from argparse import ArgumentParser

def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise

def get_args():
    parser = ArgumentParser(description='Glove with PyTorch')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.6B.100d')
    args = parser.parse_args()
return args