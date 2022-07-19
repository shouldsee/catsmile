import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import time


BASH_CMD = '''
### execute to download dataset
wget https://github.com/zalandoresearch/fashion-mnist/tarball/master
tar -xvzf master
'''
#from markov_lm.Model_pretrain import lazy_load_pretrain_model
DIR = os.path.dirname(os.path.realpath(__file__))

class fashion_mnist(torch.utils.data.Dataset):
    def __init__(self,CUDA=False):
        super(fashion_mnist, self).__init__()

        'zalandoresearch-fashion-mnist-b2617bb/data/fashion/t10k-images-idx3-ubyte.gz'
        'zalandoresearch-fashion-mnist-b2617bb/data/fashion/t10k-labels-idx1-ubyte.gz'
        'zalandoresearch-fashion-mnist-b2617bb/data/fashion/train-images-idx3-ubyte.gz'
        'zalandoresearch-fashion-mnist-b2617bb/data/fashion/train-labels-idx1-ubyte.gz'
        print("LOADING GERMAN SENTENCES")
        self.device = torch.device('cpu' if not CUDA else 'cuda:0')
        self.trainData=dict()
        self.trainData['images'], self.trainData['labels'] = self.load_mnist(f'{DIR}/zalandoresearch-fashion-mnist-b2617bb/data/fashion','train')
        self.testData=dict()
        self.testData['images'], self.testData['labels'] = self.load_mnist(f'{DIR}/zalandoresearch-fashion-mnist-b2617bb/data/fashion','t10k')
        self.graph_dim = self.trainData['images'].shape[1]
        self.train()

    def test(self):
        self.mode = "test"
    def train(self):
        self.mode = "train"

    def load_mnist(self, path, kind='train'):
        import os
        import gzip
        import numpy as np

        """Load MNIST data from `path`"""
        labels_path = os.path.join(path,
                                   '%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)
        images = torch.tensor(images,device=self.device,dtype=torch.float)
        images = images - images.mean(-1,keepdims=True)                
        return (
        images,
        torch.tensor(labels,device=self.device).long())

    def logit_to_sentence(self,logits,language="german"):
        if(language=="german"):
            vocab = self.german_vocab_reversed
        else:
            vocab = self.english_vocab_reversed
        sentence = []
        for l in logits:
            idx = torch.argmax(l)
            word = vocab[idx]
            sentence.append(word)
        return "".join(sentence)

    def test(self):
        self.mode = "test"
    def train(self):
        self.mode = "train"
    def __getitem__(self, idx):
        # torch.set_default_tensor_type(torch.FloatTensor)
        if(self.mode=="test"):
            data =self.testData
        else:
            data = self.trainData
        #
        # Tooo slow
        # # gl = german_logits*logit_mask
        # if self.mode=='test':
        #     idx = idx + len(self.german_sentences_train)

        return {
        "index":idx,
        "images":data['images'][idx],
        "labels":data['labels'][idx],
                }
    def __len__(self):
        if(self.mode=="test"):
            return self.testData['images'].__len__()
        else:
            return self.trainData['images'].__len__()

    def total_length(self):
        assert 0
        return  len(self.german_sentences_test)+len(self.german_sentences_train)

                    # def __getitem__()
if __name__ == '__main__':
    x= fashion_mnist()
    x = BertMiddleLayer(BertTok, BertModel, CUDA=True)
    # x[range(5)]
    # print(x[range(5)]['input'].shape)
    # print(len(x))
    # x.test()
    # x[range(5)]
    # print(len(x))
