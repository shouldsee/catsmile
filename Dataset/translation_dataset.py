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


# source_lang_file = "German_sentences.pkl"
# dest_lang_file  = "English_sentences.pkl"
source_lang_file = "English_sentences.pkl"
dest_lang_file  = "German_sentences.pkl"
DIR = os.path.dirname(os.path.realpath(__file__))
class EnglishToGermanDataset(torch.utils.data.Dataset):
    def __init__(self,CUDA=False):
        super(EnglishToGermanDataset, self).__init__()
        print("LOADING GERMAN SENTENCES")
        load = torch.load(os.path.join(DIR,dest_lang_file))
        self.german_sentences_train = load["train_data"]
        self.german_sentences_test = load["test_data"]
        self.german_max_len = load["max_len"]
        self.german_min_len = load["min_len"]
        self.german_vocab_len = load["vocab_len"]
        self.german_vocab = load["vocab"]
        self.german_vocab_reversed = load["vocab_reversed"]
        self.german_eos = self.german_vocab["<end>"]
        print("LOADING ENGLISH SENTENCES")
        load = torch.load(os.path.join(DIR,source_lang_file))
        self.english_sentences_train = load["train_data"]
        self.english_sentences_test = load["test_data"]
        self.english_max_len = load["max_len"]
        self.english_min_len = load["min_len"]
        self.english_vocab_len = load["vocab_len"]
        self.english_vocab = load["vocab"]
        self.english_vocab_reversed = load["vocab_reversed"]
        self.mode = "train"
        self.english_eos = self.english_vocab["<end>"]
        # self.min_len = 30#min(self.german_min_len,self.english_min_len)
        self.min_len = 15#min(self.german_min_len,self.english_min_len)
        self.CUDA = CUDA
        self.device = torch.device('cuda:0' if CUDA else 'cpu')

        ## truncate or pad to min_len
        for k in ["english_sentences_train","english_sentences_test","german_sentences_test","german_sentences_train"]:
            xx = getattr(self,k)
            for i,x in enumerate(xx):
                x = x[:self.min_len]
                x = torch.cat([x,torch.tensor([1]*(self.min_len - len(x))).long()] ,dim=0)
                xx[i] = x
            y= torch.stack(xx,dim=0).to(self.device)
            setattr(self,k,y)

        # import pdb; pdb.set_trace()

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
            german_item = self.german_sentences_test[idx]
            english_item = self.english_sentences_test[idx]
        else:
            german_item = self.german_sentences_train[idx]
            english_item = self.english_sentences_train[idx]
        #
        #
        # min_len = min(len(german_item),len(english_item))
        # start_token = torch.tensor([self.german_vocab["<start>"]],dtype=torch.int64)
        # if(min_len>self.min_len):
        #     #Crop randomly
        #     crop_range = min(len(german_item),len(english_item)) - self.min_len
        #     crop = random.randint(0,crop_range)
        #     german_item = german_item[crop:self.min_len+crop]
        #     english_item = english_item[crop:self.min_len+crop]
        #     german_item = torch.cat((start_token,german_item))
        #     logit_mask = torch.ones((len(german_item),1),dtype=torch.bool)
        # else:
        #     german_item = F.pad(german_item,(0,self.min_len-len(german_item)),"constant", self.german_vocab_reversed.index('<end>'))
        #     # german_item = F.pad(german_item,(0,self.min_len-len(german_item)),"constant", self.german_eos)
        #     english_item = F.pad(english_item,(0,self.min_len-len(english_item)),"constant",self.english_vocab_reversed.index('<end>'))
        #     english_item = F.pad(english_item,(0,self.min_len-len(english_item)),"constant",self.english_eos)
        #     german_item = torch.cat((start_token,german_item))
        #     #Logit Mask For Training
        #     logit_mask = torch.ones((len(german_item),1),dtype=torch.bool)
        #     logit_mask[min_len+1:,:] = 0
        # german_logits = torch.zeros((len(german_item),self.german_vocab_len))
        # index = torch.arange(0,len(german_item))
        # german_logits[index,german_item] = 1
        # if(self.CUDA):
        #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
        #
        # # gl = german_logits*logit_mask
        # # german_logits = (gl).mean(dim=1,keepdims=True)*0.2+gl * 0.8
        # if self.mode=='test':
        #     idx = idx + len(self.german_sentences_train)

        return {
        "index":idx,
        "german":german_item,
                # "english":english_item.to(self.device)[::2],
                "english":english_item,
                # "logits":german_logits.to(self.device),
                # "logit_mask":logit_mask.to(self.device)
                }
    def __len__(self):
        if(self.mode=="test"):
            return len(self.german_sentences_test)
        else:
            return len(self.german_sentences_train)
    def total_length(self):
        return  len(self.german_sentences_test)+len(self.german_sentences_train)
