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
        # Tooo slow
        # # gl = german_logits*logit_mask
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

import random
class RefillDataset(EnglishToGermanDataset):
    def __init__(self,CUDA=False):
        super().__init__(CUDA)
        self.english_vocab_len += 1
        idx = len(self.english_vocab_reversed)
        self.english_vocab_reversed.append('<mask>')
        self.english_vocab['<mask>'] =idx
        self.op_extract_and_mask(2)

    def op_extract_and_mask(self,n_mask):
        #### randomly take out tokens and put mask at its position
        train_idx = range(0,len(self.english_sentences_train))
        test_idx =range(len(self.english_sentences_train), len(self.english_sentences_train)+len(self.english_sentences_test))
        x = torch.cat([self.english_sentences_train,self.english_sentences_test],dim=0)
        idx = [random.sample(range(x.size(1)), n_mask) for _ in range(x.size(0))]
        idx = torch.tensor(idx).to(self.device)
        y = torch.gather(x,index=idx,dim=1)
        z = x.clone()
        z = torch.scatter(z,index=idx,dim=1,src=torch.tensor(self.english_vocab['<mask>']).to(self.device)[None,None].repeat(idx.shape))
        # import pdb; pdb.set_trace()
        self.english_extracted_train = y[train_idx]
        self.english_masked_train = z[train_idx]
        self.english_extracted_test = y[test_idx]
        self.english_masked_test = z[test_idx]

    def __getitem__(self, idx):
        # torch.set_default_tensor_type(torch.FloatTensor)
        # print(idx)
        if(self.mode=="test"):
            german_item = self.german_sentences_test[idx]
            x = english_item = self.english_sentences_test[idx]
            extracted = self.english_extracted_test[idx]
            masked    = self.english_masked_test[idx]
        else:
            german_item = self.german_sentences_train[idx]
            x = english_item = self.english_sentences_train[idx]
            extracted = self.english_extracted_train[idx]
            masked     = self.english_masked_train[idx]
        #
        # Tooo slow
        # # gl = german_logits*logit_mask
        # if self.mode=='test':
        #     idx = idx + len(self.german_sentences_train)

        return {
                "index":idx,
                "extracted":extracted,
                "masked":masked,
                "english":x,
                "german":german_item,
                }
                    # def __getitem__()
