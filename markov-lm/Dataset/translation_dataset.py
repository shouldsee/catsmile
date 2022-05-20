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

from markov_lm.Model_pretrain import lazy_load_pretrain_model
modelName = "bert-base-uncased"
BertTok,BertModel= BertBaseUncased = lazy_load_pretrain_model(modelName)

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
        self.min_len = 15  #min(self.german_min_len,self.english_min_len)
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
        self.graph_dim = self.english_vocab_len
        self.mask_token_idx=idx

    def op_extract_and_mask(self,n_mask):
        #### randomly take out tokens and put mask at its position
        train_idx = range(0,len(self.english_sentences_train))
        test_idx =range(len(self.english_sentences_train), len(self.english_sentences_train)+len(self.english_sentences_test))
        x = torch.cat([self.english_sentences_train,self.english_sentences_test],dim=0)
        idx = [random.sample(range(x.size(1)), n_mask) for _ in range(x.size(0))]
        idx = torch.tensor(idx).to(self.device)
        y = torch.gather(x,index=idx,dim=1)
        z = x.clone()
        # print(idx)
        z = torch.scatter(z,index=idx,dim=1,src=torch.tensor(self.english_vocab['<mask>']).to(self.device)[None,None].repeat(idx.shape))
        # import pdb; pdb.set_trace()
        self.english_extracted_train = y[train_idx]
        self.english_masked_train = z[train_idx]
        self.english_extracted_test = y[test_idx]
        self.english_masked_test = z[test_idx]
        self.mask_index_train = idx[train_idx]
        self.mask_index_test  = idx[test_idx]
        # import pdb; pdb.set_trace()
        # print(z[train_idx].shape,z[test_idx].shape)
        # print(self.english_sentences_train.shape, self.english_sentences_test.shape)
    def __getitem__(self, idx):
        # torch.set_default_tensor_type(torch.FloatTensor)
        # print(idx)
        if(self.mode=="test"):
            # print(f'[testing]{idx}')
            german_item = self.german_sentences_test[idx]
            x = english_item = self.english_sentences_test[idx]
            extracted = self.english_extracted_test[idx]
            masked    = self.english_masked_test[idx]
            mask_index= self.mask_index_test[idx]
        else:
            # print(f'[training]{idx}')
            german_item = self.german_sentences_train[idx]
            x = english_item = self.english_sentences_train[idx]
            extracted = self.english_extracted_train[idx]
            masked     = self.english_masked_train[idx]
            mask_index= self.mask_index_train[idx]
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
                "mask":mask_index,
                }
                    # def __getitem__()



class ArithmeticTest(torch.utils.data.Dataset):
    def __init__(self,B = 1000,CUDA=False):
        # self.mask_token_idx = 12
        v = torch.randint(10,size=(B,2))
        v3 = (v[:,0]-v[:,1])%10
        v1 = v[:,0]
        v2 = v[:,1]
        B = len(v)
        L = 50
        varr = torch.cat([v,v3[:,None]],dim=1)
        out = torch.zeros((B,L,)).int() + 11
        for i in range(B):
            v = torch.randperm(L-3,)
            vi = -1
            for vv in v:
                vi+=1
                if vv <= 2:
                    out[i,vi]= 20+vv
                    vi+=1
                    out[i,vi]= varr[i,vv]
            # print(out[i])
            # # break
            # import pdb; pdb.set_trace()
        # ['.',None,'A',]
        idx2letter = ['X']*30
        for i in range(10):
            idx2letter[i] = str(i)
        idx2letter[20]='A'
        idx2letter[21]='B'
        idx2letter[22]='B'
        idx2letter[23]='M';self.mask_token_idx=23
        idx2letter[11]='.'

        # import pdb; pdb.set_trace()
        self.train()
        self.test_idx_offset = 800
        # import pdb; pdb.set_trace()

        self.graph_dim = len(idx2letter)+1
        self.B = B
        self.device = torch.device('cpu' if not CUDA else 'cuda:0')
        out = out.to(self.device)

        self.data = out.long()
        v = out
        vt = 20+ torch.randint(3,(len(v),1),device=self.device)
        self.m = m = (1+((v==vt).max(1)[1]))[:,None].long()
        self.vm = torch.scatter(v, src=torch.ones(v.shape,device=self.device).int()*self.mask_token_idx, index=m,dim=1).long()


    def test(self):
        self.is_train=False

    def train(self):
        self.is_train=True

    def __getitem__(self,idx):
        if self.is_train:
            idx = idx
            # v = self.data[idx]
        else:
            idx= idx+self.test_idx_offset
            # v = self.data[idx+self.test_idx_offset]

        return dict(unmasked =self.data[idx] ,masked = self.vm[idx],mask = self.m[idx])
    def __len__(self):
        if self.is_train:
            return self.test_idx_offset -0
        else:
            return self.B - self.test_idx_offset
        pass
dat = ArithmeticTest(1000)
# dat[10:20]

# from markov_lm.Model_pretrain import BertTok,BertModel
import random
import shutil

class BertMiddleLayer(EnglishToGermanDataset):
    '''
    Cache input as encoder.layers.[7].output
    and output as encoder.layers.[8].output
    '''
    def __init__(self, layerIndex, BertTok=BertTok, BertModel=BertModel, CUDA=False):
        super().__init__(CUDA)

        BertModel = BertModel.to(self.device)
        self.mimic = BertModel.encoder.layer[layerIndex]

        PKL =f'{__file__}.{__class__.__name__}.{layerIndex}.pkl'
        if 0 and os.path.exists(PKL):
            loaded = torch.load(PKL, map_location=self.device)
            self.__dict__.update(loaded)
        else:
            for k in ["english_sentences_train","english_sentences_test",]:
                xx = getattr(self,k)
                sents = []
                # sents += [ 'move ' + (''.join([self.english_vocab_reversed[i] for i in sent_idx])) for sent_idx in xx][::2]
                # sents += [ 'move this ' + (''.join([self.english_vocab_reversed[i] for i in sent_idx])) for sent_idx in xx][::2]
                sents += [ '' + (''.join([self.english_vocab_reversed[i] for i in sent_idx])) for sent_idx in xx]

                x = BertTok(sents,max_length=self.min_len,padding='max_length',truncation=True)
                x = torch.tensor(x.input_ids,device=self.device)
                y = BertModel(x, output_hidden_states=True).hidden_states
                x1,x2 = y[layerIndex].detach(),y[layerIndex+1].detach()
                y= dict(input=x1,output=x2,input0=x1,output0=x2)
                setattr(self,k,y)
            torch.save(vars(self),PKL+'.temp')
            shutil.move(PKL+'.temp',PKL)

        self.train()
        return
    def jiggle_data(self):
        sigma = 0.02
        for k in ["english_sentences_train","english_sentences_test",]:
            xx = getattr(self,k)
            x1 = xx['input0']
            # x2 = xx['output0']
            noise = torch.normal(0,sigma,x1.shape)
            mask = torch.arange((len(x1)))[:,None,None]>0
            x1 = x1+ (noise * mask).to(self.device)
            x2 = self.mimic(x1)[0]
            y = xx
            y['input'] = x1.detach()
            y['output']=x2.detach()
            # setattr(self,k,y)
        pass

    @property
    def data(self):
        if(self.mode=="test"):
            x = self.english_sentences_test
        else:
            x = self.english_sentences_train
        return x

    def __getitem__(self, idx):
        x= self.data
        return {"index":idx,"input":x['input'][idx],"output":x['output'][idx]}

    def __len__(self):
        return self.data['input'].__len__()

                    # def __getitem__()
if __name__ == '__main__':
    x = BertMiddleLayer(BertTok, BertModel, CUDA=True)
    x[range(5)]
    print(x[range(5)]['input'].shape)
    print(len(x))
    x.test()
    x[range(5)]
    print(len(x))
