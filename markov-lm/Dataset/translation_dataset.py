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

# source_lang_file = "German_sentences.pkl"
# dest_lang_file  = "English_sentences.pkl"
source_lang_file = "English_sentences.pkl"
dest_lang_file  = "German_sentences.pkl"
DIR = os.path.dirname(os.path.realpath(__file__))
class DeviceDataset(torch.utils.data.Dataset):
    def __init__(self,CUDA=False):
        super().__init__()
        self.device = torch.device('cuda:0' if CUDA else 'cpu')
        self.train()

    def test(self):
        self.mode = "test"
    def train(self):
        self.mode = "train"

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
        # self.min_len = 30 #min(self.german_min_len,self.english_min_len)
        self.min_len = 15  #min(self.german_min_len,self.english_min_len)
        self.data_dim = self.min_len
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

class GermanToEnglishDatasetRenamed(EnglishToGermanDataset):
    def __init__(self,CUDA):
        # import pdb; pdb.set_trace()
        super().__init__(CUDA)
        self.graph_dim = self.german_vocab.__len__() + self.english_vocab.__len__()

    def __getitem__(self, idx):
        # torch.set_default_tensor_type(torch.FloatTensor)
        if(self.mode=="test"):
            german_item = self.german_sentences_test[idx]
            english_item = self.english_sentences_test[idx]
        else:
            german_item = self.german_sentences_train[idx]
            english_item = self.english_sentences_train[idx]

        return {
        "index":idx,
        "source":german_item,
        "target":english_item + self.german_vocab.__len__(),
                }

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
#

class BertMiddleLayer(EnglishToGermanDataset):
    '''
    Cache input as encoder.layers.[7].output
    and output as encoder.layers.[8].output
    '''
    BertTok = None
    BertModel = None

    def __init__(self, layerIndex, CUDA=False):
        super().__init__(CUDA)
        cls = self.__class__
        if self.__class__.BertTok is None:
            modelName = "bert-base-uncased"
            cls.BertTok,cls.BertModel = BertBaseUncased = lazy_load_pretrain_model(modelName)

        BertModel = cls.BertModel.to(self.device)
        self.mimic = BertModel.encoder.layer[layerIndex]

        PKL =f'{__file__}.{__class__.__name__}.{layerIndex}.pkl'
        if 1 and os.path.exists(PKL):
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



class Vocab(object):
    def __init__(self,vocab,offset):
        self.offset = offset
        self.i2w = list(sorted(vocab))
        self.w2i = {w:i+offset for i,w in enumerate(self.i2w)}
    def __len__(self):
        return self.i2w.__len__()
    def tokenize(self,k):
        return self.w2i[k]

    def wordize(self,i):
        return self.i2w[i-self.offset]

import collections

import pickle
class WMT14(DeviceDataset):
    def __init__(self, source='de', target='en', B = 1000,CUDA=False,test_ratio=0.2, min_len=50):
        super().__init__(CUDA=CUDA)
        # self.B = B
        test_count = int(B*test_ratio)
        train_count = B - test_count
        self.root= DIR + '/wmt14/train'
        self.source = source
        self.target = target
        # self.train()
        device= self.device
        sv,tv,sc,tc = self.get_cached_data(min_len,B)
        self.src_codes = torch.tensor(sc,dtype=torch.long,device=self.device)
        self.tgt_codes = torch.tensor(tc,dtype=torch.long,device=self.device)
        # import pdb; pdb.set_trace()
        self.src_vocab = sv
        self.tgt_vocab = tv
        # self.test_offset = train_count
        self.test_offset = self.train_count =train_count
        self.test_count = test_count
        self.end_offset = test_count + self.test_offset

        self.graph_dim = len(sv)+len(tv)
        self.data_dim  = min_len
        print(f'[{self.__class__.__name__}]INIT_FINISH')

    def p(self,v):
        print(f'[{self.__class__.__name__}]:{v}')
        # processing {i}/{B}')
    def get_cached_data(self,min_len,B):
        CACHE_FILE = f'{self.root}.cache.{min_len}.{B}.pkl'
        if 0 and os.path.exists(CACHE_FILE):
            with open(CACHE_FILE,'rb') as f:
                v = pickle.load(f)
            src_vocab,tgt_vocab,src_codes,tgt_codes =v
            # import pdb; pdb.set_trace()
        else:
            src_list = []
            tgt_list = []
            with open(self.root+'.'+ self.source, 'r') as f:
                with open(self.root+'.'+self.target, 'r') as f2:
                    # f.read().splitlines()
                    for i in range(B):
                        if i%500==0:
                            print(f'{self.__class__.__name__}:processing {i}/{B}')
                        src = src_line = f.readline().strip()
                        tgt = tgt_line = f2.readline().strip()
                        src = src.split()[:min_len]
                        src = [x.upper() for x in src ]

                        tgt = tgt.split()[:min_len]
                        tgt = [x.upper() for x in tgt ]

                        src = src + ['<UNK>']*max(0,(min_len - len(src)))
                        tgt = tgt + ['<UNK>']*max(0,(min_len - len(tgt)))

                        src_list.append(src)
                        tgt_list.append(tgt)
                        # if 'This hotel has a contemporary character with a classy 1930s feel . The Au Palais De' in tgt_line:
                            # import pdb; pdb.set_trace()

            self.p('VOCAB 1')
            src_vocab = set()
            [src_vocab.update(x) for x in src_list]

            tgt_vocab = set()
            self.p('VOCAB 2')
            [tgt_vocab.update(x) for x in tgt_list]

            self.p('VOCAB 3')
            src_vocab = Vocab(src_vocab,0)
            tgt_vocab = Vocab(tgt_vocab, src_vocab.__len__())
            # src_codes =

            src_codes = [[src_vocab.tokenize(x) for x in xx] for xx in src_list]
            tgt_codes = [[tgt_vocab.tokenize(x) for x in xx] for xx in tgt_list]
            # import pdb; pdb.set_trace()
            with open(CACHE_FILE+'.temp','wb') as f:
                pickle.dump((src_vocab,tgt_vocab,src_codes,tgt_codes),f)
            shutil.move(CACHE_FILE+'.temp',CACHE_FILE)
            v = src_vocab,tgt_vocab,src_codes,tgt_codes

        for vocab_key in 'src_vocab tgt_vocab'.split():
            vocab = eval(vocab_key)
            print(f'[{vocab_key}]{len(vocab)} {list(vocab.i2w)[:5]}')
        cts = {}
        cts['src'] = collections.Counter()
        cts['tgt'] = collections.Counter()
        [cts['src'].update(x) for x in src_codes]
        [cts['tgt'].update(x) for x in tgt_codes]
        for x in 'src tgt'.split():
            self.p( cts[x].most_common(10))
        # self.p( collections.Counter(sum(tgt_codes,[])).most_common(10))
        return v
            # return src_vocab,tgt_vocab,src_codes,tgt_codes

    def src_wordize(self,v):
        return self.src_vocab.wordize(v)
    def tgt_wordize(self,v):
        return self.tgt_vocab.wordize(v)
    def __len__(self):
        if(self.mode=="test"):
            return self.test_count
        else:
            return self.train_count
    def __getitem__(self,index):
        if(self.mode=="test"):
            index = index + self.test_offset
        else:
            pass
        # print(index)
        target=self.tgt_codes[index]
        xsource=self.src_codes[index]
        # if index==3604:
        if 1:
            pass
        return dict(
            target=target,
            source=xsource,
            index=index)

import torchtext
from torchtext import data


class Multi30k(torch.utils.data.Dataset):
    pass

if __name__ == '__main__':
    fix_length = 50
    src = tgt = torchtext.data.Field(lower=False, include_lengths=False, batch_first=True,fix_length=fix_length)
    root = DIR
    ret = torchtext.datasets.Multi30k.download(root)
    m30k = torchtext.datasets.Multi30k(root+'/multi30k/train',('.de','.en'),(src,tgt))
    src.build_vocab(m30k, max_size=80000)
    tgt.build_vocab(m30k, max_size=40000)
    it = data.BucketIterator(dataset=m30k, batch_size=32)
    it = iter(it)
    it.__next__()
    self = m30k
    def to_code(x,self=self):
        return {
        'source':self.fields['src'].numericalize([x['src']]),
        'target':self.fields['trg'].numericalize([x['trg']]),
        }
    # m30k.fields src.numericalize([m30k[0].src])
    import pdb; pdb.set_trace()
    x = self.examples[0].__dict__
    to_code(x)


    # ,sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))
    # train_iter = data.BucketIterator(dataset=m30k, batch_size=32,sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))
    # m30k[[0,1,2]]
    m30k[0]
    import pdb; pdb.set_trace()
    dat = WMT14()
    print(dat[torch.tensor([0,1,2])])
    dat.test()
    print(dat[torch.tensor([0,1,2])])
    # print(dat[[0,1,2]])

    # x = BertMiddleLayer(BertTok, BertModel, CUDA=True)
    # x[range(5)]
    # print(x[range(5)]['input'].shape)
    # print(len(x))
    # x.test()
    # x[range(5)]
    # print(len(x))
