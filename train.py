
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from Dataset.translation_dataset import EnglishToGermanDataset

import os,sys

import numpy as np
from tqdm import tqdm
from torch import autograd

from util_html import write_png_tag

import pandas as pd
import sys
import glob
from pprint import pprint
import shutil
from tqdm import tqdm

CUDA = 1
batch_size = 60
dataset = EnglishToGermanDataset(CUDA=CUDA)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# device = torch.device('cuda:0' if CUDA else 'cpu')






class MixtureOfHMM(nn.Module):
    def __init__(self, graph_dim,embed_dim,mixture_count,state_count,device):
        super().__init__()
        # graph_dim = english_vocab_len
        self.device = device
        self.embed = nn.Embedding(graph_dim,embed_dim).to(self.device)
        # self.vocab = nn.Linear(graph_dim,embed_dim).to(self.device)
        self.vocab = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.mixture_count = mixture_count
        self.state_count = state_count

        x = nn.Linear(state_count,state_count*mixture_count).to(self.device)
        self.init_dist = nn.Parameter(x.bias.reshape((1,mixture_count,state_count)))
        self.transition= nn.Parameter(x.weight.reshape((1,mixture_count,state_count,state_count)))
        x = nn.Linear(embed_dim*state_count,mixture_count).to(self.device)
        self.state_vect= nn.Parameter(x.weight.reshape((1,mixture_count,state_count,embed_dim)))
        # self.emission  = nn.Linear(state_count**2,mixture_count).to(self.device)
        # print(self.transition.shape)
        # self.init_dist

    def log_prob(self,x):
        x0 = x
        xx = self.embed(x)
        zs = torch.tensor([],requires_grad=True)
        z = torch.log_softmax(self.init_dist[:,:,:,None]/2.,dim=2)
        logtransition = torch.log_softmax(self.transition/2.,2)
        emit = torch.log_softmax(self.vocab(0*self.state_vect+xx.mean(1)[:,None,None]),dim=-1)
        # emit = torch.log_softmax(self.vocab(self.state_vect)/2.,dim=-1)
        # state_vect = xx
        # import pdb; pdb.set_trace()
        for i in range(x.shape[1]):
            # xxx = ( xx[:,i:i+1,:])
            z = torch.logsumexp(logtransition + z,dim=2)[:,:,:,None]
            x = torch.cat([x0[:,None,i:i+1,None] for _ in range(self.mixture_count)],dim=1)
            x = torch.cat([x for _ in range(self.state_count)],dim=2)
            x = torch.gather(emit+(0*x),index=x,dim=-1)
            z = x + z
        # x.shape[1]
            # [0][:,0:])
        z = z/1./(i+1)
        # print(z[0,:,0,:])
        # print(z[0,0,:,:])
        # print(z[0,:,0,:])
        # print(self.transition[0])
        # z = torch.logsumexp(z,dim=2)
        # z = torch.logsumexp(1*z,dim=1)
        z = torch.logsumexp(z,dim=(1,2))
        # z =  z.max(dim=(2))[0].max(dim=1)[0]
            # log_emit = x
            # z = log_emit +
        # torch.logsumexp(self.transition * self.alpha,dim=)
        # torch.log(torch.sum(torch.exp()))/
        # z = x
        # import pdb; pdb.set_trace()
        return z

    def decode(self,z):
        y = z
        return y

    def sample(self,size):
        torch.random()

def cross_entropy(targ,pred):
    return

class Config(object):
    def __init__(self):
        return
conf = Config()

conf.criterion = cross_entropy
conf.embed_dim = 25
conf.mixture_count = 10
conf.state_count = 10
conf.device =  torch.device('cuda:0' if CUDA else 'cpu')
conf.num_epoch = 1000
model = MixtureOfHMM(graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device)

conf.learning_rate = 0.001

params = list(model.parameters())
print(dict(model.named_parameters()).keys())
#### using Adam with high learning_rate is catastrophic
optimizer = torch.optim.Adagrad( params, lr=conf.learning_rate)

def main():
    for epoch in range(conf.num_epoch):
        loss_train_sum = 0
        loss_test_sum = 0

        model.eval()
        dataset.test()
        conf.tsi_max = 10
        for tsi,item in enumerate(dataloader_test):
            x = item['english']
            loss =  -model.log_prob(x).mean()
            loss_test_sum +=  float(loss.item())
            if tsi==conf.tsi_max:
                break
                # print(tsi)

        model.train()
        dataset.train()
        for tri,item in enumerate(tqdm(dataloader)):
            x = item['english']
            # z = model.encode(x)
            # y = model.decode(z)
            loss =  -model.log_prob(x).mean()
            # loss.mean()
            loss_train_sum += float(loss.item())
            loss.backward()
            optimizer.step()
            # break


        loss_train_mean = loss_train_sum/(1+tri)
        loss_test_mean = loss_test_sum/(1+tsi)
        print(f'Epoch: {epoch}')
        print(f'Training Loss: {loss_train_mean}')
        print(f'Testing Loss: {loss_test_mean}')
            # loss = cross_entropy(x,y)
if __name__=='__main__':
    main()
