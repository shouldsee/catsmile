import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import os

from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.bert.modeling_bert import BertLayer,BertIntermediate,BertOutput


PKL = __file__+'.temp.pkl'
if os.path.exists(PKL):
    tokenizer,model = torch.load(PKL)
else:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    torch.save((tokenizer,model),PKL)
BertTok = tokenizer
BertModel = model

class NoAttention(nn.Module):
    '''
    BertLayer without the attention module
    '''
    def __init__(self, bconf):
        super().__init__()
        self.intermediate = BertIntermediate(bconf)
        self.output = BertOutput(bconf)

    def forward(self, x):
        return self.feed_forward_chunk(x)

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class TransAttention(nn.Module):
    '''
    BertLayer without the attention module
    '''
    def __init__(self, bconf):
        super().__init__()
        self.intermediate = BertIntermediate(bconf)
        self.output = BertOutput(bconf)
        self.K = K    = bconf.num_attention_heads
        self.E = E    = bconf.hidden_size
        self.EOK= E//K
        x = nn.Linear(K,E)
        self.mu = nn.Parameter(x.weight.T[None]) # (1,K,E)
        self.val= nn.Linear(E,E)

    def forward(self, x):
        # x (B,L,E)
        # xq (B,L,K,E)
        B = x.size(0)
        L = x.size(1)
        K = self.K
        E = self.E
        EOK = self.EOK
        # (B,L,K,L)

        xq = x[:,:,None,:] + self.mu[:,None,:,:]
        xd = xq[:,:,:,None,] - x[:,None,None,:,:]
        nmsq = -xd.square().mean(-1)

        # (B,L,K,L)
        # (B,K,L,L)
        # (BK,L,L)
        att = nmsq.softmax(-1).transpose(1,2).reshape((-1,L,L))

        # (B,K,L,EOK)
        # (BK,L,EOK)
        xv = self.val(x).view((B,L,K,EOK)).permute((0,2,1,3)).reshape((-1,L,EOK))
        # (BK,L,EOK)
        xsel = att.matmul(xv).reshape((B,K,L,EOK)).permute((0,2,1,3)).reshape((B,L,E))
        # return xsel
        x = x + xsel
        # x = xsel
        # return x

        # xsel = att.matmul()
        # import pdb; pdb.set_trace()
        # xdiff = x[:,:,None,:] - x[:,None,:,:]

        # attention_output =
        return self.feed_forward_chunk(x)

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output



class FirstOrderLowRankEnergy(nn.Module):
    def __init__(self,K,E,D):
        super().__init__()
        x = nn.Linear(K,E)
        self.E = E
        self.mu = nn.Parameter(x.weight.T[None])
        self.kedr = nn.Linear(K*E,   D ,  bias=False)
        self.kedl = nn.Linear(D,   K*E ,  bias=False)
        self.k = nn.Parameter(torch.tensor(1.))
        self.b = nn.Parameter(torch.tensor(0.))
        pass

    def forward(self,x):
        # x of shape (B,L,E)
        B = x.size(0)
        mu = self.mu #(1, K, E)

        # (B, K, L)
        att = mu.matmul(x.transpose(2,1)/self.E**0.5).softmax(-1)

        # (B, K, E)
        sel = att.matmul(x)

        # dsel = self.kedl(F.relu(self.kedr(sel.reshape((B,-1)))))
        xr = (self.kedr(sel.reshape((B,-1))))
        xr = F.relu(xr)
        dsel = self.kedl(xr)
        dsel = dsel.reshape(sel.shape)

        dx = att.transpose(2,1).matmul(sel)
        x = x + dx
        x = self.k * x + self.b
        return x



class FirstOrderLowRankEnergyWithLimit(nn.Module):
    def __init__(self,K,E,D):
        super().__init__()
        x = nn.Linear(E,K)
        self.E = E
        self.mu = nn.Parameter(x.weight[None])
        self.bias = nn.Parameter(x.bias[None])

        self.kedr = nn.Linear(K*E,   D ,  bias=False)
        self.kedl = nn.Linear(D,   K*E ,  bias=False)
        self.k = nn.Parameter(torch.tensor(1.))
        self.b = nn.Parameter(torch.tensor(0.))
        pass

    def forward(self,x):
        # x of shape (B,L,E)
        B = x.size(0)
        mu = self.mu #(1, K, E)

        # (B, K, L)
        att = mu.matmul(x.transpose(2,1)/self.E**0.5)

        part = att.logsumexp(-1)
        att  = att.softmax(-1)

        # (B, K, E)
        sel = att.matmul(x)
        lre = torch.sigmoid(part - self.bias)[:,:,None]
        sel = lre*sel + (1-lre) * sel

        # dsel = self.kedl(F.relu(self.kedr(sel.reshape((B,-1)))))
        xr = (self.kedr(sel.reshape((B,-1))))
        xr = F.relu(xr)
        dsel = self.kedl(xr)
        dsel = dsel.reshape(sel.shape)

        dx = att.transpose(2,1).matmul(sel)
        x = x + dx
        x = self.k * x + self.b
        return x



class FirstOrderLowRankEnergyExpAttention(nn.Module):
    def __init__(self,K,E,D):
        super().__init__()
        x = nn.Linear(E,K)
        self.E = E
        self.mu = nn.Parameter(x.weight[None])
        # self.bias = nn.Parameter(x.bias[None])
        # weight.T[None])
        self.kedr = nn.Linear(K*E,   D ,  bias=False)
        self.kedl = nn.Linear(D,   K*E ,  bias=False)
        self.k = nn.Parameter(torch.tensor(1.))
        self.b = nn.Parameter(torch.tensor(0.))
        pass

    def forward(self,x):
        # x of shape (B,L,E)
        B = x.size(0)
        mu = self.mu #(1, K, E)

        # (B, K, L)
        att = mu.matmul(x.transpose(2,1)/self.E**0.5)
        # att = torch.exp(- att.square() - self.bias[:,:,None])
        att = torch.exp(- att.square())

        # (B, K, E)
        sel = att.matmul(x)

        # dsel = self.kedl(F.relu(self.kedr(sel.reshape((B,-1)))))
        xr = (self.kedr(sel.reshape((B,-1))))
        xr = F.relu(xr)
        dsel = self.kedl(xr)
        dsel = dsel.reshape(sel.shape)

        dx = att.transpose(2,1).matmul(sel)
        x = x + dx
        x = self.k * x + self.b
        return x


class SimpleLowRankEnergy(nn.Module):
    def __init__(self,L,E,D):
        super().__init__()
        # x = nn.Linear(K,E)
        # self.E = E
        # self.mu = nn.Parameter(x.weight.T[None])
        self.kedr = nn.Linear(L*E,   D ,  bias=False)
        self.kedl = nn.Linear(D,   L*E ,  bias=False)
        # self.outlayer  = nn.Linear(E,E)
        self.k = nn.Parameter(torch.tensor(1.))
        self.b = nn.Parameter(torch.tensor(0.))
        pass

    def forward(self,x):
        # x of shape (B, L, E)
        B = x.size(0)
        sel = x.reshape(B,-1)

        # dsel = self.kedl(F.relu(self.kedr(sel.reshape((B,-1)))))
        xr = self.kedr(sel.reshape((B,-1)))
        dsel = self.kedl(F.relu(xr))
        dsel = self.kedl((self.kedr(sel.reshape((B,-1)))))
        dsel = dsel.reshape(x.shape)
        dx = dsel

        x = x + dx
        x = self.k * x + self.b
        return x
