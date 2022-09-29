import torch
import torch.nn as nn


'''
Sample from a vector-parametrised HHMM model
Language Prior

A sampling model generate samples associated with their energy
with energy being log(P) can be used to weight a cost function
in reinforce

The process is ill-defined so I am not inerested in writing
futher. Bascially it tries to project vectors into first-order
markov chains with states as vectors. Then sample and re-expand.
The hard bit is defining absorbing states and figuring out the
exact normalisation of the probability.
'''


class VHMM(nn.Module):
    def __init__(self, embed_dim, observe_dim,seq_dim.max_expand):
        self.embed_dim   = embed_dim
        self.seq_dim     = seq_dim
        self.observe_dim = observe_dim
        self.max_expand  = max_expand
        self.K = nn.Linear(embed_dim, seq_dim*embed_dim)
        self.V = nn.Linear(embed_dim, seq_dim*embed_dim)
        self.T = nn.Linear(embed_dim, seq_dim*embed_dim)

    def forward(self,x,n_sample):
        kx = self.K(x).reshape((len(x),seq_dim,embed_dim))
        vx = self.V(x).reshape((len(x),seq_dim,embed_dim))
        tx = self.T(x).reshape((len(x),seq_dim,embed_dim))
        for i in range(self.max_expand):
            if i ==0:
                pass
            pass

VHMM(20,100,15)
