import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

class RNNWithVectorSelection(nn.Module):
    '''
    Instead of attempting decomposition, I now 
    '''
    def __init__(self, device, graph_dim,embed_dim,mixture_count,state_count,total_length,min_len):
        super().__init__()
        state_count = 5
        # state_count = 15
        self.device = device
        self.total_length = total_length
        self.min_len = min_len
        self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        self.state_count = state_count

        x = nn.Linear(embed_dim*state_count,total_length).to(self.device)
        self.latent     = nn.Parameter(x.weight)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.anchor = nn.Linear(embed_dim,mixture_count).to(self.device)
        self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.n_step     = min_len
        self.extractor = nn.Linear(embed_dim,state_count).to(self.device)

    def sample_tokens(self,z,n_step):
        zs = self.sample_trajectory(z,n_step)
        ys = self.vocab(zs).log_softmax(-1)
        return ys

    def sample_trajectory(self,z,n_step):
        lat = z.reshape((len(z),self.state_count,self.embed_dim))
        lat    = lat / (0.00001 + lat.std(-1,keepdims=True)) *0.113
        z   = lat[:,:1,]
        zs = torch.tensor([],requires_grad=True).to(self.device)
        for i in range(n_step):
            z    = z / (0.00001 + z.std(-1,keepdims=True)) *0.113
            # emit = z[:,:]
            cand = torch.cat([z[:], lat[:,1:]],dim=1)
            att  = self.extractor(z)
            # att  = att.softmax(-1)
            emit = att.matmul(cand)

            zs = torch.cat([zs,emit],dim=1)
            z = (z + self.transition(z))/2.
        return zs

    def log_prob(self,zi,y):
        z = self.latent[zi]
        ys = self.sample_tokens(z,self.n_step)
        yp = torch.gather(ys,index=y[:,:,None],dim=-1)[:,:,0]
        return yp

    log_prob_grad = log_prob
