import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim



class RefillModelRNNSwitch(nn.Module):
    '''
    Calculate f_{ik}(Y,Z)
    where Y a set of tokens
          Z a sequence with masks at position of extraction

    '''
    def __init__(self, device, graph_dim,embed_dim,mixture_count,state_count,total_length,min_len,mask_token_idx):
        super().__init__()
        state_count = 5
        # state_count = 15
        self.device = device
        self.total_length = total_length
        self.min_len = min_len
        self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        self.state_count = state_count

        # x = nn.Linear(embed_dim*state_count,total_length).to(self.device)
        # self.latent     = nn.Parameter(x.weight)
        # self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.anchor = nn.Linear(embed_dim,mixture_count).to(self.device)
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.selector   = nn.Linear(embed_dim, mixture_count).to(self.device)
        self.selector_q = nn.Linear(embed_dim, mixture_count).to(self.device)
        # self.selector_q = nn.Linear(embed_dim, mixture_count*embed_dim).to(self.device)
        self.selector_k = nn.Linear(embed_dim, embed_dim).to(self.device)
        # self.extractor = nn.Linear(embed_dim,state_count).to(self.device)
        # self.kernel    = nn.Bilinear(embed_dim,embed_dim,mixture_count).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater_v  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.query_init = nn.Linear(1,embed_dim).to(self.device)
        self.query_init      = nn.Linear(1,embed_dim).to(self.device)
        self.query_k    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.query_v    = nn.Linear(embed_dim,embed_dim).to(self.device)

    def nembed(self,y):
        y = self.embed(y)
        y = self.norm(y)
        return y
    def norm(self,y):
        y = y / (0.00001 + y.std(-1,keepdims=True)) *1.0
        return y

    def vocab(self,x):
        y = x.matmul(self.embed.weight.T)
        return y


    def target_energy(self,lptok,yt):
        yp = torch.gather(lptok,index=yt[:,:,None],dim=-1)[:,:,0]
        return yp

    def loss(self,zi,x,y,z):
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        xs = self.init_state.weight.T[None,0:1]
        xs = self.norm(xs)
        # xs =
        y  = self.norm(y)
        z  = self.norm(z)
        fs = torch.tensor([],requires_grad=True).to(self.device)

        # self.selector = nn.Linear(embed_dim, mixture_count).to(self.device)
        ### Uses an explicit RNN to switch between copying z and extract y
        for i in range(z.size(1)):
            xz = z[:,i:i+1]
            xs = xs + self.transition(xs) + self.updater(xz)
            xs = self.norm(xs)
            sel = self.selector(xs).softmax(-1)
            ### maybe make query vector a function of state?
            cand = self.selector_q.weight[None].matmul(self.selector_k(y).transpose(2,1)).softmax(-1).matmul(y)
            # cand = self.selector_q(xs).reshape((len(xs),-1,self.embed_dim)).matmul(self.selector_k(y).transpose(2,1)).softmax(-1).matmul(y)
            cand = torch.cat([xz,cand[:,1:]],dim=1)
            xq = sel.matmul(cand)
            fs = torch.cat([fs,xq],dim=1)


        fs = self.norm(fs)
        lptok =  self.vocab(fs).log_softmax(-1)
        cent  = self.target_energy(lptok,x)
            # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        ### return NLL
        return -cent.mean(-1)


        # return ll
    grad_loss = loss
    def corrupt(self,zi,y):
        # self.sigma = 1.5
        # self.sigma = 1.0
        y = y / (0.00001 + y.std(-1,keepdims=True)) *1.0
        y = y + torch.normal(0, self.sigma, y.shape).to(self.device)
        return y



class RefillModelCopy(nn.Module):
    '''
    Calculate f_{ik}(Y,Z)
    where Y a set of tokens
          Z a sequence with masks at position of extraction

    '''
    def __init__(self, device, graph_dim,embed_dim,mixture_count,state_count,total_length,min_len,mask_token_idx):
        super().__init__()
        state_count = 5
        # state_count = 15
        self.device = device
        self.total_length = total_length
        self.min_len = min_len
        self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        self.state_count = state_count

        # x = nn.Linear(embed_dim*state_count,total_length).to(self.device)
        # self.latent     = nn.Parameter(x.weight)
        # self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.anchor = nn.Linear(embed_dim,mixture_count).to(self.device)
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        # self.extractor = nn.Linear(embed_dim,state_count).to(self.device)
        # self.kernel    = nn.Bilinear(embed_dim,embed_dim,mixture_count).to(self.device)
        # kernel_size = 5
    def nembed(self,y):
        y = self.embed(y)
        y = self.norm(y)
        return y
    def norm(self,y):
        y = y / (0.00001 + y.std(-1,keepdims=True)) *1.0
        return y

    def vocab(self,x):
        y = x.matmul(self.embed.weight.T)
        return y


    def target_energy(self,lptok,yt):
        yp = torch.gather(lptok,index=yt[:,:,None],dim=-1)[:,:,0]
        return yp

    def loss(self,zi,x,y,z):
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        # xs = self.init_state.weight.T[None]
        # xs = self.norm(xs)
        y  = self.norm(y)
        z  = self.norm(z)
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        fs = z
        lptok =  self.vocab(fs).log_softmax(-1)
        cent  = self.target_energy(lptok,x)
            # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        ### return NLL
        return -cent.mean(-1)


        # return ll
    grad_loss = loss
    def corrupt(self,zi,y):
        # self.sigma = 1.5
        # self.sigma = 1.0
        y = y / (0.00001 + y.std(-1,keepdims=True)) *1.0
        y = y + torch.normal(0, self.sigma, y.shape).to(self.device)
        return y

import random

class RefillModelCopyWithRandomFill(nn.Module):
    '''
    Calculate f_{ik}(Y,Z)
    where Y a set of tokens
          Z a sequence with masks at position of extraction

    '''
    def __init__(self, device, graph_dim,embed_dim,mixture_count,state_count,total_length,min_len,mask_token_idx):
        super().__init__()
        state_count = 5
        # state_count = 15
        self.device = device
        self.mask_token_idx = mask_token_idx
        self.total_length = total_length
        self.min_len = min_len
        self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        self.state_count = state_count

        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
    def nembed(self,y):
        y = self.embed(y)
        y = self.norm(y)
        return y
    def norm(self,y):
        y = y / (0.00001 + y.std(-1,keepdims=True)) *1.0
        return y

    def vocab(self,x):
        y = x.matmul(self.embed.weight.T)
        return y


    def target_energy(self,lptok,yt):
        yp = torch.gather(lptok,index=yt[:,:,None],dim=-1)[:,:,0]
        return yp

    def loss(self,zi,x,y,z):
        ### state init
        # z = self.embed(z)
        # y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        # xs = self.init_state.weight.T[None]
        # xs = self.norm(xs)
        # y  = self.norm(y)
        # torch.randperm()
        # z  = self.norm(z)
        fs = torch.tensor([],requires_grad=True).to(self.device)
        # import pdb; pdb.set_trace()
        ismask = (z == self.mask_token_idx).double().topk(k=3,dim=1)[1]
        xperm = torch.tensor([random.sample(range(y.size(1)),y.size(1)) for _ in range(y.size(0))]).to(self.device)
        # yperm = torch.gather(y,index=xperm[:,:,None].repeat((1,1,y.size(2))),dim=1)
        yperm = torch.gather(y,index=xperm,dim=1)
        xz = torch.scatter(z,index=ismask,src=yperm,dim=1)

        fs = self.norm(self.embed(xz))
        lptok =  self.vocab(fs).log_softmax(-1)
        cent  = self.target_energy(lptok,x)
            # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        ### return NLL
        return -cent.mean(-1)


        # return ll
    grad_loss = loss
    def corrupt(self,zi,y):
        # self.sigma = 1.5
        # self.sigma = 1.0
        y = y / (0.00001 + y.std(-1,keepdims=True)) *1.0
        y = y + torch.normal(0, self.sigma, y.shape).to(self.device)
        return y

class RefillModelOld(nn.Module):
    '''
    Calculate f_{ik}(Y,Z)
    where Y a set of tokens
          Z a sequence with masks at position of extraction

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

        # x = nn.Linear(embed_dim*state_count,total_length).to(self.device)
        # self.latent     = nn.Parameter(x.weight)
        # self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.anchor = nn.Linear(embed_dim,mixture_count).to(self.device)
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        # self.extractor = nn.Linear(embed_dim,state_count).to(self.device)
        # self.kernel    = nn.Bilinear(embed_dim,embed_dim,mixture_count).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater_v  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.query_init = nn.Linear(1,embed_dim).to(self.device)
        self.query_init      = nn.Linear(1,embed_dim).to(self.device)
        self.query_k    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.query_v    = nn.Linear(embed_dim,embed_dim).to(self.device)

    def nembed(self,y):
        y = self.embed(y)
        y = self.norm(y)
        return y
    def norm(self,y):
        y = y / (0.00001 + y.std(-1,keepdims=True)) *1.0
        return y

    def vocab(self,x):
        y = x.matmul(self.embed.weight.T)
        return y


    def target_energy(self,lptok,yt):
        yp = torch.gather(lptok,index=yt[:,:,None],dim=-1)[:,:,0]
        return yp

    def loss(self,zi,x,y,z):
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        xs = self.init_state.weight.T[None]
        xs = self.norm(xs)
        y  = self.norm(y)
        z  = self.norm(z)
        fs = torch.tensor([],requires_grad=True).to(self.device)
        for i in range(z.size(1)):
            xz = z[:,i:i+1]

            ### state update
            for iu in range(1):
                xpu = self.updater(xz).matmul(xs.transpose(2,1)).softmax(-1)
                zv  = self.updater_v(xz)
                xs  = xs + xpu.transpose(2,1).matmul(zv)
                xs  = self.norm(xs)

            ### sequence output
            xas = torch.cat([xs,y],dim=1)
            ### simple query gen
            xq  = self.query_init.weight.T[None]
            for iq in  range(1):
                xq  = self.norm(xq)
                xqk = self.query_k(xq)
                xqatt = xqk.matmul(xas.transpose(2,1)).softmax(-1)
                xq  = xqatt.matmul(self.query_v(xas))

            xq  = self.norm(xq)
            fs = torch.cat([fs,xq],dim=1)

        lptok =  self.vocab(fs).log_softmax(-1)
        cent  = self.target_energy(lptok,x)
            # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        ### return NLL
        return -cent.mean(-1)


        # return ll
    grad_loss = loss
    def corrupt(self,zi,y):
        # self.sigma = 1.5
        # self.sigma = 1.0
        y = y / (0.00001 + y.std(-1,keepdims=True)) *1.0
        y = y + torch.normal(0, self.sigma, y.shape).to(self.device)
        return y
