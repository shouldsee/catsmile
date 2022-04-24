import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

class PerturbModel(nn.Module):
    '''
    Instead of attempting decomposition, let's try the
    super duper popular peruturbation-based pretraining!

    Type of noise introduced in BART already. Here we try both

    '''
    def __init__(self, device, graph_dim,embed_dim,mixture_count,state_count,total_length,min_len,optimizer_factory):
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
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.extractor = nn.Linear(embed_dim,state_count).to(self.device)
        self.kernel    = nn.Bilinear(embed_dim,embed_dim,mixture_count).to(self.device)
        kernel_size = 5
        # self.conv1     = nn.Conv1d(embed_dim,mixture_count,kernel_size).to(self.device)
        self.conv1_hand= nn.Linear(embed_dim*kernel_size,mixture_count).to(self.device)
        self.conv1_hand_v= nn.Linear(embed_dim*kernel_size,mixture_count).to(self.device)
        # nn.Conv1d(embed_dim,mixture_count,kernel_size).to(self.device)
        self.optimizer_factory = optimizer_factory
        self.sigma = 2.0
        self.lin   = nn.Linear(1,1).to(self.device)
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

    def sample_tokens(self,zi,y,n_step,is_embed):
        if not is_embed:
            y = self.embed(y)
        # zi = None
        corrupted = self.corrupt(zi,y)
        recovered = self.recover(zi,corrupted)
        lptok =  self.vocab(recovered).log_softmax(-1)

        # zs = self.sample_trajectory(z,n_step)
        # ys = self.vocab(zs).log_softmax(-1)
        return lptok

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

    def corrupt(self,zi,y):
        # self.sigma = 1.5
        # self.sigma = 1.0
        y = y / (0.00001 + y.std(-1,keepdims=True)) *1.0
        y = y + torch.normal(0, self.sigma, y.shape).to(self.device)
        return y

    def energy(self,xc):
        return self.energy_and_grad(xc)[0].mean()

    def energy_and_grad(self,xc):
        # xx = self.conv1(xc.transpose(2,1)).transpose(2,1)
        # eng = xx.logsumexp(-1)
        # .sum()
        xcc = torch.stack(
        [
            xc.roll(-2,1),
            xc.roll(-1,1),
            xc.roll(0,1),
            xc.roll(1,1),
            xc.roll(2,1),

        ],-1)
        xccc = xcc.reshape(xc.shape[:2]+(-1,))
        # xs = (xccc.matmul(self.conv1_hand.weight.T))
        xs = self.conv1_hand(xccc)
        lse = xs.logsumexp(-1)

        xp = xs.softmax(-1)
        # import pdb; pdb.set_trace()
        rxs = xp.matmul(self.conv1_hand_v.weight)
        rxss = rxs.reshape(xcc.shape)
        dxc = rxss.mean(-1)
        return lse,dxc
        # eng

    def recover(self,zi,xc,nstep=20,lr=0.1):
        '''
        explict descent on a energy function
        xs = lse( W * conv(X))
        # dxs / dX

        '''

        # lr = 0.1
        # lr = 0.01
        # xc = torch.tensor(xc,requires_grad=True)
        # xc.retain_grad()
        # xc = xc[:,:,:,None]
        xc0 = xc
        for i in range(nstep):
            lse,dxc = self.energy_and_grad(xc)
            xc = xc -  lr * dxc
            xc = self.norm(xc)
        return xc

            # print((lse.mean()*100).long())
        #### This does not work properly
        # import pdb; pdb.set_trace()
        # for i in range(0):
        #     # eng =
        #     xx = self.conv1(xc.transpose(2,1)).transpose(2,1)
        #     eng = xx.logsumexp(-1)
        #     import pdb; pdb.set_trace()
        #     # xc.retain_grad()
        #     # eng = self.energy(xc).sum()
        #     # eng = xx.mean()
        #     # print(eng)
        #     eng.backward(retain_graph=True)
        #     xc = xc - lr * xc.grad
        #     # xc.data.sub_(lr * xc.grad.data)
        #     # xc.grad.data.zero_()
    def target_energy(self,lptok,yt):
        yp = torch.gather(lptok,index=yt[:,:,None],dim=-1)[:,:,0]
        return yp

    def log_prob(self,zi,y,is_embed=False):
        '''
        The log_prob
        Ask the recovered to be better than the perturbed.

        Original embedding needs to be good.
        recovered also needs to be good.

        Once you get token series y, you needs to (randomly?) corrupt it and recover it.

        To ensure energy locking, ask the fitted energy function
        to follows real energy function, up to a linear change
        '''
        # yp = 0
        ll = 0

        yn = self.nembed(y)
        corrupted = self.corrupt(zi,yn)
        recovered = self.recover(zi,corrupted)

        xc = recovered
        lptok =  self.vocab(xc).log_softmax(-1)
        y1 = self.target_energy(lptok,y)
        e1,gd = self.energy_and_grad(xc)
        xe1m = e1.mean(-1)
        # ll = ll - torch.square(y1.mean(-1) - self.lin(xe1m[:,None]))
        y1a = y1.mean(-1)
        e1a = e1.mean(-1)
        # ll = y1a - gd.square().mean(-1).mean(-1)

        xc = corrupted
        lptok =  self.vocab(xc).log_softmax(-1)
        y1 = self.target_energy(lptok,y)
        e1 = self.energy_and_grad(xc)[0]
        xe1m = e1.mean(-1)
        # ll = ll - torch.square(y1.mean(-1) - self.lin(xe1m[:,None]))
        y1b = y1.mean(-1)
        e1b = e1.mean(-1)



        xc = yn
        lptok =  self.vocab(xc).log_softmax(-1)
        y1 = self.target_energy(lptok,y)
        e1 = self.energy_and_grad(xc)[0]
        xe1m = e1.mean(-1)
        # ll = ll - torch.square(y1.mean(-1) - self.lin(xe1m[:,None]))
        y1c = y1.mean(-1)
        e1c = e1.mean(-1)


        ### white noise
        xc =  torch.normal(0,1,yn.shape).to(self.device)
        xc = self.norm(xc)

        lptok =  self.vocab(xc).log_softmax(-1)
        y1 = self.target_energy(lptok,y)
        e1 = self.energy_and_grad(xc)[0]
        xe1m = e1.mean(-1)
        # ll = ll - torch.square(y1.mean(-1) - self.lin(xe1m[:,None]))
        y1d = y1.mean(-1)
        e1d = e1.mean(-1)

        xc = self.recover(None,xc)
        lptok =  self.vocab(xc).log_softmax(-1)
        y1 = self.target_energy(lptok,y)
        e1 = self.energy_and_grad(xc)[0]
        xe1m = e1.mean(-1)
        # ll = ll - torch.square(y1.mean(-1) - self.lin(xe1m[:,None]))
        y1e = y1.mean(-1)
        e1e = e1.mean(-1)

        # xes = torch.stack([e1a,e1b,e1c,],-1)
        # xys = torch.stack([y1a,y1b,y1c],-1)
        # xes = xes / 10.
        return y1a
        
        xes = torch.stack([e1a,e1b,e1d,e1e],-1)
        xys = torch.stack([y1a,y1b,y1d,y1e],-1)
        '''
        The loss function says that RECOVERED samples needs to be
        better than CORRUPTED samples

        If the original input must be masked otherwise the model would cheat!
        '''

        sampled_energy = ((-xes).softmax(-1) * xys).sum(-1)
        return sampled_energy

        # import pdb; pdb.set_trace()

        # ll = y1a +y1c
        # ll = ll + (y1a-y1b)*(e1a-e1b)
        #
        #
        # xc = yn
        # lptok =  self.vocab(xc).log_softmax(-1)
        # y1 = self.target_energy(lptok,y)
        # e1 = self.energy_and_grad(xc)[0]
        # xe1m = e1.mean(-1)
        # ll = ll - torch.square(y1.mean(-1) - self.lin(xe1m[:,None]))

        # ll = ll - torch.square(y1.mean(-1) - self.lin(xe1m[:,None]))

        # y1.mean() - e1.mean()
        # import pdb; pdb.set_trace()


        # yn = self.nembed(y)
        # recovered = self.recover(zi,yn)
        # lptok = self.vocab(recovered).log_softmax(-1)
        # yp = yp + torch.gather(lptok,index=y[:,:,None],dim=-1)[:,:,0]

        return ll

    log_prob_grad = log_prob
