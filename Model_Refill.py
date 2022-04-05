import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim





class RefillModelRNNBase(nn.Module):
    '''
    Calculate f_{ik}(Y,Z)
    where Y a set of tokens
          Z a sequence with masks at position of extraction

    '''
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):
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
        # self.updater_v  = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.query_init = nn.Linear(1,embed_dim).to(self.device)
        # self.query_init      = nn.Linear(1,embed_dim).to(self.device)
        # self.query_k    = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.query_v    = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.callback = lambda zi,x,y,z,sel: None
    def callback_step(self,outer,inner):
        [zi,x,y,z,fs],[i,sel,xz,xs] = outer,inner
        return
    def callback_init(self,outer):
        return
    def callback_end(self,outer):
        return
        # self.callback_init = lambda zi,x,y,z,sel: None

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
        return self._loss(zi,x,y,z,out='loss')
    def get_tokens(self,zi,x,y,z):
        return self._loss(zi,x,y,z,out='token')

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''
        return outer,inner


    def _batch_init(self,zi,x,y,z):
        #### batch_init part
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
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        outer = [zi,x,y,z,fs]
        i = -1
        sel = None
        xz = None
        inner = [i,sel,xz,xs]
        return outer,inner

    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        for i in range(z.size(1)):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        self.callback_end(outer)
        (zi,x,y,z,fs) = outer


        fs = self.norm(fs)
        if out=='traj': return fs
        lptok =  self.vocab(fs).log_softmax(-1)
        if out=='token': return lptok

        cent  = self.target_energy(lptok,x)

        if out=='loss': return -cent.mean(-1)
        assert 0

    grad_loss = loss

    def corrupt(self,zi,y):
        # self.sigma = 1.5
        # self.sigma = 1.0
        y = y / (0.00001 + y.std(-1,keepdims=True)) *1.0
        y = y + torch.normal(0, self.sigma, y.shape).to(self.device)
        return y

class RefillModelRNNAdditive(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
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

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)


        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.selector   = nn.Linear(embed_dim, mixture_count).to(self.device)
        # self.selector_2   = nn.Linear(embed_dim, embed_dim).to(self.device)
        self.selector_q = nn.Linear(embed_dim, mixture_count).to(self.device)
        # self.selector_q = nn.Linear(embed_dim, mixture_count*embed_dim).to(self.device)
        self.selector_k = nn.Linear(embed_dim, embed_dim).to(self.device)
        # self.extractor = nn.Linear(embed_dim,state_count).to(self.device)
        # self.kernel    = nn.Bilinear(embed_dim,embed_dim,mixture_count).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
    # def _step(self,outer,inner):
    def _batch_init(self,zi,x,y,z):
        #### batch_init part
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        xs = self.init_state.weight.T[None,0:1]
        xs = xs.repeat((len(z),1,1))
        xs = self.norm(xs)
        # xs =
        y  = self.norm(y)
        z  = self.norm(z)
        fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xz = None
        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xs]
        return outer,inner

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xz,xs) = inner
        sel = None
        xz  = None

        xz = z[:,i:i+1]

        # xs = xs + self.updater(xz)
        # xs = xs + self.transition(xs) + self.updater(xz)
        if i>=2:
            ### Uses the minus two token for prediction
            ### nearly a CNN fashion
            xs = self.transition(z[:,i-2:i-1])

        xs = self.transition(xs) + self.updater(xz)
        xs = self.norm(xs)

        ### the hidden states can be considered as updating the most relevant neuron according to the
        # sel = self.selector_2(xs).relu()
        sel = self.selector(xs).softmax(-1)
        ### maybe make query vector a function of state?
        # cand
        cand = self.selector_q.weight[None].matmul(self.selector_k(y).transpose(2,1)).softmax(-1).matmul(y)
        # cand = self.selector_q(xs).reshape((len(xs),-1,self.embed_dim)).matmul(self.selector_k(y).transpose(2,1)).softmax(-1).matmul(y)
        # cand = torch.cat([xz,cand[:,1:]],dim=1)
        cand = torch.cat([xz,xs,cand[:,2:]],dim=1)
        xq   = sel.matmul(cand)

        # xs = xs + self.transition(xs) + self.updater(xq)
        # xs = self.norm(xs)

        fs   = torch.cat([fs,xq],dim=1)
        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xs]
        return outer,inner


class RefillModelRNNAdditiveWithPseudoSampling(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
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

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)


        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.selector   = nn.Linear(embed_dim, mixture_count).to(self.device)
        # self.selector_2   = nn.Linear(embed_dim, embed_dim).to(self.device)
        self.selector_q = nn.Linear(embed_dim, mixture_count).to(self.device)
        # self.selector_q = nn.Linear(embed_dim, mixture_count*embed_dim).to(self.device)
        self.selector_k = nn.Linear(embed_dim, embed_dim).to(self.device)
        # self.extractor = nn.Linear(embed_dim,state_count).to(self.device)
        # self.kernel    = nn.Bilinear(embed_dim,embed_dim,mixture_count).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
    # def _step(self,outer,inner):
    def _batch_init(self,zi,x,y,z):
        #### batch_init part
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        xs = self.init_state.weight.T[None,0:1]
        xs = xs.repeat((len(z),1,1))
        xs = self.norm(xs)
        # xs =
        y  = self.norm(y)
        z  = self.norm(z)
        fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xz = None
        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xs]
        return outer,inner

    def _step(self,outer,inner):
        '''
        Pesudo-sampling is to plug in the likelihood into the expectation function.
        Since the final emission is not linear, taking expectation before emission might
        cause collapsing problem...
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xz,xs) = inner
        sel = None
        xz  = None

        xz = z[:,i:i+1]

        # xs = xs + self.updater(xz)
        # xs = xs + self.transition(xs) + self.updater(xz)
        if i>=2:
            ### Uses the minus two token for prediction
            ### nearly a CNN fashion
            xs = self.transition(z[:,i-2:i-1])

        xs = self.transition(xs) + self.updater(xz)
        xs = self.norm(xs)

        ### the hidden states can be considered as updating the most relevant neuron according to the
        # sel = self.selector_2(xs).relu()
        ### maybe make query vector a function of state?
        # cand
        cand = self.selector_q.weight[None].matmul(self.selector_k(y).transpose(2,1)).softmax(-1).matmul(y)
        # cand = self.selector_q(xs).reshape((len(xs),-1,self.embed_dim)).matmul(self.selector_k(y).transpose(2,1)).softmax(-1).matmul(y)
        # cand = torch.cat([xz,cand[:,1:]],dim=1)
        cand = torch.cat([xz,xs,cand[:,2:]],dim=1)
        # xq   = sel.matmul(cand)
        lptok = self.vocab(cand).log_softmax(-1)
        # import pdb; pdb.set_trace()
        # cent  = self.target_energy(lptok,x[:,i:i+1].repeat((1,self.mixture_count)))

        sel   = self.selector(xs).log_softmax(-1)
        xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
        # cent  = (cent+sel[:,0]).logsumexp(-1,keepdims=True)
        # mean(-1,keepdims=True)
        fs    = torch.cat([fs,xq],dim=1)

        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xs]
        return outer,inner

    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        for i in range(z.size(1)):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0


class RefillModelRNNAdditiveDirect(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
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

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)


        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.selector   = nn.Linear(embed_dim, mixture_count).to(self.device)
        # self.selector_2   = nn.Linear(embed_dim, embed_dim).to(self.device)
        self.selector_q = nn.Linear(embed_dim, mixture_count).to(self.device)
        # self.selector_q = nn.Linear(embed_dim, mixture_count*embed_dim).to(self.device)
        self.selector_k = nn.Linear(embed_dim, embed_dim).to(self.device)
        self.xkey_static = nn.Linear(embed_dim, 2).to(self.device)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim).to(self.device)
        # self.extractor = nn.Linear(embed_dim,state_count).to(self.device)
        # self.kernel    = nn.Bilinear(embed_dim,embed_dim,mixture_count).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
    # def _step(self,outer,inner):
    def _batch_init(self,zi,x,y,z):
        #### batch_init part
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        xs = self.init_state.weight.T[None,0:1]
        xs = xs.repeat((len(z),1,1))
        xs = self.norm(xs)
        # xs =
        y  = self.norm(y)
        z  = self.norm(z)
        fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xz = None
        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xs]
        return outer,inner

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xz,xs) = inner
        sel = None
        xz  = None

        xz = z[:,i:i+1]

        # xs = xs + self.updater(xz)
        # xs = xs + self.transition(xs) + self.updater(xz)
        # if i>=2:
        #     ### Uses the minus two token for prediction
        #     ### nearly a CNN fashion
        #     xs = self.transition(z[:,i-2:i-1])

        xs = self.transition(xs) + self.updater(xz)
        xs = self.norm(xs)
        ### The selector is a direct manner
        ### Input is under a static key, whereas the reservoir is self-keyed
        ### under a projection matrix
        ### This should allow a direction interaction between hidden and
        ### the output
        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey = torch.cat([xkey_static, xkey_dynamic],dim=1)
        sel = xs.matmul(xkey.transpose(2,1)).softmax(-1)
        cand = torch.cat([xz,xs,y],dim=1)
        xq   = sel.matmul(cand)
        #### I think the expectation aggregation here is too harsh...
        #### it's probably better to calculate emission probability by sampling, then aggregate the
        #### Emission.

        # xs = xs + self.transition(xs) + self.updater(xq)
        # xs = self.norm(xs)
        fs   = torch.cat([fs,xq],dim=1)
        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xs]
        return outer,inner



class RefillModelRNNAdditiveDirectMixing(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
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

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)


        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.selector   = nn.Linear(embed_dim, mixture_count).to(self.device)
        # self.selector_2   = nn.Linear(embed_dim, embed_dim).to(self.device)
        self.selector_q = nn.Linear(embed_dim, mixture_count).to(self.device)
        # self.selector_q = nn.Linear(embed_dim, mixture_count*embed_dim).to(self.device)
        self.selector_k = nn.Linear(embed_dim, embed_dim).to(self.device)
        self.xkey_static = nn.Linear(embed_dim, 2).to(self.device)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim).to(self.device)
        # self.extractor = nn.Linear(embed_dim,state_count).to(self.device)
        # self.kernel    = nn.Bilinear(embed_dim,embed_dim,mixture_count).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
    # def _step(self,outer,inner):
    def _batch_init(self,zi,x,y,z):
        #### batch_init part
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        xs = self.init_state.weight.T[None,0:1]
        xs = xs.repeat((len(z),1,1))
        xs = self.norm(xs)
        # xs =
        y  = self.norm(y)
        z  = self.norm(z)
        fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xz = None
        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xs]
        return outer,inner

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xz,xs) = inner
        sel = None
        xz  = None

        xz = z[:,i:i+1]

        # xs = xs + self.updater(xz)
        # xs = xs + self.transition(xs) + self.updater(xz)
        # if i>=2:
        #     ### Uses the minus two token for prediction
        #     ### nearly a CNN fashion
        #     xs = self.transition(z[:,i-2:i-1])

        xs = self.transition(xs) + self.updater(xz)
        xs = self.norm(xs)
        ### The selector is a direct manner
        ### Input is under a static key, whereas the reservoir is self-keyed
        ### under a projection matrix
        ### This should allow a direction interaction between hidden and
        ### the output
        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)


        sel   = xs.matmul(xkey.transpose(2,1)).log_softmax(-1)
        cand  = torch.cat([xz,xs,y],dim=1)
        lptok = self.vocab(cand).log_softmax(-1)

        xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
        # xq   = sel.matmul(cand)
        #### I think the expectation aggregation here is too harsh...
        #### it's probably better to calculate emission probability by sampling, then aggregate the
        #### Emission.

        # xs = xs + self.transition(xs) + self.updater(xq)
        # xs = self.norm(xs)
        fs   = torch.cat([fs,xq],dim=1)
        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xs]
        return outer,inner



    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        for i in range(z.size(1)):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0


class RefillModelRNNAttention(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):
        super().__init__(device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
        state_count = 5
        # state_count = 15
        self.device = device
        self.total_length = total_length
        self.min_len = min_len
        self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        self.state_count = state_count

        # x = nn.Linear(embed_dim*state_count,total_length).to(self.device)
        # self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.anchor = nn.Linear(embed_dim,mixture_count).to(self.device)
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.selector   = nn.Linear(embed_dim*mixture_count, mixture_count).to(self.device)
        self.selector_q = nn.Linear(embed_dim, mixture_count).to(self.device)
        # self.selector_q = nn.Linear(embed_dim, mixture_count*embed_dim).to(self.device)
        self.selector_k = nn.Linear(embed_dim, embed_dim).to(self.device)
        # self.extractor = nn.Linear(embed_dim,state_count).to(self.device)
        # self.kernel    = nn.Bilinear(embed_dim,embed_dim,mixture_count).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.project    = nn.Linear(embed_dim,embed_dim*mixture_count).to(self.device)
        self.project_v  = nn.Linear(embed_dim,embed_dim*mixture_count).to(self.device)
        self.state_key  = nn.Linear(mixture_count,embed_dim).to(self.device)

    # def _step(self,outer,inner):
    def _batch_init(self,zi,x,y,z):
        #### batch_init part
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
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xz = None
        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xs]
        return outer,inner

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xz,xs) = inner
        sel = None
        xz  = None
        xz = z[:,i:i+1]
        L =len(z)

        ### the hidden states can be considered as updating the most relevant neuron according to the
        ### the presented vector is memorised into the reservoir according to their
        ### instead of simple summation, we construct a KV container that
        xs = xs + self.transition(xs)
        # xs = xs + self.updater(xz)
        ### update the most relevant projection of the
        # xs = self.norm(xs)
        xsk = self.state_key.weight.T[None]

        xsaff = xz.matmul(xsk.transpose(2,1)).softmax(-1)
        xsd   = xsaff.transpose(2,1).matmul(xz)
        xs = xs + xsd
        xs = self.norm(xs)


        sel = self.selector(xs.reshape((L,1,-1))).softmax(-1)
        # print(xs.shape)
        # print(sel.shape)
        # import pdb; pdb.set_trace()
        ### maybe make query vector a function of state?
        cand = self.selector_q.weight[None].matmul(self.selector_k(y).transpose(2,1)).softmax(-1).matmul(y)
        # cand = self.selector_q(xs).reshape((len(xs),-1,self.embed_dim))
        # cand = self.norm(cand).matmul(self.selector_k(y).transpose(2,1)).softmax(-1).matmul(y)
        # cand = torch.cat([xz,cand[:,1:]],dim=1)
        cand = torch.cat([xz,cand[:,1:]],dim=1)
        xq   = sel.matmul(cand)
        xs   = xs + self.updater(xq)

        fs   = torch.cat([fs,xq],dim=1)
        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xs]
        return outer,inner



class RefillModelNGRAM(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
        state_count = 5
        # state_count = 15
        self.device = device
        self.total_length = total_length
        self.min_len = min_len
        self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        self.state_count = state_count

        # x = nn.Linear(embed_dim*state_count,total_length).to(self.device)
        # self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.anchor = nn.Linear(embed_dim,mixture_count).to(self.device)
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.kernel_size = kernel_size = 5
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.selector   = nn.Linear(embed_dim*kernel_size, mixture_count).to(self.device)
        self.selector_q = nn.Linear(embed_dim, mixture_count).to(self.device)
        # self.selector_q = nn.Linear(embed_dim*kernel_size, mixture_count*embed_dim).to(self.device)
        self.selector_k = nn.Linear(embed_dim, embed_dim).to(self.device)
        # self.extractor = nn.Linear(embed_dim,state_count).to(self.device)
        # self.kernel    = nn.Bilinear(embed_dim,embed_dim,mixture_count).to(self.device)
        self.init_state = nn.Linear(kernel_size, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.project    = nn.Linear(embed_dim,embed_dim*mixture_count).to(self.device)
        self.project_v  = nn.Linear(embed_dim,embed_dim*mixture_count).to(self.device)
        self.state_key  = nn.Linear(mixture_count,embed_dim).to(self.device)

    # def _step(self,outer,inner):
    def _batch_init(self,zi,x,y,z):
        #### batch_init part
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        L = len(z)
        xs = self.init_state.weight.T[None,:].repeat((L,1,1))*0
        xs = self.norm(xs)
        # xs =
        y  = self.norm(y)
        z  = self.norm(z)
        fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xz = None
        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xs]
        return outer,inner

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xz,xs) = inner
        sel = None
        xz  = None
        xz  = z[:,i:i+1]
        L   = len(z)

        xs = torch.cat([self.transition(xs[:,:-1]),self.updater(xz)],dim=1)
        ### the hidden states can be considered as updating the most relevant neuron according to the
        ### the presented vector is memorised into the reservoir according to their
        ### instead of simple summation, we construct a KV container that


        sel = self.selector(xs.reshape((L,1,-1))).softmax(-1)
        # import pdb; pdb.set_trace()
        ### maybe make query vector a function of state?
        cand = self.selector_q.weight[None].matmul(self.selector_k(y).transpose(2,1)).softmax(-1).matmul(y)
        # cand = self.selector_q(xs.reshape((L,1,-1))).reshape((len(xs),-1,self.embed_dim))
        # cand = cand.matmul(self.selector_k(y).transpose(2,1)).softmax(-1).matmul(y)
        # cand = torch.cat([xz,cand[:,1:]],dim=1)
        cand = torch.cat([xz,cand[:,1:]],dim=1)
        xq   = sel.matmul(cand)
        # xs   = xs + self.updater(xq)

        fs   = torch.cat([fs,xq],dim=1)
        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xs]
        return outer,inner

class RefillModelRNNGRU(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):
        super().__init__(device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
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
        # self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater=nn.Linear(embed_dim*2,embed_dim).to(self.device)
        self.resetter=nn.Linear(embed_dim*2,embed_dim).to(self.device)
        self.hid1=nn.Linear(embed_dim,embed_dim).to(self.device)
        self.hid2=nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.updater_v  = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.query_init = nn.Linear(1,embed_dim).to(self.device)
        # self.query_init      = nn.Linear(1,embed_dim).to(self.device)
        # self.query_k    = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.query_v    = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.callback = lambda zi,x,y,z,sel: None

    '''Try implementing Gated-Recurrent-Unit'''
    # def _step(self,outer,inner):
    def _batch_init(self,zi,x,y,z):
        #### batch_init part
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        xs = self.init_state.weight.T[None,0:1]
        xs = self.norm(xs)
        xs = xs.repeat((len(z),1,1))
        # xs =
        y  = self.norm(y)
        z  = self.norm(z)
        fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xz = None
        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xs]
        return outer,inner

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xz,xs) = inner
        sel = None
        xz  = None
        xz = z[:,i:i+1]


        hax = h_and_x = torch.cat([xz,xs],-1)
        v_up  = torch.sigmoid(self.updater(hax))
        v_res = torch.sigmoid(self.resetter(hax))
        v_hidn= torch.tanh(self.hid1(xz)+self.hid2(v_res * xs))
        xs = (1-v_up)*xs+ v_up * v_hidn

        sel = self.selector(xs).softmax(-1)
        ### maybe make query vector a function of state?
        cand = self.selector_q.weight[None].matmul(self.selector_k(y).transpose(2,1)).softmax(-1).matmul(y)
        # cand = self.selector_q(xs).reshape((len(xs),-1,self.embed_dim)).matmul(self.selector_k(y).transpose(2,1)).softmax(-1).matmul(y)
        cand = torch.cat([xz,cand[:,1:]],dim=1)
        xq   = sel.matmul(cand)
        fs   = torch.cat([fs,xq],dim=1)
        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xs]
        return outer,inner


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

    def loss(self,zi,x,y,z,):
        ### state init
        fs = torch.tensor([],requires_grad=True).to(self.device)
        # import pdb; pdb.set_trace()
        ismask = (z == self.mask_token_idx).double().topk(k=3,dim=1)[1]
        xperm = torch.tensor([random.sample(range(y.size(1)),y.size(1)) for _ in range(y.size(0))]).to(self.device)
        # yperm = torch.gather(y,index=xperm[:,:,None].repeat((1,1,y.size(2))),dim=1)
        yperm = torch.gather(y,index=xperm,dim=1)
        xz = torch.scatter(z,index=ismask,src=yperm,dim=1)

        # fs = self.norm(self.embed(x))
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
