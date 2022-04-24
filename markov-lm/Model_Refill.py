import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim





class RefillModelRNNBase(nn.Module):
    '''
    Symbolic Module that defines Loss function

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

        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.selector   = nn.Linear(embed_dim, mixture_count).to(self.device)
        self.selector_q = nn.Linear(embed_dim, mixture_count).to(self.device)
        # self.selector_q = nn.Linear(embed_dim, mixture_count*embed_dim).to(self.device)
        self.selector_k = nn.Linear(embed_dim, embed_dim).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)

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


class RefillModelRNNAdditiveDirectSampling(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):

        state_count = 1
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
        # state_count = 15
        self.device = device
        self.total_length = total_length
        self.min_len = min_len
        self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        self.state_count = state_count

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        #### the vector vocabulary needs a more flexibile management [TBC]
        #### here the extra tokens are tied to this model
        self.embed_extra= nn.Embedding(5,embed_dim).to(self.device)

        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2).to(self.device)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.fs_type    = 'sampled_traj'
        self.K = 100

    def _batch_init(self,zi,x,y,z):
        ### k is number of chain
        #### batch_init part
        ### state init
        k = self.K
        z = self.embed(z)
        B = y.size(0)
        self.max_mem = M = 9

        ### construct reservoir
        y = self.embed(y)
        y = torch.cat([y,self.embed_extra(torch.zeros((B, M -y.shape[1])).long().to(self.device)) ],dim=1)
        y = y[:,None].repeat((1,k,1,1))

        # xs = torch.cat([y,init_state],dim=1)
        xs = self.init_state.weight.T[None,0:1]
        xs = xs.repeat((len(z),k,1))
        xs = self.norm(xs)
        # xs =

        ## the reservoir set is now a variable
        y  = self.norm(y)
        z  = self.norm(z)
        fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        # xz = None
        lp = 0*xs[:,:,0]
        outer = [zi,x,y,z,fs]
        inner = [i,sel,lp,xs]
        return outer,inner

    def _step(self,outer,inner):
        '''

        ##### The sampling model factorize the loss function as
        ##### an expectation to allow flexibility in the underlying
        ##### probability function.
        #####
        ##### instead of directly calculate L = f(x,y,z)
        ##### we caclculate L = E_p(t)[f(x,y,z,t)]
        ##### where t is a hidden variable generated from sampling
        ##### and the expectation is approximated by normalising the probability
        ##### of generated samples


        For expecation models, must maintain
        a probabilistic representation of the states

        here the probabilistic part is the reservoir vector
        if the t-step predicts a END state, then the selected
        vector needs to be deleted from the reservoir.

        Each state needs to be associated with a score, so
        that later can be used to calculate expecation


        adds a dimension after batch to indicates which sample is this

        This model is not quite working yet, unable to learn when to copy the input
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,lp,xs) = inner

        ### xs needs to be expanded into
        sel = None
        xz  = None

        K = self.K
        B = len(xs)
        E = self.embed_dim
        M = self.max_mem

        ##shared
        xz = z[:,i:i+1]
        # print(xs.shape)
        # print(xz.shape)
        xs = self.transition(xs) + self.updater(xz)
        xs = self.norm(xs)

        ##### Action Selection
        ### selector can be static or dynamic
        ### Input is under a static key, whereas the reservoir is self-keyed
        ### under a projection matrix
        ### This should allow a direction interaction between hidden and
        ### the output

        ## static keys are shared
        xkey_static = self.xkey_static.weight[None,None, :2,:self.embed_dim].repeat((len(z),K,1,1))

        ### dynamic keys are dependent on the
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=2)


        ### calculate actions
        ### einsum maybe slow, but maybe reshape?
        sel   = xs .reshape((B*K,-1,E)).matmul(xkey.reshape((B*K,-1,E)).transpose(2,1))
        # sel   = sel / 0.1
        sel   = sel.log_softmax(-1)
        sel   = sel.reshape((B,K,-1))
        cand  = torch.cat([ xz[:,None].repeat((1,K,1,1)),xs[:,:,None],y],dim=2)
        ###

        ### performs random sampling and record loglp
        ### temperature should be a parameter?
        xpc = sel.exp().cumsum(-1)
        xr = torch.rand(sel.shape[:-1]+(1,)).to(self.device)
        _ ,xi = (xr<=xpc).max(dim=-1)

        lp = lp + torch.gather(sel,index=xi[:,:,None],dim=-1)[:,:,0]

        #### modify reservoir to erase accessed memory
        xq = torch.gather(cand,index=xi[:,:,None,None].repeat((1,1,1,E)),dim=2)

        xi = (xi - 2)% self.max_mem
        # 0.3980,  1.7758, -0.6467,
        y = torch.scatter(y,index=xi[:,:,None,None].repeat((1,1,1,E)),dim=2,src=self.embed_extra.weight[0:1,None,None,:].repeat((B,K,1,1)))
        # y = self.norm(y)


        ### maybe we can still use a mixture in the calculation of emission? [TBC]


        # xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
        #### I think the expectation aggregation here is too harsh...
        #### it's probably better to calculate emission probability by sampling, then aggregate the
        #### Emission.
        #### indeed it's much better to not aggregate the expectation

        #### fs represents lptok
        fs   = torch.cat([fs,xq],dim=2)
        outer = [zi,x,y,z,fs]
        inner = [i,sel,lp,xs]
        return outer,inner


    def loss(self,zi,x,y,z):
        return self._loss(zi,x,y,z,out='loss')
    # grad_loss = loss

    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj grad_loss'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        for i in range(z.size(1)):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        self.callback_end(outer)
        (zi,x,y,z, sampled_traj) = outer
        (i,sel,lp,xs) = inner
        lptok = self.vocab(sampled_traj).log_softmax(-1)
        xc = cent  = torch.gather(lptok,index=x[:,None,:,None].repeat((1,self.K,1,1)),dim=-1).mean((-1))
        # import pdb; pdb.set_trace()

        if out == 'grad_loss':
            ## REINFORCE
            wloss = xc.mean(-1) * lp
            # wloss = xc.mean(-1)/
            # wloss = -xc.mean(-1) * lp.log_softmax(-1)
            # wloss = -xc.mean(-1) * lp.softmax(-1)
            return wloss
        # cent = self.target_energy(lptok,x)
        # loss  = -cent.mean(-1)
        loss  = -xc.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0
    def grad_loss(self,zi,x,y,z):
        return self._loss(zi,x,y,z,out='grad_loss')


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

        state_count = 1
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
        # state_count = 15
        self.device = device
        self.total_length = total_length
        self.min_len = min_len
        self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        self.state_count = state_count

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2).to(self.device)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.fs_type    = 'lptok'

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

        # if i>=2:
        #     ### Uses the minus two token for prediction
        #     ### nearly a CNN fashion
        #     xs = self.transition(z[:,i-2:i-1])
        xz = z[:,i:i+1]

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
        #### I think the expectation aggregation here is too harsh...
        #### it's probably better to calculate emission probability by sampling, then aggregate the
        #### Emission.
        #### indeed it's much better to not aggregate the expectation

        #### fs represents lptok
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


class RefillModelRNNAdditiveDirectMixingWithGate(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):

        state_count = 1
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
        # state_count = 15
        self.device = device
        self.total_length = total_length
        self.min_len = min_len
        self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        self.state_count = state_count

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2).to(self.device)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.update_gate = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.fs_type    = 'lptok'

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

        # if i>=2:
        #     ### Uses the minus two token for prediction
        #     ### nearly a CNN fashion
        #     xs = self.transition(z[:,i-2:i-1])
        xz = z[:,i:i+1]
        xg = self.update_gate(xz)[:,:,0:1].sigmoid()
        xs = self.transition(xs) + xg * self.updater(xz)
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
        #### I think the expectation aggregation here is too harsh...
        #### it's probably better to calculate emission probability by sampling, then aggregate the
        #### Emission.
        #### indeed it's much better to not aggregate the expectation

        #### fs represents lptok
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


class RefillModelRNNAdditiveDirectMixingBidirectional(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):

        state_count = 1
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
        # state_count = 15
        self.device = device
        self.total_length = total_length
        self.min_len = min_len
        self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        self.state_count = state_count

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2).to(self.device)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.transition_gate = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.fs_type    = 'lptok'

    def _batch_init(self,zi,x,y,z):
        #### batch_init part
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        xsa = self.init_state.weight.T[None,0:1]
        xsa = xsa.repeat((len(z),z.size(1),1))
        xsa = self.norm(xsa)

        y  = self.norm(y)
        z  = self.norm(z)
        fs = self.vocab(xsa)*0
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xz = None
        outer = [zi, x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xz,xsa) = inner
        sel = None
        xz  = None

        xz = z[:,i:i+1]

        L = x.shape[1]
        xs = 0
        if i>=1:
            xsl = xsa[:,i-1:i]
            xg = 1
            # xg = xs.matmul(self.transition_gate.weight).matmul(xsl.transpose(2,1)).sigmoid()
            xs = xs + xg * xsl.matmul(self.transition.weight)
        if i+1<=L-1:
            xsl = xsa[:,i+1:i+2]
            xg = 1
            # xg = xsl.matmul(self.transition_gate.weight).matmul(xs.transpose(2,1)).sigmoid()
            xs = xs + xg * xsl.matmul(self.transition.weight.transpose(1,0  ))

        xs = xs + self.updater(xz)
        xs = self.norm(xs)

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)

        sel   = xs.matmul(xkey.transpose(2,1)).log_softmax(-1)
        cand  = torch.cat([xz,xs,y],dim=1)
        lptok = self.vocab(cand).log_softmax(-1)

        xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
        # fs   = torch.cat([fs,xq],dim=1)
        fs  = torch.scatter(fs,index=(xq*0).long()+i,src=xq,dim=1)
        # import pdb; pdb.set_trace()
        xsa = torch.scatter(xsa,index=(xs*0).long()+i,src=xs,dim=1)

        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        for i in range(L):
            inner[0]=L-1-i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        for i in range(L):
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


class RefillModelRNNAdditiveDirectMixingWithAttention(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):

        state_count = 1
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
        # state_count = 15
        self.device = device
        self.total_length = total_length
        self.min_len = min_len
        self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        self.state_count = state_count

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2).to(self.device)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_kernel = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_energy = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_prob   = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.fs_type    = 'lptok'

    def _batch_init(self,zi,x,y,z):
        #### batch_init part
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        xsa = self.init_state.weight.T[None,0:1]
        xsa = xsa.repeat((len(z),z.size(1),1))
        xsa = self.norm(xsa)

        y  = self.norm(y)
        z  = self.norm(z)
        fs = self.vocab(xsa)*0
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xz = None
        outer = [zi, x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xz,xsa) = inner
        sel = None
        xz  = None

        xz = z[:,i:i+1]

        L = x.shape[1]

        xs = xsa[:,i:i+1]
        ##### adds attention interaction term
        # import pdb; pdb.set_trace()
        att = self.att_kernel(xs).matmul(xsa.transpose(2,1)).softmax(-1)
        att = att * (self.att_prob(xs)[:,:,0:1].sigmoid())
        val = att.matmul(xsa)
        xs  = xs+ (val).matmul(self.att_energy.weight.T)

        if i>=1:
            xsl = xsa[:,i-1:i]
            xs = xs + xsl.matmul(self.transition.weight)
        if i+1<=L-1:
            xsl = xsa[:,i+1:i+2]
            xs = xs + xsl.matmul(self.transition.weight.transpose(1,0  ))

        xs = xs + self.updater(xz)


        # att = self.att_kernel(xsa).matmul(xs.transpose(2,1)).softmax(-2)
        # val = att.transpose(2,1).matmul(xsa)
        # xs = xs+ val.matmul(self.att_energy.weight)
        # self.att_energy(val)


        xs = self.norm(xs)

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)

        sel   = xs.matmul(xkey.transpose(2,1)).log_softmax(-1)
        cand  = torch.cat([xz,xs,y],dim=1)
        lptok = self.vocab(cand).log_softmax(-1)

        xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
        # fs   = torch.cat([fs,xq],dim=1)
        fs  = torch.scatter(fs,index=(xq*0).long()+i,src=xq,dim=1)
        # import pdb; pdb.set_trace()
        xsa = torch.scatter(xsa,index=(xs*0).long()+i,src=xs,dim=1)

        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        for i in range(L):
            inner[0]=L-1-i
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


class RefillModelRNNAdditiveDirectMixingBidirectionalFixedEmission(RefillModelRNNAdditiveDirectMixingBidirectional):
    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xz,xsa) = inner
        sel = None
        xz  = None

        xz = z[:,i:i+1]

        L = x.shape[1]
        xs = 0
        if i>=1:
            xsl = xsa[:,i-1:i]
            xs = xs + xsl.matmul(self.transition.weight)
        if i+1<=L-1:
            xsl = xsa[:,i+1:i+2]
            xs = xs + xsl.matmul(self.transition.weight.transpose(1,0  ))

        xs = xs + (xz)
        xs = self.norm(xs)

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)

        sel   = xs.matmul(xkey.transpose(2,1)).log_softmax(-1)
        cand  = torch.cat([xz,xs,y],dim=1)
        lptok = self.vocab(cand).log_softmax(-1)

        xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
        # fs   = torch.cat([fs,xq],dim=1)
        fs  = torch.scatter(fs,index=(xq*0).long()+i,src=xq,dim=1)
        # import pdb; pdb.set_trace()
        xsa = torch.scatter(xsa,index=(xs*0).long()+i,src=xs,dim=1)

        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        for i in range(L):
            inner[0]=L-1-i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        for i in range(L):
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


class RefillModelRNNAdditiveSweeping(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):

        state_count = 1
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
        # state_count = 15
        self.device = device
        self.total_length = total_length
        self.min_len = min_len
        self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        self.state_count = state_count

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2).to(self.device)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.fs_type    = 'lptok'

    def _batch_init(self,zi,x,y,z):
        #### batch_init part
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        xsa = self.init_state.weight.T[None,0:1]
        xsa = xsa.repeat((len(z),z.size(1),1))
        xsa = self.norm(xsa)

        y  = self.norm(y)
        z  = self.norm(z)
        fs = self.vocab(xsa)*0
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xz = None
        outer = [zi, x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xz,xsa) = inner
        sel = None
        xz  = None

        xz = z[:,i:i+1]

        L = x.shape[1]
        xs = 0
        if i>=1:
            xsl = xsa[:,i-1:i]
            xs = xs + xsl.matmul(self.transition.weight)
        if i+1<=L-1:
            xsl = xsa[:,i+1:i+2]
            xs = xs + xsl.matmul(self.transition.weight.transpose(1,0   ))

        xs = xs + self.updater(xz)
        xs = self.norm(xs)

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)

        sel   = xs.matmul(xkey.transpose(2,1)).log_softmax(-1)
        cand  = torch.cat([xz,xs,y],dim=1)
        lptok = self.vocab(cand).log_softmax(-1)

        xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
        fs  = torch.scatter(fs,index=(xq*0).long()+i,src=xq,dim=1)
        xsa = torch.scatter(xsa,index=(xs*0).long()+i,src=xs,dim=1)

        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        for i in range(L):
            inner[0]=L-1-i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        for i in range(L):
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

class RefillModelThreeWayRNNAdditiveSweeping(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):

        state_count = 1
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
        # state_count = 15
        self.device = device
        self.total_length = total_length
        self.min_len = min_len
        self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        self.state_count = state_count

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2).to(self.device)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        E = self.embed_dim
        # self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim*embed_dim).to(self.device)
        self.W = nn.Parameter(self.transition.weight.reshape((E,E,E)))

        self.transition2 = nn.Linear(embed_dim,embed_dim*embed_dim).to(self.device)
        self.W2 = nn.Parameter(self.transition.weight.reshape((E,E,E)))
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.fs_type    = 'lptok'

    def _batch_init(self,zi,x,y,z):
        #### batch_init part
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        xsa = self.init_state.weight.T[None,0:1]
        xsa = xsa.repeat((len(z),z.size(1),1))
        xsa = self.norm(xsa)

        xsb = self.init_state.weight.T[None,1:2]
        xsb = xsb.repeat((len(z),z.size(1),1))
        xsb = self.norm(xsb)


        y  = self.norm(y)
        z  = self.norm(z)
        fs = self.vocab(xsa)*0
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xz = None
        outer = [zi, x,y,z,fs]
        inner = [i,sel,xsb,xsa]
        return outer,inner

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xsb,xsa) = inner
        sel = None
        xz  = None

        xz = z[:,i:i+1]

        L = x.shape[1]
        xs = 0
        assert  i>=1

        beta = 0.1

        if self.sense ==1:
            ### infer from left
            xsl = xsa[:,i-1:i]
            xx = torch.tensordot(xsl,self.W,1)
            xxx = (xx * xz [:,:,:,None]).sum(2)

        # xs = xs + xsl.matmul(self.transition.weight)
        # xsr = xsa[:,i:i+1]
        # xs = xs + xsl.matmul(self.transition.weight.transpose(1,0   ))
        # import pdb; pdb.set_trace()
        # xs = xs + self.updater(xz)
            xsr = xsa[:,i:i+1]
            xsr = self.norm(xsr + beta*xxx)

            xx = torch.tensordot(xsl,self.W2,1)
            xxx = (xx * xsr[:,:,None]).sum(2)
            xsa = torch.scatter(xsa,index=(xxx*0).long()+i,src=xsr,dim=1)

            # xs = self.norm(xxx+xsb[:,i:i+1])
            # xsb = torch.scatter(xsb,index=(xs*0).long()+i,src=xs,dim=1)
            xs = self.norm(xxx)
            # xsb = torch.scatter(xsb,index=(xs*0).long()+i,src=xs,dim=1)
            # xsb = torch.scatter
            # print(i)
        elif self.sense==-1 :
            # print(i)
            xsr = xsa[:,i:i+1]
            xsl = xsa[:,i-1:i]
            xx = torch.tensordot(xsr,self.W.transpose(2,0),1)
            xxx = (xx * xz [:,:,:,None]).sum(2)

        # xs = xs + xsl.matmul(self.transition.weight)
        # xsr = xsa[:,i:i+1]
        # xs = xs + xsl.matmul(self.transition.weight.transpose(1,0   ))
        # import pdb; pdb.set_trace()
        # xs = xs + self.updater(xz)

            xsl = self.norm(xsl + beta* xxx)

            xx = torch.tensordot(xsl,self.W2,1)
            xxx = (xx * xsr[:,:,None]).sum(2)
            # xs = self.norm(xxx)
            xsa = torch.scatter(xsa,index=(xxx*0).long()+i-1,src=xsl,dim=1)

            # xs = self.norm(xxx+xsb[:,i:i+1])
            # xsb = torch.scatter(xsb,index=(xs*0).long()+i,src=xs,dim=1)
            xs = self.norm(xxx)



        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)

        sel   = xs.matmul(xkey.transpose(2,1)).log_softmax(-1)
        cand  = torch.cat([xz,xs,y],dim=1)
        lptok = self.vocab(cand).log_softmax(-1)

        xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
        fs  = torch.scatter(fs,index=(xq*0).long()+i,src=xq,dim=1)

        outer = [zi,x,y,z,fs]
        inner = [i,sel,xsb,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        self.sense = 1
        for i in range(1,L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        self.sense = -1
        for i in range(0,L-1):
            inner[0]=L-1-i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        # for i in range(L):
        #     inner[0]=i
        #     outer,inner = self._step(outer,inner) #### do what ever with your hidden state
        #     self.callback_step(outer,inner)

        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0

class RefillModelThreeWayRNNAdditiveSweeping2(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):

        state_count = 1
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
        # state_count = 15
        self.device = device
        self.total_length = total_length
        self.min_len = min_len
        self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        self.state_count = state_count

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2).to(self.device)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        E = self.embed_dim
        # self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim*embed_dim).to(self.device)
        self.W = nn.Parameter(self.transition.weight.reshape((E,E,E)))

        self.transition2 = nn.Linear(embed_dim,embed_dim*embed_dim).to(self.device)
        self.W2 = nn.Parameter(self.transition.weight.reshape((E,E,E)))
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.fs_type    = 'lptok'

    def _batch_init(self,zi,x,y,z):
        #### batch_init part
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        xsa = self.init_state.weight.T[None,0:1]
        xsa = xsa.repeat((len(z),z.size(1),1))
        xsa = self.norm(xsa)

        xsb = self.init_state.weight.T[None,1:2]
        xsb = xsb.repeat((len(z),z.size(1),1))
        xsb = self.norm(xsb)


        y  = self.norm(y)
        z  = self.norm(z)
        fs = self.vocab(xsa)*0
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xz = None
        outer = [zi, x,y,z,fs]
        inner = [i,sel,xsb,xsa]
        return outer,inner

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xsb,xsa) = inner
        sel = None
        xz  = None

        xz = z[:,i:i+1]

        L = x.shape[1]
        xs = 0
        assert  i>=1

        beta = 0.01

        if self.sense ==1:
            ### infer from left
            xsl = xsa[:,i-1:i]
            xx = torch.tensordot(xsl,self.W,1)
            xxx = (xx * xz [:,:,:,None]).sum(2)

        # xs = xs + xsl.matmul(self.transition.weight)
        # xsr = xsa[:,i:i+1]
        # xs = xs + xsl.matmul(self.transition.weight.transpose(1,0   ))
        # import pdb; pdb.set_trace()
        # xs = xs + self.updater(xz)
            xsr = xsa[:,i:i+1]
            xsr = self.norm(xsr + beta*xxx)

            xx = torch.tensordot(xsl,self.W2,1)
            xxx = (xx * xsr[:,:,None]).sum(2)
            xsa = torch.scatter(xsa,index=(xxx*0).long()+i,src=xsr,dim=1)

            # xs = self.norm(xxx+xsb[:,i:i+1])
            # xsb = torch.scatter(xsb,index=(xs*0).long()+i,src=xs,dim=1)
            xs = self.norm(xxx)
            # xsb = torch.scatter(xsb,index=(xs*0).long()+i,src=xs,dim=1)
            # xsb = torch.scatter
            # print(i)
        elif self.sense==-1 :
            # print(i)
            xsr = xsa[:,i:i+1]
            xsl = xsa[:,i-1:i]
            xx = torch.tensordot(xsr,self.W.transpose(2,0),1)
            xxx = (xx * xz [:,:,:,None]).sum(2)

        # xs = xs + xsl.matmul(self.transition.weight)
        # xsr = xsa[:,i:i+1]
        # xs = xs + xsl.matmul(self.transition.weight.transpose(1,0   ))
        # import pdb; pdb.set_trace()
        # xs = xs + self.updater(xz)

            xsl = self.norm(xsl + beta* xxx)

            xx = torch.tensordot(xsl,self.W2,1)
            xxx = (xx * xsr[:,:,None]).sum(2)
            # xs = self.norm(xxx)
            xsa = torch.scatter(xsa,index=(xxx*0).long()+i-1,src=xsl,dim=1)

            # xs = self.norm(xxx+xsb[:,i:i+1])
            # xsb = torch.scatter(xsb,index=(xs*0).long()+i,src=xs,dim=1)
            xs = self.norm(xxx)



        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)

        sel   = xs.matmul(xkey.transpose(2,1)).log_softmax(-1)
        cand  = torch.cat([xz,xs,y],dim=1)
        lptok = self.vocab(cand).log_softmax(-1)

        xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
        fs  = torch.scatter(fs,index=(xq*0).long()+i,src=xq,dim=1)

        outer = [zi,x,y,z,fs]
        inner = [i,sel,xsb,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        self.sense = 1
        for i in range(1,L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        self.sense = -1
        for i in range(0,L-1):
            inner[0]=L-1-i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        # for i in range(L):
        #     inner[0]=i
        #     outer,inner = self._step(outer,inner) #### do what ever with your hidden state
        #     self.callback_step(outer,inner)

        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0


class RefillModelCrossRNNAdditiveSweeping(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):

        state_count = 1
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
        # state_count = 15
        self.device = device
        self.total_length = total_length
        self.min_len = min_len
        self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        self.state_count = state_count

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.xkey_static = nn.Linear((embed_dim//2)**2, 2).to(self.device)
        self.xkey_dynamic= nn.Linear(embed_dim, (embed_dim//2)**2).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,(embed_dim//2)**2).to(self.device)
        self.fs_type    = 'lptok'

    def _batch_init(self,zi,x,y,z):
        #### batch_init part
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        xsa = self.init_state.weight.T[None,0:1]
        xsa = xsa.repeat((len(z),z.size(1),1))
        xsa = self.norm(xsa)

        y  = self.norm(y)
        z  = self.norm(z)
        fs = self.vocab(xsa)*0
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xz = None
        outer = [zi, x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner

    def _step(self,outer,inner):
        '''

        Outer should be non-mutable?
        '''

        (zi, x,y,z,fs) = outer
        (i,sel,xz,xsa) = inner
        sel = None
        xz  = None

        xz = z[:,i:i+1]

        L = x.shape[1]
        dx = 0
        xs = xsa[:,i:i+1]
        if i>=1:
            xsl = xsa[:,i-1:i]
            dx = dx +xsl.matmul(self.transition.weight)
        if i+1<=L-1:
            xsl = xsa[:,i+1:i+2]
            dx = dx + xsl.matmul(self.transition.weight.transpose(1,0   ))

        ### use xz to parametrise the interaction between two state vector
        E = self.embed_dim
        B = len(x)
        xs1,xs2 = xs[:,:,:E//2],xs[:,:,E//2:]
        W = self.updater(xz).reshape((B,E//2,E//2))
#        xs1 = xs1+
        dx1 = dx[:,:,:E//2]+xs2.matmul(W)
        dx2 = dx[:,:,E//2:]+xs1.matmul(W.transpose(2,1))
        xs1 = self.norm(xs1 + dx1)
        xs2 = self.norm(xs2 + dx2)

        xs  = torch.cat([xs1,xs2],dim=-1)

        xkey_static = self.xkey_static.weight[None,:1,:].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        xscross = ( xs1.transpose(2,1)*xs2).reshape((B,1,-1))

        # xp = xs1[:,0].matmul(xkey)
        # import pdb; pdb.set_trace()
        sel   = xscross.matmul(xkey.transpose(2,1)).log_softmax(-1)
        cand  = torch.cat([xz,y],dim=1)
        lptok = self.vocab(cand).log_softmax(-1)

        xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
        fs  = torch.scatter(fs,index=(xq*0).long()+i,src=xq,dim=1)
        xsa = torch.scatter(xsa,index=(xs*0).long()+i,src=xs,dim=1)

        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)

        for i in range(L):
            inner[0]=L-1-i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)

        for i in range(L):
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



class RefillModelRNNAdditiveSweepingWithResidual(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx):

        state_count = 1
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
        # state_count = 15
        self.device = device
        self.total_length = total_length
        self.min_len = min_len
        self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        self.state_count = state_count

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2).to(self.device)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.fs_type    = 'lptok'

    def _batch_init(self,zi,x,y,z):
        #### batch_init part
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        xsa = self.init_state.weight.T[None,0:1]
        xsa = xsa.repeat((len(z),z.size(1),1))
        xsa = self.norm(xsa)

        y  = self.norm(y)
        z  = self.norm(z)
        fs = self.vocab(xsa)*0
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xz = None
        outer = [zi, x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xz,xsa) = inner
        sel = None
        xz  = None

        xz = z[:,i:i+1]

        L = x.shape[1]
        xs = xsa[:,i:i+1]
        if i>=1:
            xsl = xsa[:,i-1:i]
            xs = xs + xsl.matmul(self.transition.weight)
        if i+1<=L-1:
            xsl = xsa[:,i+1:i+2]
            xs = xs + xsl.matmul(self.transition.weight.transpose(1,0   ))

        xs = xs + self.updater(xz)
        xs = self.norm(xs)

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)

        sel   = xs.matmul(xkey.transpose(2,1)).log_softmax(-1)
        cand  = torch.cat([xz,xs,y],dim=1)
        lptok = self.vocab(cand).log_softmax(-1)

        xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
        fs  = torch.scatter(fs,index=(xq*0).long()+i,src=xq,dim=1)
        xsa = torch.scatter(xsa,index=(xs*0).long()+i,src=xs,dim=1)

        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        for i in range(L):
            inner[0]=L-1-i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        for i in range(L):
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

class RefillModelMixtureRNNSweepingOldEmission(RefillModelRNNBase):
    '''
    Implements a MRF ICM where the base interaction is
    a dynamic MRF, at each location, the MRF is selected
    from a mixture to avoid forcing nearest neighbor interaction.

    To achieve this, we needs to manage
         - a set of vectors, varied at each step
         - a distribution, varied at each step

    '''
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        mask_token_idx):

        state_count  = 1
        total_length = 1
        min_len      = 1
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
        # state_count = 15
        self.device = device
        self.total_length  = total_length
        self.min_len       = min_len
        self.mixture_count = mixture_count
        self.K = mixture_count
        self.embed_dim   = embed_dim
        self.state_count = state_count

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed        = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.embed_extra  = nn.Embedding(10,embed_dim,).to(self.device)
        self.n_step       = min_len
        self.xkey_static  = nn.Linear(embed_dim, 2).to(self.device)
        self.xkey_dynamic = nn.Linear(embed_dim, embed_dim).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        # self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.We         = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.Wr         = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.fs_type    = 'lptok'



    def _batch_init(self,zi,x,y,z):
        '''
        :type zi: shape of (B,)
        :type x:  shape of (B, L)
        :type y:  shape of (B, K), K <= self.mixture_count
        :type z:  shape of (B, L)
        :type xsa: shape of (B, L, E). this is stored as vector state
        :type xl: shape of (B, L, K, E) this is the left image
        :type xr: shape of (B, L, K, E) this is the right image
        :type xlr: shape of (B, L, K, E) this is the left right image
        ## xl and xr can be combined as long as the sweeping is simple
        '''
        #### batch_init part
        #### state init
        z = self.embed(z)
        y = self.embed(y)

        ### construct left rightimage
        K = self.K
        B = y.size(0)
        L = x.size(1)
        E = self.embed_dim
        # xlr = torch.cat([y,self.embed_extra(torch.zeros((B, K -y.shape[1])).long().to(self.device)) ],dim=1).reshape((B,1,K,E)).repeat((1,L,1,1))
        # xlr = torch.cat([self.embed_extra(torch.zeros((B, K)).long().to(self.device)) ],dim=1).reshape((B,1,K,E)).repeat((1,L,1,1))
        xlr = self.embed_extra(torch.arange(K).long().to(self.device)).reshape((1,1,K,E)).repeat((B,L,1,1))

        ### inferred from xlr thus empty for now
        xsa  = xlr[:,:,0,:]*0

        ### fs should store emitted token propensity, better to do in parallel

        y  = self.norm(y)
        z  = self.norm(z)
        fs = self.vocab(xsa)*0

        i   = -1
        sel = None
        lr  = 1 ## 1 for right, -1 for left
        outer = [zi, x,y,z,fs]
        inner = [(i,lr),  sel,xlr,xsa]
        return outer,inner

    def _step(self,outer,inner):
        '''
        :param fs: token emission at each position
        :param xsa: hidden state at each position
        :param xz:  input state at current position
        :param sel: selection probability at current position
        :param i: pointer to current position
        :param x: ground truth output sequence
        :param y: candidate sets containing masked tokens
        :param z: input sequence with tokens masked

        ### needs a

        ### Unlike simple MRF
            The states propagate to left and right without equalisation
            to ensure long-range interaction.
        ### At each position, we choose one chain to place a node
            this node takes a value that minimise the LRC 3-way interaction,
            and is thus determined by summing the conditional interaction.
            The choice is manifested as a probability, parametrising the
            update of L and R channels

        ### pseudo-code
            for pos in seg:
                lp = f(L(t-1),R(t+1),Ct)
                xp = softmax(lp)  ### choose the most likely node
                                        ### When propagating to right, left image is updated
                                        ### When propagating to left, right image is updated
                                        ### the most likely node is calculated as the one maximising the
                                        ### three way interaction

                L(t) = L(xp,L(t-1))     ### propagate left image towards right
                                        ### if a node is absent,  directly copy
                                        ### if a node is present, copy the node vector into left image
                                        ### scatter into t

                R(t) = R(xp,R(t+1))     ### propagate right image towards left
                                        ### if a node is absent,  directly copy
                                        ### if a node is present, copy the node vector into right image
                                        ### scatter into t

                E(t) = g(xp,L,R)        ### scatter into t

                ### problem:
                #### how to model emission? Should select between copying and the candidates

                - old emission: extract the state vector, pass through a dynamic selector, mix the output likelihood.
                    - this is closer to the old model
                    - easier for comparision for now

                - new emission: emit between the original token and the node token, selected by the node.
                    - this is much simpler since the selection is only between 2 nodes.

                #### the Wr matrix needs to be different for each chain...
        '''

        (zi,x,  y,  z, fs) = outer
        ((i,lr), sel,  xlr,   xsa) = inner



        '''
        Computing probability as maximising exp(xWL+xWR+xWz)

                 z
                 |
                (We)
                 |
        L--(Wr)--x--(Wr)--R

        '''


        xz = z[:,i:i+1,None]
        # trying for all K components
        # (B,L,K,E)
        B  = x.size(0)
        L  = x.shape[1]
        K  = self.K
        E  = self.embed_dim

        xss = 0.
        ### always have Z
        xss = xss + xz.matmul(self.We.weight.T)
        if i>=1:
            xl = xlr[:,i-1:i]
            xss = xss + xl.matmul(self.Wr.weight.T)
        if i+1<=L-1:
            xr = xlr[:,i+1:i+2]
            xss = xss + xr.matmul(self.Wr.weight)
        xss = self.norm(xss)

        xe = 0
        xe = xe + (xss * xz.matmul(self.We.weight.T)).mean(-1)
        if i>=1:
            xl = xlr[:,i-1:i]
            xe  = xe  + (xss * xl.matmul(self.Wr.weight.T)).mean(-1)
        if i+1<=L-1:
            xr = xlr[:,i+1:i+2]
            xe  = xe  + (xss * xr.matmul(self.Wr.weight)).mean(-1)
        xp = xe.softmax(-1)[:,:,:,None]  ### select the best node
        # import pdb; pdb.set_trace()
        #### propagate the images
        if lr==1:
            ## copy from left or new vector
            val = xlr[:,i-1:i] if i>=1 else 0.
            val  = xp*xss + (1-xp) * val
            xlr  = torch.scatter(xlr,index=(val*0).long()+i,src=val,dim=1)
        elif lr==-1:
            ### copy from right or new vector
            val = xlr[:,i+1:i+2] if i+1<=L-1 else 0.
            val  = xp*xss + (1-xp) * val
            xlr  = torch.scatter(xlr,index=(val*0).long()+i,src=val,dim=1)
        else:
            raise Exception(f'Unknown lr={lr}')




        ### old emission function to populate (fs)
        if 1:
            xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
            xkey_dynamic= self.xkey_dynamic(y)
            xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)

            xs = self.norm((xss*xp).sum(2))

            sel   = xs.matmul(xkey.transpose(2,1)).log_softmax(-1)
            cand  = torch.cat([xz[:,0],xs,y],dim=1)
            lptok = self.vocab(cand).log_softmax(-1)

            xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
            fs    = torch.scatter(fs,index=(xq*0).long()+i,src=xq,dim=1)

        outer = [zi,x,y,z,fs]
        # inner = d
        inner = [i,xp.reshape((B,1,K)).log(),xlr,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        for i in range(L):
            inner[0]=(i,1)
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        for i in range(L):
            inner[0]=(L-1-i,-1)
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


class RefillModelMixtureRNNSweepingOldEmissionDifferentTransition(RefillModelRNNBase):
    '''
    Implements a MRF ICM where the base interaction is
    a dynamic MRF, at each location, the MRF is selected
    from a mixture to avoid forcing nearest neighbor interaction.

    To achieve this, we needs to manage
         - a set of vectors, varied at each step
         - a distribution, varied at each step

    '''
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        mask_token_idx):

        state_count  = 1
        total_length = 1
        min_len      = 1
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
        # state_count = 15
        self.device = device
        self.total_length  = total_length
        self.min_len       = min_len
        self.mixture_count = mixture_count
        self.K = mixture_count
        self.embed_dim   = embed_dim
        self.state_count = state_count

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed        = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.embed_extra  = nn.Embedding(10,embed_dim,).to(self.device)
        self.n_step       = min_len
        self.xkey_static  = nn.Linear(embed_dim, 2).to(self.device)
        self.xkey_dynamic = nn.Linear(embed_dim, embed_dim).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        # self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.We         = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.Wr         = nn.Linear(mixture_count*embed_dim,embed_dim).to(self.device)
        self.fs_type    = 'lptok'



    def _batch_init(self,zi,x,y,z):
        '''
        :type zi: shape of (B,)
        :type x:  shape of (B, L)
        :type y:  shape of (B, K), K <= self.mixture_count
        :type z:  shape of (B, L)
        :type xsa: shape of (B, L, E). this is stored as vector state
        :type xl: shape of (B, L, K, E) this is the left image
        :type xr: shape of (B, L, K, E) this is the right image
        :type xlr: shape of (B, L, K, E) this is the left right image
        ## xl and xr can be combined as long as the sweeping is simple
        '''
        #### batch_init part
        #### state init
        z = self.embed(z)
        y = self.embed(y)

        ### construct left rightimage
        K = self.K
        B = y.size(0)
        L = x.size(1)
        E = self.embed_dim
        # xlr = torch.cat([y,self.embed_extra(torch.zeros((B, K -y.shape[1])).long().to(self.device)) ],dim=1).reshape((B,1,K,E)).repeat((1,L,1,1))
        # xlr = torch.cat([self.embed_extra(torch.zeros((B, K)).long().to(self.device)) ],dim=1).reshape((B,1,K,E)).repeat((1,L,1,1))
        xlr = self.embed_extra(torch.arange(K).long().to(self.device)).reshape((1,1,K,E)).repeat((B,L,1,1))

        ### inferred from xlr thus empty for now
        xsa  = xlr[:,:,0,:]*0

        ### fs should store emitted token propensity, better to do in parallel

        y  = self.norm(y)
        z  = self.norm(z)
        fs = self.vocab(xsa)*0

        i   = -1
        sel = None
        lr  = 1 ## 1 for right, -1 for left
        outer = [zi, x,y,z,fs]
        inner = [(i,lr),  sel,xlr,xsa]
        return outer,inner

    def _step(self,outer,inner):
        '''
        :param fs: token emission at each position
        :param xsa: hidden state at each position
        :param xz:  input state at current position
        :param sel: selection probability at current position
        :param i: pointer to current position
        :param x: ground truth output sequence
        :param y: candidate sets containing masked tokens
        :param z: input sequence with tokens masked

        ### needs a

        ### Unlike simple MRF
            The states propagate to left and right without equalisation
            to ensure long-range interaction.
        ### At each position, we choose one chain to place a node
            this node takes a value that minimise the LRC 3-way interaction,
            and is thus determined by summing the conditional interaction.
            The choice is manifested as a probability, parametrising the
            update of L and R channels

        ### pseudo-code
            for pos in seg:
                lp = f(L(t-1),R(t+1),Ct)
                xp = softmax(lp)  ### choose the most likely node
                                        ### When propagating to right, left image is updated
                                        ### When propagating to left, right image is updated
                                        ### the most likely node is calculated as the one maximising the
                                        ### three way interaction

                L(t) = L(xp,L(t-1))     ### propagate left image towards right
                                        ### if a node is absent,  directly copy
                                        ### if a node is present, copy the node vector into left image
                                        ### scatter into t

                R(t) = R(xp,R(t+1))     ### propagate right image towards left
                                        ### if a node is absent,  directly copy
                                        ### if a node is present, copy the node vector into right image
                                        ### scatter into t

                E(t) = g(xp,L,R)        ### scatter into t

                ### problem:
                #### how to model emission? Should select between copying and the candidates

                - old emission: extract the state vector, pass through a dynamic selector, mix the output likelihood.
                    - this is closer to the old model
                    - easier for comparision for now

                - new emission: emit between the original token and the node token, selected by the node.
                    - this is much simpler since the selection is only between 2 nodes.

                #### the Wr matrix needs to be different for each chain...
        '''

        (zi,x,  y,  z, fs) = outer
        ((i,lr), sel,  xlr,   xsa) = inner



        '''
        Computing probability as maximising exp(xWL+xWR+xWz)

                 z
                 |
                (We)
                 |
        L--(Wr)--x--(Wr)--R

        '''


        xz = z[:,i:i+1,None]
        # trying for all K components
        # (B,L,K,E)
        B  = x.size(0)
        L  = x.shape[1]
        K  = self.K
        E  = self.embed_dim

        Wr = self.Wr.weight.reshape((K,E,E))
        # import pdb; pdb.set_trace()


        xss = 0.
        ### always have Z
        xss = xss + xz.matmul(self.We.weight.T)
        def trans_wr(xr,Wr):
            x = xr[:,:,:,:,None] * Wr[None,None]
            x = x.sum(-2)
            return x

        if i>=1:
            xl = xlr[:,i-1:i]
            xss = xss + trans_wr(xl,Wr)
            # xl.matmul(self.Wr.weight.T)
        if i+1<=L-1:
            xr = xlr[:,i+1:i+2]
            xss = xss +  trans_wr(xr,Wr.transpose(2,1))
            # xr.matmul(self.Wr.weight)
        xss = self.norm(xss)

        xe = 0
        xe = xe + (xss * xz.matmul(self.We.weight.T)).mean(-1)
        if i>=1 and 1==1:
            xl = xlr[:,i-1:i]
            xe  = xe  + (xss * trans_wr(xl,Wr)).mean(-1)
            # xe  = xe  + (xss * xl.matmul(self.Wr.weight.T)).mean(-1)
        if i+1<=L-1 and -1==-1:
            xr = xlr[:,i+1:i+2]
            xe  = xe  + (xss *  trans_wr(xr,Wr.transpose(2,1))).mean(-1)

        xp = xe.softmax(-1)[:,:,:,None]  ### select the best node
        # import pdb; pdb.set_trace()
        #### propagate the images
        if lr==1:
            ## copy from left or new vector
            val = xlr[:,i-1:i] if i>=1 else 0.
            val  = xp*xss + (1-xp) * val
            xlr  = torch.scatter(xlr,index=(val*0).long()+i,src=val,dim=1)
        elif lr==-1:
            ### copy from right or new vector
            val = xlr[:,i+1:i+2] if i+1<=L-1 else 0.
            val  = xp*xss + (1-xp) * val
            xlr  = torch.scatter(xlr,index=(val*0).long()+i,src=val,dim=1)
        else:
            raise Exception(f'Unknown lr={lr}')




        ### old emission function to populate (fs)
        if 0:
            xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
            xkey_dynamic= self.xkey_dynamic(y)
            xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)

            xs = self.norm((xss*xp).sum(2))

            sel   = xs.matmul(xkey.transpose(2,1)).log_softmax(-1)
            cand  = torch.cat([xz[:,0],xs,y],dim=1)
            lptok = self.vocab(cand).log_softmax(-1)

            xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
            fs    = torch.scatter(fs,index=(xq*0).long()+i,src=xq,dim=1)

        if 1:

            xkey_static = self.xkey_static.weight[None,:1,:self.embed_dim].repeat((len(z),1,1))
            xkey_dynamic= self.xkey_dynamic(y)
            xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)

            #### all xss are equally possible to be the vector, thus should consider all
            #### consider all possiblilities equally. expand the nodes then compute lptok
            ####
            # xss[:,0].matmul(xkey.transpose(2,1))
            sel = xss[:,0].matmul(xkey.transpose(2,1))
            sel = sel.log_softmax(-1)
            sel = sel + xp[:,0].log()
            sel = sel.logsumexp(-2,keepdims=True)
            # sel = sel.reshape((B,-1)).log_softmax(-1).reshape(sel.shape)

            # import pdb; pdb.set_trace()
            # xs = self.norm((xss*xp).sum(2))
            # sel   = xs.matmul(xkey.transpose(2,1)).log_softmax(-1)
            cand  = torch.cat([xz[:,0],y],dim=1)
            lptok = self.vocab(cand).log_softmax(-1)

            xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
            fs    = torch.scatter(fs,index=(xq*0).long()+i,src=xq,dim=1)



        outer = [zi,x,y,z,fs]
        inner = [i,xp.reshape((B,1,K)).log(),xlr,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        for i in range(L):
            inner[0]=(i,1)
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        for i in range(L):
            inner[0]=(L-1-i,-1)
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


class RefillModelMixtureRNNSweepingOldEmission2(RefillModelMixtureRNNSweepingOldEmission):
    pass

class RefillModelMixtureRNNSweepingNewEmission(RefillModelRNNBase):
    '''
    Implements a MRF ICM where the base interaction is
    a dynamic MRF, at each location, the MRF is selected
    from a mixture to avoid forcing nearest neighbor interaction.

    To achieve this, we needs to manage
         - a set of vectors, varied at each step
         - a distribution, varied at each step

    '''
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        mask_token_idx):

        state_count  = 1
        total_length = 1
        min_len      = 1
        super().__init__(
            device,
            graph_dim,
            embed_dim,
            mixture_count,
            state_count,
            total_length,
            min_len,
            mask_token_idx)
        # state_count = 15
        self.device = device
        self.total_length  = total_length
        self.min_len       = min_len
        self.mixture_count = mixture_count
        self.K = mixture_count
        self.embed_dim   = embed_dim
        self.state_count = state_count

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed        = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.embed_extra  = nn.Embedding(10,embed_dim,).to(self.device)
        self.n_step       = min_len
        self.xkey_static  = nn.Linear(embed_dim, 2).to(self.device)
        self.xkey_dynamic = nn.Linear(embed_dim, embed_dim).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        # self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.We         = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.Wr         = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.fs_type    = 'lptok'



    def _batch_init(self,zi,x,y,z):
        '''
        :type zi: shape of (B,)
        :type x:  shape of (B, L)
        :type y:  shape of (B, K), K <= self.mixture_count
        :type z:  shape of (B, L)
        :type xsa: shape of (B, L, E). this is stored as vector state
        :type xl: shape of (B, L, K, E) this is the left image
        :type xr: shape of (B, L, K, E) this is the right image
        :type xlr: shape of (B, L, K, E) this is the left right image
        ## xl and xr can be combined as long as the sweeping is simple
        '''
        #### batch_init part
        #### state init
        z = self.embed(z)
        y = self.embed(y)

        ### construct left rightimage
        K = self.K
        B = y.size(0)
        L = x.size(1)
        E = self.embed_dim
        xlr = torch.cat([y,self.embed_extra(torch.zeros((B, K -y.shape[1])).long().to(self.device)) ],dim=1).reshape((B,1,K,E)).repeat((1,L,1,1))

        ### inferred from xlr thus empty for now
        xsa  = xlr[:,:,0,:]*0

        ### fs should store emitted token propensity, better to do in parallel

        y  = self.norm(y)
        z  = self.norm(z)
        fs = self.vocab(xsa)*0

        i   = -1
        sel = None
        lr  = 1 ## 1 for right, -1 for left
        outer = [zi, x,y,z,fs]
        inner = [(i,lr),  sel,xlr,xsa]
        return outer,inner

    def _step(self,outer,inner):
        '''
        :param fs: token emission at each position
        :param xsa: hidden state at each position
        :param xz:  input state at current position
        :param sel: selection probability at current position
        :param i: pointer to current position
        :param x: ground truth output sequence
        :param y: candidate sets containing masked tokens
        :param z: input sequence with tokens masked

        ### needs a

        ### Unlike simple MRF
            The states propagate to left and right without equalisation
            to ensure long-range interaction.
        ### At each position, we choose one chain to place a node
            this node takes a value that minimise the LRC 3-way interaction,
            and is thus determined by summing the conditional interaction.
            The choice is manifested as a probability, parametrising the
            update of L and R channels

        ### pseudo-code
            for pos in seg:
                lp = f(L(t-1),R(t+1),Ct)
                xp = softmax(lp)  ### choose the most likely node
                                        ### When propagating to right, left image is updated
                                        ### When propagating to left, right image is updated
                                        ### the most likely node is calculated as the one maximising the
                                        ### three way interaction

                L(t) = L(xp,L(t-1))     ### propagate left image towards right
                                        ### if a node is absent,  directly copy
                                        ### if a node is present, copy the node vector into left image
                                        ### scatter into t

                R(t) = R(xp,R(t+1))     ### propagate right image towards left
                                        ### if a node is absent,  directly copy
                                        ### if a node is present, copy the node vector into right image
                                        ### scatter into t

                E(t) = g(xp,L,R)        ### scatter into t

                ### problem:
                how to model emission? Should select between copying and the candidates

                old emission: extract the state vector, pass through a dynamic selector, mix the output likelihood.
                    - this is closer to the old model

                new emission: emit between the original token and the node token, selected by the node.
                    - this is much simpler since the selection is only between 2 nodes.

        '''

        (zi,x,  y,  z, fs) = outer
        ((i,lr), sel,  xlr,   xsa) = inner



        '''
        Computing probability as maximising exp(xWL+xWR+xWz)

                 z
                 |
                (We)
                 |
        L--(Wr)--x--(Wr)--R

        '''


        xz = z[:,i:i+1,None]
        # trying for all K components
        # (B,L,K,E)
        B  = x.size(0)
        L  = x.shape[1]
        K  = self.K
        E  = self.embed_dim

        xss = 0.
        ### always have Z
        xss = xss + xz.matmul(self.We.weight.T)
        if i>=1:
            xl = xlr[:,i-1:i]
            xss = xss + xl.matmul(self.Wr.weight.T)
        if i+1<=L-1:
            xr = xlr[:,i+1:i+2]
            xss = xss + xr.matmul(self.Wr.weight)
        xss = self.norm(xss)

        xe = 0
        xe = xe + (xss * xz.matmul(self.We.weight.T)).mean(-1)
        if i>=1:
            xl = xlr[:,i-1:i]
            xe  = xe  + (xss * xl.matmul(self.Wr.weight.T)).mean(-1)
        if i+1<=L-1:
            xr = xlr[:,i+1:i+2]
            xe  = xe  + (xss * xr.matmul(self.Wr.weight)).mean(-1)
        xp = xe.softmax(-1)[:,:,:,None]  ### select the best node

        #### propagate the images
        if lr==1:
            ## copy from left or new vector
            val  = xlr[:,i-1:i] if i>=1 else 0.
            val  = xp*xss + (1-xp) * val
            val  = self.norm(val)
            xlr  = torch.scatter(xlr,index=(val*0).long()+i,src=val,dim=1)
        elif lr==-1:
            ### copy from right or new vector
            val  = xlr[:,i+1:i+2] if i+1<=L-1 else 0.
            val  = xp*xss + (1-xp) * val
            val  = self.norm(val)
            xlr  = torch.scatter(xlr,index=(val*0).long()+i,src=val,dim=1)
        else:
            raise Exception(f'Unknown lr={lr}')




        ### new emission function to populate (fs)
        if 1:
            # xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
            xs    = self.norm((xss*xp).sum(2))
            cand  = torch.cat([xz[:,0],xs,],dim=1)
            xkey  = xkey_dynamic= self.xkey_dynamic(cand)

            sel   = xkey[:,:,0:1].transpose(2,1).log_softmax(-1)
            lptok = self.vocab(cand).log_softmax(-1)

            xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
            fs    = torch.scatter(fs,index=(xq*0).long()+i,src=xq,dim=1)

        if 0:
            # xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
            # xs    = self.norm((xss*xp).sum(2))
            # cand  = torch.cat([xz[:,0],xs,],dim=1)

            # import pdb; pdb.set_trace()
            cand = torch.cat([ xz[:,0],xss[:,0]],dim=1)
            # xkey_static = self.xkey_static.weight[None,:21:self.embed_dim].repeat((len(z),1,1))
            xe   = torch.cat([self.xkey_dynamic(xz[:,0])[:,:,0:1], xe],dim=2)
            sel  = xe.log_softmax(-1)
            # import pdb; pdb.set_trace()
            # xkey = xkey_dynamic= self.xkey_dynamic(cand)
            # sel   = xs.matmul(xkey.transpose(2,1)).log_softmax(-1)
            lptok = self.vocab(cand).log_softmax(-1)

            xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
            fs    = torch.scatter(fs,index=(xq*0).long()+i,src=xq,dim=1)

        '''
        Epoch: 45
        ModelClassName: RefillModelMixtureRNNSweepingNewEmission
        Training Loss: 0.3286665997334889
        Testing Loss: 0.3549345460805026

        '''
        outer = [zi,x,y,z,fs]
        inner = [(i,lr),sel,xlr,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        for i in range(L):
            inner[0]=(i,1)
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        for i in range(L):
            inner[0]=(L-1-i,-1)
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
