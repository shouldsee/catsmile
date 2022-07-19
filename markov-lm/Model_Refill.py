import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


from transformers.models.bert.modeling_bert import BertLayer,BertConfig

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

class RefillModelRNNConvolveWithHiddenVector(RefillModelRNNBase):
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
        self.embed      = nn.Embedding(graph_dim,embed_dim,)
        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim)
        self.kernel_size = kernel_size = 4
        self.init_state = nn.Linear(mixture_count, embed_dim)
        self.transition = nn.Linear(embed_dim*1,embed_dim)
        self.transcore  = nn.Linear(embed_dim,embed_dim)
        self.updater    = nn.Linear(embed_dim,embed_dim)
        self.emittor    = nn.Linear(embed_dim,embed_dim)
        self.attention_probe = nn.Linear(embed_dim,kernel_size)
        self.mu = nn.Linear(embed_dim,1)
        self.KE = KE = embed_dim//kernel_size//2

        self.attention_head  = nn.Linear(embed_dim,kernel_size*KE)
        self.attention_head_l  = nn.Linear(kernel_size,embed_dim*KE)
        self.attention_head_r  = nn.Linear(kernel_size,embed_dim*KE)
        # self.attention_head_r  = nn.Linear(embed_dim,kernel_size*KE)
        # self.K = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.U = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.K2 = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.U2= nn.Linear(embed_dim//2,embed_dim).to(self.device)


        self.K = nn.Linear(embed_dim,embed_dim)
        self.U = nn.Linear(embed_dim,embed_dim)
        self.K2 = nn.Linear(embed_dim,embed_dim)
        self.U2= nn.Linear(embed_dim,embed_dim)

        self.att_prob   = nn.Linear(embed_dim,embed_dim)
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
        mu = self.mu.weight[None,0:1]+(xsa[:,0:1]*0)

        # mu = self.attention_head
        xsa = z
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        # xz = None
        xsa = xsa*0
        gradsq = xsa*0
        # xzw = self.updater.weight
        xzw = self.transcore.weight
        xzw = xzw[None,].repeat((len(z),1,1))
        # xw =
        outer = [zi, x,y,z,fs]
        inner = [i,mu,xzw,xsa]
        return outer,inner
    # def att_func(self,att):
    #     return att
    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,mu,xzw,xsa) = inner
        xzw= None
        # sel = None
        xz  = None

        E = xsa.size(-1)
        L = xsa.size(1)
        B = xsa.size(0)
        K = self.kernel_size
        KE =self.KE

        # wK = torch.eye(E).to(self.device)
        # wK = self.K.weight
        # wK = wK.matmul(wK.T)
        wT = self.transition.weight
        # wT = wT.matmul(wT.T)/wT.shape[1]**0.5
        zk = self.attention_probe.weight.T
        akl = self.attention_head_l.weight.T
        akr = self.attention_head_r.weight.T
        ak = akl.reshape((K,E,KE)).matmul(akr.reshape((K,KE,E))).reshape((E,K*E))/KE**0.5
        rzk = zk.std(0,keepdims=True)*K**0.5
        peki =  (xsa.matmul(zk/rzk).transpose(2,1))
        # peki = peki/5.
        # peki = peki.clip(-10,10)
        # import pdb; pdb.set_trace()
        pki = peki.softmax(-1)
        pik = pki.transpose(2,1)
        pkix  = pki.matmul(xsa)
        #(K,E,KE)
        mud = pkix.reshape((B,-1,K*E)).matmul(ak.T)/1./K

        akmu = mu.matmul(ak).reshape((B,-1,E))

        xnew=1. *(
            xsa.roll(1,1).matmul(wT)    +
            xsa.roll(-1,1).matmul(wT.T) +
            # xsa.roll(1,0).matmul(xzw)+
            # xsa.roll(0,1).matmul(self.transcore.weight)     +
            z.matmul(self.updater.weight.T) +
            (pik*(1-pik)*xsa.matmul(akmu.transpose(2,1))).matmul(zk.T)/K +
            pik.matmul(akmu)/K +
            self.transition.bias +
            0.
            )



        if i%2==1:
            xsa = 0.8 * xsa + 0.2 * xnew
            # xsa = xsa/(0.001 + mu.std(-1,keepdims=True))
            # xsa = 1.0 * xnew
            # xsa = 1.0 * xnew + 0.0 * xsa
        # xsa = 0.5 * xnew
        if i%2!=1:
            # mu = 1.0*mud
            mu = 0.8 *mu +  mud
            # mu = mu/(0.001 + mu.std(-1,keepdims=True))
            # xzwd = z.transpose(2,1).matmul(xsa)/z.shape[1]
            # xzwd = xsa.transpose(2,1).matmul(xsa)/z.shape[1]
            # xzw = 1.0*xzw + 0.0*xzwd
            # # xzw = 0.5*xzw + 0.5*xzwd
        # xzw = xzwd
        # print((xzw[0][:10][:10]*10).int())

        # print(mu.std())
        # if i+1==L:
        #     import pdb; pdb.set_trace()
        outer = [zi,x,y,z,fs]
        inner = [peki,mu,xzw,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        B = len(z)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)


        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (peki,mu,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()
        # print(100*(mu[0:2,0,:10].detach().cpu().int()).numpy())
        # print()

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)

        # .matmul(lptok)
        # lptok = self.vocab(cand).log_softmax(-1)
        # if i+1==L:
        # if 0:
        if zi.min()==0:
            for k,xx in dict(mu=mu,xsa=xsa,peki=peki).items():
                print(xx.shape)
                print((xx[:3,0:2,:10]*10).int())
                print((xx.std(0)[0:2,:10]))
                print(f'{k}_std:{xx.std()}')
                print(f'{k}_max:{xx.max()}')
                print(f'{k}_min:{xx.min()}')
                print()
            for k,v in self.named_parameters():

                # print(f'{k}:{self.embed.weight.max()}')
                print(f'{k}:\t\t{v.max():.3f}')
                # import pdb; pdb.set_trace()
            # print(self.state)
            print()


        # lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok
class RefillModelWithBert(RefillModelRNNBase):
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
        self.embed      = nn.Embedding(graph_dim,embed_dim,)
        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim)
        self.kernel_size = kernel_size = 4
        self.init_state = nn.Linear(mixture_count, embed_dim)
        self.transition = nn.Linear(embed_dim*1,embed_dim)
        self.transcore  = nn.Linear(embed_dim,embed_dim)
        self.updater    = nn.Linear(embed_dim,embed_dim)
        self.emittor    = nn.Linear(embed_dim,embed_dim)
        self.attention_probe = nn.Linear(embed_dim,kernel_size)
        self.mu = nn.Linear(embed_dim,1)
        self.KE = KE = embed_dim//kernel_size//2
        self.E = E = embed_dim

        self.attention_head  = nn.Linear(embed_dim,kernel_size*KE)
        self.attention_head_l  = nn.Linear(kernel_size,embed_dim*KE)
        self.attention_head_r  = nn.Linear(kernel_size,embed_dim*KE)
        # self.attention_head_r  = nn.Linear(embed_dim,kernel_size*KE)
        # self.K = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.U = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.K2 = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.U2= nn.Linear(embed_dim//2,embed_dim).to(self.device)


        self.K = nn.Linear(embed_dim,embed_dim)
        self.U = nn.Linear(embed_dim,embed_dim)
        self.K2 = nn.Linear(embed_dim,embed_dim)
        self.U2= nn.Linear(embed_dim,embed_dim)

        self.att_prob   = nn.Linear(embed_dim,embed_dim)
        bconf = BertConfig()
        bconf.hidden_size=E
        bconf.intermediate_size = E*2
        bconf.num_attention_heads = 10
        # self.blayer = blayer = BertLayer(bconf)
        self.blayer_list = nn.ModuleList([BertLayer(bconf) for _ in range(20)])
        # import pdb; pdb.set_trace()

        # self.blayer =
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
        mu = self.mu.weight[None,0:1]+(xsa[:,0:1]*0)

        # mu = self.attention_head
        xsa = z
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        # xz = None
        xsa = xsa
        gradsq = xsa*0
        # xzw = self.updater.weight
        xzw = self.transcore.weight
        xzw = xzw[None,].repeat((len(z),1,1))
        # xw =
        outer = [zi, x,y,z,fs]
        inner = [i,mu,xzw,xsa]
        return outer,inner
    # def att_func(self,att):
    #     return att
    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,mu,xzw,xsa) = inner
        peki = xsa

        E = xsa.size(-1)
        L = xsa.size(1)
        B = xsa.size(0)
        K = self.kernel_size
        KE =self.KE
        # for blayer in self.
        blayer = self.blayer_list[i//2]

        xsad = blayer(xsa)[0]
        # xsa = 0.0*xsa + 1.0*xsad
        xsa = 0.5*xsa + 0.5*xsad
#
        outer = [zi,x,y,z,fs]
        inner = [peki,mu,xzw,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        B = len(z)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)


        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (peki,mu,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()
        # print(100*(mu[0:2,0,:10].detach().cpu().int()).numpy())
        # print()

        # xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        # xkey_dynamic= self.xkey_dynamic(y)
        # xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # # import pdb; pdb.set_trace()
        # cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        # sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        # lptok = self.vocab(cand).log_softmax(-1)
        # lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)

        # .matmul(lptok)
        # lptok = self.vocab(cand).log_softmax(-1)
        # if i+1==L:
        # if 0:

        lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0

        if out=='loss': return loss
        assert 0


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

class RefillModelRNNAdditiveDirectEmission(RefillModelRNNBase):
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
        self.emittor = nn.Linear(embed_dim,embed_dim).to(self.device)
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


        # sel   = xs.matmul(xkey.transpose(2,1)).log_softmax(-1)
        # cand  = torch.cat([xz,xs,y],dim=1)
        lptok = self.vocab(self.emittor(xs)).log_softmax(-1)

        # xq    = (lptok+sel.transpose(2,1)).logsumexp(1,keepdims=True)
        #### I think the expectation aggregation here is too harsh...
        #### it's probably better to calculate emission probability by sampling, then aggregate the
        #### Emission.
        #### indeed it's much better to not aggregate the expectation

        #### fs represents lptok
        fs   = torch.cat([fs,lptok],dim=1)
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

class RefillModelRNNConvolveGrad(RefillModelRNNBase):
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
        self.wa = nn.Linear(embed_dim-1,embed_dim).to(self.device)
        self.wb = nn.Linear(embed_dim-1,embed_dim).to(self.device)
        self.wc = nn.Linear(embed_dim-1,embed_dim).to(self.device)
        self.wd = nn.Linear(embed_dim-1,embed_dim).to(self.device)
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
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

        wa = self.wa.weight.T
        wb = self.wb.weight.T
        wc = self.wc.weight.T
        wd = self.wd.weight.T
        xd = 0
        xi = torch.inverse((wa.T.matmul(wa)+wc.T.matmul(wc)+wd.T.matmul(wd)).T)
        # xd = xd - xsa.matmul ((wa.T.matmul(wa)+wc.T.matmul(wc)+wd.T.matmul(wd)).T)
        xd = xd - z.matmul((wa.T.matmul(wb)).T) - wa.T.matmul(self.wa.bias[:-1])
        xd = xd - xsa.roll(1, 1).matmul((wc.T.matmul(wd)).T) - wc.T.matmul(self.wb.bias[:-1])
        xd = xd - xsa.roll(1,-1).matmul((wd.T.matmul(wc)).T) - wd.T.matmul(self.wb.bias[:-1 ])
        xs = xd.matmul(xi)
        # xsa = xsa  + 0.5
        # xsa = xd

        # 0.01*xd
        xsa = 0.5* xsa + 0.5*xd
        # xsa = xnew
        # xsa + 0.5*xnew


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
        # for i in range(L):
        #     inner[0]=L-1-i
        #     outer,inner = self._step(outer,inner) #### do what ever with your hidden state
        #     self.callback_step(outer,inner)

        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,sel,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0


class RefillModelRNNConvolve(RefillModelRNNBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        mixture_count,
        state_count,
        total_length,
        min_len,
        mask_token_idx,
        use_mixture=1):

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
        self.device        = device
        self.total_length  = total_length
        self.min_len       = min_len
        self.mixture_count = mixture_count
        self.embed_dim     = embed_dim
        self.state_count   = state_count

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2).to(self.device)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim).to(self.device)
        kernel_size = 5
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.transcore  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_kernel = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_energy = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_prob   = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.use_mixture= use_mixture
        self.fs_type    = 'lptok'

    def _batch_init(self,zi,x,y,z):
        #### batch_init part
        ### state init
        z = self.embed(z)
        # xs = torch.cat([y,init_state],dim=1)
        xsa = self.init_state.weight.T[None,0:1]
        xsa = xsa.repeat((len(z),z.size(1),1))
        xsa = self.norm(xsa)
        if y is not None:
            y = self.embed(y)
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

        xnew=1./3 *(
            xsa.roll(1,1).matmul(self.transition.weight)+
            xsa.roll(-1,1).matmul(self.transition.weight.T)+
            xsa.roll(0,1).matmul(self.transcore.weight)+
            z.matmul(self.updater.weight.T)
            +self.transition.bias
            )
        # xsa = 0.5* xsa + 0.5*xnew
        xsa = xnew
        if i+1 == xsa.size(1):
            # print((xsa[0,:3,:30]).int())
            pass
        # xsa + 0.5*xnew


        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner

    def loss(self,zi,x,y,z,mask =None):
        return self._loss(zi,x,y,z,out='loss',mask =mask)

    def _loss(self,zi,x,y,z,out='loss',mask =None):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        # for i in range(L):
        #     inner[0]=L-1-i
        #     outer,inner = self._step(outer,inner) #### do what ever with your hidden state
        #     self.callback_step(outer,inner)

        self.callback_end(outer,)
        (zi,x,y,z,lptok) = outer
        (i,sel,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # f1 = self.vocab(xsa).log_softmax(-1)
        # # .matmul(self.emittor.weight)
        # f2 = self.vocab(z).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()
        if self.use_mixture:
            xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
            xkey_dynamic= self.xkey_dynamic(y)
            xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
            # import pdb; pdb.set_trace()
            cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
            sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
            lptok = self.vocab(cand).log_softmax(-1)
            lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)
        # import pdb; pdb.set_trace()
        else:
            if self.use_mask:
                _mask = mask[:,:,None].repeat((1,1, xsa.shape[2]))
                # lptok= torch.gather(lptok,index=mask[:,:,None].repeat((1,1,lptok.shape[2])),dim=1)
                lptok = self.vocab(torch.gather(xsa,index=_mask,dim=1).matmul(self.emittor.weight)).log_softmax(-1)
                x    = torch.gather(x,index=mask[:,:],dim=1)
                # print(lptok.shape,x.shape)
            else:
                lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)
        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0





class RefillModelRNNBigramMixture(RefillModelRNNBase):
    '''
    This does not work for some reason, probably because the bigram
    is not easily fittable and require large samples.
    bigram requires identifying the most possible bigram given the
    current hidden states then transmit the information.
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
        # embed_dim = embed_dim*5
        embed_dim = 50
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
        self.K = K = kernel_size = 10
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim//K,embed_dim//K).to(self.device)
        self.transcore  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.wl    = nn.Linear(embed_dim//K,embed_dim).to(self.device)
        self.wr    = nn.Linear(embed_dim//K,embed_dim).to(self.device)
        self.we    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.mul    = nn.Linear(embed_dim,embed_dim-1).to(self.device)
        self.mur    = nn.Linear(embed_dim,embed_dim-1).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)

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
        fs = z
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
        B = xsa.shape[0]
        L = xsa.shape[1]
        K = self.K
        # E = xsa.shape[2]
        # xsak = xsa.reshape((B,L,K,-1)).permute((0,2,1,3)).reshape((B*K,L,-1))
        _norm = lambda  xsa,dim=-1: xsa /(1+xsa.std(dim,keepdims=True))

        xsa = _norm(xsa)
        # xsak = xsak /(1+xsak.std(-1,keepdims=True))
        fs   = self.norm(fs)

        mul = self.mul.weight.T
        mur = self.mur.weight.T
        mul = _norm(mul,0)
        mur = _norm(mur,0)


        # xslk = xsak.roll(1,1)
        # xsrk = xsak.roll(-1,1)
        zk   = z[:,None].repeat((1,K,1,1)).reshape((B*K,L,-1))
        fsk  = fs[:,None].repeat((1,K,1,1)).reshape((B*K,L,-1))
        # reg =
        # fsk = zk
        tau = 1./xsa.shape[-1]**0.5/2.

        '''E = softmax(hik) * hik'''
        xsl = xsa.roll(1,1)
        xsr = xsa.roll(-1,1)
        hikl = 5*xsa @ mul
        hikr = 5*xsr @ mur
        # hik  = hik*5
        pikl = hikl.softmax(-1)
        pikr = hikr.softmax(-1)
        # xdl = pik @ mul.T
        # xdr = pik @ mur.T
        xsad = 0.
        xsad = xsad + pikr @ mul.T + (pikl @ mur.T).roll(1,1)
        # xsad = xsad + xdl + xdr.roll(1,1)
        # xsad = xsad + ((pik*(1-pik)*hik) @ mul.T + ((pik*(1-pik)*hik) @ mur.T).roll(1,1))
        xsad = xsad + z @ (self.updater.weight)
        # import pdb; pdb.set_trace()
        # xsad = _norm(xsad)

        xsa = 0.5* xsa + 0.5*xsad
        xsa = _norm(xsa)

        # import pdb; pdb.set_trace()
        # xsa = 1.0* xsa + 0.2*self.norm(xsad)
        # xsa = xsad
        DEBUG = '--debug' in sys.argv
        if DEBUG:
            if i+1 >= xsa.size(1) or i ==3:
                print(f'[{i}]')
                _x = xsa.square().mean(-1)[:5][:10]
                print((100*_x).int())
                print('[xsa]')

                _x = xsa[:2,:10,:10]
                print((100*_x).int())
                print('[xsa2]')


                _x = xsad[:2,:10,:10]
                print((100*_x).int())
                print('[xsad]')
                # _x = fs.square().mean(-1)[:5][:10]
                # print((100*_x).int())
                # _x = hik[:2,:10,]
                # print((100*_x).int())
                _x = pikl[:2,:10,]
                print((100*_x).int())
                # print((xsa[0,:3,:30]).int())
                pass
        # xsa + 0.5*xnew


        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj var'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        # for i in range(L):
        #     inner[0]=L-1-i
        #     outer,inner = self._step(outer,inner) #### do what ever with your hidden state
        #     self.callback_step(outer,inner)

        self.callback_end(outer)
        (zi,x,y,z,fs) = outer
        (i,sel,xz,xsa) = inner
        xsa = fs
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # f1 = self.vocab(xsa).log_softmax(-1)
        # # .matmul(self.emittor.weight)
        # f2 = self.vocab(z).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)
        # import pdb; pdb.set_trace()
        # lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        if out=='var': return inner,outer

        if out=='token': return lptok

        if out=='loss': return loss
        assert 0

    def get_hidden(self,zi,x,y,z,out='var'):
        return self._loss(zi,x,y,z,out='var')

class RefillModelRNNClusterAndRotate(RefillModelRNNBase):
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
        # embed_dim = embed_dim*5
        embed_dim = 50
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
        self.K = K = kernel_size = 10
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim//K,embed_dim//K).to(self.device)
        self.transcore  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.wl    = nn.Linear(embed_dim//K,embed_dim).to(self.device)
        self.wr    = nn.Linear(embed_dim//K,embed_dim).to(self.device)
        self.we    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)

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
        fs = z
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
        B = xsa.shape[0]
        L = xsa.shape[1]
        K = self.K
        # E = xsa.shape[2]
        xsak = xsa.reshape((B,L,K,-1)).permute((0,2,1,3)).reshape((B*K,L,-1))
        # xsak = self.norm(xsak)
        xsak = xsak /(1+xsak.std(-1,keepdims=True))
        fs   = self.norm(fs)

        wt = self.transition.weight.T
        wl = self.wl.weight.T
        wr = self.wr.weight.T
        we = self.we.weight.T
        # xsak.matmul(wt)

        xslk = xsak.roll(1,1)
        xsrk = xsak.roll(-1,1)
        zk   = z[:,None].repeat((1,K,1,1)).reshape((B*K,L,-1))
        fsk  = fs[:,None].repeat((1,K,1,1)).reshape((B*K,L,-1))
        # reg =
        # fsk = zk
        tau = 1./xsak.shape[-1]**0.5/2.


        #### calculate derivate wrt xk
        hik = (xslk.matmul(wt)*(xsak)).sum(-1) + (xslk.matmul(wl)*fsk).sum(-1) + (xsak.matmul(wr)*fsk).sum(-1) - (xslk*xsak).sum(-1)
        hik = tau * hik
        # import pdb; pdb.set_trace()
        pik = hik.reshape((B,K,L)).softmax(1).reshape((B*K,L,1))
        # [:,:,None]

        xdl = tau*(xslk + pik            * (xslk.matmul(wt)   + fsk.matmul(wr.T)            - xslk ))
        xdr = tau*(xsrk + pik.roll(-1,1) * (xsrk.matmul(wt.T) + fsk.roll(-1,1).matmul(wl.T) - xsrk ))
        # if i<=L//2:
        # # if i%2==0:
        #     xdr = 0.
        # else:
        #     xdl = 0.
        # if i%2==0:
        if 1:
            grad = xdl + xdr
            # grad = self.norm(grad)
            xsad = grad.reshape((B,K,L,-1)).permute((0,2,1,3)).reshape((B,L,-1))
            # xsa = 1.0* xsa + 0.1*xsad
            xsa = 0.5* xsa + 0.5*xsad
        # else:

            ### update fsk
            fsd = (pik * ( xslk @ wl+xsak @ wr)).reshape((B,K,L,-1)).sum(1) + z @ we.T
            fs = 0.5 * fs + 0.5 * tau* fsd
            # fs =  fsd

        # import pdb; pdb.set_trace()
        # xsa = 1.0* xsa + 0.2*self.norm(xsad)
        # xsa = xsad
        DEBUG = '--debug' in sys.argv
        if DEBUG:
            if i+1 >= xsa.size(1) or i ==3:
                _x = xsak.square().mean(-1)[:5][:10]
                print((100*_x).int())
                _x = fs.square().mean(-1)[:5][:10]
                print((100*_x).int())
                _x = pik[:10,:10,0]
                print((100*_x).int())
                # print((xsa[0,:3,:30]).int())
                pass
        # xsa + 0.5*xnew


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
        # for i in range(L):
        #     inner[0]=L-1-i
        #     outer,inner = self._step(outer,inner) #### do what ever with your hidden state
        #     self.callback_step(outer,inner)

        self.callback_end(outer)
        (zi,x,y,z,fs) = outer
        (i,sel,xz,xsa) = inner
        xsa = fs
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # f1 = self.vocab(xsa).log_softmax(-1)
        # # .matmul(self.emittor.weight)
        # f2 = self.vocab(z).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)
        # import pdb; pdb.set_trace()
        # lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0


class RefillModelRNNDynamicCluster(RefillModelRNNBase):
    '''
    Fitting a constrained vector whose projections describes abundance of vectors
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
        # embed_dim = embed_dim*5
        embed_dim = 50
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
        self.K = K = kernel_size = 10
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.transcore  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.wa    = nn.Linear(embed_dim,embed_dim*K).to(self.device)
        self.wr    = nn.Linear(embed_dim//K,embed_dim).to(self.device)
        self.we    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)

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
        fs = xsa[:,0:1]*0+1.
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
        B = xsa.shape[0]
        L = xsa.shape[1]
        E = xsa.shape[2]
        K = self.K
        # E = xsa.shape[2]
        # xsak = xsa.reshape((B,L,K,-1)).permute((0,2,1,3)).reshape((B*K,L,-1))
        # xsak = self.norm(xsak)
        # xsak = xsak /(1+xsak.std(-1,keepdims=True))
        _norm = lambda xsa: xsa /(1+xsa.std(-1,keepdims=True))
        xsa = _norm(xsa)
        fs  = _norm(fs)
        # self.norm(fs)

        wa = self.wa.weight.T
        wt = self.transition.weight.T
        # wl = self.wl.weight.T
        wr = self.wr.weight.T
        we = self.we.weight.T

        # xsak.matmul(wt)
        zwa = (fs @ wa ).reshape((B,K,E))
        zwa = _norm(zwa)
        hki = zwa @ xsa.transpose(2,1)
        # hki = hki/E**0.5
        hki = hki
        pki = hki.softmax(1)
        pik = pki.transpose(2,1)
        xsade = pik @ (zwa)
        # import pdb; pdb.set_trace()
        # pik =

        xsad=1./3 *(
            xsa.roll(1,1).matmul(self.transition.weight)+
            xsa.roll(-1,1).matmul(self.transition.weight.T)+
            xsa.roll(0,1).matmul(self.transcore.weight)+
            xsade+
            z.matmul(self.updater.weight.T)
            +self.transition.bias
            )


        xsa = 0.5* xsa + 0.5*xsad

        fsd= ((xsa[:,:,None,:] * pik.unsqueeze(-1)).reshape((B,L,K*E)) @ wa.reshape((E,K,E)).permute((1,2,0)).reshape((K*E,E)))
        fsd = fsd.mean(1,keepdims=True) + fs@we
        # import pdb; pdb.set_trace()
        fs = 0.5 * fs + 0.5*fsd
        xsa = _norm(xsa)
        fs  = _norm(fs)


        DEBUG = '--debug' in sys.argv
        if DEBUG:
            if i+1 >= xsa.size(1) or i ==-1:
                print(f'[i]={i}')
                _x = z
                _x = _x[0][:10,:10]
                print((100*_x).int())

                _x =pik
                _x = _x[0][:10,:10]
                print((100*_x).int())
                _x = fs
                _x = _x[:5,:10,:10]
                print((100*_x).int())
                _x = zwa
                _x = _x[0][:10,:10]
                print((100*_x).int())
                print()
                print()

                # _x = fs.square().mean(-1)[:5][:10]
                # print((100*_x).int())
                # _x = pik[:10,:10,0]
                # print((100*_x).int())
                # print((xsa[0,:3,:30]).int())
                pass

        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner

    def get_hidden(self,zi,x,y,z,out='var'):
        return self._loss(zi,x,y,z,out='var')
    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj var'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        # for i in range(L):
        #     inner[0]=L-1-i
        #     outer,inner = self._step(outer,inner) #### do what ever with your hidden state
        #     self.callback_step(outer,inner)

        self.callback_end(outer)
        (zi,x,y,z,fs) = outer
        (i,sel,xz,xsa) = inner
        # xsa = fs
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # f1 = self.vocab(xsa).log_softmax(-1)
        # # .matmul(self.emittor.weight)
        # f2 = self.vocab(z).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)
        # import pdb; pdb.set_trace()
        # lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        if out=='var': return inner,outer

        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0


class RefillModelRNNMixture(RefillModelRNNBase):
    '''
    Fitting a constrained vector whose projections describes abundance of vectors
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
        # embed_dim = embed_dim*5
        embed_dim = 50
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
        self.K = K = kernel_size = 10
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.transcore  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.wt    = nn.Linear(embed_dim,embed_dim*K).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)

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
        fs = xsa[:,0:1]*0+1.
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
        B = xsa.shape[0]
        L = xsa.shape[1]
        E = xsa.shape[2]
        K = self.K
        # E = xsa.shape[2]
        # xsak = xsa.reshape((B,L,K,-1)).permute((0,2,1,3)).reshape((B*K,L,-1))
        # xsak = self.norm(xsak)
        # xsak = xsak /(1+xsak.std(-1,keepdims=True))
        _norm = lambda xsa: xsa /(1+xsa.std(-1,keepdims=True))
        xsa = _norm(xsa)
        fs  = _norm(fs)
        # self.norm(fs)

        # wa = self.wa.weight.T
        wt = self.wt.weight.T
        xsr = xsa.roll(-1,1)
        xsl = xsa.roll( 1,1)
        # hik = ((xsa @ wt).reshape((B,L,K,E)) * xsr[:,:,None,:]).mean(-1)
        hik = ((xsa @ wt).reshape((B,L,K,E)) * xsr[:,:,None,:]).sum(-1)
        pik = hik.softmax(-1)
        pikl = pik.roll(1,1)

        xsad = 0.
        xsad = xsad + (pik.unsqueeze(-1)*xsr.unsqueeze(2)).reshape((B,L,K*E))@(wt.T)
        xsad = xsad + (pikl.unsqueeze(-1)*xsl.unsqueeze(2)).reshape((B,L,K*E))@(wt.reshape(E,K,E).permute((1,0,2)).reshape(K*E,E))

        # pik
        # xsad =
        # # xsak.matmul(wt)
        # zwa = (fs @ wa ).reshape((B,K,E))
        # zwa = _norm(zwa)
        # hki = zwa @ xsa.transpose(2,1)
        # # hki = hki/E**0.5
        # hki = hki
        # pki = hki.softmax(1)
        # pik = pki.transpose(2,1)
        # xsade = pik @ (zwa)
        # import pdb; pdb.set_trace()
        # pik =

        xsad=1./3 *(
        xsad +
        # xsad.roll()
            # xsa.roll(1,1).matmul(self.transition.weight)+
            # xsa.roll(-1,1).matmul(self.transition.weight.T)+
            xsa.roll(0,1).matmul(self.transcore.weight)+
            # xsade+
            z.matmul(self.updater.weight.T)
            +self.transition.bias
            )


        xsa = 0.5* xsa + 0.5*xsad
        #
        # fsd= ((xsa[:,:,None,:] * pik.unsqueeze(-1)).reshape((B,L,K*E)) @ wa.reshape((E,K,E)).permute((1,2,0)).reshape((K*E,E)))
        # fsd = fsd.mean(1,keepdims=True) + fs@we
        # # import pdb; pdb.set_trace()
        # fs = 0.5 * fs + 0.5*fsd


        DEBUG = '--debug' in sys.argv
        if DEBUG:
            if i+1 >= xsa.size(1) or i ==-1:
                print(f'[i]={i}')
                _x = z
                _x = _x[0][:10,:10]
                print((100*_x).int())

                _x =pik
                _x = _x[0][:10,:10]
                print((100*_x).int())

                _x = wt.reshape((E,K,E))[:10,:3,:10]
                print((100*_x).int())
            # _x = fs
                # _x = _x[:5,:10,:10]
                # print((100*_x).int())
                # _x = zwa
                # _x = _x[0][:10,:10]
                # print((100*_x).int())
                # print()
                # print()

                # _x = fs.square().mean(-1)[:5][:10]
                # print((100*_x).int())
                # _x = pik[:10,:10,0]
                # print((100*_x).int())
                # print((xsa[0,:3,:30]).int())
                pass

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
        # for i in range(L):
        #     inner[0]=L-1-i
        #     outer,inner = self._step(outer,inner) #### do what ever with your hidden state
        #     self.callback_step(outer,inner)

        self.callback_end(outer)
        (zi,x,y,z,fs) = outer
        (i,sel,xz,xsa) = inner
        # xsa = fs
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # f1 = self.vocab(xsa).log_softmax(-1)
        # # .matmul(self.emittor.weight)
        # f2 = self.vocab(z).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)
        # import pdb; pdb.set_trace()
        # lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0


class RefillModelRNNConvolveWithMixedEmission(RefillModelRNNBase):
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
        self.transcore  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater2   = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_kernel = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_energy = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_prob   = nn.Linear(embed_dim,embed_dim).to(self.device)

        self.fs_type    = 'lptok'
        self.loss_type  ='simplelog'
        self.loss_type  ='mixture'

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
        z = self.norm(z)
        xsa = self.norm(xsa)
        # xeu =   z.matmul(self.updater.weight.T)
        Wu  =  self.updater.weight.T[:,:-1]
        Wu2 =  self.updater2.weight.T[:,:-1]
        Wu = self.norm(Wu.T).T
        att0 = (xsa * z).matmul(Wu)/self.embed_dim**0.5
        att2 = (xsa * z).matmul(Wu2)/self.embed_dim**0.5
        # att = (xsa ).matmul(Wu)

        att = (att0).softmax(-1)
        # xeu = att.matmul(self.updater2.weight[:,:-1].T)*z
        xeu = att.matmul(Wu2.T)*z

        # xeu = xeu + (att * (1-att) * att2 ).matmul(Wu.T) * z
        # import pdb; pdb.set_trace()
        if i == x.size(1)-1:
            pass
            # print((100*att [0,:2,:800]).int())
            # print((100*att[0,:2,:800]).int())


        xnew=1./3 *(
            xsa.roll(1,1).matmul(self.transition.weight)+
            xsa.roll(-1,1).matmul(self.transition.weight.T)+
            xsa.roll(0,1).matmul(self.transcore.weight)+
            xeu +
            # z.matmul(self.updater.weight)+
            self.transition.bias +
            0.
            )
        # xsa = 0.5* xsa + 0.5*xnew
        xsa = xnew
        # xsa + 0.5*xnew


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
        # for i in range(L):
        #     inner[0]=L-1-i
        #     outer,inner = self._step(outer,inner) #### do what ever with your hidden state
        #     self.callback_step(outer,inner)

        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,sel,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # f1 = self.vocab(xsa).log_softmax(-1)
        # # .matmul(self.emittor.weight)
        # f2 = self.vocab(z).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()

        if self.loss_type == 'mixture':
            xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
            xkey_dynamic= self.xkey_dynamic(y)
            xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
            # import pdb; pdb.set_trace()
            cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
            sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
            lptok = self.vocab(cand).log_softmax(-1)
            lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)
        elif self.loss_type=='simplelog':
            lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)
        else:
            assert 0,self.loss_type

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0

class RefillModelRNNConvolveWithMixedEmissionMatrix(RefillModelRNNBase):
    '''
    Temperature tuning is crucial... Needs to prove mixture's capability
    of dynamic modelling. pik captures token class, but whether this is
    useful is another question.

    The second mixture component is within the transition model,
    with the null model being no transition at all, and the
    hypothesis model being applying a shift, we can probe
    whether the null model will be used at all?
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
        self.kernel_size = kernel_size = mixture_count
        self.init_state = nn.Linear(mixture_count, embed_dim).to(self.device)
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.transcore  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim*kernel_size).to(self.device)
        self.updater2   = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_kernel = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_energy = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_prob   = nn.Linear(embed_dim,embed_dim).to(self.device)

        self.fs_type    = 'lptok'
        self.loss_type  ='simplelog'
        self.loss_type  ='mixture'

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
        B = len(xsa)
        L = xsa.size(1)
        E = xsa.size(2)
        K = self.kernel_size

        z = self.norm(z)
        xsa = self.norm(xsa)
        wu = self.updater.weight.T
        zwu = (z @ wu).reshape((B,L,K,E))
        hik = (zwu * xsa[:,:,None].repeat((1,1,K,1))).sum(-1)
        hik = hik / E **0.5
        pik = hik.softmax(-1)
        xsadu = (pik[:,:,None] @ zwu ).squeeze(2)
        _norm = lambda x,dim=-1:x/(1+x.std(dim=dim,keepdims=True))

        useMixedTransition = 2
        if useMixedTransition == 0:
            xsadt = 0.
            xsadt =  xsadt + xsa.roll(1,1).matmul(self.transition.weight)
            xsadt =  xsadt + xsa.roll(-1,1).matmul(self.transition.weight.T)
        elif useMixedTransition==1:
            xsadt = 0.
            ## maximise (log exp(xsa W xsr ) + exp(xsa I xsr ))
            ## (B,L,K,E)
            w1 = torch.eye(E).to(self.device)
            w2 = self.transition.weight
            wt = torch.stack([w1,w2],dim=1)
            xsr = xsa.roll(-1,1)
            xsrw = _norm(xsr.matmul(wt.reshape((E,2*E)))).reshape((B,L,2,E))
            hik2 = (xsrw * xsa[:,:,None]).sum(-1)
            hik2 = hik2 / E**0.5
            pik2 = hik2.softmax(-1)
            xsaw = _norm(xsa @ wt.permute((2,1,0)).reshape((E,2*E))).reshape((B,L,2,E))

            xsadt = xsadt+( pik2.unsqueeze(2) @ xsrw ).squeeze(2) + ( pik2.unsqueeze(2) @ xsaw ).squeeze(2).roll(1,1)
        elif useMixedTransition==2:
            xsadt = 0.
            xsadt =  xsadt + xsa.roll(1,1).matmul(self.transition.weight)
            xsadt =  xsadt + xsa.roll(-1,1).matmul(self.transition.weight.T)
            # xsadt =  xsadt + xsa.roll(1,1)
            # xsadt =  xsadt + xsa.roll(-1,1)


            # wt = torch.cat()
            # import pdb; pdb.set_trace()
            # xsa.matmul(self.transition.weight)

        if ('--debug' in sys.argv) and (i == L-1 or i ==3):
            print(f'[i={str(i)*30}]')
            # _x = xsa.square().mean(-1)[:5][:10]
            # print((100*_x).int())
            print('[xsa]')
            _x = xsa[:2,:10,:10]
            # print((100*_x).int())
            # print('[xsa2]')
            (100*wu).reshape((E,K,E))[0:2,:10,:10].int()
            # _x = fs.square().mean(-1)[:5][:10]
            # print((100*_x).int())
            # _x = hik[:2,:10,]
            # print((100*_x).int())
            print('[pik]')
            _x = pik[:2,:,]
            print((100*_x).int())

            # print('[pik2]')
            # _x = pik2[:2,:,]
            # print((100*_x).int())

            pass


        xnew=1./3 *(
            xsa.roll(0,1).matmul(self.transcore.weight)+
            xsadt +
            xsadu +
            # xeu +

            # z.matmul(self.updater.weight)+
            self.transition.bias +
            0.
            )
        xsa = 0.5* xsa + 0.5*xnew
        xsa = _norm(xsa)
        # xsa = xnew
        # xsa + 0.5*xnew


        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj var'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)
        # for i in range(L):
        #     inner[0]=L-1-i
        #     outer,inner = self._step(outer,inner) #### do what ever with your hidden state
        #     self.callback_step(outer,inner)

        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,sel,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # f1 = self.vocab(xsa).log_softmax(-1)
        # # .matmul(self.emittor.weight)
        # f2 = self.vocab(z).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()

        if self.loss_type == 'mixture':
            xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
            xkey_dynamic= self.xkey_dynamic(y)
            xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
            # import pdb; pdb.set_trace()
            cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
            sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
            lptok = self.vocab(cand).log_softmax(-1)
            lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)
        elif self.loss_type=='simplelog':
            lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)
        else:
            assert 0,self.loss_type

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='var': return inner,outer

        if out=='token': return lptok

        if out=='loss': return loss
        assert 0

    def get_hidden(self,zi,x,y,z,out='var'):
        return self._loss(zi,x,y,z,out='var')





class RefillModelRNNConvolveWithMixedEmissionAndMixedTransition(RefillModelRNNBase):
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
        self.transition2= nn.Linear(embed_dim,embed_dim).to(self.device)
        self.transcore  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater2   = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_kernel = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_energy = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_prob   = nn.Linear(embed_dim,embed_dim).to(self.device)

        self.fs_type    = 'lptok'
        self.loss_type  ='simplelog'

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
        # xeu =   z.matmul(self.updater.weight.T)
        Wu  =  self.updater.weight.T[:,:-1]
        Wu2 =  self.updater2.weight.T[:,:-1]
        att0 = (xsa * z).matmul(Wu)/self.embed_dim**0.5
        att2 = (xsa * z).matmul(Wu2)/self.embed_dim**0.5
        # att = (xsa ).matmul(Wu)

        att = (att0).softmax(-1)
        # xeu = att.matmul(self.updater2.weight[:,:-1].T)*z
        xeu = att.matmul(Wu2.T)*z

        # xeu = xeu + (att * (1-att) * att2 ).matmul(Wu.T) * z

        # import pdb; pdb.set_trace()


        xr = xsa.roll(1,1)
        Wt = self.transition.weight.T[:,:-1]
        Wt2 = self.transition2.weight.T[:,:-1]
        att = (xsa*xr).matmul(Wt)/self.embed_dim**0.5
        att = att.softmax(-1)
        xer = att.matmul(Wt2.T) * xr


        xnew=1./3 *(
            xer+
            # xsa.roll(-1,1).matmul(self.transition.weight.T)+
            xsa.roll(0,1).matmul(self.transcore.weight)+
            xeu +
            # z.matmul(self.updater.weight)+
            self.transition.bias +
            0.
            )
        # xsa = 0.5* xsa + 0.5*xnew
        xsa = xnew
        # xsa + 0.5*xnew


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
        # for i in range(L):
        #     inner[0]=L-1-i
        #     outer,inner = self._step(outer,inner) #### do what ever with your hidden state
        #     self.callback_step(outer,inner)

        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,sel,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # f1 = self.vocab(xsa).log_softmax(-1)
        # # .matmul(self.emittor.weight)
        # f2 = self.vocab(z).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()

        if self.loss_type == 'mixture':
            xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
            xkey_dynamic= self.xkey_dynamic(y)
            xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
            # import pdb; pdb.set_trace()
            cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
            sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
            lptok = self.vocab(cand).log_softmax(-1)
            lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)
        elif self.loss_type=='simplelog':
            lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)
        else:
            assert 0,self.loss_type

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0




class RefillModelRNNConvolveTwoLayer(RefillModelRNNBase):
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
        self.transcore  = nn.Linear(embed_dim,embed_dim).to(self.device)

        self.transcore2 = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.transition2= nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater2   = nn.Linear(embed_dim,embed_dim).to(self.device)

        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
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
        xz = xsa
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        # xz = None
        outer = [zi, x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xz,xsa) = inner
        xznew = 1./3 *(
            xz.roll(1,1).matmul(self.transition2.weight)+
            xz.roll(-1,1).matmul(self.transition2.weight.T)+
            xz.roll(1,1).matmul(self.transcore2.weight)+
            z.matmul(self.updater2.weight.T)
            +self.transition2.bias
            )

        xnew=1./3 *(
            xsa.roll(1,1).matmul(self.transition.weight)+
            xsa.roll(-1,1).matmul(self.transition.weight.T)+
            xsa.roll(1,1).matmul(self.transcore.weight)+
            xz.matmul(self.updater.weight.T)
            +self.transition.bias
            )
        # xsa = 0.5* xsa + 0.5*xnew
        xsa = xnew
        xz = xznew
        # xsa + 0.5*xnew


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
        # for i in range(L):
        #     inner[0]=L-1-i
        #     outer,inner = self._step(outer,inner) #### do what ever with your hidden state
        #     self.callback_step(outer,inner)

        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,sel,xz,xsa) = inner
        xsa = xz
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # f1 = self.vocab(xsa).log_softmax(-1)
        # # .matmul(self.emittor.weight)
        # f2 = self.vocab(z).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()

        # xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        # xkey_dynamic= self.xkey_dynamic(y)
        # xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # # import pdb; pdb.set_trace()
        # cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        # sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        # lptok = self.vocab(cand).log_softmax(-1)
        # lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)

        lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0

class RefillModelRNNConvolveWithAttention7(RefillModelRNNBase):
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
        self.transcore  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
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

        yy = xsa.matmul(self.att_kernel.weight).matmul(xsa.transpose(2,1)).softmax(-1).matmul(xsa)
        # yy = yy*self.att_prob(xsa)[:,:,:1].sigmoid()

        xnew=1./3 *(
            xsa.roll(1,1).matmul(self.transition.weight)+
            xsa.roll(1,-1).matmul(self.transition.weight.T)+
            xsa.roll(1,0).matmul(self.transcore.weight)+
            self.att_energy(yy)+
            # yy.matmul(self.att_kernel.weight)+
            z.matmul(self.updater.weight.T)
            +self.transition.bias
            )
        # xsa = 0.5* xsa + 0.5*xnew
        # xsa = torch.cat([xsa,xnew],dim=2)
        xsa = xnew
        # xsa + 0.5*xnew



        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        B = len(z)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)



        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,sel,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # f1 = self.vocab(xsa).log_softmax(-1)
        # # .matmul(self.emittor.weight)
        # f2 = self.vocab(z).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()

        # xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        # xkey_dynamic= self.xkey_dynamic(y)
        # xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # # import pdb; pdb.set_trace()
        # cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        # sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        # lptok = self.vocab(cand).log_softmax(-1)
        # lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)

        # .matmul(lptok)
        # lptok = self.vocab(cand).log_softmax(-1)


        lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0

class RefillModelRNNConvolveWithAttentionSymm(RefillModelRNNBase):
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
        self.transcore  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
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
        gradsq = xsa*0
        outer = [zi, x,y,z,fs]
        inner = [i,gradsq,xz,xsa]
        return outer,inner
    # def att_func(self,att):
    #     return att
    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,gradsq,xz,xsa) = inner
        # sel = None
        xz  = None

        E = xsa.size(-1)
        L = xsa.size(1)
        att = xsa.matmul(self.att_kernel.weight).matmul(xsa.transpose(2,1))
        # att = (att /E ).abs()
        # attr = att.transpose(2,1)
        # attl = att
        # attl = attl/(1+attl.sum(-1,keepdims=True))
        # attr = attr/(1+attr.sum(-1,keepdims=True))
        # /(1+attl.std(-1,keepdims=True))
        # attl = (attl-attl.mean(-1,keepdims=True))/(1+attl.std(-1,keepdims=True))
        # attr = (attr-attr.mean(-1,keepdims=True))/(1+attr.std(-1,keepdims=True))

        # att = att / E**0.5
        # att = att.softmax(-1)

        ##### notoriously more difficult to train
        att = (att).abs()
        # att = att / 1
        # att = att*(1-torch.eye(L)[None].to(self.device))
        attr = att.transpose(2,1)
        attl = att
        # .softmax(-1)
        # att = (att + att.transpose(2,1))/2
        sumnorm = lambda x:x/x.sum(-1,keepdims=True)
        # yy = yy*self.att_prob(xsa)[:,:,:1].sigmoid()

        xnew=1./3 *(
            xsa.roll(1,1).matmul(self.transition.weight)+
            xsa.roll(1,-1).matmul(self.transition.weight.T)+
            xsa.roll(1,0).matmul(self.transcore.weight)+
            attl.matmul(xsa).matmul(self.att_energy.weight.T)+
            attr.matmul(xsa).matmul(self.att_energy.weight)+
            # self.att_energy(att.matmul(xsa))+

            z.matmul(self.updater.weight.T)
            +self.transition.bias
            )
        # xsa = 0.5* xsa + 0.5*xnew

        # xnew = xnew/(1.+xnew.std(-1,keepdims=True))
        # xsa = 1.0*xsa + 0.1*(-xsa + xnew)

        grad = (-0.1*xsa + xnew)
        grad = grad/(1.+grad.std(-1,keepdims=True))
        xsa = 1.0*xsa + 0.1*grad

        # grad = (0.1*xsa + xnew)
        # gradsq = 0.3*gradsq+0.7*grad.square()
        # xsa = 1.0*xsa + 0.1*grad/gradsq.sqrt()
        # (-xsa + xnew)

        # xsa = torch.cat([xsa,xnew],dim=2)

        # mask = torch.randint(2,xsa.shape[:-1])[:,:,None].to(self.device)
        # arange(0,1)==1
        # import pdb; pdb.set_trace()

        # xsa = mask * xnew + (1-mask ) * xsa
        # xsa + 0.5*xnew

        outer = [zi,x,y,z,fs]
        inner = [i,gradsq,xz,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        B = len(z)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)



        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,sel,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # f1 = self.vocab(xsa).log_softmax(-1)
        # f2 = self.vocab(z).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)

        # .matmul(lptok)
        # lptok = self.vocab(cand).log_softmax(-1)


        # lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0


class RefillModelRNNConvolveWithLinearAttention(RefillModelRNNBase):
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
        self.transcore  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.K = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        self.U = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        self.K2 = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        self.U2= nn.Linear(embed_dim//2,embed_dim).to(self.device)

        self.K = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.U = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.K2 = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.U2= nn.Linear(embed_dim,embed_dim).to(self.device)

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

        xsa = z
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xz = None
        gradsq = xsa*0
        outer = [zi, x,y,z,fs]
        inner = [i,gradsq,xz,xsa]
        return outer,inner
    # def att_func(self,att):
    #     return att
    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,gradsq,xz,xsa) = inner
        # sel = None
        xz  = None

        E = xsa.size(-1)
        L = xsa.size(1)

        # wK = torch.eye(E).to(self.device)
        wK = self.K.weight
        wK = wK.matmul(wK.T)

        # wK2 = self.K2.weight
        # wK2 = wK.matmul(wK2.T)

        aK = xsa.matmul(wK).matmul(xsa.transpose(2,1))/E
        aU = xsa.matmul(self.U.weight.matmul(self.U.weight.T)).matmul(xsa.transpose(2,1))/E
        # aU2= xsa.matmul(self.U2.weight).matmul(xsa.transpose(2,1))/E
        # aU = aU/
        # aU = xsa.matmul(self.U.weight).matmul(xsa.transpose(2,1)).relu()

        # aK = aK.relu()
        # aK = aK/(0.001 + aK.sum(-1,keepdims=True))

        # aU= aU.softmax(-1)

        # aU
        # # aU = torch.minimum(aU,aU2)
        sumnorm = lambda aU:aU/(0.001 + aU.sum(-1,keepdims=True))
        aU = aU.relu()
        aU = aU/(0.001 + aU.sum(-1,keepdims=True))



        xnew=1./3 *(
            xsa.roll(1,1).matmul(self.transition.weight)+
            xsa.roll(1,-1).matmul(self.transition.weight.T)+
            xsa.roll(1,0).matmul(self.transcore.weight)+

            # aK.matmul(xsa).matmul(self.U.weight) /E +
            # aK.transpose(2,1).matmul(xsa).matmul(self.U.weight.T)/E +

            aU.matmul(xsa).matmul(wK) +
            # sumnorm( xsa.matmul(self.U2.weight.matmul(self.U2.weight.T)).matmul(xsa.transpose(2,1)).relu()/E).matmul(xsa).matmul(self.K2.weight.matmul(self.K2.weight.T)) +
            # aU.transpose(2,1).matmul(xsa).matmul(wK.T) / E +

            # attl.matmul(xsa).matmul(self.att_energy.weight.T)+
            # attr.matmul(xsa).matmul(self.att_energy.weight)+
            # self.att_energy(att.matmul(xsa))+

            z.matmul(self.updater.weight.T)
            +self.transition.bias
            )
        # xsa = 0.5* xsa.abs() + 0.5*xnew.abs()

        # xnew = xnew/(1.+xnew.std(-1,keepdims=True))
        # xsa = 1.0*xsa + 0.1*(-xsa + xnew)
        #
        # grad = (-0.1*xsa + xnew)
        # # grad = grad/(1.+grad.std(-1,keepdims=True))
        # xsa = 1.0*xsa + 0.1*grad

        xsa =xnew

        # grad = (0.1*xsa + xnew)
        # gradsq = 0.3*gradsq+0.7*grad.square()
        # xsa = 1.0*xsa + 0.1*grad/gradsq.sqrt()
        # (-xsa + xnew)
        # xsa = torch.cat([xsa,xnew],dim=2)

        # mask = torch.randint(2,xsa.shape[:-1])[:,:,None].to(self.device)
        # arange(0,1)==1
        # import pdb; pdb.set_trace()

        # xsa = mask * xnew + (1-mask ) * xsa
        # xsa + 0.5*xnew



        outer = [zi,x,y,z,fs]
        inner = [i,gradsq,xz,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        B = len(z)
        for i in range(L//2):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)


        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,sel,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)

        # .matmul(lptok)
        # lptok = self.vocab(cand).log_softmax(-1)


        lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0




class RefillModelRNNConvolveWithLSTM(RefillModelRNNBase):
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
        self.transcore  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.K = nn.Linear(embed_dim,embed_dim*4,).to(self.device)
        # self.U = nn.Linear(embed_dim,embed_dim*4).to(self.device)
        self.K = nn.Linear(embed_dim*4,embed_dim,).to(self.device)
        self.U = nn.Linear(embed_dim*4,embed_dim).to(self.device)
        self.K2 = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.K3 = nn.Linear(embed_dim,embed_dim*4).to(self.device)
        self.U2= nn.Linear(embed_dim,embed_dim).to(self.device)

        # self.K = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.U = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.K2 = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.U2= nn.Linear(embed_dim,embed_dim).to(self.device)

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

        xsa = z
        xc = xsa*0
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xz = None
        gradsq = xsa*0
        outer = [zi, x,y,z,fs]
        inner = [i,xc,xz,xsa]
        return outer,inner
    # def att_func(self,att):
    #     return att
    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,xc,xz,xsa) = inner
        # sel = None
        xz  = None

        E = xsa.size(-1)
        L = xsa.size(1)

        U = self.U.weight
        K = self.K.weight
        # fgio = (self.U(xsa)+self.K(z)).sigmoid().reshape(xsa.shape+(-1,))
        fgio = (torch.roll(xsa,1,1).matmul(U)+(z).matmul(K)+self.K3.bias[None,None]).sigmoid().reshape(xsa.shape+(-1,))
        # fgio = torch.roll(fgio,1,1)
        xcr = torch.roll(xc,-1,1,)
        xcl = torch.roll(xc,1,1)
        xf,xg,xi,xo = fgio.permute((3,0,1,2))
        wf,wg,wi,wo = U.reshape((E,E,4)).permute((2,0,1))

        if i%2==1:
        # if 1:

            xsad = 0
            xsad = xsad + (xf*(1-xf)*xcl *xc)        .matmul(wf.T)
            xsad = xsad + (xi*(1-xi)*xc  *(xg-0.5))  .matmul(wi.T)
            xsad = xsad + (xg*(1-xg)*xc  *xi)        .matmul(wg.T)
            xsad = xsad + (xo*(1-xo)*xc.tanh()*xsa)  .matmul(wo.T)
            # # xsad = xsad + (xo*(1-xo)*xc.tanh()).matmul(wo.T)
            xsad = xsad + xcl.tanh() * torch.roll(xo,1,1)

            xsad = torch.roll(xsad,-1,1)

            # xsad = xsad + self.K2(xsa)

        else:
            xcd = (
             xcl * xf +
             xcr * torch.roll(xf,-1,1,) +
             xi  * xg +
             xsa * xo *0.5*(1+xc.tanh())*(1-xc.tanh()) +
             0.
             )

            # xcd = xcd + self.U2(xc)


        # grad = (0 + xsad)
        # grad = grad/(1.+grad.std(-1,keepdims=True))
        # xsa = 0.9*xsa + 0.1*grad
        #
        # # xsa = 1.0*xsa + 0.1*grad/gradsq.sqrt()
        #
        # grad = (0 + xcd)
        # grad = grad/(1.+grad.std(-1,keepdims=True))
        # xc = 0.9*xc + 0.1*grad

        if i%2==1:
        # if 1:
            # xsa =xsad
            xsa = 0.5*xsa + 0.5*xsad
        else:
            # xc =xcd
        # else:
            xc = 0.5*xc + 0.5*xcd

        # if i%2==1:
        #
        #     xsa = 0.5*xsa + 0.5*xsad
        # else:
        #     xc = 0.5*xc + 0.5*xcd

        # #torch.roll(xsad,1,1)
        # xc = 0.5*xc + 0.5*xcd
        # # xsa = torch.roll(xsad,1,1)



        # import pdb; pdb.set_trace()

        # xsa =xnew

        # grad = (0.1*xsa + xnew)
        # gradsq = 0.3*gradsq+0.7*grad.square()
        # xsa = 1.0*xsa + 0.1*grad/gradsq.sqrt()
        # (-xsa + xnew)
        # xsa = torch.cat([xsa,xnew],dim=2)

        # mask = torch.randint(2,xsa.shape[:-1])[:,:,None].to(self.device)
        # arange(0,1)==1
        # import pdb; pdb.set_trace()

        # xsa = mask * xnew + (1-mask ) * xsa
        # xsa + 0.5*xnew



        outer = [zi,x,y,z,fs]
        inner = [i,xc,xz,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        B = len(z)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)


        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,sel,xz,xsa) = inner
        # # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # # sel = self.emittor(xsa)[:,:,0].sigmoid()
        #
        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)

        lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0

class RefillModelRNNConvolveWithLSTMWithMemory(RefillModelRNNBase):
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
        self.transcore  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.K = nn.Linear(embed_dim,embed_dim*4,).to(self.device)
        # self.U = nn.Linear(embed_dim,embed_dim*4).to(self.device)
        self.K = nn.Linear(embed_dim*4,embed_dim,).to(self.device)
        self.U = nn.Linear(embed_dim*4,embed_dim).to(self.device)
        self.K2 = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.K3 = nn.Linear(embed_dim,embed_dim*4).to(self.device)
        self.U2= nn.Linear(embed_dim,embed_dim).to(self.device)

        # self.K = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.U = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.K2 = nn.Linear(embed_dim,embed_dim).to(self.device)
        # self.U2= nn.Linear(embed_dim,embed_dim).to(self.device)

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

        xsa = z
        xc = xsa*0
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        xw = self.U2.weight[None].repeat((len(y),1,1))
        gradsq = xsa*0
        outer = [zi, x,y,z,fs]
        inner = [i,xc,xw,xsa]
        return outer,inner
    # def att_func(self,att):
    #     return att
    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,xc,xw,xsa) = inner
        # sel = None
        # xz  = None

        E = xsa.size(-1)
        L = xsa.size(1)

        U = self.U.weight
        K = self.K.weight
        # fgio = (self.U(xsa)+self.K(z)).sigmoid().reshape(xsa.shape+(-1,))
        # self.K3.bias[None,None]
        bias = self.K3.bias[None,None]
        fgio = (torch.roll(xsa,1,1).matmul(U)+(z).matmul(K)+bias).sigmoid().reshape(xsa.shape+(-1,))
        # fgio = torch.roll(fgio,1,1)
        xcr = torch.roll(xc,-1,1,)
        xcl = torch.roll(xc,1,1)
        xf,xg,xi,xo = fgio.permute((3,0,1,2))
        wf,wg,wi,wo = U.reshape((E,E,4)).permute((2,0,1))

        xcw = xsa.matmul(xw)

        if i%2==1:
        # if 1:

            xsad = 0
            xsad = xsad + (xf*(1-xf)*xcl *xc)        .matmul(wf.T)
            xsad = xsad + (xi*(1-xi)*xc  *(xg-0.5))  .matmul(wi.T)
            xsad = xsad + (xg*(1-xg)*xc  *xi)        .matmul(wg.T)
            xsad = xsad + (xo*(1-xo)*xcw.tanh()*xsa)  .matmul(wo.T)
            # # xsad = xsad + (xo*(1-xo)*xc.tanh()).matmul(wo.T)
            xsad = xsad + xcl.tanh() * torch.roll(xo,1,1)

            xsad = torch.roll(xsad,-1,1)
            # xsad = xsad + self.K2(xsa)
            xwd = xc.transpose(2,1).matmul((xsa * xo *0.5*(1+xcw.tanh())*(1-xcw.tanh())))

        else:
            xcd = (
             xcl * xf +
             xcr * torch.roll(xf,-1,1,) +
             xi  * xg +
             (xsa * xo *0.5*(1+xcw.tanh())*(1-xcw.tanh())).matmul(xw.transpose(2,1)) +
             0.
             )
            # xcd = xcd + self.U2(xc)


        # grad = (0 + xsad)
        # grad = grad/(1.+grad.std(-1,keepdims=True))
        # xsa = 0.9*xsa + 0.1*grad
        #
        # # xsa = 1.0*xsa + 0.1*grad/gradsq.sqrt()
        #
        # grad = (0 + xcd)
        # grad = grad/(1.+grad.std(-1,keepdims=True))
        # xc = 0.9*xc + 0.1*grad

        if i%2==1:
        # if 1:
            # xsa =xsad
            xsa = 0.5*xsa + 0.5*xsad
            xw  = 0.5*xw + 0.5*xwd
        else:
            # xc =xcd
        # else:
            xc = 0.5*xc + 0.5*xcd

        # if i%2==1:
        #
        #     xsa = 0.5*xsa + 0.5*xsad
        # else:
        #     xc = 0.5*xc + 0.5*xcd

        # #torch.roll(xsad,1,1)
        # xc = 0.5*xc + 0.5*xcd
        # # xsa = torch.roll(xsad,1,1)



        # import pdb; pdb.set_trace()

        # xsa =xnew

        # grad = (0.1*xsa + xnew)
        # gradsq = 0.3*gradsq+0.7*grad.square()
        # xsa = 1.0*xsa + 0.1*grad/gradsq.sqrt()
        # (-xsa + xnew)
        # xsa = torch.cat([xsa,xnew],dim=2)

        # mask = torch.randint(2,xsa.shape[:-1])[:,:,None].to(self.device)
        # arange(0,1)==1
        # import pdb; pdb.set_trace()

        # xsa = mask * xnew + (1-mask ) * xsa
        # xsa + 0.5*xnew



        outer = [zi,x,y,z,fs]
        inner = [i,xc,xw,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        B = len(z)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)


        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,sel,xz,xsa) = inner
        # # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # # sel = self.emittor(xsa)[:,:,0].sigmoid()
        #
        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)

        # lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0

class RefillModelRNNConvolveWithDynamicWeight(RefillModelRNNBase):
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
        self.transcore  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.K = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        self.U = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        self.K2 = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        self.U2= nn.Linear(embed_dim//2,embed_dim).to(self.device)

        self.K = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.U = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.K2 = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.U2= nn.Linear(embed_dim,embed_dim).to(self.device)

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

        xsa = z
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        # xz = None
        gradsq = xsa*0
        # xzw = self.updater.weight
        xzw = self.transcore.weight
        xzw = xzw[None,].repeat((len(z),1,1))
        # xw =
        outer = [zi, x,y,z,fs]
        inner = [i,gradsq,xzw,xsa]
        return outer,inner
    # def att_func(self,att):
    #     return att
    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,gradsq,xzw,xsa) = inner
        # sel = None
        xz  = None

        E = xsa.size(-1)
        L = xsa.size(1)

        # wK = torch.eye(E).to(self.device)
        wK = self.K.weight
        wK = wK.matmul(wK.T)

        xnew=1./3 *(
            xsa.roll(1,1).matmul(self.transition.weight)+
            xsa.roll(1,-1).matmul(self.transition.weight.T)+
            # xsa.roll(1,0).matmul(xzw)+
            # xsa.roll(1,0).matmul(self.transcore.weight)+
            z.matmul(xzw)
            # z.matmul(self.updater.weight.T)
            +self.transition.bias
            )


        if i%2==0:
            # xsa = 0.5 * xnew + 0.5 * xsa
            xsa = 1.0 * xnew + 0.0 * xsa
        # xsa = 0.5 * xnew
        if i%2==1:
            # xzwd = z.transpose(2,1).matmul(xsa)/z.shape[1]
            xzwd = xsa.transpose(2,1).matmul(xsa)/z.shape[1]
            xzw = 1.0*xzw + 0.0*xzwd
            # xzw = 0.5*xzw + 0.5*xzwd
        # xzw = xzwd
        # print((xzw[0][:10][:10]*10).int())


        outer = [zi,x,y,z,fs]
        inner = [i,gradsq,xzw,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        B = len(z)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)


        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,sel,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)

        # .matmul(lptok)
        # lptok = self.vocab(cand).log_softmax(-1)


        lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0

class RefillModelRNNConvolveWithHiddenVector(RefillModelRNNBase):
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
        self.embed      = nn.Embedding(graph_dim,embed_dim,)
        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim)
        self.kernel_size = kernel_size = 4
        self.init_state = nn.Linear(mixture_count, embed_dim)
        self.transition = nn.Linear(embed_dim*1,embed_dim)
        self.transcore  = nn.Linear(embed_dim,embed_dim)
        self.updater    = nn.Linear(embed_dim,embed_dim)
        self.emittor    = nn.Linear(embed_dim,embed_dim)
        self.attention_probe = nn.Linear(embed_dim,kernel_size)
        self.mu = nn.Linear(embed_dim,1)
        self.KE = KE = embed_dim//kernel_size//2

        self.attention_head  = nn.Linear(embed_dim,kernel_size*KE)
        self.attention_head_l  = nn.Linear(kernel_size,embed_dim*KE)
        self.attention_head_r  = nn.Linear(kernel_size,embed_dim*KE)
        # self.attention_head_r  = nn.Linear(embed_dim,kernel_size*KE)
        # self.K = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.U = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.K2 = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.U2= nn.Linear(embed_dim//2,embed_dim).to(self.device)


        self.K = nn.Linear(embed_dim,embed_dim)
        self.U = nn.Linear(embed_dim,embed_dim)
        self.K2 = nn.Linear(embed_dim,embed_dim)
        self.U2= nn.Linear(embed_dim,embed_dim)

        self.att_prob   = nn.Linear(embed_dim,embed_dim)
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
        mu = self.mu.weight[None,0:1]+(xsa[:,0:1]*0)

        # mu = self.attention_head
        xsa = z
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        # xz = None
        xsa = xsa*0
        gradsq = xsa*0
        # xzw = self.updater.weight
        xzw = self.transcore.weight
        xzw = xzw[None,].repeat((len(z),1,1))
        # xw =
        outer = [zi, x,y,z,fs]
        inner = [i,mu,xzw,xsa]
        return outer,inner
    # def att_func(self,att):
    #     return att
    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,mu,xzw,xsa) = inner
        xzw= None
        # sel = None
        xz  = None

        E = xsa.size(-1)
        L = xsa.size(1)
        B = xsa.size(0)
        K = self.kernel_size
        KE =self.KE

        # wK = torch.eye(E).to(self.device)
        # wK = self.K.weight
        # wK = wK.matmul(wK.T)
        wT = self.transition.weight
        # wT = wT.matmul(wT.T)/wT.shape[1]**0.5
        zk = self.attention_probe.weight.T
        akl = self.attention_head_l.weight.T
        akr = self.attention_head_r.weight.T
        ak = akl.reshape((K,E,KE)).matmul(akr.reshape((K,KE,E))).reshape((E,K*E))/KE**0.5
        rzk = zk.std(0,keepdims=True)*K**0.5
        peki =  (xsa.matmul(zk/rzk).transpose(2,1))
        # peki = peki/5.
        # peki = peki.clip(-10,10)
        # import pdb; pdb.set_trace()
        pki = peki.softmax(-1)
        pik = pki.transpose(2,1)
        pkix  = pki.matmul(xsa)
        #(K,E,KE)
        mud = pkix.reshape((B,-1,K*E)).matmul(ak.T)/1./K

        akmu = mu.matmul(ak).reshape((B,-1,E))

        xnew=1. *(
            xsa.roll(1,1).matmul(wT)    +
            xsa.roll(-1,1).matmul(wT.T) +
            # xsa.roll(1,0).matmul(xzw)+
            # xsa.roll(0,1).matmul(self.transcore.weight)     +
            z.matmul(self.updater.weight.T) +
            (pik*(1-pik)*xsa.matmul(akmu.transpose(2,1))).matmul(zk.T)/K +
            pik.matmul(akmu)/K +
            self.transition.bias +
            0.
            )



        if i%2==1:
            xsa = 0.8 * xsa + 0.2 * xnew
            # xsa = xsa/(0.001 + mu.std(-1,keepdims=True))
            # xsa = 1.0 * xnew
            # xsa = 1.0 * xnew + 0.0 * xsa
        # xsa = 0.5 * xnew
        if i%2!=1:
            # mu = 1.0*mud
            mu = 0.8 *mu +  mud
            # mu = mu/(0.001 + mu.std(-1,keepdims=True))
            # xzwd = z.transpose(2,1).matmul(xsa)/z.shape[1]
            # xzwd = xsa.transpose(2,1).matmul(xsa)/z.shape[1]
            # xzw = 1.0*xzw + 0.0*xzwd
            # # xzw = 0.5*xzw + 0.5*xzwd
        # xzw = xzwd
        # print((xzw[0][:10][:10]*10).int())

        # print(mu.std())
        # if i+1==L:
        #     import pdb; pdb.set_trace()
        outer = [zi,x,y,z,fs]
        inner = [peki,mu,xzw,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        B = len(z)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)


        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (peki,mu,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()
        # print(100*(mu[0:2,0,:10].detach().cpu().int()).numpy())
        # print()

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)

        # .matmul(lptok)
        # lptok = self.vocab(cand).log_softmax(-1)
        # if i+1==L:
        # if 0:
        if zi.min()==0:
            for k,xx in dict(mu=mu,xsa=xsa,peki=peki).items():
                print(xx.shape)
                print((xx[:3,0:2,:10]*10).int())
                print((xx.std(0)[0:2,:10]))
                print(f'{k}_std:{xx.std()}')
                print(f'{k}_max:{xx.max()}')
                print(f'{k}_min:{xx.min()}')
                print()
            for k,v in self.named_parameters():

                # print(f'{k}:{self.embed.weight.max()}')
                print(f'{k}:\t\t{v.max():.3f}')
                # import pdb; pdb.set_trace()
            # print(self.state)
            print()


        # lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0


class RefillModelRNNConvolveWithVariableHiddenVector(RefillModelRNNBase):
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
        self.embed      = nn.Embedding(graph_dim,embed_dim,)
        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim)
        self.kernel_size = kernel_size = 10
        self.init_state = nn.Linear(mixture_count, embed_dim)
        self.transition = nn.Linear(embed_dim*1,embed_dim)
        self.transcore  = nn.Linear(embed_dim,embed_dim)
        self.updater    = nn.Linear(embed_dim,embed_dim)
        self.emittor    = nn.Linear(embed_dim,embed_dim)
        self.attention_probe = nn.Linear(embed_dim,kernel_size)
        self.mu = nn.Linear(embed_dim,1)
        self.KE = KE = embed_dim//kernel_size
        # self.HV = embed_dim*2
        self.attention_head     = nn.Linear(embed_dim,kernel_size*KE)
        self.attention_head_l  = nn.Linear(kernel_size,embed_dim*KE)
        self.attention_head_r  = nn.Linear(kernel_size,embed_dim*KE)
        # self.attention_head_r  = nn.Linear(embed_dim,kernel_size*KE)
        # self.K = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.U = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.K2 = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.U2= nn.Linear(embed_dim//2,embed_dim).to(self.device)


        self.K = nn.Linear(embed_dim,embed_dim)
        self.U = nn.Linear(embed_dim,embed_dim)
        self.K2 = nn.Linear(embed_dim,embed_dim)
        self.U2= nn.Linear(embed_dim,embed_dim)

        self.att_prob   = nn.Linear(embed_dim,embed_dim)
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
        mu = self.mu.weight[None,0:1]+(xsa[:,0:1]*0)

        # mu = self.attention_head
        xsa = z
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        # xz = None
        xsa = xsa*0
        gradsq = xsa*0
        # xzw = self.updater.weight
        xzw = self.transcore.weight
        xzw = xzw[None,].repeat((len(z),1,1))
        # xw =
        outer = [zi, x,y,z,fs]
        inner = [i,mu,xzw,xsa]
        return outer,inner
    # def att_func(self,att):
    #     return att
    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,mu,xzw,xsa) = inner
        xzw= None
        # sel = None
        xz  = None

        E = xsa.size(-1)
        L = xsa.size(1)
        B = xsa.size(0)
        K = self.kernel_size
        KE =self.KE

        # wK = torch.eye(E).to(self.device)
        # wK = self.K.weight
        # wK = wK.matmul(wK.T)
        wT = self.transition.weight
        # wT = wT.matmul(wT.T)/wT.shape[1]**0.5
        zk = self.attention_probe.weight.T
        akl = self.attention_head_l.weight.T
        akr = self.attention_head_r.weight.T
        ak = akl.reshape((K,E,KE)).matmul(akr.reshape((K,KE,E))).reshape((E,K*E))/KE**0.5

        pki = (xsa.matmul(zk).transpose(2,1)).softmax(-1)
        pik = pki.transpose(2,1)
        pkix  = pki.matmul(xsa)
        #(K,E,KE)
        mud = pkix.reshape((B,-1,K*E)).matmul(ak.T)/1./K

        akmu = mu.matmul(ak).reshape((B,-1,E))

        xnew=1. *(
            xsa.roll(1,1).matmul(wT)    +
            xsa.roll(-1,1).matmul(wT.T) +
            # xsa.roll(1,0).matmul(xzw)+
            # xsa.roll(0,1).matmul(self.transcore.weight)     +
            z.matmul(self.updater.weight.T) +
            (pik*(1-pik)*xsa.matmul(akmu.transpose(2,1))).matmul(zk.T)/K +
            pik.matmul(akmu)/K +
            self.transition.bias +
            0.
            )



        if i%2==1:
            xsa = 0.5 * xsa + 0.2 * xnew
            # xsa = 1.0 * xnew
            # xsa = 1.0 * xnew + 0.0 * xsa
        # xsa = 0.5 * xnew
        if i%2!=1:
            mu = 0.5 *mu + 0.2 * mud
            # mu = 1.0*mud
            # xzwd = z.transpose(2,1).matmul(xsa)/z.shape[1]
            # xzwd = xsa.transpose(2,1).matmul(xsa)/z.shape[1]
            # xzw = 1.0*xzw + 0.0*xzwd
            # # xzw = 0.5*xzw + 0.5*xzwd
        # xzw = xzwd
        # print((xzw[0][:10][:10]*10).int())


        outer = [zi,x,y,z,fs]
        inner = [i,mu,xzw,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        B = len(z)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)


        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,mu,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()
        # print(100*(mu[0:2,0,:10].detach().cpu().int()).numpy())
        # print()

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)

        # .matmul(lptok)
        # lptok = self.vocab(cand).log_softmax(-1)
        # if i+1==L:
        if 0:
        # if zi.min()==0:
            for xx in [mu,xsa]:
                print(xx.shape)
                print((xx[:6,0:2,:10]*10).int())
                print(f'xx_std:{xx.std()}')


        # lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0

class RefillModelRNNConvolveWithHiddenVectorDynamic(RefillModelRNNBase):
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
        self.embed      = nn.Embedding(graph_dim,embed_dim,)
        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim)
        self.kernel_size = kernel_size = 10
        self.init_state = nn.Linear(mixture_count, embed_dim)
        self.transition = nn.Linear(embed_dim*1,embed_dim)
        self.transcore  = nn.Linear(embed_dim,embed_dim)
        self.updater    = nn.Linear(embed_dim,embed_dim)
        self.emittor    = nn.Linear(embed_dim,embed_dim)
        self.attention_probe = nn.Linear(embed_dim,kernel_size)
        self.mu = nn.Linear(embed_dim,1)
        self.KE = KE = embed_dim//kernel_size
        self.attention_head  = nn.Linear(embed_dim,kernel_size*KE)
        self.attention_head_l  = nn.Linear(kernel_size,embed_dim*KE)
        self.attention_head_r  = nn.Linear(kernel_size,embed_dim*KE)
        # self.attention_head_2  = nn.Linear(embed_dim,kernel_size*embed_dim)
        self.attention_head_l2  = nn.Linear(kernel_size,embed_dim*KE)
        self.attention_head_r2  = nn.Linear(kernel_size,embed_dim*KE)
        # self.attention_head_r  = nn.Linear(embed_dim,kernel_size*KE)
        # self.K = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.U = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.K2 = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.U2= nn.Linear(embed_dim//2,embed_dim).to(self.device)


        self.K = nn.Linear(embed_dim,embed_dim)
        self.U = nn.Linear(embed_dim,embed_dim)
        self.K2 = nn.Linear(embed_dim,embed_dim)
        self.U2= nn.Linear(embed_dim,embed_dim)

        self.att_prob   = nn.Linear(embed_dim,embed_dim)
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
        mu = self.mu.weight[None,0:1]+(xsa[:,0:1]*0)
        # mu = xsa.mean(1,keepdims=True)
        # mu = self.attention_head
        xsa = z
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        # xz = None
        xsa = xsa*0
        gradsq = xsa*0
        # xzw = self.updater.weight
        xzw = self.transcore.weight
        xzw = xzw[None,].repeat((len(z),1,1))
        # xw =
        outer = [zi, x,y,z,fs]
        inner = [i,mu,xzw,xsa]
        return outer,inner
    # def att_func(self,att):
    #     return att
    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,mu,xzw,xsa) = inner
        xzw= None
        # sel = None
        xz  = None

        E = xsa.size(-1)
        L = xsa.size(1)
        B = xsa.size(0)
        K = self.kernel_size
        KE =self.KE

        # wK = torch.eye(E).to(self.device)
        # wK = self.K.weight
        # wK = wK.matmul(wK.T)
        wT = self.transition.weight
        # wT = wT.matmul(wT.T)/wT.shape[1]**0.5
        # zk = self.attention_probe.weight.T

        akl = self.attention_head_l.weight.T
        akr = self.attention_head_r.weight.T
        ak = akl.reshape((K,E,KE)).matmul(akr.reshape((K,KE,E))).reshape((E,K*E))/KE**0.5

        akl2 = self.attention_head_l2.weight.T
        akr2 = self.attention_head_r2.weight.T
        ak2 = akl2.reshape((K,E,KE)).matmul(akr2.reshape((K,KE,E))).reshape((E,K*E))/KE**0.5
        # E ( K E )
        zk  = mu.matmul(ak2).reshape((B,K,E))

        peki =  (xsa.matmul(zk.transpose(2,1)).transpose(2,1))
        pki = (xsa.matmul(zk.transpose(2,1)).transpose(2,1)).softmax(-1)
        pik = pki.transpose(2,1)
        pkix  = pki.matmul(xsa)
        #(K,E,KE)

        akmu = mu.matmul(ak).reshape((B,-1,E))
        xakmu = xsa.matmul(akmu.transpose(2,1))
        # import pdb; pdb.set_trace()
        mud = 0
        # mud
        mud = mud + pkix.reshape((B,-1,K*E)).matmul(ak.T)
        v = (pki*(1-pki)*xakmu.transpose(2,1)).transpose(2,1)
        vv = v.reshape((B,1,L*K)).matmul(xsa.matmul(ak2.reshape((E,K,E)).permute((2,1,0)).reshape((E,K*E))).reshape((B,L*K,E)))
        # .matmul(ak2.reshape((E,K,E)).permute((1,2,0)).reshape((K,E*E)))
        mud = mud +vv
        mud = mud /K
        # xsa.matmul(ak2))

        xnew=1. *(
            xsa.roll(1,1).matmul(wT)    +
            xsa.roll(-1,1).matmul(wT.T) +
            # xsa.roll(1,0).matmul(xzw)+
            # xsa.roll(0,1).matmul(self.transcore.weight)     +
            z.matmul(self.updater.weight.T) +
            (pik*(1-pik)*xakmu).matmul(zk)/K +
            pik.matmul(akmu)/K +
            self.transition.bias +
            0.
            )

        # if i==L-1:
            # import pdb; pdb.set_trace()
        if i%2==1:
            xsa = 0.8 * xsa + 0.2 * xnew
            # xsa = 1.0 * xnew
            # xsa = 1.0 * xnew + 0.0 * xsa
        # xsa = 0.5 * xnew
        if i%2!=1:
            mu = 0.8 *mu + 0.2 * mud
            # mu = 1.0*mud
            # xzwd = z.transpose(2,1).matmul(xsa)/z.shape[1]
            # xzwd = xsa.transpose(2,1).matmul(xsa)/z.shape[1]
            # xzw = 1.0*xzw + 0.0*xzwd
            # # xzw = 0.5*xzw + 0.5*xzwd
        # xzw = xzwd
        # print((xzw[0][:10][:10]*10).int())


        outer = [zi,x,y,z,fs]
        inner = [i,mu,xzw,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        B = len(z)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)


        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,mu,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()
        # print(100*(mu[0:2,0,:10].detach().cpu().int()).numpy())
        # print()

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)

        # .matmul(lptok)
        # lptok = self.vocab(cand).log_softmax(-1)


        # lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0

class RefillModelRNNConvolveWithHiddenVectorDynamicFitting(RefillModelRNNBase):
    '''
    Adding logsumexp energy to the model
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
        self.embed      = nn.Embedding(graph_dim,embed_dim,)
        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim)
        self.kernel_size = kernel_size = 10
        self.init_state = nn.Linear(mixture_count, embed_dim)
        self.transition = nn.Linear(embed_dim*1,embed_dim)
        self.transcore  = nn.Linear(embed_dim,embed_dim)
        self.updater    = nn.Linear(embed_dim,embed_dim)
        self.emittor    = nn.Linear(embed_dim,embed_dim)
        # self.attention_probe = nn.Linear(embed_dim,kernel_size)
        self.mu = nn.Linear(embed_dim,1)
        self.KE = KE = embed_dim//kernel_size
        self.attention_head  = nn.Linear(embed_dim,kernel_size*KE)
        self.attention_head_l  = nn.Linear(kernel_size,embed_dim*KE)
        self.attention_head_r  = nn.Linear(kernel_size,embed_dim*KE)
        # self.attention_head_2  = nn.Linear(embed_dim,kernel_size*embed_dim)
        # self.attention_head_l2  = nn.Linear(kernel_size,embed_dim*KE)
        # self.attention_head_r2  = nn.Linear(kernel_size,embed_dim*KE)
        # self.attention_head_r  = nn.Linear(embed_dim,kernel_size*KE)
        # self.K = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.U = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.K2 = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.U2= nn.Linear(embed_dim//2,embed_dim).to(self.device)


        # self.K = nn.Linear(embed_dim,embed_dim)
        # self.U = nn.Linear(embed_dim,embed_dim)
        # self.K2 = nn.Linear(embed_dim,embed_dim)
        # self.U2= nn.Linear(embed_dim,embed_dim)

        # self.att_prob   = nn.Linear(embed_dim,embed_dim)
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
        mu = self.mu.weight[None,0:1]+(xsa[:,0:1]*0)
        mu = xsa.mean(1,keepdims=True)

        # mu = self.attention_head
        xsa = z
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        # xz = None
        xsa = xsa*0
        gradsq = xsa*0
        # xzw = self.updater.weight
        xzw = self.transcore.weight
        xzw = xzw[None,].repeat((len(z),1,1))
        # xw =
        outer = [zi, x,y,z,fs]
        inner = [i,mu,xzw,xsa]
        return outer,inner
    # def att_func(self,att):
    #     return att
    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,mu,xzw,xsa) = inner
        xzw= None
        # sel = None
        xz  = None

        E = xsa.size(-1)
        L = xsa.size(1)
        B = xsa.size(0)
        K = self.kernel_size
        KE =self.KE

        # wK = torch.eye(E).to(self.device)
        # wK = self.K.weight
        # wK = wK.matmul(wK.T)
        wT = self.transition.weight
        # wT = wT.matmul(wT.T)/wT.shape[1]**0.5
        # zk = self.attention_probe.weight.T

        akl = self.attention_head_l.weight.T
        akr = self.attention_head_r.weight.T
        ak = akl.reshape((K,E,KE)).matmul(akr.reshape((K,KE,E))).reshape((E,K*E))/KE**0.5

        # akl2 = self.attention_head_l2.weight.T
        # akr2 = self.attention_head_r2.weight.T
        # ak2 = akl2.reshape((K,E,KE)).matmul(akr2.reshape((K,KE,E))).reshape((E,K*E))/KE**0.5
        # E ( K E )
        zk  = mu.matmul(ak).reshape((B,K,E))

        peki =  (xsa.matmul(zk.transpose(2,1)).transpose(2,1))
        # peki = peki - 0.5*mu.matmul(mu.transpose(2,1))
        # peki = peki - 0.5*xsa.square().sum(-1,keepdims=True).transpose(2,1)
        peki = peki /E**0.5
        pki = peki.softmax(-1)
        pik = pki.transpose(2,1)
        pkix  = pki.matmul(xsa)
        #(K,E,KE)

        # akmu = mu.matmul(ak).reshape((B,-1,E))
        # akmu.reshape((B,-1,E))
        akmu = zk
        xakmu = xsa.matmul(akmu.transpose(2,1))
        # import pdb; pdb.set_trace()
        mud = 0
        # mud
        mud = mud + pkix.reshape((B,-1,K*E)).matmul(ak.T) #+ self.updater.bias[None,None]
        #
        # v = (pki*(1-pki)*xakmu.transpose(2,1)).transpose(2,1)
        # vv = v.reshape((B,1,L*K)).matmul(xsa.matmul(ak2.reshape((E,K,E)).permute((2,1,0)).reshape((E,K*E))).reshape((B,L*K,E)))
        # # .matmul(ak2.reshape((E,K,E)).permute((1,2,0)).reshape((K,E*E)))
        # mud = mud +vv
        mud = mud /K
        # xsa.matmul(ak2))

        xnew=1. *(
            xsa.roll(1,1).matmul(wT)    +
            xsa.roll(-1,1).matmul(wT.T) +
            # # xsa.roll(1,0).matmul(xzw)+
            # xsa.roll(0,1).matmul(self.transcore.weight)     +
            z.matmul(self.updater.weight.T) +
            # (pik*(1-pik)*xakmu).matmul(zk)/K +
            pik.matmul(akmu)/K +
            self.transition.bias +
            0.
            )

        # if i==L-1:
            # import pdb; pdb.set_trace()
        if i%2==1:
            xsa = 0.8 * xsa + 0.2 * xnew
            # xsa = 1.0 * xnew
            # xsa = 1.0 * xnew + 0.0 * xsa
            xsa = xsa/ (0.001+xsa.std(-1,keepdims=True))
        # xsa = 0.5 * xnew
        if i%2!=1:
            mu = 0.8 *mu + 0.2 * mud
            # mu = mu - mu.mean(-1)
            # mu = 1.0*mud
            # xzwd = z.transpose(2,1).matmul(xsa)/z.shape[1]
            # xzwd = xsa.transpose(2,1).matmul(xsa)/z.shape[1]
            # xzw = 1.0*xzw + 0.0*xzwd
            # # xzw = 0.5*xzw + 0.5*xzwd
        # xzw = xzwd
        # print((xzw[0][:10][:10]*10).int())

        xdict= dict(mu=mu,xsa=xsa,peki=peki.softmax(-1),xsaq=xsa.square().mean(-1,keepdim=True),

        zk = zk,
        )
        outer = [zi,x,y,z,fs]
        inner = [i,mu,xdict,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        B = len(z)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)


        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,mu,xdict,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()
        # print(100*(mu[0:2,0,:10].detach().cpu().int()).numpy())
        # print()

        if zi.min()==0:
            for k,xx in xdict.items():
                print(xx.shape)
                print((xx[:3,0:9,:10]*100).int())
                print((xx.std(0)[0:2,:10]))
                print(f'{k}_std:{xx.std()}')
                print(f'{k}_max:{xx.max()}')
                print(f'{k}_min:{xx.min()}')
                print()
            for k,v in self.named_parameters():

                # print(f'{k}:{self.embed.weight.max()}')
                print(f'{k}:\t\t{v.max():.3f}')
                # import pdb; pdb.set_trace()
            # print(self.state)
            print()


        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)

        # .matmul(lptok)
        # lptok = self.vocab(cand).log_softmax(-1)


        # lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0
import sys

class RefillModelRNNConvolveWithHiddenVectorDynamicFittingPaired(RefillModelRNNBase):
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
        self.embed      = nn.Embedding(graph_dim,embed_dim,)
        self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim)
        self.kernel_size = kernel_size = 10
        self.init_state = nn.Linear(mixture_count, embed_dim)
        self.transition = nn.Linear(embed_dim*1,embed_dim)
        self.transcore  = nn.Linear(embed_dim,embed_dim)
        self.updater    = nn.Linear(embed_dim,embed_dim)
        self.emittor    = nn.Linear(embed_dim,embed_dim)
        # self.attention_probe = nn.Linear(embed_dim,kernel_size)
        self.mu = nn.Linear(embed_dim,1)
        self.KE = KE = embed_dim//kernel_size
        self.attention_head  = nn.Linear(embed_dim,kernel_size*KE)
        self.attention_head_l  = nn.Linear(kernel_size,embed_dim*KE)
        self.attention_head_r  = nn.Linear(kernel_size,embed_dim*KE)
        self.attention_head_2  = nn.Linear(embed_dim,kernel_size*embed_dim)
        self.attention_head_l2  = nn.Linear(kernel_size,embed_dim*KE)
        self.attention_head_r2  = nn.Linear(kernel_size,embed_dim*KE)
        self.debug = '--debug' in sys.argv
        # self.attention_head_r  = nn.Linear(embed_dim,kernel_size*KE)
        # self.K = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.U = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.K2 = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        # self.U2= nn.Linear(embed_dim//2,embed_dim).to(self.device)


        # self.K = nn.Linear(embed_dim,embed_dim)
        # self.U = nn.Linear(embed_dim,embed_dim)
        # self.K2 = nn.Linear(embed_dim,embed_dim)
        # self.U2= nn.Linear(embed_dim,embed_dim)

        # self.att_prob   = nn.Linear(embed_dim,embed_dim)
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
        mu = self.mu.weight[None,0:1]+(xsa[:,0:1]*0)

        # mu = self.attention_head
        xsa = z
        # fs = torch.tensor([],requires_grad=True).to(self.device)
        # outer,inner = self.batch_init(zi,x,y,z,fs)
        i = -1
        sel = None
        # xz = None
        xsa = xsa*0
        gradsq = xsa*0
        # xzw = self.updater.weight
        xzw = self.transcore.weight
        xzw = xzw[None,].repeat((len(z),1,1))
        # xw =
        outer = [zi, x,y,z,fs]
        inner = [i,mu,xzw,xsa]
        return outer,inner
    # def att_func(self,att):
    #     return att
    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,mu,xzw,xsa) = inner
        xzw= None
        # sel = None
        xz  = None

        E = xsa.size(-1)
        L = xsa.size(1)
        B = xsa.size(0)
        K = self.kernel_size
        KE =self.KE

        # wK = torch.eye(E).to(self.device)
        # wK = self.K.weight
        # wK = wK.matmul(wK.T)
        wT = self.transition.weight
        # wT = wT.matmul(wT.T)/wT.shape[1]**0.5
        # zk = self.attention_probe.weight.T

        akl = self.attention_head_l.weight.T
        akr = self.attention_head_r.weight.T
        ak = akl.reshape((K,E,KE)).matmul(akr.reshape((K,KE,E))).reshape((E,K*E))/KE**0.5

        akl2 = self.attention_head_l2.weight.T
        akr2 = self.attention_head_r2.weight.T
        ak2 = akl2.reshape((K,E,KE)).matmul(akr2.reshape((K,KE,E))).reshape((E,K*E))/KE**0.5


        # ak2 = ak
        # E ( K E )
        zk  = mu.matmul(ak).reshape((B,K,E))

        peki =  (xsa.matmul(zk.transpose(2,1)).transpose(2,1))
        # peki = peki - 0.5*mu.matmul(mu.transpose(2,1))
        # peki = peki - 0.5*xsa.square().sum(-1,keepdims=True).transpose(2,1)
        peki = peki /E**0.5
        pki = peki.softmax(-1)
        pik = pki.transpose(2,1)
        pkix  = pki.matmul(xsa)
        #(K,E,KE)

        akmu = mu.matmul(ak2).reshape((B,-1,E))
        # akmu.reshape((B,-1,E))
        # akmu = zk
        xakmu = xsa.matmul(akmu.transpose(2,1))
        # import pdb; pdb.set_trace()
        mud = 0
        # mud
        mud = mud + pkix.reshape((B,-1,K*E)).matmul(ak2.T) #+ self.updater.bias[None,None]
        #
        withCorrcetion=0
        if withCorrcetion:
            v = (pki*(1-pki)*xakmu.transpose(2,1)).transpose(2,1)
            vv = v.reshape((B,1,L*K)).matmul(xsa.matmul(ak2.reshape((E,K,E)).permute((2,1,0)).reshape((E,K*E))).reshape((B,L*K,E)))
            # # .matmul(ak2.reshape((E,K,E)).permute((1,2,0)).reshape((K,E*E)))
            mud = mud +vv
        mud = mud /K
        # xsa.matmul(ak2))

        xnew=1. *(
            xsa.roll(1,1).matmul(wT)    +
            xsa.roll(-1,1).matmul(wT.T) +
            # # xsa.roll(1,0).matmul(xzw)+
            # xsa.roll(0,1).matmul(self.transcore.weight)     +
            z.matmul(self.updater.weight.T) +
            ((pik*(1-pik)*xakmu).matmul(zk)/K if withCorrcetion else 0) +
            pik.matmul(akmu)/K +
            self.transition.bias +
            0.
            )

        # if i==L-1:
            # import pdb; pdb.set_trace()
        if i%2==1:
            xsa = 0.8 * xsa + 0.2 * xnew
            # xsa = 1.0 * xnew
            # xsa = 1.0 * xnew + 0.0 * xsa
            xsa = xsa/(0.001+xsa.std(-1,keepdims=True))
        # xsa = 0.5 * xnew
        if i%2!=1:
            mu = 0.8 *mu + 0.2 * mud
            # mu = mu - mu.mean(-1)
            mu = mu/(0.001+mu.std(-1,keepdims=True))*K**0.5
            # mu = 1.0*mud
            # xzwd = z.transpose(2,1).matmul(xsa)/z.shape[1]
            # xzwd = xsa.transpose(2,1).matmul(xsa)/z.shape[1]
            # xzw = 1.0*xzw + 0.0*xzwd
            # # xzw = 0.5*xzw + 0.5*xzwd
        # xzw = xzwd
        # print((xzw[0][:10][:10]*10).int())

        xdict= dict(mu=mu,xsa=xsa,peki=peki.softmax(-1),
        # xsaq=xsa.square().mean(-1,keepdim=True),

        zk = zk,
        )
        outer = [zi,x,y,z,fs]
        inner = [i,mu,xdict,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        B = len(z)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)


        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,mu,xdict,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()
        # print(100*(mu[0:2,0,:10].detach().cpu().int()).numpy())
        # print()
        # import pdb; pdb.set_trace()
        if self.debug and zi.min()==0:
            for k,xx in xdict.items():
                print(xx.shape)
                print((xx[:3,0:9,:8]*100).int())
                print((xx.std(0)[0:2,:8]))
                print(f'{k}_std:{xx.std()}')
                print(f'{k}_max:{xx.max()}')
                print(f'{k}_min:{xx.min()}')
                print()
            for k,v in self.named_parameters():

                # print(f'{k}:{self.embed.weight.max()}')
                print(f'{k}:\t\t{v.max():.3f}')
                # import pdb; pdb.set_trace()
            # print(self.state)
            print()


        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)

        # .matmul(lptok)
        # lptok = self.vocab(cand).log_softmax(-1)


        # lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0


class RefillModelRNNConvolveWithAttentionWithGrad(RefillModelRNNBase):
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
        self.transcore  = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
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
        gradsq = xsa*0
        outer = [zi, x,y,z,fs]
        inner = [i,gradsq,xz,xsa]
        return outer,inner
    # def att_func(self,att):
    #     return att
    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,gradsq,xz,xsa) = inner
        # sel = None
        xz  = None

        E = xsa.size(-1)
        L = xsa.size(1)
        att = xsa.matmul(self.att_kernel.weight).matmul(xsa.transpose(2,1))
        # att = (att /E ).abs()
        # attr = att.transpose(2,1)
        # attl = att
        # attl = attl/(1+attl.sum(-1,keepdims=True))
        # attr = attr/(1+attr.sum(-1,keepdims=True))
        # /(1+attl.std(-1,keepdims=True))
        # attl = (attl-attl.mean(-1,keepdims=True))/(1+attl.std(-1,keepdims=True))
        # attr = (attr-attr.mean(-1,keepdims=True))/(1+attr.std(-1,keepdims=True))

        # att = att / E**0.5
        # att = att.softmax(-1)

        ##### notoriously more difficult to train
        att = (att).abs()
        # att = att / 1
        # att = att*(1-torch.eye(L)[None].to(self.device))
        attr = att.transpose(2,1)
        attl = att
        # .softmax(-1)
        # att = (att + att.transpose(2,1))/2
        sumnorm = lambda x:x/x.sum(-1,keepdims=True)
        # yy = yy*self.att_prob(xsa)[:,:,:1].sigmoid()

        xnew=1./3 *(
            xsa.roll(1,1).matmul(self.transition.weight)+
            xsa.roll(1,-1).matmul(self.transition.weight.T)+
            xsa.roll(1,0).matmul(self.transcore.weight)+
            attl.matmul(xsa).matmul(self.att_energy.weight.T)+
            attr.matmul(xsa).matmul(self.att_energy.weight)+
            # self.att_energy(att.matmul(xsa))+

            z.matmul(self.updater.weight.T)
            +self.transition.bias
            )
        # xsa = 0.5* xsa + 0.5*xnew

        # xnew = xnew/(1.+xnew.std(-1,keepdims=True))
        # xsa = 1.0*xsa + 0.1*(-xsa + xnew)

        grad = (-0.1*xsa + xnew)
        grad = grad/(1.+grad.std(-1,keepdims=True))
        xsa = 1.0*xsa + 0.1*grad

        # grad = (0.1*xsa + xnew)
        # gradsq = 0.3*gradsq+0.7*grad.square()
        # xsa = 1.0*xsa + 0.1*grad/gradsq.sqrt()
        # (-xsa + xnew)

        # xsa = torch.cat([xsa,xnew],dim=2)

        # mask = torch.randint(2,xsa.shape[:-1])[:,:,None].to(self.device)
        # arange(0,1)==1
        # import pdb; pdb.set_trace()

        # xsa = mask * xnew + (1-mask ) * xsa
        # xsa + 0.5*xnew



        outer = [zi,x,y,z,fs]
        inner = [i,gradsq,xz,xsa]
        return outer,inner


    def _loss(self,zi,x,y,z,out='loss'):
        assert out in 'loss token traj'.split()

        outer,inner = self._batch_init(zi,x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = z.size(1)
        B = len(z)
        for i in range(L):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)



        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,sel,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # f1 = self.vocab(xsa).log_softmax(-1)
        # # .matmul(self.emittor.weight)
        # f2 = self.vocab(z).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)

        # .matmul(lptok)
        # lptok = self.vocab(cand).log_softmax(-1)


        # lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0




class RefillModelRNNConvolveWithAttentionDist(RefillModelRNNBase):
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
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_kernel = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_energy = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_prob   = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.gate       = nn.Linear(embed_dim,embed_dim).to(self.device)
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

        # y = self.att_kernel(xsa).matmul(xsa.transpose(2,1)).softmax(-1).matmul(xsa)
        att = -(self.att_kernel(xsa)[:,:,None,:,] - self.att_prob(xsa)[:,None,:,:]).square().softmax(-1)
        kl = self.att_energy.weight
        kr = self.att_prob.weight
        kj = xsa.matmul(kl)

        yy = (att*kj[:,None]).sum(-2)
        yyy = yy.matmul(kr)

        # yyy = yyy * self.gate(xsa)[:,:1].sigmoid()
        # y.matmul(kl)
        # import pdb; pdb.set_trace()
        # y =
        # .matmul(xsa.transpose(2,1)).softmax(-1).matmul(xsa)

        xnew=1./3 *(
            xsa.roll(1,1).matmul(self.transition.weight)+
            xsa.roll(1,-1).matmul(self.transition.weight.T)+
            # self.att_energy(y)+
            yyy +
            z.matmul(self.updater.weight.T)
            +self.transition.bias
            )
        # xsa = 0.5* xsa + 0.5*xnew
        # xsa = xnew
        # xsa + 0.5*xnew

        grad = (-0.1*xsa + xnew)
        grad = grad/(1.+grad.std(-1,keepdims=True))
        xsa = 1.0*xsa + 0.1*grad


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
        # for i in range(L):
        #     inner[0]=L-1-i
        #     outer,inner = self._step(outer,inner) #### do what ever with your hidden state
        #     self.callback_step(outer,inner)

        self.callback_end(outer)
        (zi,x,y,z,lptok) = outer
        (i,sel,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # f1 = self.vocab(xsa).log_softmax(-1)
        # # .matmul(self.emittor.weight)
        # f2 = self.vocab(z).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()

        xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
        xkey_dynamic= self.xkey_dynamic(y)
        xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
        # import pdb; pdb.set_trace()
        cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
        sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
        lptok = self.vocab(cand).log_softmax(-1)
        lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)

        # lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0

class RefillModelRNNConvolveSmall(RefillModelRNNConvolve):
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
        self.embed_dim = embed_dim - 5
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
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_kernel = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_energy = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_prob   = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.fs_type    = 'lptok'




class RefillModelRNNConvolveWithSelection(RefillModelRNNBase):
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
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.selector    = nn.Linear(embed_dim,embed_dim).to(self.device)
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

        xnew=1./3 *(
            xsa.roll(1,1).matmul(self.transition.weight)+
            xsa.roll(1,-1).matmul(self.transition.weight.T)+
            z.matmul(self.updater.weight.T)
            +self.transition.bias
            )
        # xsa = 0.5* xsa + 0.5*xnew
        xsa = xnew
        # xsa + 0.5*xnew


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
        (i,sel,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # f1 = self.vocab(xsa).log_softmax(-1)
        # # .matmul(self.emittor.weight)
        # f2 = self.vocab(z).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()
        # torch.cat([])
        g = self.selector(xsa)[:,:,:1].sigmoid()
        f1 = self.vocab(xsa.matmul(self.emittor.weight)).softmax(-1)
        f2 = self.vocab(z).softmax(-1)
        v = f1*g+(1-g)*f2
        lptok = v.log()



        # lptok = self.vocab(v).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0

class RefillModelRNNConvolveLowRank(RefillModelRNNBase):
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
        self.transition = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        self.transition2 = nn.Linear(embed_dim//2,embed_dim).to(self.device)
        self.updater    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.emittor    = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_kernel = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_energy = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.att_prob   = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.fs_type    = 'lptok'

    def vocab(self,x):
        y = (x).matmul(self.embed.weight.T) -  self.embed.weight.square().mean(-1)[None,None]
        return y

    def _batch_init(self,zi,x,y,z):
        #### batch_init part
        ### state init
        z = self.embed(z)
        y = self.embed(y)
        # xs = torch.cat([y,init_state],dim=1)
        xsa = self.init_state.weight.T[None,0:1]
        xsa = xsa.repeat((len(z),z.size(1),1))
        xsa = self.norm(xsa)*0

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
        W = self.transition.weight.matmul(self.transition2.weight.T)

        xnew=1./3 *(
            xsa.roll(1,1).matmul(W)+
            xsa.roll(1,-1).matmul(W.T)+
            z.matmul(self.updater.weight.T)
            +self.transition.bias
            )
        # xsa = 0.5* xsa + 0.5*xnew
        xsa = xnew
        # xsa + 0.5*xnew


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
        (i,sel,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # f1 = self.vocab(xsa).log_softmax(-1)
        # # .matmul(self.emittor.weight)
        # f2 = self.vocab(z).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()


        lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0

class RefillModelRNNConvolveHighRank(RefillModelRNNConvolveLowRank):
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
        self.transition = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.transition2 = nn.Linear(embed_dim,embed_dim).to(self.device)
    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xz,xsa) = inner
        sel = None
        xz  = None
        W = self.transition.weight.matmul(self.transition2.weight.T)

        xnew=1./3 *(
            xsa.roll(1,1).matmul(self.transition.weight),
            # .relu().matmul(self.transition2.weight.T)+
            xsa.roll(1,-1).matmul(self.transition.weight.T),
            # .relu().matmul(self.transition.weight.T)+
            z.matmul(self.updater.weight.T)
            +self.transition.bias
            )
        xsa = 0.5* xsa + 0.5*xnew
        # xsa = xnew
        # xsa + 0.5*xnew


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
        (i,sel,xz,xsa) = inner
        # lptok = self.vocab(xsa.matmul(self.updater.weight)).log_softmax(-1)
        # f1 = self.vocab(xsa).log_softmax(-1)
        # # .matmul(self.emittor.weight)
        # f2 = self.vocab(z).log_softmax(-1)
        # sel = self.emittor(xsa)[:,:,0].sigmoid()


        lptok = self.vocab(self.emittor(xsa)).log_softmax(-1)

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0




class RefillModelRNNConvolveHighRank2(RefillModelRNNConvolveLowRank):
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
        self.transition = nn.Linear(embed_dim*2,embed_dim).to(self.device)
        self.transition2 = nn.Linear(embed_dim*2,embed_dim).to(self.device)

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (zi,x,y,z,fs) = outer
        (i,sel,xz,xsa) = inner
        sel = None
        xz  = None
        W = self.transition.weight.matmul(self.transition2.weight.T)

        xnew=(xsa.roll(1,1).matmul(self.transition.weight).relu().matmul(self.transition2.weight.T)+
            xsa.roll(1,-1).matmul(self.transition2.weight).relu().matmul(self.transition.weight.T)+
            z.matmul(self.updater.weight.T)
            +self.transition.bias)
        xsa = 0.5* xsa + 0.5*xnew
        # xsa = xnew
        # xsa + 0.5*xnew


        outer = [zi,x,y,z,fs]
        inner = [i,sel,xz,xsa]
        return outer,inner

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
        xs  = xs + (val).matmul(self.att_energy.weight.T)

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



class RefillModelRNNAdditiveDirectMixingWithRegressionAttention(RefillModelRNNBase):
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
        # (B,L,E)
        # (B,1,E)
        xq  = self.att_kernel(xs)
        att = (-(xq  - xsa).square().mean(-1)).softmax(-1)[:,None]
        # att = xq.matmul(xsa.transpose(2,1)).softmax(-1)
        att = att * (self.att_prob(xs)[:,:,0:1].sigmoid())
        val = att.matmul(xsa)
        xs  = xs + (val).matmul(self.att_energy.weight.T)

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

class RefillModelRNNAdditiveDirectMixingWithKAttention(RefillModelRNNBase):
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
        self.att_kernel = nn.Linear(embed_dim,embed_dim-1).to(self.device)
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

        W = self.att_kernel.weight.T
        ky  = xsa.matmul(W)
        kxb = self.att_kernel(xs)
        #(B,1,E)
        #(B,L,E)
        att = ((kxb - 0.5 * ky)*ky).mean(-1)
        att = att.softmax(-1)[:,None]

        att = att * (self.att_prob(xs)[:,:,0:1].sigmoid())
        # import pdb; pdb.set_trace()

        #####

        # att = xq.matmul().softmax(-1)

        val = att.matmul(xsa)
        xs  = xs + (val).matmul(self.att_energy.weight.T)

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
