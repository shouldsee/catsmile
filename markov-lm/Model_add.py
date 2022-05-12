import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


from transformers.models.bert.modeling_bert import BertLayer,BertConfig


class AddModelBase(nn.Module):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        depth,
        mask_token_idx):

        state_count = 1
        super().__init__()
        # state_count = 15
        self.device = device
        # self.total_length = total_length
        # self.min_len = min_len
        # self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        # self.state_count = state_count
        self.depth = depth
        self.D = depth

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,)
        # self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim)
        self.kernel_size = kernel_size = 4
        # self.init_state = nn.Linear(mixture_count, embed_dim)
        self.transition = nn.Linear(embed_dim*1,embed_dim)
        self.transcore  = nn.Linear(embed_dim,embed_dim)
        self.updater    = nn.Linear(embed_dim,embed_dim)
        self.emittor    = nn.Linear(embed_dim,embed_dim)
        self.attention_probe = nn.Linear(embed_dim,kernel_size)
        self.mu = nn.Linear(embed_dim,1)
        self.KE = KE = embed_dim//kernel_size//2
        self.E = E = embed_dim
        self.D = depth

        self.attention_head  = nn.Linear(embed_dim,kernel_size*KE)
        self.attention_head_l  = nn.Linear(kernel_size,embed_dim*KE)
        self.attention_head_r  = nn.Linear(kernel_size,embed_dim*KE)


        self.K = nn.Linear(embed_dim,embed_dim)
        self.U = nn.Linear(embed_dim,embed_dim)
        self.K2 = nn.Linear(embed_dim,embed_dim)
        self.U2= nn.Linear(embed_dim,embed_dim)

        self.att_prob   = nn.Linear(embed_dim,embed_dim)

        bconf = BertConfig()
        bconf.hidden_size=E
        bconf.intermediate_size = E*2
        bconf.num_attention_heads = 10
        self.blayer_list = nn.ModuleList([BertLayer(bconf) for _ in range(20)])
        self.fs_type    = 'lptok'

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (i,z,xsa) = inner
        # (zi,x,y,z,fs) = outer
        # (i,mu,xzw,xsa) = inner
        peki = xsa

        E = xsa.size(-1)
        L = xsa.size(1)
        B = xsa.size(0)
        K = self.kernel_size
        KE =self.KE
        # for blayer in self.
        blayer = self.blayer_list[i]

        xsad = 0.
        xsad = xsad + xsa.roll(1,1) @ self.transition.weight
        xsad = xsad + xsa.roll(-1,1) @ self.transition.weight.T

        xsa = blayer(xsa)[0]
        xsa = xsa + xsad
        xsa = xsa / (0.1 + xsa.std(-1,keepdims=True))

        outer = (None,)
        inner = [i,z,xsa]

        # outer = [zi,x,y,z,fs]
        # inner = [peki,mu,xzw,xsa]
        return outer,inner

    def _batch_init(self,zi,masked):
        #### batch_init part
        ### state init
        z = masked
        z = self.embed(z)
        z = self.norm(z)
        y = None
        xsa = z

        i = -1
        outer = (None,)
        inner = [i,z,xsa]
        # outer = [zi, x,y,z,fs]
        # inner = [i,mu,xzw,xsa]
        return outer,inner
    # def att_func(self,att):
    #     return att
    def callback_step(self,outer,inner):
        # [zi,x,y,z,fs],[i,sel,xz,xs] = outer,inner
        return
    def callback_init(self,outer):
        return
    def callback_end(self,outer):
        return

    def norm(self,y):
        y = y / (0.00001 + y.std(-1,keepdims=True)) *1.0
        return y

    def vocab(self,x):
        y = x.matmul(self.embed.weight.T)
        return y


    def target_energy(self,lptok,yt):
        yp = torch.gather(lptok,index=yt[:,:,None],dim=-1)[:,:,0]
        return yp

    def loss(self,item):
        return self._loss(item,out='loss')
    grad_loss = loss




    def _loss(self,item,out='loss'):
        masked = item['masked']
        unmasked = item['unmasked']
        mask = item['mask']
        zi = None
        assert out in 'loss token traj'.split()
        # zi = None
        outer,inner = self._batch_init(zi,masked)
        # x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer)
        L = masked.size(1)
        B = masked.size(0)
        for i in range(self.D):
            inner[0]=i
            outer,inner = self._step(outer,inner) #### do what ever with your hidden state
            self.callback_step(outer,inner)


        self.callback_end(outer)
        _ = outer
        # (i,z,[r*I/xsa) = inner
        xsa = inner[-1]

        lptok = torch.gather(xsa,index=mask[:,None,None].repeat((1,1,xsa.shape[-1])),dim=1)
        lptok = self.vocab(lptok.matmul(self.emittor.weight)).log_softmax(-1)
        x = torch.gather(unmasked,index=mask[:,None,],dim=1).long()

        cent = self.target_energy(lptok,x)
        loss  = -cent.mean(-1)
        # if out=='token': return lptok
        if out=='token': return lptok

        if out=='loss': return loss
        assert 0

        if out=='loss': return loss
        assert 0

class QKV_Attention(nn.Module):
    def __init__(self,embed_dim,kernel_size,kernel_dim):
        super().__init__()
        assert kernel_dim == embed_dim
        self.attq = nn.Linear(embed_dim,kernel_size*kernel_dim)
        self.attk = nn.Linear(embed_dim,kernel_size*kernel_dim)
        self.bias = nn.Linear(1,kernel_size)
        self.kernel_size = kernel_size

    def forward(self,xsa):
        B,L,E = xsa.shape[:3]
        K = self.kernel_size
        hijk = (xsa @ self.attq.weight.T).reshape((B,L,K,E)) @ xsa.reshape((B,1,L,E)).transpose(-2,-1)
        hijk = hijk / E**0.5
        pijk = hijk.softmax(-1)
        yik = pijk @ xsa.reshape((B,1,L,E))

        # eijk = hijk.logsumexp(-1)
        # yik = yik* (eijk-self.bias.bias[None,None,:]).sigmoid().unsqueeze(-1)

        xid = yik.reshape((B,L,K*E))
        return xid,pijk


class Gaussian_Attention(nn.Module):
    def __init__(self,embed_dim,kernel_size,kernel_dim):
        super().__init__()
        assert kernel_dim == embed_dim
        kernel_dim = embed_dim//2
        self.attq = nn.Linear(embed_dim,kernel_size*kernel_dim)
        self.attk = nn.Linear(embed_dim,kernel_size*kernel_dim)
        self.attv = nn.Linear(embed_dim*kernel_size,embed_dim)
        # kernel_dim)
        self.bias = nn.Linear(kernel_size,kernel_dim)
        self.bias2 = nn.Linear(kernel_size,embed_dim)
        self.kernel_size = kernel_size
        self.kernel_dim = kernel_dim

    def forward(self,xsa):
        B,L,E = xsa.shape[:3]
        K = self.kernel_size
        KE = self.kernel_dim
        # xsa@
        xq=  (xsa @ self.attq.weight.T).reshape((B,L,1,K,KE))
        xk = (xsa @ self.attk.weight.T).reshape((B,1,L,K,KE))
        xd=  (xq - xk - self.bias.weight.T[None,None,None])
        hikj = (-xd.square().sum(-1)).transpose(2,-1)
        # import pdb; pdb.set_trace()
        # hikj = (xsa @ self.attq.weight.T).reshape((B,L,K,E)) @ xsa.reshape((B,1,L,E)).transpose(-2,-1)
        hikj = hikj / KE**0.5
        pikj = hikj.softmax(-1)
        yik  = pikj @ xsa.reshape((B,1,L,E))
        yik = (yik.unsqueeze(-1)*self.attv.weight.reshape((1,1,K,E,E))).sum(-2)
        yik = yik+self.bias2.weight.T[None,None]
        # import pdb; pdb.set_trace()

        # eijk = hijk.logsumexp(-1)
        # yik = yik* (eijk-self.bias.bias[None,None,:]).sigmoid().unsqueeze(-1)

        xid = yik.reshape((B,L,K*E))
        return xid

AddModelWithBert = AddModelBase
class AddModelWithAttention(AddModelBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        depth,
        kernel_size,
        use_dropout,
        use_dense_relu,
        use_layernorm,

        mask_token_idx):

        state_count = 1
        super().__init__(device,
            graph_dim,
            embed_dim,
            depth,
            mask_token_idx)
        self.use_dropout = use_dropout
        self.use_dense_relu = use_dense_relu
        self.use_layernorm = use_layernorm
        # state_count = 15
        self.device = device
        # self.total_length = total_length
        # self.min_len = min_len
        # self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        # self.state_count = state_count
        self.depth = depth
        self.D = depth

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,)
        # self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim)
        self.kernel_size = kernel_size = 4
        # self.init_state = nn.Linear(mixture_count, embed_dim)
        self.transition = nn.Linear(embed_dim*1,embed_dim)
        self.transcore  = nn.Linear(embed_dim,embed_dim)
        self.updater    = nn.Linear(embed_dim,embed_dim)
        self.emittor    = nn.Linear(embed_dim,embed_dim)
        # self.attention_probe = nn.Linear(embed_dim,kernel_size)
        self.mu = nn.Linear(embed_dim,1)
        self.KE = KE = embed_dim//kernel_size//2
        self.E = E = embed_dim
        self.D = depth

        # self.attention_head  = nn.Linear(embed_dim,kernel_size*KE)
        # self.attention_head_l  = nn.Linear(kernel_size,embed_dim*KE)
        # self.attention_head_r  = nn.Linear(kernel_size,embed_dim*KE)
        # kernel_size =
        self.attention = QKV_Attention(embed_dim,kernel_size,embed_dim)
        # self.attention = Gaussian_Attention(embed_dim,kernel_size,embed_dim)
        # self.attq = nn.Linear(embed_dim,kernel_size*embed_dim)
        # self.attk = nn.Linear(embed_dim,kernel_size*embed_dim)
        self.attv = nn.Linear(embed_dim,kernel_size*embed_dim)
        self.atto = nn.Linear(embed_dim,kernel_size*embed_dim)
        self.att_dropout = nn.Dropout(0.1)
        self.att_dense = nn.Linear(kernel_size*embed_dim,kernel_size*embed_dim)
        latent_size = 4*embed_dim
        self.attk2 =  nn.Linear(kernel_size*embed_dim,latent_size)
        self.attv2 =  nn.Linear(embed_dim, latent_size )


        self.K = nn.Linear(embed_dim,embed_dim)
        self.U = nn.Linear(embed_dim,embed_dim)
        self.K2 = nn.Linear(embed_dim,embed_dim)
        self.U2= nn.Linear(embed_dim,embed_dim)

        self.att_prob   = nn.Linear(embed_dim,embed_dim)
        self.update_dropout = nn.Dropout(0.5)
        # self.dense1 = nn.Linear()

        bconf = BertConfig()
        bconf.hidden_size=E
        bconf.intermediate_size = E*2
        bconf.num_attention_heads = 10
        # self.blayer_list = nn.ModuleList([BertLayer(bconf) for _ in range(20)])
        self.fs_type    = 'lptok'

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (i,z,gradsq,xsa) = inner
        # (zi,x,y,z,fs) = outer        graph_dim,
        # embed_dim,
        # depth,
        # mask_token_idx)

        # (i,mu,xzw,xsa) = inner
        peki = xsa

        E = xsa.size(-1)
        L = xsa.size(1)
        B = xsa.size(0)
        K = self.kernel_size
        KE =self.KE
        # for blayer in self.
        # blayer = self.blayer_list[i]

        _norm = lambda  xsa,dim=-1: xsa /(1+xsa.std(dim,keepdims=True))
        xsad = 0.
        # xsad = xsad + xsa.roll(1,1) @ self.transition.weight
        # xsad = xsad + xsa.roll(-1,1) @ self.transition.weight.T

        xsad = xsad + (xsa.roll(1,1) @ self.transition.weight).relu()@self.transcore.weight
        xsad = xsad + (xsa.roll(-1,1) @ self.transcore.weight.T).relu() @ self.transition.weight.T

        xid,pijk = self.attention(xsa)

        # withCorrection = 1
        if self.use_dense_relu==1:
            # xidact = self.att_dense(xid).sin()
            xidact =  xid @ self.att_dense.weight.T
            xidact = xidact.relu()
            xid = xidact @ self.atto.weight.T.T

        elif self.use_dense_relu==4:
            xidact = self.att_dense(xid).relu()

            xid = xidact @ self.atto.weight.T.T
            # xid = 0.
            xc  = xsa @ self.atto.weight.T
            xcc = (xc * (xidact>0))@self.att_dense.weight
            # xc.shape
            xp  = pijk.reshape((B,L*K,L)).transpose(2,1) @ (xcc.reshape((B,L*K,E)))
            val = xcc.reshape((B,L*K,E))@xsa.transpose(2,1)
            xw  = xsa @ self.attention.attq.weight.T
            xp  = xp + (((1-pijk)*pijk).reshape((B,L*K,L))* val).transpose(2,1) @ xw.reshape((B,L*K,E))/KE**0.5

            # xsa@
            xid = xid + xp

        elif self.use_dense_relu==2:
            xid = (xid @ self.attk2.weight.T /(E*K)**0.5).softmax(-1) @ self.attv2.weight.T.T
            pass
        elif self.use_dense_relu==3:
            xid = xid@self.attention.attq.weight
        elif self.use_dense_relu==5:
            xid = xid @ self.atto.weight.T.T
        else:
            assert 0,self.use_dense_relu


        if self.use_dropout: xid = self.att_dropout(xid)

        # xid = xid.tanh()
        # if self.use_dropout:
        #     xid = self.att_dropout(xid)
        xsad = xsad+ xid

        # gradsq = xsad.square()+gradsq * 0.3
        # xsad = xsad / gradsq
        # if self.use_dropout: xsad = self.att_dropout(xsad)
        # xsad = xsad / (0.1 + xsad.std(-1,keepdims=True))
        # if self.use_dropout: xsad = xsad  + torch.normal(0,0.1,xsad.shape,device=self.device)
        # gradsq = gradsq + xsad.square()

        # xsa = xsa + 0.01* xsad / (0.01 + gradsq**0.5)
        # mask = self.att_dropout(torch.ones(xsa.shape[:2]+(1,),device=self.device))
        mask = torch.ones(xsa.shape[:2]+(1,),device=self.device)
        if self.use_dropout:
            mask = self.update_dropout(mask)
        # import pdb; pdb.set_trace()
        # mask = self.update_dropout(mask)
        xsad = 0.05* xsad / (1 + xsad.std(-1,keepdims=True))
        xsa = xsa + xsad * mask
        if self.use_layernorm:
            xsa = xsa / (1. + xsa.std(-1,keepdims=True))

        outer = (None,)
        inner = [i,z,gradsq,xsa]

        # outer = [zi,x,y,z,fs]
        # inner = [peki,mu,xzw,xsa]
        return outer,inner

    def _batch_init(self,zi,masked):
        #### batch_init part
        ### state init
        z = masked
        z = self.embed(z)
        z = self.norm(z)
        y = None
        xsa = z

        gradsq = xsa *0.
        i = -1
        outer = (None,)
        inner = [i,z,gradsq,xsa]
        # outer = [zi, x,y,z,fs]
        # inner = [i,mu,xzw,xsa]
        return outer,inner


class AddModelWithAttentionSimpleUpdate(AddModelBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        depth,
        kernel_size,
        use_dropout,
        use_dense_relu,
        use_layernorm,

        mask_token_idx):

        state_count = 1
        super().__init__(device,
            graph_dim,
            embed_dim,
            depth,
            mask_token_idx)
        self.use_dropout = use_dropout
        self.use_dense_relu = use_dense_relu
        # state_count = 15
        self.device = device
        # self.total_length = total_length
        # self.min_len = min_len
        # self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        # self.state_count = state_count
        self.depth = depth
        self.D = depth

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,)
        # self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim)
        self.kernel_size = kernel_size = 4
        # self.init_state = nn.Linear(mixture_count, embed_dim)
        self.transition = nn.Linear(embed_dim*1,embed_dim)
        self.transcore  = nn.Linear(embed_dim,embed_dim)
        self.updater    = nn.Linear(embed_dim,embed_dim)
        self.emittor    = nn.Linear(embed_dim,embed_dim)
        # self.attention_probe = nn.Linear(embed_dim,kernel_size)
        self.mu = nn.Linear(embed_dim,1)
        self.KE = KE = embed_dim//kernel_size//2
        self.E = E = embed_dim
        self.D = depth

        # self.attention_head  = nn.Linear(embed_dim,kernel_size*KE)
        # self.attention_head_l  = nn.Linear(kernel_size,embed_dim*KE)
        # self.attention_head_r  = nn.Linear(kernel_size,embed_dim*KE)
        # kernel_size =
        self.attention = QKV_Attention(embed_dim,kernel_size,embed_dim)
        # self.attention = Gaussian_Attention(embed_dim,kernel_size,embed_dim)
        # self.attq = nn.Linear(embed_dim,kernel_size*embed_dim)
        # self.attk = nn.Linear(embed_dim,kernel_size*embed_dim)
        self.attv = nn.Linear(embed_dim,kernel_size*embed_dim)
        self.atto = nn.Linear(embed_dim,kernel_size*embed_dim)
        self.att_dropout = nn.Dropout(0.1)
        self.att_dense = nn.Linear(kernel_size*embed_dim,kernel_size*embed_dim)
        latent_size = 4*embed_dim
        self.attk2 =  nn.Linear(kernel_size*embed_dim,latent_size)
        self.attv2 =  nn.Linear(embed_dim, latent_size )


        self.K = nn.Linear(embed_dim,embed_dim)
        self.U = nn.Linear(embed_dim,embed_dim)
        self.K2 = nn.Linear(embed_dim,embed_dim)
        self.U2= nn.Linear(embed_dim,embed_dim)

        self.att_prob   = nn.Linear(embed_dim,embed_dim)
        self.update_dropout = nn.Dropout(0.5)
        # self.dense1 = nn.Linear()
        self.use_layernorm = use_layernorm

        bconf = BertConfig()
        bconf.hidden_size=E
        bconf.intermediate_size = E*2
        bconf.num_attention_heads = 10
        # self.blayer_list = nn.ModuleList([BertLayer(bconf) for _ in range(20)])
        self.fs_type    = 'lptok'

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (i,z,gradsq,xsa) = inner
        # (zi,x,y,z,fs) = outer        graph_dim,
        # embed_dim,
        # depth,
        # mask_token_idx)

        # (i,mu,xzw,xsa) = inner
        peki = xsa

        E = xsa.size(-1)
        L = xsa.size(1)
        B = xsa.size(0)
        K = self.kernel_size
        KE =self.KE
        # for blayer in self.
        # blayer = self.blayer_list[i]

        _norm = lambda  xsa,dim=-1: xsa /(1+xsa.std(dim,keepdims=True))
        xsad = 0.
        # xsad = xsad + xsa.roll(1,1) @ self.transition.weight
        # xsad = xsad + xsa.roll(-1,1) @ self.transition.weight.T

        xsad = xsad + (xsa.roll(1,1) @ self.transition.weight).relu()@self.transcore.weight
        xsad = xsad + (xsa.roll(-1,1) @ self.transcore.weight.T).relu() @ self.transition.weight.T

        xid,pijk = self.attention(xsa)

        # withCorrection = 1
        if self.use_dense_relu==1:
            xidact = self.att_dense(xid).relu()
            xid = xidact @ self.atto.weight.T.T

        elif self.use_dense_relu==4:
            xidact = self.att_dense(xid).relu()
            xid = xidact @ self.atto.weight.T.T
            # xid = 0.
            xc  = xsa @ self.atto.weight.T
            xcc = (xc * (xidact>0))@self.att_dense.weight
            # xc.shape
            xp  = pijk.reshape((B,L*K,L)).transpose(2,1) @ (xcc.reshape((B,L*K,E)))
            val = xcc.reshape((B,L*K,E))@xsa.transpose(2,1)
            xw  = xsa @ self.attention.attq.weight.T
            xp  = xp +  (((1-pijk)*pijk).reshape((B,L*K,L))* val).transpose(2,1) @ xw.reshape((B,L*K,E))/KE**0.5

            # xsa@
            xid = xid + xp

        elif self.use_dense_relu==2:
            xid = (xid @ self.attk2.weight.T /(E*K)**0.5).softmax(-1) @ self.attv2.weight.T.T
            pass
        elif self.use_dense_relu==3:
            xid = xid@self.attention.attq.weight
        elif self.use_dense_relu==5:
            xid = xid @ self.atto.weight.T.T
        else:
            assert 0,self.use_dense_relu

#
        # if self.use_dropout: xid = self.att_dropout(xid)

        # xid = xid.tanh()
        # if self.use_dropout:
        #     xid = self.att_dropout(xid)
        xsad = xsad + xid

        # xsa = xsa + 0.01* xsad / (0.01 + gradsq**0.5)
        # mask = self.att_dropout(torch.ones(xsa.shape[:2]+(1,),device=self.device))
        mask = torch.ones(xsa.shape[:2]+(1,),device=self.device)
        if self.use_dropout:
            mask = self.update_dropout(mask)
        # xsad = 0.05* xsad / (0.1 + xsad.std(-1,keepdims=True))
        xsa = (1-mask*0.5)*xsa + xsad * mask * 0.5
        if self.use_layernorm:
            xsa = xsa / (0.1 + xsa.std(-1,keepdims=True))

        outer = (None,)
        inner = [i,z,gradsq,xsa]

        # outer = [zi,x,y,z,fs]
        # inner = [peki,mu,xzw,xsa]
        return outer,inner

    def _batch_init(self,zi,masked):
        #### batch_init part
        ### state init
        z = masked
        z = self.embed(z)
        z = self.norm(z)
        y = None
        xsa = z

        gradsq = xsa *0.
        i = -1
        outer = (None,)
        inner = [i,z,gradsq,xsa]
        # outer = [zi, x,y,z,fs]
        # inner = [i,mu,xzw,xsa]
        return outer,inner


class AddModelWithReluAttention(AddModelBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        depth,
        kernel_size,
        use_dropout,
        use_dense_relu,

        mask_token_idx):

        state_count = 1
        super().__init__(device,
            graph_dim,
            embed_dim,
            depth,
            mask_token_idx)
        self.use_dropout = use_dropout
        self.use_dense_relu = use_dense_relu
        # state_count = 15
        self.device = device
        # self.total_length = total_length
        # self.min_len = min_len
        # self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        # self.state_count = state_count
        self.depth = depth
        self.D = depth

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,)
        # self.n_step     = min_len
        self.xkey_static = nn.Linear(embed_dim, 2)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim)
        self.kernel_size = kernel_size = 4
        # self.init_state = nn.Linear(mixture_count, embed_dim)
        self.transition = nn.Linear(embed_dim*1,embed_dim)
        self.transcore  = nn.Linear(embed_dim,embed_dim)
        self.updater    = nn.Linear(embed_dim,embed_dim)
        self.emittor    = nn.Linear(embed_dim,embed_dim)
        # self.attention_probe = nn.Linear(embed_dim,kernel_size)
        self.mu = nn.Linear(embed_dim,1)
        self.KE = KE = embed_dim//kernel_size//2
        self.E = E = embed_dim
        self.D = depth

        # self.attention_head  = nn.Linear(embed_dim,kernel_size*KE)
        # self.attention_head_l  = nn.Linear(kernel_size,embed_dim*KE)
        # self.attention_head_r  = nn.Linear(kernel_size,embed_dim*KE)
        # kernel_size =
        self.attq = nn.Linear(embed_dim,kernel_size*embed_dim)
        self.attk = nn.Linear(embed_dim,kernel_size*embed_dim)
        self.attv = nn.Linear(embed_dim,kernel_size*embed_dim)
        self.atto = nn.Linear(embed_dim,kernel_size*embed_dim)
        self.att_dropout = nn.Dropout(0.1)
        self.att_dense = nn.Linear(kernel_size*embed_dim,kernel_size*embed_dim)
        latent_size = 4*embed_dim
        self.attk2 =  nn.Linear(kernel_size*embed_dim,latent_size)
        self.attv2 =  nn.Linear(embed_dim, latent_size )


        self.K = nn.Linear(embed_dim,embed_dim)
        self.U = nn.Linear(embed_dim,embed_dim)
        self.K2 = nn.Linear(embed_dim,embed_dim)
        self.U2= nn.Linear(embed_dim,embed_dim)

        self.att_prob   = nn.Linear(embed_dim,embed_dim)
        # self.dense1 = nn.Linear()

        bconf = BertConfig()
        bconf.hidden_size=E
        bconf.intermediate_size = E*2
        bconf.num_attention_heads = 10
        # self.blayer_list = nn.ModuleList([BertLayer(bconf) for _ in range(20)])
        self.fs_type    = 'lptok'

    def _step(self,outer,inner):
        '''
        Outer should be non-mutable?
        '''

        (i,z,xsa) = inner
        # (zi,x,y,z,fs) = outer        graph_dim,
        # embed_dim,
        # depth,
        # mask_token_idx)

        # (i,mu,xzw,xsa) = inner
        peki = xsa

        E = xsa.size(-1)
        L = xsa.size(1)
        B = xsa.size(0)
        K = self.kernel_size
        KE =self.KE
        # for blayer in self.
        # blayer = self.blayer_list[i]

        _norm = lambda  xsa,dim=-1: xsa /(1+xsa.std(dim,keepdims=True))
        xsad = 0.
        xsad = xsad + xsa.roll(1,1) @ self.transition.weight
        xsad = xsad + xsa.roll(-1,1) @ self.transition.weight.T

        ### Consider qJ vector as neurons clipped by relu. recovered by vJ vector
        ### construct helper neurons KJ,

        ## (B,E,L,K)
        kj = self.attk(xsa).reshape((B,L,K,E)).transpose(-1,1).reshape((B,E,L*K))
        vj = self.attv(xsa).reshape((B,L*K,E))
        wt = (xsa @ kj).sigmoid()
        xid = wt@vj

        #
        # hijk = (xsa @ self.attk.weight.T).reshape((B,L,K,E)) @ xsa.reshape((B,1,L,E)).transpose(-2,-1)
        # vj = self.attv(xsa)
        # import pdb; pdb.set_trace()
        # hijk = hijk / E**0.5
        # pijk = hijk.softmax(-1)
        # yik = pijk @ xsa.reshape((B,1,L,E))
        # xid = yik.reshape((B,L,K*E))
        #
        # if self.use_dense_relu==1:
        #     xid = self.att_dense(xid).relu()
        #     xid = xid @ self.atto.weight.T.T
        #
        # elif self.use_dense_relu==2:
        #     xid = (xid @ self.attk2.weight.T /(E*K)**0.5).softmax(-1) @ self.attv2.weight.T.T
        #     pass
        # else:
        #     xid = xid @ self.atto.weight.T.T
        # # xid = xid.tanh()

        if self.use_dropout:
            xid = self.att_dropout(xid)
        xsad = xsad+ 0.3 * _norm(xid)
        # # pijk @ xsa
        # # att
        # import pdb; pdb.set_trace()
        # xsa = blayer(xsa)[0]

        xsa = xsa + 0.3* xsad / (0.1 + xsad.std(-1,keepdims=True))
        # xsa = xsa + 1./ xsad
        # 0.3* xsad / (0.1 + xsad.std(-1,keepdims=True))
        xsa = xsa / (0.1 + xsa.std(-1,keepdims=True))

        outer = (None,)
        inner = [i,z,xsa]

        # outer = [zi,x,y,z,fs]
        # inner = [peki,mu,xzw,xsa]
        return outer,inner

#
