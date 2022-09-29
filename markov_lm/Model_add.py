import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from markov_lm.Model_pretrain import lazy_load_pretrain_model

#from markov_lm.Model_gmm import
from transformers.models.bert.modeling_bert import BertLayer,BertConfig
import math

class AddModelBase(nn.Module):
    '''
    Abstract Class
    needs to implement self._step
    '''
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        depth,

        iter_per_layer,
        mask_token_idx,
        n_choice,
        use_first_mask
                ):

        super().__init__()
        self.device=device
        self.graph_dim = graph_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.D = depth
        self.iter_per_layer = iter_per_layer
        self.mask_token_idx = mask_token_idx
        self.n_choice = n_choice

        self.embed      = nn.Embedding(graph_dim,embed_dim,device=device)
        self.emittor    = nn.Linear(embed_dim,embed_dim,device=device)
        # self.emittor    = nn.Identity()
        self.kchoice    = nn.Linear(embed_dim,embed_dim*n_choice,device=device)
        self.use_first_mask = use_first_mask


    def callback_step(self,outer,inner):
        # [zi,x,y,z,fs],[i,sel,xz,xs] = outer,inner
        return
    def callback_init(self,outer,inner):
        return
    def callback_end(self,outer,inner):
        return
    def callback_end_simple(self,model,item,inner):
        return

    def norm(self,y):
        y = y / (1 + y.std(-1,keepdims=True)) *1.0
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

    def _step(self,outer,inner):
        '''
        implement variable forwarding
        '''
        return outer,inner

    def forward(self,item,detach=False):
        return self._loss(item,out='forward',detach=detach)

    # def input_transform
    def get_simple_token(self,item):
        return self._loss(item,out='simple_token')
    def _loss(self,item,out='loss',detach=False):
        '''
        Inference of masked token with softmax loss.

        item.masked:   token sequence with masked token corrupted
        item.unmasked: token sequence without mask
        item.mask:     position of mask
        '''
        masked   = item['masked']
        summer = item.get('summer',None)
        if out not in 'forward'.split():
            unmasked = item['unmasked']
            mask     = item['mask']

        zi = None
        assert out in 'loss token traj forward simple_token'.split()
        # zi = None
        outer,inner = self._batch_init(zi,masked)
        # x,y,z)
        ### Uses an explicit RNN to switch between copying z and extract y
        self.callback_init(outer,inner)
        L = masked.size(1)
        B = masked.size(0)
        for i in range(self.D):
            inner[0]=i
            outer,inner = self._step(outer,inner,detach) #### do what ever with your hidden state
            self.callback_step(outer,inner)

        self.callback_end_simple(self,item,inner)

        if out=='forward': return outer,inner
        if out=='simple_token': return self.vocab(inner[-1]).argmax(-1)

        _ = outer
        ### use last variable from inner to infer masked token
        xsa = inner[-1]


        if self.use_first_mask>0:
            mask = mask[:,:self.use_first_mask]
        cent,lptok,x = self.callback_cent(xsa,mask,unmasked)

        self.callback_end(None,(item,lptok, x))

        if out=='token': return lptok

        assert summer is not None
        if summer is not None:
            # assert 0
            loss = -(cent*summer).sum(-1)/summer.sum(-1).clip(1,None)
        else:
            loss  = -cent.mean(-1)

        if out=='loss': return loss
        assert 0

        if out=='loss': return loss
        assert 0

    def callback_cent(self,xsa,mask,unmasked):
        lptok = torch.gather(xsa,index=mask[:,:,None].repeat((1,1,xsa.shape[-1])),dim=1)

        if self.n_choice>0:
            KN=self.n_choice
            E = self.embed_dim
            LM = lptok.shape[1]
            G = self.graph_dim
            xx = self.kchoice(lptok).reshape((B,LM*KN,E))
            xx = xx @ xsa.transpose(2,1)
            xx = xx @ xsa

            # import pdb; pdb.set_trace()
            k_lptok = self.vocab( xx.matmul(self.emittor.weight)).log_softmax(-1)
            lptok = k_lptok.reshape((B,LM,KN,G)).logsumexp(-2) - math.log(KN)

        else:
            lptok = self.vocab(lptok.matmul(self.emittor.weight)).log_softmax(-1)

        x = torch.gather(unmasked,index=mask,dim=1).long()
        cent = self.target_energy(lptok,x)
        return cent, lptok, x

class AddModelWithBert(AddModelBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        depth,
        iter_per_layer,
        mask_token_idx,
        n_choice = 5):

        state_count = 1
        super().__init__(device,
            graph_dim,
            embed_dim,
            depth,
            iter_per_layer,
            mask_token_idx,
            n_choice)
        # state_count = 15
        self.device = device
        # self.total_length = total_length
        # self.min_len = min_len
        # self.mixture_count = mixture_count
        self.embed_dim = embed_dim
        # self.state_count = state_count
        self.depth = depth
        self.iter_per_layer= iter_per_layer
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
        # print(bconf)
        bconf.hidden_dropout_prob = 0.1
        self.bconf = bconf
        # import pdb; pdb.set_trace()
        self.blayer_list = nn.ModuleList([BertLayer(bconf) for _ in range(self.D)])
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

        blayer = self.blayer_list[i//self.iter_per_layer]

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
        return outer,inner


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
from dataclasses import dataclass
@dataclass
class LayerConfig(object):
    depth:int
    graph_dim:int
    embed_dim:int
    depth:int
    iter_per_layer:int
    kernel_size:int
    use_dense_relu:int
    use_layernorm:int
    use_gradnorm:int
    use_input_image:int
    mask_token_idx:int
    kernel_dim:int = 0
    use_dropout:float = 0.5
    step_size:float = 0.05
    eps:float = 1.
    n_choice:int =0
    max_position_length: int=0
    step_size_keep: float=1.0
    use_first_mask: int = -1
    mixture_count :int =1

    def to_model(self,device,charset,cls=None):
        if cls is None:
            cls = AddModelWithAttentionStacked
        return cls(device,self,None)
SASConfig = LayerConfig

class AddModelWithAttentionStacked(AddModelBase):
    def __init__(self,
        device,
        config = None,
        *a,
        **kw
        ):
        '''
        use_gradnorm: wheteher to normlise pseudo-gradient
        use_dropout: 0.5 seems good enough for now
        use_dense_relu: switch between modes. empirical best are mode 1 and mode 11
            - 1: QK Attention -> Dense -> Relu -> Denes -> per-position dropout
            - 11: CK attention -> Dense -> Relu -> CV attention -> PP dropout
        '''

        state_count = 1
        if config is None:
            xc = LayerConfig(**kw)
        else:
            xc = config
        self.config = xc
        super().__init__(device,
            xc.graph_dim,
            xc.embed_dim,
            xc.depth,
            xc.iter_per_layer,
            xc.mask_token_idx,
            xc.n_choice,
            xc.use_first_mask)
        self.layers = nn.ModuleList([ AddModelWithAttention(device=device, **xc.__dict__).to(device)
         for i in range(xc.depth//xc.iter_per_layer+1)])
        self.project = nn.Identity()

    # @property
    # def project(self):
    #     return self.layers[0].embed
    @property
    def embed(self):
        return self.layers[0].embed

    def _step(self,outer,inner,detach):
        '''
        Outer should be non-mutable?
        '''

        i = inner[0]
        ret = self.layers[i//self.config.iter_per_layer]._step(outer,inner,detach)
        return ret
        # (i,z,gradsq,xsa) = inner
    def _batch_init(self,*a,**kw):
        # zi,masked):
        return self.layers[0]._batch_init(*a,**kw)
        # zi,masked)
# from dataclasses import dataclass
@dataclass
class AddModelBertInterfaceConfig(object):
    embed_dim: int
    graph_dim: int
    mask_token_idx: int
    pretrain_model_name: str = 'bert-base-chinese'
    use_original_embedding: int = 0
    attach: int=0
    def to_alias(self):
        return f'E{self.embed_dim}-P{self.pretrain_model_name}-UOE{self.use_original_embedding}-G{self.graph_dim}-M{self.mask_token_idx}-A{self.attach}'
    def to_model(self,device,charset):
        return AddModelBertInterface(device,self,charset)
SimpleConfig = AddModelBertInterfaceConfig

class AddModelBertInterface(AddModelBase):
    def __init__(self,device,config,charset):
        tok,model = BertBasePair = lazy_load_pretrain_model(config.pretrain_model_name)
        self.bconf = model.config
        if config.embed_dim==-1:
            # pass
            _project = nn.Identity()
            config.embed_dim = self.bconf.hidden_size
        else:
            _project = nn.Linear(self.bconf.hidden_size,config.embed_dim,bias=False)

        super().__init__(device,
            config.graph_dim,
            config.embed_dim,
            depth=1,
            iter_per_layer=1,
            mask_token_idx = config.mask_token_idx,
            n_choice=0)
        self.project = _project

        self.embed_dim = config.embed_dim
        self.graph_dim = config.graph_dim
        self.config = config
        self.submodel = [model.to(self.device)]
        if self.config.attach:
            self.submodel = nn.ModuleList(self.submodel)
        self.tok      = [  tok]
        model.requires_grad = False
        if charset is None:
            charset = [x[0] for x in sorted( self.tok[0].get_vocab().items(),key = lambda x:x[1] )]
            # import pdb; pdb.set_trace()
            toks = torch.arange(len(charset),device=self.device,dtype=torch.long)
            self.proj_charset = 0

        else:
            toks = self.tok[0](charset,return_tensors='pt',padding=True,)['input_ids'][:,1].to(self.device)
            self.proj_charset = 1

        bos,v,eos = self.tok[0]('[MASK]')['input_ids']
        charset = [x.upper() for x in charset]
        toks[charset.index('[MASK]')] = v

        self.charset = charset
        self.token2bert  = toks
        self.bos =bos
        self.eos =eos

        BG = self.submodel[0].embeddings.word_embeddings.weight.shape[0]
        ct = torch.zeros((BG,),device=self.device)
        ctt=torch.scatter_add(ct,index=self.token2bert,src=ct*0+1,dim=0)
        self.bert_dup_token_count = ctt

        # self.embed
        # bhidden = BertBaseChinese[1].config["hidden_size"]
    def _step(self,outer,inner,detach):
        '''
        Outer should be non-mutable?
        '''

        (i,z,gradsq,xsa) = inner
        # tok,bert = BertBaseChinese
        if i==0:
            model = self
            # v = item['masked'][:,0:1]*0
            v = z[:,0:1]*0
            if self.proj_charset:
                z = torch.cat([ v*model.bos, self.token2bert[z],v*model.eos],dim=1)

            # import pdb; pdb.set_trace()
            # list_of_sents = z
            xm = self.submodel[0]
            xsa = self.submodel[0]( z)[0][:,1:-1]
            # self.tok(list_of_sents,return_tensors='pt')['input_ids'])[0][1:-1]
            # xsa = self.project(enc)
            # import pdb; pdb.set_trace()
                # embed = xm.embeddings.word_embeddings
                # xm.word_embeddings
        # z
        energy  =None
        outer = (None,)
        inner = [i,z,energy,xsa]

        return outer,inner

    def callback_cent(self,xsa,mask,unmasked):
        lptok = torch.gather(xsa,index=mask[:,:,None].repeat((1,1,xsa.shape[-1])),dim=1)
        if self.config.use_original_embedding:
            lptok = (lptok @ self.submodel[0].embeddings.word_embeddings.weight.T).log_softmax(-1)   ### in bert index

            # lptok = self.vocab(lptok.matmul(self.emittor.weight)).log_softmax(-1)
            x = torch.gather(unmasked,index=mask,dim=1).long()
            xx = self.token2bert[x]
            cent = self.target_energy(lptok,xx)
            cent = cent - torch.log(self.bert_dup_token_count[xx])

        else:
            lptok = self.project(lptok)
            # xsa = (xsa)@ self.project.weight.T
            lptok = self.vocab(lptok.matmul(self.emittor.weight)).log_softmax(-1)
            x = torch.gather(unmasked,index=mask,dim=1).long()
            cent = self.target_energy(lptok,x)

        return cent, lptok, x

    def _batch_init(self,zi,masked):
        #### batch_init part
        ### state init
        z = masked
        # z = self.embed(z)
        # z = self.norm(z)
        y = None
        xsa = z
        # xsa = self.embed(z)
        gradsq = xsa *0.
        i = -1
        outer = (None,)
        inner = [i,z,gradsq,xsa]
        return outer,inner


class AddModelWithAttentionMixed(AddModelBase):
    def __init__(self,
        device,
        config = None,
        *a,
        **kw
        ):
        '''
        use_gradnorm: wheteher to normlise pseudo-gradient
        use_dropout: 0.5 seems good enough for now
        use_dense_relu: switch between modes. empirical best are mode 1 and mode 11
            - 1: QK Attention -> Dense -> Relu -> Denes -> per-position dropout
            - 11: CK attention -> Dense -> Relu -> CV attention -> PP dropout
        '''

        state_count = 1
        if config is None:
            xc = LayerConfig(**kw)
        else:
            xc = config
        self.config = xc
        super().__init__(device,
            xc.graph_dim,
            xc.embed_dim,
            xc.depth,
            xc.iter_per_layer,
            xc.mask_token_idx,
            xc.n_choice,
            xc.use_first_mask)
        self.layers = nn.ModuleList([ AddModelWithAttentionStacked(device=device, **xc.__dict__).to(device)
         for i in range(xc.mixture_count)])
        self.project = nn.Identity()
        # self.K = len(self.layers)

    # @property
    # def project(self):
    #     return self.layers[0].embed
    @property
    def embed(self):
        return self.layers[0].embed

    def _batch_init(self,*a,**kw):
        K = len(self.layers)
        xsas = []
        for k in range(K):
            outer,inner = self.layers[k]._batch_init(*a,**kw)
            xsas.append(inner[-1])
        # inner
        inner[-1] = xsas[0][None].repeat((K,1,1,1))
        return outer,inner

    def _step(self,outer,inner,detach):
        '''
        Outer should be non-mutable?
        '''
        K = len(self.layers)
        i = inner[0]
        xsas = inner[-1]
        xsas_new = xsas * 0
        idx = torch.ones(xsas[0:1].shape,device=self.device).long()
        for k in range(K):
            inner[-1] =xsas[k]
            outer,inner = self.layers[k]._step(outer,inner,detach)
            xsas_new = torch.scatter(xsas_new,src=inner[-1][None],dim=0,index=k*idx)
        inner[-1] =xsas_new
        return outer,inner

    def forward(self,item,detach=False):
        outer, inner =  self._loss(item,out='forward',detach=detach)
        inner[-1] = inner[-1][0]
        return outer,inner

    def callback_cent(self,xsa,mask,unmasked):
        K =len(self.layers)

        lptok = torch.gather(xsa,index=mask[None,:,:,None].repeat((K,1,1,xsa.shape[-1])),dim=2)

        lptokk = self.vocab(lptok).logsumexp(0).log_softmax(-1)

        # lptokk = self.vocab(lptok).logsumexp(0).relu()
        # lptokk = (lptokk).log_softmax(-1)

        # lptokk = (-lptokk).log_softmax(-1)

        # lptokk = 0*lptokk
        # lptokk = lptokk.log_softmax(-1)

        # lptok = self.vocab(lptok.matmul(self.emittor.weight)).log_softmax(-1)

        x = torch.gather(unmasked,index=mask,dim=1).long()
        cent = self.target_energy(lptokk,x)
        return cent, lptokk, x


class AddModelWithAttention(AddModelBase):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,
        depth,
        iter_per_layer,
        kernel_size,
        use_dense_relu,
        use_layernorm,
        use_gradnorm,
        use_input_image,

        mask_token_idx,
        kernel_dim = None,
        use_dropout = 0.5,
        step_size = 0.05,
        eps = 1.,
        n_choice=0,
        max_position_length=0,
        step_size_keep=1.0,
        use_first_mask=0,

        **kw
        ):
        '''
        use_gradnorm: wheteher to normlise pseudo-gradient
        use_dropout: 0.5 seems good enough for now
        use_dense_relu: switch between modes. empirical best are mode 1 and mode 11
            - 1: QK Attention -> Dense -> Relu -> Denes -> per-position dropout
            - 11: CK attention -> Dense -> Relu -> CV attention -> PP dropout
        '''

        state_count = 1
        super().__init__(device,
            graph_dim,
            embed_dim,
            depth,
            iter_per_layer,
            mask_token_idx,
            n_choice,
            use_first_mask)
        self.device = device

        self.step_size_keep = step_size_keep
        self.use_input_image = use_input_image
        self.use_dropout = use_dropout
        self.use_dense_relu = use_dense_relu
        self.use_layernorm = use_layernorm
        self.use_gradnorm = use_gradnorm
        self.eps = eps
        self.step_size = step_size

        self.embed_dim = embed_dim
        self.depth = depth
        self.D = depth
        if kernel_dim is None:
            kernel_dim = embed_dim
        self.kernel_dim = self.KE = KE = kernel_dim

        #### share embed usually works
        self.max_position_length =max_position_length

        self.embed      = nn.Embedding(graph_dim,embed_dim,)
        self.pos_embed  = nn.Embedding(self.max_position_length,embed_dim)
        self.xkey_static = nn.Linear(embed_dim, 2)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim)
        self.kernel_size = kernel_size
        # self.init_state = nn.Linear(mixture_count, embed_dim)
        self.transition = nn.Linear(embed_dim,embed_dim)
        self.transcore  = nn.Linear(embed_dim,embed_dim)
        # self.transcore2 = nn.Linear(embed_dim,embed_dim)
        self.emittor    = nn.Linear(embed_dim,embed_dim)
        self.updater    = nn.Linear(embed_dim,embed_dim)

        # self.KE = KE = embed_dim//kernel_size//2
        self.E = E = embed_dim
        self.D = depth
        if self.use_dense_relu<=10:
            self.attention = QKV_Attention(embed_dim,kernel_size,embed_dim)
            # self.attention_list = nn.ModuleList([QKV_Attention(embed_dim,kernel_size,embed_dim) for _ in range(1)])
            # self.attention = Gaussian_Attention(embed_dim,kernel_size,embed_dim)
            # self.attq = nn.Linear(embed_dim,kernel_size*embed_dim)
            # self.attk = nn.Linear(embed_dim,kernel_size*embed_dim)
            self.attv = nn.Linear(embed_dim,kernel_size*embed_dim)
            self.atto = nn.Linear(embed_dim,kernel_size*embed_dim)
            # self.att_dropout = nn.Dropout(0.1)
            self.att_dense = nn.Linear(kernel_size*embed_dim,kernel_size*KE)
        elif self.use_dense_relu in (35,36,37,38,39,40,41):
            self.attv = nn.Linear(embed_dim,kernel_dim*5*embed_dim)
            self.atto = nn.Linear(embed_dim,kernel_dim*5*embed_dim)


        if self.use_dense_relu in (11,34):

            self.att_dense = nn.Linear(kernel_size*embed_dim,kernel_size*KE)
        # embed_dim)

        if self.use_dense_relu in (11,34) or self.use_dense_relu>=20:
            self.muk = nn.Linear(embed_dim,kernel_size)
            self.vk = nn.Linear(embed_dim,kernel_size)
            self.muk2 = nn.Linear(embed_dim,kernel_size)
            self.vk2 = nn.Linear(embed_dim,kernel_size)
        if self.use_dense_relu == 22:
            self.wk = nn.Linear(embed_dim,kernel_size*embed_dim)
        if self.use_dense_relu == 24:
            self.vkk2 = nn.Linear(embed_dim,kernel_size)
            self.vkk = nn.Linear(embed_dim,kernel_size)
            # self.wk = nn.Linear(embed_dim,kernel_size)

        if self.use_dense_relu in (26,27,28):
            assert self.max_position_length!=0

            self.wq = nn.Linear(max_position_length*embed_dim, kernel_size*embed_dim)
            self.wv = nn.Linear(max_position_length*embed_dim, kernel_size*embed_dim)


        self.update_dropout = nn.Dropout(max(0,self.use_dropout))

        self.fs_type    = 'lptok'

    def norm (self, xsa,dim=-1):
        return xsa /(self.eps +xsa.std(dim,keepdims=True))

    def _step(self,outer,inner,detach):
        '''
        Outer should be non-mutable?
        '''

        (i,z,gradsq,xsa) = inner

        E = xsa.size(-1)
        L = xsa.size(1)
        B = xsa.size(0)
        K = self.kernel_size
        # KE =self.KE

        _norm = self.norm
        xsad = 0.
        def _d(x):
            if detach:
                return x.detach()
            else:
                return x

        if self.use_dense_relu<20:
            ### local connection
            xsad = xsad + (xsa.roll(1,1) @ _d(self.transition.weight)).relu()@_d(self.transcore.weight)
            xsad = xsad + (xsa.roll(-1,1) @ _d(self.transcore.weight.T)).relu() @ _d(self.transition.weight.T)



        if self.use_input_image:
        # xsad = xsad + xsa.roll(0,1).matmul(self.transcore2.weight)
            xsad = xsad + z.matmul(self.updater.weight.T)
            xsad = xsad + self.transition.bias


        ### calculate attention if required
        if self.use_dense_relu<10:
            _attention = self.attention
            # _attention = self.attention_list[0]
            xid,pijk = _attention(xsa)



        if self.use_dense_relu==1:
            '''
            Dense -> Relu -> Dense
            '''
            # xidact = self.att_dense(xid).sin()
            xidact =  xid @ self.att_dense.weight.T
            xidact = xidact.relu()
            xid = xidact @ self.atto.weight.T.T[:xidact.size(-1)]
            xsadd  = xid

        elif self.use_dense_relu==2:
            '''
            Use a KV transformation instead of RELU
            '''
            # xid = (xid @ self.attk2.weight.T /(E*K)**0.5).softmax(-1) @ self.attv2.weight.T.T
            xid = (xid @ self.att_dense.weight.T /(E*K)**0.5).softmax(-1) @ self.atto.weight.T.T
            xsadd  = xid

        elif self.use_dense_relu==3:
            '''
            Use attq transpose to approx gradient of logsumexp
            '''
            xid = xid@ _attention.attq.weight
            xsadd  = xid

        elif self.use_dense_relu==4:
            '''
            Assume a constructed energy, then correct the graident wrt xsa
            '''
            xidact = self.att_dense(xid).relu()

            xid = xidact @ self.atto.weight.T.T
            xc  = xsa @ self.atto.weight.T
            xcc = (xc * (xidact>0))@self.att_dense.weight
            # xc.shape
            xp  = pijk.reshape((B,L*K,L)).transpose(2,1) @ (xcc.reshape((B,L*K,E)))
            val = xcc.reshape((B,L*K,E))@xsa.transpose(2,1)
            xw  = xsa @ _attention.attq.weight.T
            xp  = xp + (((1-pijk)*pijk).reshape((B,L*K,L))* val).transpose(2,1) @ xw.reshape((B,L*K,E))/KE**0.5

            xid = xid + xp
            xsadd  = xid

        elif self.use_dense_relu==5:

            '### No RELU activation'
            xid = xid @ self.atto.weight.T.T
            xsadd  = xid


        elif self.use_dense_relu==6:
            '''
            Recover gradient then rotate
            '''
            xid = xid@_attention.attq.weight @ self.atto.weight[:E,:E]
            xsadd  = xid

        elif self.use_dense_relu==7:
            '''
            Recover gradient then rotate
            '''
            xid = xid@_attention.attq.weight @ self.atto.weight[:E,].relu() @ self.atto.weight[:E,].T
            xsadd  = xid


        elif self.use_dense_relu==8:
            '''
            Linear output with a different matrix
            '''
            # xidact = self.att_dense(xid).sin()
            xidact =  xid
            xid = xidact @ self.atto.weight.T.T
            xsadd  = xid

        elif self.use_dense_relu==9:
            '''
            Relu then linear
            '''
            xidact =  xid.relu()
            xid = xidact @ self.atto.weight.T.T
            xsadd  = xid

        elif self.use_dense_relu==10:
            '''
            K vector to extract context, transformed into bias,RELU, dense            xsadd  = xid

            then distributed according to K vectors
            '''

            muk = self.muk.weight.T
            muk = _norm(muk)
            hik = xsa @ muk
            pik = hik.softmax(1)
            pki = pik.transpose(2,1)
            yk  = pki @ xsa
            ykd = (yk.reshape((B,K*E)) @ self.att_dense.weight.T).relu() @ self.att_dense.weight.T.T

            pik = hik.softmax(-1)
            xid = pik @ ykd.reshape((B,K,E))
            xsadd  = xid

        elif self.use_dense_relu==11:
            '''
            K vector to extract context, transformed into bias,RELU
            then distributed according to V vectors
            '''

            muk = self.muk.weight.T
            muk = _norm(muk)
            vk = self.vk.weight.T
            vk = _norm(vk)

            hik = xsa @ _d(muk)
            pik = hik.softmax(1)
            pki = pik.transpose(2,1)

            hikv = xsa @ _d(vk)
            pikv = hikv.softmax(-1)
            # xid = pikv @ ( pki @xsa )
            # yk  = pki @ xsa
            yk  = pki @ xsa
            ykd = (yk.reshape((B,K*E)) @ _d(self.att_dense.weight.T)).relu() @ _d(self.att_dense.weight.T.T)

            xid = pikv @ ykd.reshape((B,K,E))
            xsadd  = xid

        elif self.use_dense_relu==34:
            '''
            K vector to extract context, transformed into bias,RELU
            then distributed according to V vectors
            '''

            muk = self.muk.weight.T
            muk = _norm(muk)
            vk = self.vk.weight.T
            vk = _norm(vk)

            hik = xsa @ _d(muk)
            # pik = hik.softmax(1)
            pik = hik.relu()
            pki = pik.transpose(2,1)

            hikv = xsa @ _d(vk)
            # pikv = hikv.softmax(-1)
            pikv = hikv.relu()
            # xid = pikv @ ( pki @xsa )
            # yk  = pki @ xsa
            yk  = pki @ xsa
            ykd = (yk.reshape((B,K*E)) @ _d(self.att_dense.weight.T)).relu() @ _d(self.att_dense.weight.T.T)

            xid = pikv @ ykd.reshape((B,K,E))
            xsadd  = xid

        elif self.use_dense_relu==12:
            '''
            Same as 11 without the transposed dense
            '''
            # xsad = xsad + (xsa.roll(1,1) @ _d(self.transition.weight)).relu()@_d(self.transcore.weight)
            # xsad = xsad + (xsa.roll(-1,1) @ _d(self.transcore.weight.T)).relu() @ _d(self.transition.weight.T)

            muk = self.muk.weight.T
            muk = _norm(muk)
            hik = xsa @ muk
            pik = hik.softmax(1)
            pki = pik.transpose(2,1)
            yk  = pki @ xsa
            ykd = (yk.reshape((B,K*E)) @ self.att_dense.weight.T).relu()

            pik = hik.softmax(-1)
            xid = pik @ ykd.reshape((B,K,E))
            xsadd  = xid

        elif self.use_dense_relu == 13:
            xid = 0.
            xsadd  = xid

        elif self.use_dense_relu==21:
            ### local connection
            muk = self.muk.weight.T
            muk = _norm(muk)
            vk = self.vk.weight.T
            vk = _norm(vk)

            # xsl = xsa.roll(1,1)
            xsr = xsa.roll(-1,1)
            hik = xsa @ muk  + xsr @ vk
            hik = hik*0.1
            pik = hik.softmax(-1)

            xid = pik @ muk.T + (pik @ vk.T).roll(1,1)
            xsadd  = xid

        elif self.use_dense_relu==22:
            ### local connection
            muk = self.muk.weight.T
            muk = _norm(muk)
            vk = self.vk.weight.T
            vk = _norm(vk)
            muk2 = self.muk2.weight.T
            muk2 = _norm(muk2)
            vk2 = self.vk2.weight.T
            vk2 = _norm(vk2)

            # xsl = xsa.roll(1,1)
            xsr = xsa.roll(-1,1)
            hik = xsa @ muk  + xsr @ vk
            pik = hik.softmax(-1)
            wk = self.wk.weight.T.reshape((E*E,K))
            pikw  = (pik @ wk.T).reshape((B,L,E,E))
            # import pdb; pdb.set_trace()
            xid = pik @ muk2.T + (pik @ vk2.T).roll(1,1) + (pikw*xsr.unsqueeze(-1)).sum(-2)
            xsadd  = xid

        elif self.use_dense_relu==23:
            ### local connection
            muk = self.muk.weight.T
            muk = _norm(muk)
            vk = self.vk.weight.T
            vk = _norm(vk)


            muk2 = self.muk2.weight.T
            muk2 = _norm(muk2)
            vk2 = self.vk2.weight.T
            vk2 = _norm(vk2)

            # xsl = xsa.roll(1,1)
            xsr = xsa.roll(-1,1)
            hik = xsa @ muk  + xsr @ vk
            hik = hik*0.1
            pik = hik.softmax(-1)

            xid = pik @ muk2.T + (pik @ vk2.T).roll(1,1) #+ (pikw*xsr.unsqueeze(-1)).sum(-2)

            # xid = pik @ muk.T + (pik @ vk.T).roll(1,1)
            xsadd  = xid

        elif self.use_dense_relu==24:
            ### local connection
            muk = self.muk.weight.T
            muk = _norm(muk)
            vk = self.vk.weight.T
            vk = _norm(vk)
            vkk = self.vkk.weight.T
            vkk = _norm(vkk)


            muk2 = self.muk2.weight.T
            muk2 = _norm(muk2)
            vk2 = self.vk2.weight.T
            vk2 = _norm(vk2)
            vkk2 = self.vkk2.weight.T
            vkk2 = _norm(vkk2)

            # xsl = xsa.roll(1,1)
            xsr = xsa.roll(-1,1)
            xsrr= xsa.roll(-2,1)
            hik = xsa @ muk  + xsr @ vk + xsrr@vkk
            hik = hik*0.1
            pik = hik.softmax(-1)

            xid = pik @ muk2.T + (pik @ vk2.T).roll(1,1) + (pik@vkk2.T).roll(2,1)

            # xid = pik @ muk.T + (pik @ vk.T).roll(1,1)
            xsadd  = xid



        elif self.use_dense_relu==25:
            '''
            K vector to extract context, transformed into bias,RELU
            then distributed according to V vectors
            '''
            xsad = xsad + (xsa.roll(1,1) @ _d(self.transition.weight)).relu()@_d(self.transcore.weight)
            xsad = xsad + (xsa.roll(-1,1) @ _d(self.transcore.weight.T)).relu() @ _d(self.transition.weight.T)

            # import pdb; pdb.set_trace()
            xid = xsa.roll(1,1) * xsa.roll(2,1) * self.vk.weight[None, 0:1,:]
            # self.vk.weight.T[None, :,0:1]

            xsadd  = xid

        elif self.use_dense_relu==26:
            '''
            KID parametrisation
            KID cannot discriminate between shuffled and unshuffled sequences
            because they look equally unprobable to KID.
            '''
            y_k = xsa.reshape((B,L*E)) @ self.wq.weight.T
            xsadd = (y_k.relu() @ self.wv.weight).reshape((B,L,E))

        elif self.use_dense_relu==27:
            '''
            KID parametrisation with Alignemnt
            '''
            xsaa = torch.stack([xsa.roll(i,1) for i in range(L)],dim=1)
            y_k = xsaa.reshape((B,L,L*E)) @ self.wq.weight.T
            xsal = _norm((y_k @ self.wv.weight).reshape((B,L,L,E)))
            hl = (xsaa*xsal).mean((-1,-2))
            pl = hl.softmax(-1)
            xsadd = (pl[:,:,None,None]*xsal).sum(1)
            # e = xsaa
            # .reshape((L*B,L,E))
        elif self.use_dense_relu==28:
            '''
            KID parametrisation
            KID cannot discriminate between shuffled and unshuffled sequences
            because they look equally unprobable to KID.
            '''
            y_k = xsa.reshape((B,L*E)) @ self.wq.weight.T
            # xsadd = (y_k.relu() @ self.wv.weight).reshape((B,L,E)) * xsa
            xsadd = (y_k @ self.wv.weight).reshape((B,L,E)) * xsa

        elif self.use_dense_relu==30:
            ### local connection
            xsad = xsad + (xsa.roll(1,1) @ _d(self.transition.weight))@_d(self.transcore.weight)
            xsad = xsad + (xsa.roll(-1,1) @ _d(self.transcore.weight.T))@ _d(self.transition.weight.T)
            xsadd = xsad * xsa
        elif self.use_dense_relu==31:
            ### local connection
            xsad = xsad + (xsa.roll(1,1) @ _d(self.transition.weight))@_d(self.transcore.weight)
            xsad = xsad + (xsa.roll(-1,1) @ _d(self.transcore.weight.T))@ _d(self.transition.weight.T)
            xsadd = xsad

        elif self.use_dense_relu==35:
            ### local connection
            xsas = torch.stack([
                xsa.roll(2,1),
                xsa.roll(1,1),
                xsa.roll(0,1),
                xsa.roll(-1,1),
                xsa.roll(-2,1),
                 ],dim=-1)
            wx = self.atto.weight.reshape((E*5,-1))[:E*5,:self.kernel_dim]
            wy = self.attv.weight.reshape((-1,E))[:self.kernel_dim,:E]
            score = (xsas.reshape((B,L,E*5)) @ wx).relu()
            xsadd = score@ wy

        elif self.use_dense_relu==36:
            ### local connection
            xsas = torch.stack([
                xsa.roll(2,1),
                xsa.roll(1,1),
                xsa.roll(0,1),
                xsa.roll(-1,1),
                xsa.roll(-2,1),
                 ],dim=-1)
            wx = self.atto.weight.reshape((E*5,-1))[:E*5,:self.kernel_dim]
            wy = self.attv.weight.reshape((-1,E))[:self.kernel_dim,:E]
            score = (xsas.reshape((B,L,E*5)) @ wx.relu()).relu()
            xsadd = score@ wy

        elif self.use_dense_relu==37:
            ### local connection
            xsas = torch.stack([
                xsa.roll(2,1),
                xsa.roll(1,1),
                xsa.roll(0,1),
                xsa.roll(-1,1),
                xsa.roll(-2,1),
                 ],dim=-1)
            wx = self.atto.weight.reshape((E*5,-1))[:E*5,:self.kernel_dim]
            wy = self.attv.weight.reshape((-1,E))[:self.kernel_dim,:E]
            score = xsas.reshape((B,L,E*5)) @ wx
            xsadd = score@ wy

        elif self.use_dense_relu==38:
            ### local connection
            xsas = torch.stack([
                xsa.roll(2,1),
                xsa.roll(1,1),
                xsa.roll(0,1),
                xsa.roll(-1,1),
                xsa.roll(-2,1),
                 ],dim=-1)
            wx = self.atto.weight.reshape((E*5,-1))[:E*5,:self.kernel_dim]
            wy = self.attv.weight.reshape((-1,E))[:self.kernel_dim,:E]
            score = (xsas.reshape((B,L,E*5)) @ wx.relu()).relu()
            xsadd = score@ wy.relu()


        elif self.use_dense_relu==39:
            ### local connection
            xsas = torch.stack([
                xsa.roll(4,1),
                xsa.roll(3,1),
                xsa.roll(2,1),
                xsa.roll(1,1),
                xsa.roll(0,1),
                # xsa.roll(-1,1),
                # xsa.roll(-2,1),
                 ],dim=-1)
            wx = self.atto.weight.reshape((E*5,-1))[:E*5,:self.kernel_dim]
            wy = self.attv.weight.reshape((-1,E))[:self.kernel_dim,:E]
            score = xsas.reshape((B,L,E*5)) @ wx
            xsadd = score@ wy


        elif self.use_dense_relu==40:
            '''
            Left to right auto-regression
            '''
            ### local connection
            xsas = torch.stack([
                xsa.roll(4,1),
                xsa.roll(3,1),
                xsa.roll(2,1),
                xsa.roll(1,1),
                xsa.roll(0,1),
                # xsa.roll(-1,1),
                # xsa.roll(-2,1),
                 ],dim=-1)
            wx = self.atto.weight.reshape((E*5,-1))[:E*5,:self.kernel_dim]
            wy = self.attv.weight.reshape((-1,E))[:self.kernel_dim,:E]
            score = xsas.reshape((B,L,E*5)) @ wx
            score = score * 0.1
            xsadd = score.softmax(-1)@ wy

        elif self.use_dense_relu==41:
            ### local connection
            xsas = torch.stack([
                xsa.roll(4,1),
                xsa.roll(3,1),
                xsa.roll(2,1),
                xsa.roll(1,1),
                xsa.roll(0,1),
                # xsa.roll(-1,1),
                # xsa.roll(-2,1),
                 ],dim=-1)
            wx = self.atto.weight.reshape((E*5,-1))[:E*5,:self.kernel_dim]
            wy = self.attv.weight.reshape((-1,E))[:self.kernel_dim,:E]
            score = xsas.reshape((B,L,E*5)) @ wx.abs()
            score = score * 0.1
            xsadd = score.softmax(-1)@ wy.abs()
            #
            #
            # torch.stack([xsa.roll(0,1) ])
            # xsad = xsad + (xsa.roll(2,1) @ _d(self.transition.weight))
            # xsad = xsad + (xsa.roll(1,1) @ _d(self.transition.weight))@_d(self.transcore.weight)
            # xsad = xsad + (xsa.roll(-1,1) @ _d(self.transcore.weight.T))@ _d(self.transition.weight.T)
            # xsad = xsad + (xsa.roll(-2,1) @ _d(self.transcore.weight.T))@ _d(self.transition.weight.T)
            # xsadd = xsad

            # xsadd = xsad

        else:
            assert 0,self.use_dense_relu
        xsad = xsad+ xsadd

        if self.use_dropout>=0:
            mask = torch.ones(xsa.shape[:2]+(1,),device=self.device)
            mask = self.update_dropout(mask)
            noise = 0.
        elif self.use_dropout<0:
            mask = 1.
            noise = _norm(torch.normal(0,1,size=xsa.shape,device=self.device) * -self.use_dropout)


        if self.use_gradnorm:
            xsad = _norm(xsad)

        energy = (xsa*xsad).mean(-1)
        xsad = self.step_size* xsad

        xsa = self.step_size_keep * xsa + xsad * mask + noise
        if self.use_layernorm:
            xsa = _norm(xsa)

        outer = (None,)
        inner = [i,z,energy,xsa]

        return outer,inner

    # def embed(self,z):
    #     z = self._embed(z)
    #     # z = z + self.pos_embed(torch.arange(z.shape[1],device=self.device)[None,:])
    #     return z
    def _batch_init(self,zi,masked):
        #### batch_init part
        ### state init
        z = masked
        z = self.embed(z)
        if self.max_position_length>0:
            z = z + self.pos_embed(torch.arange(z.shape[1],device=self.device)[None,:])
        z = self.norm(z)
        y = None
        # z = self.
        xsa = z

        gradsq = xsa *0.
        i = -1
        outer = (None,)
        inner = [i,z,gradsq,xsa]
        return outer,inner

def _target_energy(self,lptok,yt):
    yp = torch.gather(lptok,index=yt[:,:,None],dim=-1)[:,:,0]
    return yp


class RefillLoss(nn.Module):
    def __init__(self,embed,use_mixture=1):
        super().__init__()
        self.embed = embed_layer
        embed_dim  = embed_layer.weight.shape[1]

        self.xkey_static = nn.Linear(embed_dim, 2)
        self.xkey_dynamic= nn.Linear(embed_dim, embed_dim)
        self.emittor     = nn.Linear(embed_dim, embed_dim)
        self.use_mixture = use_mixture

    def vocab(self,x):
        y = x.matmul(self.embed.weight.T)
        return y

    def forward(self, xsa, z, y):
        if self.use_mixture:
            xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
            xkey_dynamic= self.xkey_dynamic(y)
            xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
            # import pdb; pdb.set_trace()
            cand  = torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
            sel   = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
            lptok = self.vocab(cand).log_softmax(-1)
            lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)
        else:
            lptok = self.vocab(xsa.matmul(self.emittor.weight)).log_softmax(-1)

        cent = _target_energy(lptok,x)
        loss  = -cent.mean(-1)
        return loss
