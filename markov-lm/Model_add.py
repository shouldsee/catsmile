import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


from markov_lm.Model_pretrain import lazy_load_pretrain_model
# import pdb; pdb.set_trace()
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


        cent,lptok,x = self.callback_cent(xsa,mask,unmasked)

        self.callback_end(None,(item,lptok, x))

        loss  = -cent.mean(-1)
        if out=='token': return lptok

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
            xc.n_choice)
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
            n_choice)
        self.device = device

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
        self.KE = KE = kernel_dim

        #### share embed usually works
        self.embed      = nn.Embedding(graph_dim,embed_dim,)
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

        self.attention = QKV_Attention(embed_dim,kernel_size,embed_dim)
        # self.attention_list = nn.ModuleList([QKV_Attention(embed_dim,kernel_size,embed_dim) for _ in range(1)])
        # self.attention = Gaussian_Attention(embed_dim,kernel_size,embed_dim)
        # self.attq = nn.Linear(embed_dim,kernel_size*embed_dim)
        # self.attk = nn.Linear(embed_dim,kernel_size*embed_dim)
        self.attv = nn.Linear(embed_dim,kernel_size*embed_dim)
        self.atto = nn.Linear(embed_dim,kernel_size*embed_dim)
        self.att_dropout = nn.Dropout(0.1)
        self.att_dense = nn.Linear(kernel_size*embed_dim,kernel_size*KE)
        # embed_dim)

        self.muk = nn.Linear(embed_dim,kernel_size)
        self.vk = nn.Linear(embed_dim,kernel_size)

        self.update_dropout = nn.Dropout(self.use_dropout)

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


        elif self.use_dense_relu==2:
            '''
            Use a KV transformation instead of RELU
            '''
            # xid = (xid @ self.attk2.weight.T /(E*K)**0.5).softmax(-1) @ self.attv2.weight.T.T
            xid = (xid @ self.att_dense.weight.T /(E*K)**0.5).softmax(-1) @ self.atto.weight.T.T
            pass

        elif self.use_dense_relu==3:
            '''
            Use attq transpose to approx gradient of logsumexp
            '''
            xid = xid@ _attention.attq.weight

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

        elif self.use_dense_relu==5:

            '### No RELU activation'
            xid = xid @ self.atto.weight.T.T
        elif self.use_dense_relu==6:
            '''
            Recover gradient then rotate
            '''
            xid = xid@_attention.attq.weight @ self.atto.weight[:E,:E]
        elif self.use_dense_relu==7:
            '''
            Recover gradient then rotate
            '''
            xid = xid@_attention.attq.weight @ self.atto.weight[:E,].relu() @ self.atto.weight[:E,].T
        elif self.use_dense_relu==8:
            '''
            Linear output with a different matrix
            '''
            # xidact = self.att_dense(xid).sin()
            xidact =  xid
            xid = xidact @ self.atto.weight.T.T

        elif self.use_dense_relu==9:
            '''
            Relu then linear
            '''
            xidact =  xid.relu()
            xid = xidact @ self.atto.weight.T.T

        elif self.use_dense_relu==10:
            '''
            K vector to extract context, transformed into bias,RELU, dense
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

        elif self.use_dense_relu==12:
            '''
            Same as 11 without the transposed dense
            '''

            muk = self.muk.weight.T
            muk = _norm(muk)
            hik = xsa @ muk
            pik = hik.softmax(1)
            pki = pik.transpose(2,1)
            yk  = pki @ xsa
            ykd = (yk.reshape((B,K*E)) @ self.att_dense.weight.T).relu()

            pik = hik.softmax(-1)
            xid = pik @ ykd.reshape((B,K,E))
        elif self.use_dense_relu == 13:
            xid = 0.

        else:
            assert 0,self.use_dense_relu
        xsad = xsad+ xid

        mask = torch.ones(xsa.shape[:2]+(1,),device=self.device)
        mask = self.update_dropout(mask)

        if self.use_gradnorm:
            xsad = _norm(xsad)

        energy = (xsa*xsad).mean(-1)
        xsad = self.step_size* xsad

        xsa = xsa + xsad * mask
        if self.use_layernorm:
            xsa = _norm(xsa)

        outer = (None,)
        inner = [i,z,energy,xsa]

        return outer,inner

    def _batch_init(self,zi,masked):
        #### batch_init part
        ### state init
        z = masked
        z = self.embed(z)
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
