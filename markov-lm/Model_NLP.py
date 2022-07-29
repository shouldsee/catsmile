import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from dataclasses import dataclass
from markov_lm.Model_gmm import AbstractLayerConfig
# from transformers.models.bert.modeling_bert import BertLayer,BertConfig
from markov_lm.nlp.model_seq2seq import Seq2SeqWithAttention,Seq2SeqWithNoAttention
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

class RefillModelRNNConvolveSimple(RefillModelRNNBase):
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

    def loss(self,item):
        # (zi,x,y,z,mask =None):
        zi = 1
        x   = item['unmasked']
        y   = x*0
        mask= item['mask']
        z   = item['masked']
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

        self.callback_end(outer,)
        (zi,x,y,z,lptok) = outer
        (i,sel,xz,xsa) = inner

        if self.use_mixture:
            xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
            xkey_dynamic= self.xkey_dynamic(y)
            xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
            # import pdb; pdb.set_trace()
            cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
            sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
            lptok = self.vocab(cand).log_softmax(-1)
            lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)


            lptok = (lptok/2).exp()
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

class RefillModelRNNConvolveSimple(RefillModelRNNBase):
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

    def loss(self,item):
        # (zi,x,y,z,mask =None):
        zi = 1
        x   = item['unmasked']
        y   = x*0
        mask= item['mask']
        z   = item['masked']
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

        self.callback_end(outer,)
        (zi,x,y,z,lptok) = outer
        (i,sel,xz,xsa) = inner

        if self.use_mixture:
            xkey_static = self.xkey_static.weight[None,:2,:self.embed_dim].repeat((len(z),1,1))
            xkey_dynamic= self.xkey_dynamic(y)
            xkey  = torch.cat([xkey_static, xkey_dynamic],dim=1)
            # import pdb; pdb.set_trace()
            cand= torch.cat([ self.emittor(xsa)[:,:,None], z[:,:,None],y[:,None,:,:].repeat((1,L,1,1))],dim=2)
            sel = xsa.matmul(xkey.transpose(2,1)).log_softmax(-1)
            lptok = self.vocab(cand).log_softmax(-1)
            lptok = (sel[:,:,:,None] + lptok).logsumexp(-2)


            lptok = (lptok/2).exp()
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


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    Source:
      https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x.transpose(1,0) + self.pe[:x.size(0), :]
        return self.dropout(x).transpose(1,0)

class Seq2SeqWithTransformer(nn.Module):
    task_list = ['translate-german-english']
    def __init__(self, device,config,_):
        super().__init__()
        self.device=device
        self.config=config
        max_len = config.n_step
        self.submodel = nn.Transformer(
            d_model=config.embed_dim,
            nhead=8,
            # cconfig.kernel_size,
            num_encoder_layers=config.depth,
            num_decoder_layers=config.depth,
            # dim_feedforward=config.embed_dim,
            dropout=config.beta,
            # activation=<function relu>,
            custom_encoder=None,
            custom_decoder=None,
            layer_norm_eps=1e-05,
            batch_first=True,
            # norm_first=False,
            device=self.device,
            dtype=None)
        self.pe = PositionalEncoding(d_model=config.embed_dim, max_len = max_len)
        self.embed = nn.Embedding( config.graph_dim, config.embed_dim).to(self.device)
        self.output_logit_layer  = nn.Linear( config.embed_dim,  config.graph_dim).to(self.device)

    def loss(self,item):
        return self._loss(item,'loss')
    grad_loss = loss
    def _loss(self,item,ret):
        source = item['source'] ### token sequence
        target = item['target'] ### token seq
        dec_input  = target[:,:-1]
        # hidden = torch.zeros((1, len(source), self.embed_dim),device=self.device)

        source_embed = self.embed(source)
        target_embed = self.embed(target)
        output_hidden = self.submodel.forward( self.pe(source_embed), target_embed)
        output_logit  = self.output_logit_layer(output_hidden)

        output_tok = item['target'][:,1:]
        loss = -torch.gather( output_logit.log_softmax(-1),index=output_tok.unsqueeze(-1),dim=-1).squeeze(-1)
        loss = loss.mean(-1)
        # import pdb; pdb.set_trace()
        return loss




@dataclass
class NLPLayerConfig(AbstractLayerConfig):
    graph_dim:int
    model_name:str
    window_size: int = 0
    # iter_per_layer:int = 0
    loss_name: str = 'KLD'
    grad_loss_name: str = 'KLD'
    depth:int = 1
    beta: float = 0.
    n_step: int = 1
    kernel_size:int =0
    embed_dim:int = 0
    p_null: float = 0.0001
    submodel_name: str=''
    def to_model(self,device):
        cls = eval(self.model_name)
        return cls(device,self,None)
    def to_str(self,):
        out = ''
        for k,v in self.__dict__.items():
            k = k.replace('_','-')
            out+= f'-{k}{v}'
        return out


class SimpleDenseNet(nn.Module):
    '''
    Use autoregression objective
    '''
    def __init__(self,
        device,
        config, _=None):

        state_count = 1
        super().__init__()

        self.device        = device
        self.config = config
        self.embed_dim = embed_dim = config.embed_dim
        self.graph_dim = graph_dim =  config.graph_dim
        self.loss_name = config.loss_name
        '''
        window_size should be attribute of dataset, not model
        '''
        self.window_size = window_size = config.window_size

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.layers      = nn.ModuleList([
            nn.Linear(embed_dim*window_size, embed_dim*window_size).to(self.device)
            for _ in range(self.config.depth-1)
            ])
        self.final_layer      = nn.Linear(embed_dim*window_size, embed_dim).to(self.device)


    def unembed(self,x):
        y = x.matmul(self.embed.weight.T)
        return y

    def grad_loss(self,item):
        return self._loss(item,self.config.grad_loss_name)

    def loss(self,item):
        return self._loss(item,self.loss_name)

    def _loss(self,item,loss_name):
        x = item['unmasked']
        WS = self.window_size
        lossVal = 0.
        xe = self.embed(x)
        B = len(x)
        i = 0
        for i in range(x.shape[1]-WS):
            xx = xe[:,i:WS+i]
            y = x[:,WS+i]


            xx = xx.reshape((B,-1))
            for li in range(self.config.depth-1):
                xx = xx + self.layers[li](xx)
                xx = xx.relu()
            xx = self.final_layer(xx)

            yp = self.unembed(xx)

            if loss_name=='KLD':
                yp = yp.log_softmax(-1)
                _lv = torch.gather(yp,index=y[:,None],dim=1)[:,0]
            elif loss_name == 'HSQ':
                # HellingerSquared
                yp = (yp.softmax(-1)+0.000001).sqrt()
                _lv = torch.gather(yp,index=y[:,None],dim=1)[:,0]
            elif loss_name == 'SELERR':
                yp = yp.argmax(-1)
                _lv = (yp == y)
            else:
                assert 0,loss_name

            lossVal += _lv
            # import pdb; pdb.set_trace()

        lossVal = lossVal/(i+1.)
        lossVal = -lossVal
        # lossVal = lossVal.mean(0)
        return lossVal



class SimpleDenseNetGaussDist(nn.Module):
    '''
    Use autoregression objective
    '''
    def __init__(self,
        device,
        config, _=None):

        state_count = 1
        super().__init__()

        self.device        = device
        self.config = config
        self.embed_dim = embed_dim = config.embed_dim
        self.graph_dim = graph_dim =  config.graph_dim
        self.loss_name = config.loss_name
        '''
        window_size should be attribute of dataset, not model
        '''
        self.window_size = window_size = config.window_size

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.layers      = nn.ModuleList([
            nn.Linear(embed_dim*window_size, embed_dim*window_size).to(self.device)
            for _ in range(self.config.depth-1)
            ])
        self.final_layer      = nn.Linear(embed_dim*window_size, embed_dim).to(self.device)


    def unembed(self,x):
        # import pdb; pdb.set_trace()
        xdy = x[:,None] - self.embed.weight[None]
        xdy = -xdy.square().mean(-1)
        # y = x.matmul(self.embed.weight.T)
        return xdy

    def grad_loss(self,item):
        return self._loss(item,self.config.grad_loss_name)

    def loss(self,item):
        return self._loss(item,self.loss_name)

    def _loss(self,item,loss_name):
        x = item['unmasked']
        WS = self.window_size
        lossVal = 0.
        xe = self.embed(x)
        B = len(x)
        i = 0
        for i in range(x.shape[1]-WS):
            xx = xe[:,i:WS+i]
            y = x[:,WS+i]


            xx = xx.reshape((B,-1))
            for li in range(self.config.depth-1):
                xx = xx + self.layers[li](xx)
                xx = xx.relu()
            xx = self.final_layer(xx)

            yp = self.unembed(xx)

            if loss_name=='KLD':
                yp = yp.log_softmax(-1)
                _lv = torch.gather(yp,index=y[:,None],dim=1)[:,0]
            elif loss_name == 'HSQ':
                # HellingerSquared
                yp = (yp.softmax(-1)+0.000001).sqrt()
                _lv = torch.gather(yp,index=y[:,None],dim=1)[:,0]
            elif loss_name == 'SELERR':
                yp = yp.argmax(-1)
                _lv = (yp == y)
            else:
                assert 0,loss_name

            lossVal += _lv
            # import pdb; pdb.set_trace()

        lossVal = lossVal/(i+1.)
        lossVal = -lossVal
        # lossVal = lossVal.mean(0)
        return lossVal



class SimpleDenseNetSquareSum(nn.Module):
    '''
    Use autoregression objective
    '''
    def __init__(self,
        device,
        config, _=None):

        state_count = 1
        super().__init__()

        self.device        = device
        self.config = config
        self.embed_dim = embed_dim = config.embed_dim
        self.graph_dim = graph_dim =  config.graph_dim
        self.loss_name = config.loss_name
        '''
        window_size should be attribute of dataset, not model
        '''
        self.window_size = window_size = config.window_size

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.layers      = nn.ModuleList([
            nn.Linear(embed_dim*window_size, embed_dim*window_size).to(self.device)
            for _ in range(self.config.depth-1)
            ])
        self.final_layer      = nn.Linear(embed_dim*window_size, embed_dim).to(self.device)


    def unembed(self,x):
        assert 0
        # import pdb; pdb.set_trace()
        xdy = x[:,None] - self.embed.weight[None]
        xdy = -xdy.square().mean(-1)
        # y = x.matmul(self.embed.weight.T)
        return xdy

    def grad_loss(self,item):
        return self._loss(item,self.config.grad_loss_name)

    def loss(self,item):
        return self._loss(item,self.loss_name)

    def _loss(self,item,loss_name):
        x = item['unmasked']
        WS = self.window_size
        lossVal = 0.
        xe = self.embed(x)
        B = len(x)
        i = 0
        for i in range(x.shape[1]-WS):
            xx = xe[:,i:WS+i]
            y = x[:,WS+i]


            xx = xx.reshape((B,-1))
            for li in range(self.config.depth-1):
                xx = xx + self.layers[li](xx)
                xx = xx.relu()
            xx = self.final_layer(xx)

            # yp = self.unembed(xx)
            yp = (xx.abs().matmul(self.embed.weight.T.abs())) + 1E-10

            if loss_name=='KLD':
                yp = torch.log( yp/yp.sum(dim=-1,keepdims=True))
                _lv = torch.gather(yp,index=y[:,None],dim=1)[:,0]
            elif loss_name == 'HSQ':
                # HellingerSquared
                yp = ( yp/yp.sum(dim=-1,keepdims=True) + 1E-10).sqrt()
                _lv = torch.gather(yp,index=y[:,None],dim=1)[:,0]
            elif loss_name == 'SELERR':
                yp = yp.argmax(-1)
                _lv = (yp == y)
            else:
                assert 0,loss_name

            lossVal += _lv
            # import pdb; pdb.set_trace()

        lossVal = lossVal/(i+1.)
        lossVal = -lossVal
        # lossVal = lossVal.mean(0)
        return lossVal



import math


def getAttention(q, k, v, mask=None, e=1e-12):
    # input is 4 dimension tensor
    # [batch_size, head, length, d_tensor]
    batch_size, head, length, d_tensor = k.size()

    # 1. dot product Query with Key^T to compute similarity
    k_t = k.transpose(2, 3)  # transpose
    score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

    # 2. apply masking (opt)
    if mask is not None:
        score = score.masked_fill(mask == 0, -e)

    # 3. pass them softmax to make [0, 1] range
    score = torch.softmax(score,-1)

    # 4. multiply with Value
    v = score @ v
    return v, score

class SimpleDenseNetTransformer(nn.Module):
    '''
    Use autoregression objective
    '''
    def __init__(self,
        device,
        config, _=None):

        state_count = 1
        super().__init__()

        self.device        = device
        self.config = config
        self.embed_dim = embed_dim = config.embed_dim
        self.graph_dim = graph_dim =  config.graph_dim
        self.loss_name = config.loss_name
        '''
        window_size should be attribute of dataset, not model
        '''
        self.window_size = window_size = config.window_size

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        # self.layers      = nn.ModuleList([
        #     nn.Linear(embed_dim*window_size, embed_dim*window_size).to(self.device)
        #     for _ in range(self.config.depth-1)
        #     ])
        self.layer_q = nn.Linear(embed_dim, embed_dim).to(self.device)
        self.layer_k = nn.Linear(embed_dim, embed_dim).to(self.device)
        self.layer_v = nn.Linear(embed_dim, embed_dim).to(self.device)
        self.final_layer      = nn.Linear(embed_dim*window_size, embed_dim).to(self.device)


    def unembed(self,x):
        # import pdb; pdb.set_trace()
        xdy = x[:,None] - self.embed.weight[None]
        xdy = -xdy.square().mean(-1)
        # y = x.matmul(self.embed.weight.T)
        return xdy

    def grad_loss(self,item):
        return self._loss(item,self.config.grad_loss_name)

    def loss(self,item):
        return self._loss(item,self.loss_name)

    def _loss(self,item,loss_name):
        x = item['unmasked']
        WS = self.window_size
        lossVal = 0.
        xe = self.embed(x)
        B = len(x)
        i = 0
        for i in range(x.shape[1]-WS):
            xx = xe[:,i:WS+i]
            y = x[:,WS+i]


            # xx = xx.reshape((B,-1))
            for li in range(self.config.depth-1):
                xq = self.layer_q(xx)
                xk = self.layer_k(xx)
                xv = self.layer_v(xx)
                att = (xq @ xk.transpose(2,1) /math.sqrt(self.embed_dim)).softmax(-1)
                xx = xx + att @ xv
                # import pdb; pdb.set_trace()
                # xx = xx + self.layers[li](xx)
                xx = xx.relu()
            xx = xx.reshape((B,-1))
            xx = self.final_layer(xx)

            yp = self.unembed(xx)

            if loss_name=='KLD':
                yp = yp.log_softmax(-1)
                _lv = torch.gather(yp,index=y[:,None],dim=1)[:,0]
            elif loss_name == 'HSQ':
                # HellingerSquared
                yp = (yp.softmax(-1)+0.000001).sqrt()
                _lv = torch.gather(yp,index=y[:,None],dim=1)[:,0]
            elif loss_name == 'SELERR':
                yp = yp.argmax(-1)
                _lv = (yp == y)
            else:
                assert 0,loss_name

            lossVal += _lv
            # import pdb; pdb.set_trace()

        lossVal = lossVal/(i+1.)
        lossVal = -lossVal
        # lossVal = lossVal.mean(0)
        return lossVal


class AlignmentModelPrototype(nn.Module):
    '''
    This model is to test whether attention is learnable with a simple
    optimisation algorithm

    [TBC] share interface with :class:SoftAlignmentModel
    '''
    AVG_METHOD = None
    ATT_METHOD = None
    ALIGNEMNT_METHOD = None

    def __init__(self,device,config,_=None):
        super().__init__()
        self.config = config
        self.device = device
        self.n_hidden = n_hidden = config.embed_dim
        self.n_class  = n_class  = config.graph_dim
        dropout = config.beta
        assert config.depth == 1,config.depth
        self.embed = nn.Embedding(n_class,n_hidden).to(self.device)

        self.mapping = nn.Linear(n_hidden, n_hidden).to(self.device)

        # Linear for attention
        self.attn = nn.Linear(n_hidden, n_hidden).to(self.device)
        self.out_layer  = nn.Linear(n_hidden, n_class).to(self.device)
        assert self.ALIGNEMNT_METHOD
        assert self.ATT_METHOD
        assert self.AVG_METHOD

    def loss(self,item,):
        return self._loss(item,'loss')
    grad_loss = loss

    def forward(self,item):
        return self._loss(item,'forward')



    def _loss(self,item,ret):
        source = item['source'] ### token sequence
        target = item['target'] ### token seq

        source_embed = self.embed(source)
        target_embed = self.embed(target)

        target_len = item['target_len']
        target_notnull = torch.arange(target.size(1),device=self.device)[None,:]<target_len[:,None]

        source_len = item['source_len']
        source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]


        # (B, S, E)
        out_embed = self.mapping( source_embed )
        # (B, S, C)
        out_logit = self.out_layer(out_embed).log_softmax(-1)
        # (B, 1, T)
        D = source.shape[1]
        output_tok = item['target'][:,None,:].repeat((1,D,1))

        # (B, S, T)
        logp_mat = torch.gather( out_logit, index=output_tok, dim=-1)


        '''
        Whether to mask <pad> in source sentence?
        '''
        if self.ATT_METHOD=='masked':
            INF = 1E15
            logp_mat = logp_mat + -INF * (~source_notnull[:,:,None])
        elif self.ATT_METHOD=='allow_source_pad':
            pass
        else:
            raise NotImplementedError(self.ATT_METHOD)

        # (B,T)
        '''
        Whether to use hard or soft alignment?

        Note hard alignment does not yield a proba model, meaning its loss function
        cannot be compared to soft model !!!!
        '''
        if self.ALIGNEMNT_METHOD=='soft':
            val = logp_mat.logsumexp(dim=1) - math.log(source.size(1))
        elif self.ALIGNEMNT_METHOD =='hard':
            val,which = logp_mat.max(dim=1)
        else:
            raise NotImplementedError(self.ALIGNEMNT_METHOD)

        val = - val
        attn = logp_mat.softmax(dim=1)
        # attn = attn * target_notnull[:,None,:]
        # .unsqueeze(-1)

        '''
        Whether to mask <pad> in target sentence?
        '''
        if self.AVG_METHOD=='masked':
            loss = mean_notnull(val,target_notnull)
        elif self.AVG_METHOD == 'simple_mean':
            loss = val.mean(-1)
        else:
            raise NotImplementedError(self.AVG_METHOD)

        if ret =='forward':
            return val, attn

        return loss


def mean_notnull(val, target_notnull):
    '''
    Take average on particular tokens, not all tokens.

    target_notnull = torch.arange(target.size(1),device=self.device)[None,:]<target_len[:,None]
    '''
HardAlignment
    loss =  (val * target_notnull ).sum(-1) / target_notnull.sum(-1)
    return loss


class HardAlignmentModel(AlignmentModelPrototype):
    ALIGNEMNT_METHOD = 'hard'
    AVG_METHOD = 'masked'
    ATT_METHOD = 'masked'


class SoftAlignmentModel(AlignmentModelPrototype):
    ALIGNEMNT_METHOD = 'soft'
    AVG_METHOD = 'masked'
    ATT_METHOD = 'masked'


class SoftAlignmentModelAllowSourcePad(SoftAlignmentModel):
    # AVG_METHOD = 'simple_mean'
    ATT_METHOD = 'allow_source_pad'


class SoftAlignmentModelSimpleMean(SoftAlignmentModel):
    AVG_METHOD = 'simple_mean'
    ATT_METHOD = 'allow_source_pad'
