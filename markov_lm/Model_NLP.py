import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from dataclasses import dataclass
from markov_lm.Model_gmm import AbstractLayerConfig
# from transformers.models.bert.modeling_bert import BertLayer,BertConfig
from markov_lm.nlp.model_seq2seq import Seq2SeqWithAttention


class Seq2SeqWithNoAttention(Seq2SeqWithAttention):
    # USE_ATTENTION = 0
    MODE_EMISSION = 'no_attention'
class Seq2SeqWithAttentionMixture(Seq2SeqWithAttention):
    # USE_ATTENTION = 0
    MODE_EMISSION = 'attention_mixture'
# Seq2SeqWithNoAttention

class TranslationModelPrototype(nn.Module):
    meta = {}
    def log_param(self,buf,plt):
        return


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


class AlignmentModelPrototype(TranslationModelPrototype):
    '''
    This model is to test whether attention is learnable with a simple
    optimisation algorithm

    [TBC] share interface with :class:SoftAlignmentModel
    '''
    AVG_METHOD = None
    ATT_METHOD = None
    ALIGNMENT_METHOD = 'mixture_soft'
    ALIGNMENT_PRIOR = None
    IS_RUN_BACKWARD = None
    # ALIGNMENT_MODEL = 'mixture'
    INF =1E15
    # SOURCE_ENCODER ='na'


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
        self.source_encoder = None
        assert self.ALIGNMENT_METHOD
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
        S = source.size(1)
        T = target.size(1)
        B = source.size(0)
        target_len = item['target_len']
        target_notnull = torch.arange(target.size(1),device=self.device)[None,:]<target_len[:,None]

        source_len = item['source_len']
        source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]
        if self.source_encoder is None:
             # == 'na':
            pass
        else:
            source_embed = self.source_encoder(source_embed)



        # (B, S, E)
        out_embed = self.mapping( source_embed )
        # (B, S, C)
        out_logit = self.out_layer(out_embed).log_softmax(-1)
        # (B, 1, T)
        D = source.shape[1]
        output_tok = item['target'][:,None,:].repeat((1,D,1))

        # (B, S, T, K)
        output_tok

        # (B, S, T)

        #### get logp_matrix
        logp_mat = torch.gather( out_logit, index=output_tok, dim=-1)

        if self.ALIGNMENT_METHOD in 'mixture_soft mixture_hard'.split():
            prior = torch.tensor([0],device= self.device)[None,None]
            if self.ALIGNMENT_PRIOR is None:
                '''
                Uniform prior
                '''
                prior += -math.log(source.size(1))

            elif self.ALIGNMENT_PRIOR =='gaussian':
                xs = torch.arange(S,device=self.device)[None,:,None]
                xt = torch.arange(T,device=self.device)[None,None,:,]
                diff =  -self.config.beta * (xs - xt).abs()
                prior =  prior +  diff.log_softmax(1)
                # logp_mat = logp_mat + self.alignment(item)shared
                # source_len,target_len)
            elif self.ALIGNMENT_PRIOR == 'shared':
                diff = self.shared_log_align[None]
                prior = prior + diff.log_softmax(1)
            elif self.ALIGNMENT_PRIOR == 'per_token':
                # import pdb; pdb.set_trace()
                mat = self.per_token_alignment[source]
                diff = torch.arange(S,device=self.device)[None,:,None] - torch.arange(S,device=self.device)[None,None,:] + (S -1)
                # import pdb; pdb.set_trace()
                diff = torch.gather(mat,index=diff.repeat((B,1,1)),dim=-1)
                prior = prior + diff.log_softmax(1)
                # torch.gather(self.per_token_alignment, index=,dim=0)
            else:
                raise NotImplementedError(self.ALIGNMENT_PRIOR)


            '''
            Whether to mask <pad> in source sentence?
            '''
            if self.ATT_METHOD=='masked':
                INF = 1E15
                prior = prior + -INF * (~source_notnull[:,:,None])
                # logp_mat = logp_mat +
            elif self.ATT_METHOD=='allow_source_pad':
                pass
            else:
                raise NotImplementedError(self.ATT_METHOD)

            prior = prior.log_softmax(1)
            logp_mat = logp_mat + prior
            # (B,T)
            '''
            Whether to use hard or soft alignment?

            Note hard alignment does not yield a proba model, meaning its loss function
            cannot be compared to soft model !!!!
            '''
            if self.ALIGNMENT_METHOD=='mixture_soft':
                val = logp_mat.logsumexp(dim=1)
            elif self.ALIGNMENT_METHOD =='mixture_hard':
                val,which = logp_mat.max(dim=1)
            else:
                raise NotImplementedError(self.ALIGNMENT_METHOD)

            ### get \log P(E|m) = \sum_i  \log \sum_{a_i} P(e_i,a_i|m)
            val  = - val
            attn = logp_mat.softmax(dim=1)
            self._last_att = attn
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

            loss, attn

        elif self.ALIGNMENT_METHOD=='hmm':
            #### initialise forward variable
            loga = torch.ones((B,S,1), device=self.device).log_softmax(1)
            # loga = (torch.zeros((B,S,1), device=self.device)-self.INF)
            # loga[:,0]=0
            # loga = loga.log_softmax(1)
            logb = torch.ones((B,S,1), device=self.device).log()
            # _softmax(1)
            loga_array = torch.zeros((B,S,T),device=self.device)
            logb_array = torch.zeros((B,S,T),device=self.device)
            # (1, S, S)
            xs = torch.arange(S,device=self.device)[:,None]
            if self.ALIGNMENT_PRIOR == 'gaussian':
                logt = torch.exp( -self.config.beta*(xs + 0.5 - xs.T ).abs() )[None]
            elif self.ALIGNMENT_PRIOR=='uniform':
                logt = torch.ones((1,S,S),device=self.device)
            elif self.ALIGNMENT_PRIOR == 'shared':
                logt = self.shared_log_align[None]
            else:
                raise NotImplementedError(self.ALIGNMENT_PRIOR)

            if self.ATT_METHOD=='masked':
                logt = logt + -self.INF * (~source_notnull[:,:,None])
            elif self.ATT_METHOD=='allow_source_pad':
                pass
            else:
                raise NotImplementedError(self.ATT_METHOD)
            logt = logt.log_softmax(dim=2)

            ### only count loss if not null, otherwise no loss for p(<pad>)=1
            logp_mat = logp_mat * target_notnull[:,None,:]
            for ti in range(T):
                # loga = ((loga + logt).logsumexp(dim=1)) + (logp_mat[:,:,ti] * target_notnull[:,ti:ti+1])  ### only count loss if not null, otherwise no loss for p(<pad>)=1
                loga = ((loga + logt).logsumexp(dim=1)) + logp_mat[:,:,ti]
                loga = loga[:,:,None]
                loga_array[:,:,ti:ti+1] = loga

            logp = loga.squeeze(-1).logsumexp(dim=1)
            loss = -logp / target_len

            if self.IS_RUN_BACKWARD:
                for ti in range(T):
                    ti = T-1-ti
                    logb_array[:,:,ti:ti+1]= logb
                    logb = (logb + logp_mat[:,:,ti:ti+1] + logt.transpose(2,1)).logsumexp(dim=1)
                    logb = logb[:,:,None]

                # attn = (loga_array + logb_array) - (loga_array + logb_array).logsumexp(dim=1,keepdims=True)
                attn = (loga_array + logb_array).softmax(dim=1)
                 # - (loga_array + logb_array).logsumexp(dim=1,keepdims=True)
            else:
                attn = torch.ones((B,S,T),device=self.device)
            #### recursively evaluate forward variable

        else:
            raise NotImplementedError(self.ALIGNMENT_METHOD)

        self._last_att = attn

        if ret =='forward':
            return loss, attn

        return loss


def mean_notnull(val, target_notnull):
    '''
    Take average on particular tokens, not all tokens.

    target_notnull = torch.arange(target.size(1),device=self.device)[None,:]<target_len[:,None]
    '''
    loss =  (val * target_notnull ).sum(-1) / target_notnull.sum(-1)
    return loss


class HardAlignmentModel(AlignmentModelPrototype):
    ALIGNMENT_METHOD = 'mixture_hard'
    AVG_METHOD = 'masked'
    ATT_METHOD = 'masked'


class SoftAlignmentModel(AlignmentModelPrototype):
    ALIGNMENT_METHOD = 'mixture_soft'
    AVG_METHOD = 'masked'
    ATT_METHOD = 'masked'

class GaussianSoftAlignmentModel(AlignmentModelPrototype):
    ALIGNMENT_PRIOR = 'gaussian'
    ALIGNMENT_METHOD = 'mixture_soft'
    AVG_METHOD = 'masked'
    ATT_METHOD = 'masked'


class SoftAlignmentModelAllowSourcePad(SoftAlignmentModel):
    # AVG_METHOD = 'simple_mean'
    AVG_METHOD = 'masked'
    ATT_METHOD = 'allow_source_pad'


class SoftAlignmentModelSimpleMean(SoftAlignmentModel):
    AVG_METHOD = 'simple_mean'
    ATT_METHOD = 'allow_source_pad'

from markov_lm.util_html import write_png_tag

class SharedSoftAlignmentModel(AlignmentModelPrototype):
    ALIGNMENT_PRIOR = 'shared'
    ALIGNMENT_METHOD = 'mixture_soft'
    AVG_METHOD = 'masked'
    ATT_METHOD = 'masked'
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        S = self.config.n_step
        x = nn.Linear(S,S)
        self.shared_log_align = nn.Parameter(x.weight)

    def log_param(self,buf,plt):
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        mat = self.shared_log_align.log_softmax(0).cpu().detach()

        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat,vmin=-4,vmax=0.)
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        ax.set_title(f'[log_param]self.shared_log_align\nModel_name={self.config.model_name}\nEpoch:{epoch}')
        buf.write(write_png_tag(fig))
        pass



class PerTokenSoftAlignmentModel(AlignmentModelPrototype):
    ALIGNMENT_PRIOR = 'per_token'
    ALIGNMENT_METHOD = 'mixture_soft'
    AVG_METHOD = 'masked'
    ATT_METHOD = 'masked'
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        S = self.config.n_step
        G = self.config.graph_dim
        x = nn.Linear(G,S*2)
        self.per_token_alignment = nn.Parameter(x.weight.T)

    def log_param(self,buf,plt):
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        mat = self._last_att.mean(dim=0).log().cpu().detach()
        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat,vmin=-4,vmax=0.)
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        ax.set_title(f'[log_param]self.shared_log_align\nModel_name={self.config.model_name}\nEpoch:{epoch}')
        buf.write(write_png_tag(fig))
        pass

SAM1 = SoftAlignmentModelSimpleMean
SAM2 = SoftAlignmentModelAllowSourcePad
SAM3 = SoftAlignmentModel
SAM4 = GaussianSoftAlignmentModel
SAM5 = SharedSoftAlignmentModel

class SAM7(SoftAlignmentModel):
    '''
    Implement a causal relation graph model
    aside from the alignment model

    $$
    P(e_i | f,e,a,b)  \propto \exp( e_i W_1 e_{b_i}^T +  e_i W_2 f_{a_i})
    $$

    TOOOOOOOOOO SLOW!

    '''

    ALIGNMENT_PRIOR = 'na'
    ALIGNMENT_METHOD = 'na'
    AVG_METHOD = 'na'
    ATT_METHOD = 'na'
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        n_hidden = self.n_hidden
        self.mapping2 = nn.Linear(n_hidden, n_hidden).to(self.device)

    def _loss(self,item,ret):
        source = item['source'] ### token sequence
        target = item['target'] ### token seq

        source_embed = self.embed(source)
        target_embed = self.embed(target)
        S = source.size(1)
        T = target.size(1)
        B = source.size(0)
        target_len = item['target_len']
        target_notnull = torch.arange(target.size(1),device=self.device)[None,:]<target_len[:,None]

        source_len = item['source_len']
        source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]


        # (B, S, T, E)
        out_embed = self.mapping( source_embed )[:,:,None,:] + self.mapping2(target_embed)[:,None,:,:]
        # (B, S, T, C)
        out_logit = self.out_layer(out_embed).log_softmax(-1)

        D = source.shape[1]

        # (B, S, T, 1)
        output_tok = item['target'][:,None,None,:].repeat((1,S,T,1))
        output_tok

        # (B, S, T, T)
        # logp_mat = torch.gather( out_logit, index=output_tok, dim=-1,sparse_grad=True)
        logp_mat = torch.gather( out_logit, index=output_tok, dim=-1)
        '''
        Uniform prior
        '''
        logp_mat += -math.log(S*T/2) #[approx]

        '''
        Whether to mask <pad> in source sentence?
        Mask <pad> in input seq
        '''
        INF = 1E15
        logp_mat = logp_mat + -INF * (~source_notnull[:,:,None,None])

        '''
        Force nodes to depend on left tokens
        '''
        ts = torch.arange(T,device=self.device)
        is_future_node = ts[None,None,None,:]<=ts[None,None,:,None]
        logp_mat = logp_mat + -INF * (is_future_node)

        # (B,T)
        '''
        Whether to use hard or soft alignment?

        Note hard alignment does not yield a proba model, meaning its loss function
        cannot be compared to soft model !!!!
        '''
        val = logp_mat.logsumexp(dim=(1,2))
        val = - val
        attn1 = logp_mat.logsumexp(dim=2).softmax(dim=1) ## alignment
        attn2 = logp_mat.logsumexp(dim=1).softmax(dim=1) ## dep
        self._last_att = attn1
        attn = attn1

        '''
        Whether to mask <pad> in target sentence?
        Also excludes <sos>
        '''
        loss = mean_notnull(val[:,1:],target_notnull[:,1:])

        if ret =='forward':
            return val, attn
        return loss



class SAM8(SoftAlignmentModel):
    '''
    Implement a causal relation graph model
    aside from the alignment model

    '''

    ALIGNMENT_PRIOR = 'na'
    ALIGNMENT_METHOD = 'na'
    AVG_METHOD = 'na'
    ATT_METHOD = 'na'
    USE_CAUSAL = 0
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        n_hidden = self.n_hidden
        self.mapping2 = nn.Linear(n_hidden, n_hidden).to(self.device)

    def _loss(self,item,ret):
        source = item['source'] ### token sequence
        target = item['target'] ### token seq

        source_embed = self.embed(source)
        target_embed = self.embed(target)
        S = source.size(1)
        T = target.size(1)
        B = source.size(0)
        target_len = item['target_len']
        target_notnull = torch.arange(target.size(1),device=self.device)[None,:]<target_len[:,None]

        source_len = item['source_len']
        source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]


        # (B,   S, E)
        out_embed = self.mapping( source_embed )
        if self.USE_CAUSAL:
            # (B,S,T,E)
            out_embed = out_embed[:,:,None] + target_embed.roll(1,1)[:,None,:,:]
            out_logit = self.out_layer(out_embed).log_softmax(-1)
            output_tok= item['target'][:,None,:,None].repeat((1,S,1,1))
            logp_mat  = torch.gather( out_logit, index=output_tok, dim=-1)
            # logp_mat  = torch.gather( out_logit, index=output_tok, dim=-1)
        else:
            pass
            # (B,   S, C)
            out_logit = self.out_layer(out_embed).log_softmax(-1)
            # (B, 1*S, T)
            output_tok= item['target'][:,None,:].repeat((1,S,1))
            # (B,   S, T)
            logp_mat  = torch.gather( out_logit, index=output_tok, dim=-1)
        logp_mat  += -math.log(S) #[approx]
        logp_mat1 = logp_mat

        '''
        Whether to mask <pad> in source sentence?
        Mask <pad> in input seq
        '''
        INF = 1E15
        logp_mat1 = logp_mat1 + -INF * (~source_notnull[:,:,None])



        out_embed = self.mapping2( target_embed )
        # (B,   T, C)
        out_logit = self.out_layer(out_embed).log_softmax(-1)
        # (B, 1*T, T)
        output_tok= item['target'][:,None,:].repeat((1,T,1))
        # (B,   T, T)
        logp_mat  =  torch.gather( out_logit, index=output_tok, dim=-1)
        logp_mat  += -math.log(T) #[approx]
        logp_mat2 =  logp_mat

        '''
        Force nodes to depend on left tokens
        '''
        ts = torch.arange(T,device=self.device)
        is_future_node = ts[None,None,:]<=ts[None,:,None]
        logp_mat2  = logp_mat2 + -INF * (is_future_node)

        # (B,T)
        '''
        Whether to use hard or soft alignment?

        Note hard alignment does not yield a proba model, meaning its loss function
        cannot be compared to soft model !!!!
        '''
        val = torch.stack([logp_mat1.logsumexp(dim=1),logp_mat2.logsumexp(dim=1)], dim=-1).logsumexp(-1) - math.log(2)
        # val = logp_mat.logsumexp(dim=(1,2))
        val = - val

        # attn1 = logp_mat1.softmax(dim=1) ## alignment
        # attn2 = logp_mat2.softmax(dim=1) ## dep
        attn1 = (logp_mat1 + val.unsqueeze(1)).exp()  ## alignment
        attn2 = (logp_mat2 + val.unsqueeze(1)).exp()  ## alignment
        self._last_att_1 = attn1
        self._last_att_2 = attn2
        attn = attn2

        '''
        Whether to mask <pad> in target sentence?
        Also excludes <sos>
        '''
        loss = mean_notnull(val[:,1:],target_notnull[:,1:])

        if ret =='forward':
            return val, attn
        return loss




class SAM9(SoftAlignmentModel):
    '''
    Implement a causal relation graph model
    aside from the alignment model

    '''

    ALIGNMENT_PRIOR = 'na'
    ALIGNMENT_METHOD = 'uniform'
    AVG_METHOD = 'na'
    ATT_METHOD = 'na'
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        n_hidden = self.n_hidden
        self.mapping2 = nn.Linear(n_hidden, n_hidden).to(self.device)

    def _loss(self,item,ret):
        source = item['source'] ### token sequence
        target = item['target'] ### token seq

        source_embed = self.embed(source)
        target_embed = self.embed(target)
        S = source.size(1)
        T = target.size(1)
        B = source.size(0)
        target_len = item['target_len']
        target_notnull = torch.arange(target.size(1),device=self.device)[None,:]<target_len[:,None]

        source_len = item['source_len']
        source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]
        #
        #
        # # (B,   S, E)
        # out_embed = self.mapping( source_embed )
        # # (B,   S, C)
        # out_logit = self.out_layer(out_embed).log_softmax(-1)
        # # (B, 1*S, T)
        # output_tok= item['target'][:,None,:].repeat((1,S,1))
        # # (B,   S, T)
        # logp_mat  = torch.gather( out_logit, index=output_tok, dim=-1)
        # logp_mat  += -math.log(S) #[approx]
        # logp_mat1 = logp_mat
        #
        # '''
        # Whether to mask <pad> in source sentence?
        # Mask <pad> in input seq
        # '''
        INF = 1E15
        # logp_mat1 = logp_mat1 + -INF * (~source_notnull[:,:,None])



        out_embed = self.mapping2( target_embed )
        # (B,   T, C)
        out_logit = self.out_layer(out_embed).log_softmax(-1)
        # (B, 1*T, T)
        output_tok= item['target'][:,None,:].repeat((1,T,1))
        # (B,   T, T)
        logp_mat  =  torch.gather( out_logit, index=output_tok, dim=-1)
        logp_mat  += -math.log(T) #[approx]
        logp_mat2 =  logp_mat

        '''
        Force nodes to depend on left tokens
        '''
        ts = torch.arange(T,device=self.device)
        is_future_node = ts[None,None,:]<=ts[None,:,None]
        logp_mat  = logp_mat2 + -INF * (is_future_node)

        if self.ALIGNMENT_PRIOR =='uniform':
            '''
            Uniform prior
            '''
            pass
            # logp_mat += -math.log(source.size(1))

        elif self.ALIGNMENT_PRIOR =='gaussian':
            xs = torch.arange(T,device=self.device)[None,:,None]
            xt = torch.arange(T,device=self.device)[None,None,:,]
            diff =  -self.config.beta * (xs - xt).abs()
            logp_mat = logp_mat + diff.log_softmax(1)
        # (B,T)
        '''
        Whether to use hard or soft alignment?

        Note hard alignment does not yield a proba model, meaning its loss function
        cannot be compared to soft model !!!!
        '''
        # val = torch.stack([logp_mat1.logsumexp(dim=1),logp_mat2.logsumexp(dim=1)], dim=-1).logsumexp(-1) - math.log(2)
        # val = logp_mat.logsumexp(dim=(1,2))
        val = logp_mat.logsumexp(dim=1)
        val = - val

        # attn1 = logp_mat1.softmax(dim=1) ## alignment
        # attn2 = logp_mat2.softmax(dim=1) ## dep
        # attn1 = (logp_mat1 + val.unsqueeze(1)).exp()  ## alignment
        attn2 = (logp_mat + val.unsqueeze(1)).exp()  ## alignment
        self._last_att_1 = attn2
        self._last_att_2 = attn2
        attn = attn2

        '''
        Whether to mask <pad> in target sentence?
        Also excludes <sos>
        '''
        loss = mean_notnull(val[:,1:],target_notnull[:,1:])

        if ret =='forward':
            return val, attn
        return loss


    def log_param(self,buf,plt):
        key = '_last_att_1'
        mat = self._last_att_1.mean(dim=0).log().cpu().detach()
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat,vmin=-4,vmax=0.)
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        ax.set_title(f'[log_param]self.{key}\nModel_name={self.config.model_name}\nEpoch:{epoch}')
        buf.write(write_png_tag(fig))

        key = '_last_att_2'
        mat = self._last_att_2.mean(dim=0).log().cpu().detach()
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat,vmin=-4,vmax=0.)
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        ax.set_title(f'[log_param]self.{key}\nModel_name={self.config.model_name}\nEpoch:{epoch}')
        buf.write(write_png_tag(fig))


        pass
class SAM10(SAM9):
    '''
    Implement a causal relation graph model
    aside from the alignment model

    '''

    ALIGNMENT_PRIOR = 'gaussian'
    ALIGNMENT_METHOD = 'na'
    AVG_METHOD = 'na'
    ATT_METHOD = 'na'

class SAM11(SoftAlignmentModel):
    '''
    Implement a causal relation graph model
    aside from the alignment model

    '''
    ALIGNMENT_PRIOR = 'gaussian'
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        n_hidden = self.n_hidden
        self.source_encoder_base = nn.Conv1d(n_hidden,n_hidden,config.kernel_size,padding='same')
        self.source_encoder = lambda x:self.source_encoder_base(x.transpose(2,1)).transpose(2,1)
        # Linear
        # self.mapping2 = nn.Linear(n_hidden, n_hidden).to(self.device)

    # def _loss(self,item,ret):

class SAM12(SoftAlignmentModel):
    '''
    Implement a causal relation graph model
    aside from the alignment model

    '''
    ALIGNMENT_PRIOR = 'gaussian'
    USE_CAUSAL = 1

    def log_param(self,buf,plt):
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        mat = self._last_att.mean(dim=0).log().cpu().detach()
        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat,vmin=-4,vmax=0.)
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        ax.set_title(f'[log_param]self.shared_log_align\nModel_name={self.config.model_name}\nEpoch:{epoch}')
        buf.write(write_png_tag(fig))
        pass

class HMMSoftAlignmentModel(SoftAlignmentModel):
    '''
    Implement a HMM alignment model
    '''
    ALIGNMENT_METHOD = 'hmm'
    ALIGNMENT_PRIOR = 'gaussian'
    IS_RUN_BACKWARD = 1

    def log_param(self,buf,plt):
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        mat = self._last_att.mean(dim=0).log().cpu().detach()
        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat,vmin=-4,vmax=0.)
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        ax.set_title(f'[log_param]self.shared_log_align\nModel_name={self.config.model_name}\nEpoch:{epoch}')
        buf.write(write_png_tag(fig))
        pass



class SAM14(SharedSoftAlignmentModel):
    '''
    Implement a HMM alignment model
    '''
    ALIGNMENT_METHOD = 'hmm'
    ALIGNMENT_PRIOR = 'shared'
    IS_RUN_BACKWARD = 1
    # def __init__(self,device,config,_=None):
    #     super().__init__(device,config)
    #     S = self.config.n_step
    #     x = nn.Linear(S,S)
    #     self.shared_log_align = nn.Parameter(x.weight)
    #

    def log_param(self,buf,plt):
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        key = 'self.shared_log_align \n $\log P(a_{i+1}|a_i)$'
        mat = self.shared_log_align.log_softmax(0).cpu().detach()

        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat.T,vmin=-4,vmax=0.,origin='lower')
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        ax.set_title(f'[log_param]{key}\nModel_name={self.config.model_name}\nEpoch:{epoch}')
        ax.set_xlabel('$a_i$')
        ax.set_ylabel('$a_{i+1}$')
        plt.suptitle('')
        buf.write(write_png_tag(fig))
        pass

SAM13 = HMMSoftAlignmentModel
