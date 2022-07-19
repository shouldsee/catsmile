import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from dataclasses import dataclass
from markov_lm.Model_gmm import AbstractLayerConfig
# from transformers.models.bert.modeling_bert import BertLayer,BertConfig

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



@dataclass
class NLPLayerConfig(AbstractLayerConfig):
    graph_dim:int
    model_name:str
    window_size: int
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
