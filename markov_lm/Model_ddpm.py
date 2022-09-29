'''
Implements mixture autoencoder
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from markov_lm.Model_pretrain import lazy_load_pretrain_model

#from markov_lm.Model_gmm import
from transformers.models.bert.modeling_bert import BertLayer,BertConfig
import math

from markov_lm.Model_gmm import AbstractLayerConfig
from dataclasses import dataclass
@dataclass
class DDPMModelConfig(AbstractLayerConfig):
    depth:int
    graph_dim:int
    iter_per_layer:int
    kernel_size:int
    model_name:str
    beta: float
    submodel_config: dict=lambda:{}
    n_step: int = 1
    embed_dim:int = 0
    p_null: float = 0.0001
    def to_model(self,device):
#        cls = getattr(self.model_name)
        cls = eval(self.model_name)
#        if cls is None:
#            cls = AddModelWithAttentionStacked
        return cls(device,self,None)
#SASConfig = LayerConfig

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class AutoEncoderDDPMShared(nn.Module):
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()

        self.beta = config.beta
        assert self.beta < 1.0 and self.beta>0.0,self.beta
        self.n_step = config.n_step
        self.G = config.graph_dim

        self.project = nn.Linear(config.graph_dim,config.embed_dim)
        self.rec = nn.Linear(config.embed_dim,config.graph_dim)

    def get_ab(self,):
        T = self.n_step
        betas = torch.ones((T+1,),device=self.device) * self.beta
        betas[0]=0.
        alphas = (1 - betas).cumprod(0)[:-1]
        return alphas,betas

    def loss(self,item,ret='rec'):
        x0 = images = item['images']
        B = len(x0)
        G = self.G
        #(B,G)
        T = self.n_step
        alphas,betas = self.get_ab()

        # x0 = x
        xt,xt1,xz = self.encode(x0,ret='tup')

        # xtt  = self.encode(x0,ret='rec')
        #(B,G,T,2) parallel sampling of t,t+1 pair
        # import pdb; pdb.set_trace()

        xtv = torch.arange(self.n_step,device=self.device)[None,None,:]
        pred = self._decode(xt1,xtv)
        # pred = xt1 + self.rec(self.project(xt1.transpose(2,1))).transpose(2,1)
        # wt = -0.5* xz[:,:,:,1].square().sum(dim=1,keepdims=True)
        # loss = (pred-xt).square() * torch.exp( -0.5* xz[:,:,:,1].square().sum(dim=1,keepdims=True) )/math.sqrt(2*3.1415)
        # loss = (pred-xt).square() * wt.softmax(0) * B
        loss = (pred-xt).square()
        return loss.mean(dim=(1,2))

    grad_loss = loss
    def _decode(self,x,t):
        '''
        This is just a shallow AutoEncoder
        '''

        pred = x + self.rec(self.project(x.transpose(2,1))).transpose(2,1)
        # xz = torch.normal(0,1,size=x.shape,device=self.device)
        # pred = pred + xz
        return pred

    def decode(self,x,ret='rec'):
        if x.shape.__len__()==2:
            x = x.unsqueeze(-1)
        # ones = torch.ones(x.shape,device=self.device).long()
        ones = torch.ones((1,),device=self.device).long()
        for xi in range(self.n_step):
            x = self._decode(x, xi*ones)
        return x[:,:,0]

    def encode(self,x0,ret='rec'):
        B = len(x0)
        G = self.G
        #(B,G)
        T = self.n_step
        alphas,betas = self.get_ab()

        # xtv = torch.arange(self.n_step,device=self.device)[None,None,:]
        if ret =='tup':
            xz = torch.normal(0,1,size=(B,G)+(T,2),device=self.device)
            xt = alphas.sqrt() * x0.unsqueeze(-1) + (1-alphas[None,None]).sqrt() * xz[:,:,:,0]
            xb1 = betas[None,None,1:]
            xt1 = xt*(1-xb1).sqrt()+ xb1.sqrt() *xz[:,:,:,1]
            return xt,xt1,xz

        if ret =='rec':
            '''
            Set to zero to force equilibrium
            '''
            alphas = alphas[-1:]
            xz = torch.normal(0,1,size=(B,G,1,1),device=self.device)
            xt = alphas.sqrt() * x0.unsqueeze(-1) + (1-alphas[None,None]).sqrt() * xz[:,:,:,0]
            return xt

        else:
            assert 0,ret
import copy
# from markov_lm.Model_gmm import GMMLayerConfig
class AutoEncoderDDPMWithFixedAE(nn.Module):
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()

        self.beta = config.beta
        assert self.beta < 1.0 and self.beta>0.0,self.beta
        self.n_step = config.n_step
        self.G = config.graph_dim
        # config_copy = copy.copy(config)
        # config_copy.model_name  = config_copy.submodel_name
        self.ae = config.submodel_config.to_model(self.device)

    def get_ab(self,):
        T = self.n_step
        betas = torch.ones((T+1,),device=self.device) * self.beta
        betas[0]=0.
        alphas = (1 - betas).cumprod(0)[:-1]
        return alphas,betas

    def loss(self,item,ret='rec'):
        x0 = images = item['images']
        B = len(x0)
        G = self.G
        #(B,G)
        T = self.n_step
        alphas,betas = self.get_ab()

        xt,xt1,xz = self.encode(x0,ret='tup')
        # xtt  = self.encode(x0,ret='rec')
        #(B,G,T,2) parallel sampling of t,t+1 pair
        # import pdb; pdb.set_trace()

        xtv = torch.arange(self.n_step,device=self.device)[None,None,:]
        pred = self._decode(xt1,xtv)
        loss = (pred-xt).square()
        return loss.mean(dim=(1,2))

    grad_loss = loss
    def _decode(self,x,t):
        _xshape = x.shape
        B,E,T = x.shape
        lat = self.ae.encode(x.transpose(2,1).reshape((B*T,E)),ret='rec')
        dx  = self.ae.decode(lat,ret='rec').reshape((B,T,E)).transpose(2,1)
        pred = x + dx
        return pred

    def decode(self,x,ret='rec'):
        if x.shape.__len__()==2:
            x = x.unsqueeze(-1)
        # ones = torch.ones(x.shape,device=self.device).long()
        ones = torch.ones((1,),device=self.device).long()
        for xi in range(self.n_step):
            x = self._decode(x, xi*ones)
        return x[:,:,0]

    def encode(self,x0,ret='rec'):
        B = len(x0)
        G = self.G
        #(B,G)
        T = self.n_step
        alphas,betas = self.get_ab()

        # xtv = torch.arange(self.n_step,device=self.device)[None,None,:]
        if ret =='tup':
            xz = torch.normal(0,1,size=(B,G)+(T,2),device=self.device)
            xt = alphas.sqrt() * x0.unsqueeze(-1) + (1-alphas[None,None]).sqrt() * xz[:,:,:,0]
            xb1 = betas[None,None,1:]
            xt1 = xt*(1-xb1).sqrt()+ xb1.sqrt() *xz[:,:,:,1]
            return xt,xt1,xz

        if ret =='rec':
            '''
            Set to zero to force equilibrium
            '''
            alphas = alphas[-1:]
            xz = torch.normal(0,1,size=(B,G,1,1),device=self.device)
            xt = alphas.sqrt() * x0.unsqueeze(-1) + (1-alphas[None,None]).sqrt() * xz[:,:,:,0]
            return xt

        else:
            assert 0,ret


class AutoEncoderDDPMWithFixedAEAndLabelEmbedding(nn.Module):
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()

        self.beta = config.beta
        assert self.beta < 1.0 and self.beta>0.0,self.beta
        self.n_step = config.n_step
        self.G = config.graph_dim
        # config_copy = copy.copy(config)
        # config_copy.model_name  = config_copy.submodel_name
        self.ae = config.submodel_config.to_model(self.device)
        self.ae2 = config.submodel_config.to_model(self.device)
        x = nn.Linear(config.kernel_size, config.submodel_config.embed_dim)
        self.labels_embedding = nn.Parameter(x.weight.T)

    def get_ab(self,):
        T = self.n_step
        betas = torch.ones((T+1,),device=self.device) * self.beta
        betas[0]=0.
        alphas = (1 - betas).cumprod(0)[:-1]
        return alphas,betas

    def loss(self,item,ret='rec'):
        x0 = images = item['images']
        labels =item['labels']
        # labels
        B = len(x0)
        G = self.G
        #(B,G)
        T = self.n_step
        alphas,betas = self.get_ab()

        xt,xt1,xz = self.encode(item,ret='tup')
        # xtt  = self.encode(x0,ret='rec')
        #(B,G,T,2) parallel sampling of t,t+1 pair
        # import pdb; pdb.set_trace()

        xtv = torch.arange(self.n_step,device=self.device)[None,None,:]
        pred,labels = self._decode((xt1,labels),xtv)

        loss = (pred-xt).square()
        return loss.mean(dim=(1,2))

    grad_loss = loss
    def _decode(self,x,t):
        x,labels = x

        _xshape = x.shape
        B,E,T = x.shape
        lat = self.ae.encode(x.transpose(2,1).reshape((B*T,E)),ret='rec')

        label_embedding = self.labels_embedding[labels,None].repeat((1,T,1)).reshape((B*T,self.config.submodel_config.embed_dim))
        # # label_embedding = 0.
        if isinstance(lat,tuple):
            lat2 = list(lat)[:]
            lat2[0] = lat2[0] + label_embedding #!!
        else:
            lat2 = label_embedding
            # lat = lat + label_embedding
        w = 0.5
        dx  = (1+w)*self.ae.decode(lat2,ret='rec').reshape((B,T,E)).transpose(2,1)
        dx  = dx - w*self.ae.decode(lat,ret='rec').reshape((B,T,E)).transpose(2,1)
        pred = x + dx
        return (pred,labels)

    def decode(self,x,ret='rec'):

        x,labels = x
        if x.shape.__len__()==2:
            x = x.unsqueeze(-1)
        # ones = torch.ones(x.shape,device=self.device).long()
        ones = torch.ones((1,),device=self.device).long()
        x = (x,labels)

        for xi in range(self.n_step):
            x = self._decode(x, xi*ones)

        x,labels = x
        return x[:,:,0]

    def encode(self,x0,ret='rec'):
        labels = x0['labels']
        x0 = x0['images']
        B = len(x0)
        G = self.G
        #(B,G)
        T = self.n_step
        alphas,betas = self.get_ab()

        # xtv = torch.arange(self.n_step,device=self.device)[None,None,:]
        if ret =='tup':
            xz = torch.normal(0,1,size=(B,G)+(T,2),device=self.device)
            xt = alphas.sqrt() * x0.unsqueeze(-1) + (1-alphas[None,None]).sqrt() * xz[:,:,:,0]
            xb1 = betas[None,None,1:]
            xt1 = xt*(1-xb1).sqrt()+ xb1.sqrt() *xz[:,:,:,1]
            return xt,xt1,xz

        if ret =='rec':
            '''
            Set to zero to force equilibrium
            '''
            alphas = alphas[-1:]
            xz = torch.normal(0,1,size=(B,G,1,1),device=self.device)
            xt = alphas.sqrt() * x0.unsqueeze(-1) + (1-alphas[None,None]).sqrt() * xz[:,:,:,0]
            return xt,labels

        else:
            assert 0,ret



class AutoEncoderDDPMSharedWithRelu(AutoEncoderDDPMShared):
    '''
    Decouple layers
    '''

    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__(device,config)

        # self.beta = config.beta
        # assert self.beta < 1.0 and self.beta>0.0,self.beta
        # self.n_step = config.n_step
        # self.G = config.graph_dim
        #
        # self.project = nn.Linear(config.graph_dim*self.n_step,config.embed_dim)
        # self.rec = nn.Linear(config.embed_dim,config.graph_dim*self.n_step)
    # grad_loss = loss
    def _decode(self,x,t):

        pred = x + self.rec(self.project(x.transpose(2,1)).relu()).transpose(2,1)
        # xz = torch.normal(0,1,size=x.shape,device=self.device)
        # pred = pred +xz
        return pred

class AutoEncoderDDPMSliced(AutoEncoderDDPMShared):
    '''
    Decouple layers
    '''

    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__(device,config)

        self.beta = config.beta
        assert self.beta < 1.0 and self.beta>0.0,self.beta
        self.n_step = config.n_step
        self.G = config.graph_dim

        self.project = nn.Linear(config.graph_dim*self.n_step,config.embed_dim)
        self.rec = nn.Linear(config.embed_dim,config.graph_dim*self.n_step)
    # grad_loss = loss


    def _decode(self,x,t):
        # xz = torch.normal(0,1,size=x.shape,device=self.device)
        # pred = pred + 0.1 * xz
        # import pdb; pdb.set_trace()
        t = t.reshape(-1)
        B = len(x)
        G = self.G
        T = self.n_step
        wt = self.project.weight.reshape((G,T,-1))[:,t]
        wr = self.rec.weight.reshape((G,T,-1))[:,t]
        # torch.tensordot()
        wb = self.rec.bias.reshape((1,G,T))[:,:,t]
        xx =(x.unsqueeze(-1)*wt).sum(1)
        xd = (xx[:,None] * wr[None]).sum(-1) + wb
        # import pdb; pdb.set_trace()
        pred = x + xd
        # pred = x + self.rec(self.project(x.transpose(2,1))).transpose(2,1)
        return pred


class AutoEncoder(nn.Module):
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()

        self.project = nn.Linear(config.graph_dim,config.embed_dim)
        self.rec = nn.Linear(config.graph_dim,config.embed_dim)

    def loss(self,item,ret='rec'):
        images = item['images']
        # lat = transform(images)
        # images_recon = lat @ self.rec.weight
        images_recon = self.decode(self.encode(images))
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss
    def encode(self,images,ret='rec'):
        lat = images @ self.project.weight.T
        return lat
    def decode(self,lat,ret='rec'):
        images_recon = lat @ self.project.weight
        return images_recon

class NLAutoEncoder(nn.Module):
    def __init__(self,
        device,
        config, _):
        self.device = device
        self.config = config
        super().__init__()

        self.project = nn.Linear(config.graph_dim,config.embed_dim)
        self.rec = nn.Linear(config.graph_dim,config.embed_dim)
        self.rec2 = nn.Linear(config.embed_dim,config.embed_dim)

    def loss(self,item):
        images = item['images']
        # lat = transform(images)
        lat = images @ self.project.weight.T
        lat = lat.relu()
        lat = lat @ self.rec2.weight
        images_recon = lat @ self.project.weight
        # images_recon = lat @ self.rec.weight
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss





class Gauss2DLayers(nn.Module):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,):
        pass
