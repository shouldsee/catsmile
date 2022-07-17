'''
Implements mixture autoencoder
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from markov_lm.Model_pretrain import lazy_load_pretrain_model

import markov_lm.Model_betaVAE

#from markov_lm.Model_gmm import
from transformers.models.bert.modeling_bert import BertLayer,BertConfig
import math

# from markov_lm.Model_ddpm import DDPMModelConfig
class GMMLayerConfig(object):
    pass
from dataclasses import dataclass
# import md5sum

class AbstractLayerConfig(object):
    def to_str(self,):
        out = ''
        for k,v in self.__dict__.items():
            k = k.replace('_','-')
            out+= f'-{k}{v}'
        return out

@dataclass
class GMMLayerConfig(AbstractLayerConfig):
    depth:int
    graph_dim:int
    iter_per_layer:int
    kernel_size:int
    model_name:str
    beta: float
    n_step: int = 1
    embed_dim:int = 0
    p_null: float = 0.0001
    submodel_name: str=''
    def to_model(self,device):
#        cls = getattr(self.model_name)
        cls = eval(self.model_name)
#        if cls is None:
#            cls = AddModelWithAttentionStacked
        return cls(device,self,None)
    # def to_str(self,):
    #     out = ''
    #     for k,v in self.__dict__.items():
    #         k = k.replace('_','-')
    #         out+= f'-{k}{v}'
    #     return out

#SASConfig = LayerConfig

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class EncoderBase(nn.Module):
    def grad_loss(self,*a,**kw):
        return self.loss(*a,**kw)
class SKPCA(EncoderBase):
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        self.project = nn.Linear(config.graph_dim,config.embed_dim)
        self.mod = PCA(n_components =config.embed_dim, )
        self.data_list = []
        self.data = None
        embed_dim = config.embed_dim
        self.lin = nn.Linear(embed_dim,embed_dim)

    def loss(self,item):
        images = item['images']
        if item['epoch']==0:
            self.data_list.append(item['images'])
            images_recon = images * 0 *self.lin.weight[0][0]
        else:
            if self.data is None:
                self.data = x = torch.cat(self.data_list,dim=0)
                self.mod = self.mod.fit(x.cpu())
            lat = self.mod.transform(images.cpu())
            images_recon = self.mod.inverse_transform(lat)
            images_recon = torch.tensor(images_recon).to(self.device)
            images_recon = images_recon*(1 + 0 *self.lin.weight[0][0])
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss
class SKKMEANS(EncoderBase):
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        # self.project = nn.Linear(config.graph_dim,config.embed_dim)
        self.mod = KMeans(n_clusters =config.kernel_size, n_init =1,verbose=1)
        self.data_list = []
        self.data = None
        embed_dim = config.embed_dim
        self.lin = nn.Linear(embed_dim,embed_dim)

    def loss(self,item):
        images = item['images']
        if item['epoch']==0:
            self.data_list.append(item['images'])
            images_recon = images * 0 *self.lin.weight[0][0]
        else:
            if self.data is None:
                self.data = x = torch.cat(self.data_list,dim=0)
                self.mod = self.mod.fit(x.cpu())
            lat = self.mod.predict(images.cpu())
            # import pdb; pdb.set_trace()
            images_recon = self.mod.cluster_centers_[lat]
            images_recon = torch.tensor(images_recon).to(self.device)
            images_recon = images_recon*(1 + 0 *self.lin.weight[0][0])
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss
class AutoEncoder(EncoderBase):
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

class BetaVAE(nn.Module):
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        nc = 1
        self.bvae = markov_lm.Model_betaVAE.BetaVAE_H(z_dim=config.embed_dim//2, nc=nc)
        self.G = config.graph_dim
        self.G1 = self.G[-1]

    def loss(self,item,ret='rec'):
        images = item['images']
        # lat = transform(images)
        # images_recon = lat @ self.rec.weight
        images_recon = self.decode(self.encode(images))
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss
    grad_loss = loss

    def forward(self, x):
        assert 0,'For code review only'
        '''
        We can push the stochasticity into the decoder layer
        '''
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)
        return x_recon, mu, logvar


    def encode(self,images,ret='rec'):
        B = len(images)
        im_shape= images.shape

        images = images.reshape((B,)+ self.G)
        lat = self.bvae._encode(images)
        return lat,im_shape

    def decode(self,lat,ret='rec'):
        # lat = distributions
        distributions,im_shape = lat
        mu = distributions[:, :self.bvae.z_dim]
        logvar = distributions[:, self.bvae.z_dim:]

        '''
        no need to set positivity of logvar
        '''
        # z = markov_lm.Model_betaVAE.reparametrize(mu, logvar)
        # std = logvar.div(2).exp()
        std = logvar
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        z =  mu + std*eps


        rec = self.bvae._decode(z)
        G2 = rec.shape[-1]
        dG = (G2-self.G1)//2
        xG = (G2-self.G1)%2
        rec=rec[:,:,(dG+xG):-dG,(dG+xG):-dG]

        rec = rec.reshape(im_shape)

        # import pdb; pdb.set_trace()
        return rec



class BetaVAENoNoise(BetaVAE):
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__(device,config)
        nc = 1
        self.bvae = markov_lm.Model_betaVAE.BetaVAE_H(z_dim=config.embed_dim, nc=nc)
        self.G = config.graph_dim
        self.G1 = self.G[-1]



    def encode(self,images,ret='rec'):
        B = len(images)
        im_shape= images.shape

        images = images.reshape((B,)+ self.G)
        lat = self.bvae._encode(images)[:,:self.bvae.z_dim]
        return lat,im_shape



    def decode(self,lat,ret='rec'):
        # lat = distributions
        mu,im_shape = lat
        # mu = distributions[:, :self.bvae.z_dim]
        z = mu + 0

        rec = self.bvae._decode(z)
        G2 = rec.shape[-1]
        dG = (G2-self.G1)//2
        xG = (G2-self.G1)%2
        rec=rec[:,:,(dG+xG):-dG,(dG+xG):-dG]
        rec = rec.reshape(im_shape)
        return rec

    #
    # def decode(self,lat,ret='rec'):
    #     # lat = distributions
    #     distributions,im_shape = lat
    #     mu = distributions[:, :self.bvae.z_dim]
    #     # logvar = distributions[:, self.bvae.z_dim:]
    #
    #     '''
    #     no need to set positivity of logvar
    #     '''
    #     # z = markov_lm.Model_betaVAE.reparametrize(mu, logvar)
    #     # std = logvar.div(2).exp()
    #     # std = logvar
    #     # eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    #     # z =  mu + std*eps
    #     z = mu + 0
    #     # print(z.shape)
    #
    #     rec = self.bvae._decode(z)
    #     G2 = rec.shape[-1]
    #     dG = (G2-self.G1)//2
    #     xG = (G2-self.G1)%2
    #     rec=rec[:,:,(dG+xG):-dG,(dG+xG):-dG]
    #
    #     rec = rec.reshape(im_shape)
    #
    #     # import pdb; pdb.set_trace()
    #     return rec



class BetaVAEConvLocalNoNoise(BetaVAENoNoise):
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__(device,config)
        nc = 1
        self.bvae = markov_lm.Model_betaVAE.BetaVAEConvLocal(z_dim=config.embed_dim, nc=nc)
        self.G = config.graph_dim
        self.G1 = self.G[-1]


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



import markov_lm.ae_bakeoff.building as aebb
class AutoEncoderBakeOff(nn.Module):

    def __init__(self,device,config,_ = None):
    # def build_ae(model_type, input_shape, anomaly=False):
        super().__init__()
        self.device = device
        self.config = config
        latent_dim  = config.embed_dim
        input_shape = config.graph_dim
        # model_type  = 'vanilla'
        model_type  = 'denoising'
        # latent_dim = 2 if anomaly else 20
        noise_ratio = 0.5 if model_type == 'denoising' else None
        encoder, decoder = aebb._build_networks(model_type, input_shape, latent_dim)
        bottleneck = aebb._build_bottleneck(model_type, latent_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck
        # ae = aebb.lightning.Autoencoder(encoder, bottleneck, decoder, lr=0.001, noise_ratio=noise_ratio)
        # self.ae = ae
        # print(ae)
        # import pdb; pdb.set_trace()

    def loss(self,item,ret='rec'):
        images = item['images']
        # lat = transform(images)
        # images_recon = lat @ self.rec.weight
        images_recon = self.decode(self.encode(images))
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss

    grad_loss = loss


    def encode(self,images,ret='rec'):
        B = len(images)
        lat = self.encoder.forward(images)
        return lat,images.shape

    def decode(self,lat,ret='rec'):
        lat,shape = lat
        lat,_ = self.bottleneck(lat)
        rec =self.decoder.forward(lat).reshape(shape)
        return rec




class GlobalMixtureEncoder(nn.Module):
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()

        self.project = nn.Linear(config.graph_dim,config.kernel_size)
        self.rec = nn.Linear(config.kernel_size,config.embed_dim)
#        self.rec = nn.Linear(config.graph_dim,config.embed_dim)
        # self.beta = config.beta
        self.beta = getattr(config,'beta',0.001)
        # self.beta = 0.001

    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        images_recon = self.decode(lat)
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss
    grad_loss = loss

    def encode(self,images):
        beta = self.beta
        # lat = transform(images)
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        # wrec = wrec - wrec.mean(0,keepdims=True)
        #xh = images @ self.project.weight.T
        xh  = -(images.unsqueeze(-1) - wimg[None])**2
        xh  = xh.mean(1)
        xh  = xh * beta
        xp  = xh.softmax(-1)
        lat = xp @ wrec

        print((100*xp[:5,:170]).int())
        print((100*xp[:5,:170]).cumsum(-1).int())
        print((100*xp[:5,:170]).sum(-1).int())
        # import pdb; pdb.set_trace()
        return lat

    def decode(self,lat,ret='rec'):
        beta = self.beta
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        # wrec = wrec - wrec.mean(0,keepdims=True)

        xh = -(lat.unsqueeze(-1) - wrec.T[None]).square()
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        # print((100*xp[:2,:170]).int())
        # print((100*xp[:2,:170]).cumsum(-1).int())
        # print((100*xp[:2,:170]).sum(-1).int())
        # print((100*xp[:2,100:200]).int())
        images_recon = xp @ wimg.T
        #tempself.project.weight
        return images_recon



class GlobalMixtureEncoderEOL(nn.Module):
    '''
    Change loss function to expectation of probability
    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()

        self.project = nn.Linear(config.graph_dim,config.kernel_size)
        self.rec = nn.Linear(config.kernel_size,config.embed_dim)
#        self.rec = nn.Linear(config.graph_dim,config.embed_dim)
        self.beta = config.beta
        # self.beta = getattr(config,'beta',0.001)
        # self.beta = 0.001


    def loss(self,item):
        images = item['images']
        lat = self.encode(images)

        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec')
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss

    def grad_loss(self,item):
        images = item['images']
        lat = self.encode(images)

        #### decode the latent to an ensemble with weight
        xp, irec = self.decode(lat)
        ### calculate expectation of likelihood
        loss = xp * (irec.unsqueeze(0) - images.unsqueeze(1)).square().mean(-1)
        loss = loss.sum(-1)
        return loss


    def encode(self,images):
        beta = self.beta
        # lat = transform(images)
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        # wrec = wrec - wrec.mean(0,keepdims=True)
        #xh = images @ self.project.weight.T
        xh = -(images.unsqueeze(-1) - wimg[None])**2
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        lat  = xp @ wrec
        return lat

    def decode(self,lat,ret='tup'):
        beta = self.beta
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        # wrec = wrec - wrec.mean(0,keepdims=True)

        xh = -(lat.unsqueeze(-1) - wrec.T[None]).square()
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        #tempself.project.weight
        # return images_recon
        if ret == 'rec':
            images_recon = xp @ wimg.T
            return images_recon
        elif ret=='tup':
            # import pdb; pdb.set_trace()
            return xp, wimg.T
        else:
            assert 0,ret






class LLGAE(nn.Module):
    '''
    Locally Linear Generative Autoencoder
    Hyper parameters make a big difference

        conf.lconf  =lconf= GMMLayerConfig(
            depth = 1,
            iter_per_layer=-1,
            n_step = 5,
            # beta = 1.001,
            beta = 0.01001,
            graph_dim = conf.dataset.graph_dim,
            embed_dim = 20,
            kernel_size = 15,
            model_name = 'LLGAE',
            p_null = 0.,
        )
        conf.learning_rate = 0.001

    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        self.G = G = config.graph_dim
        self.E = E = config.embed_dim
        self.K = K = config.kernel_size

        x = nn.Linear(config.graph_dim,config.kernel_size)
        self.mu = nn.Parameter(x.weight.T)
        x = nn.Linear(config.graph_dim*config.embed_dim*config.kernel_size,1)
        self.w  = nn.Parameter(x.weight.T.reshape((K,G,E)))
        self.beta = config.beta


    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec')
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss

    grad_loss = loss

    def norm_w(self,):
        return self.w / ( 0.01 + self.w.std(dim=1,keepdims=True) )

    def encode(self,images,ret='rec'):
        beta = self.beta
        G = self.config.graph_dim
        # xw = w  = self.norm_w()
        w  = self.w              #(K,G,E)
        mu = self.mu.T.unsqueeze(0)#(1,K,G)
        x = images[:,None]         #(B,1,G)
        B = len(x)
        z = torch.zeros((B,  self.K, self.E),device=self.device)
        for i in range(self.config.n_step):
            '''
            For each possible components, finds the optimal z
            '''
            # (B,K,G)
            dx = (x - mu - (z.unsqueeze(2) * w[None]).sum(-1))
            # (B,K,G,E)
            wdx = dx.unsqueeze(-1) * w.unsqueeze(0)
            z = z + beta * (wdx.sum(2) - z)

        dxsq = -dx.square().mean(-1)
        kmax = dxsq.argmax(-1)
        #(B,)
        # zmax = z[:,kmax]
        # import pdb; pdb.set_trace()
        zmax = torch.gather(z,index=kmax[:,None,None].expand((-1,-1,self.E)),dim=1)
        #### no detach makes it easier to fit the model
        # zmax = zmax.detach()
        return zmax,kmax

        # wimg = self.project.weight.T
        # wrec = self.rec.weight.T
        # xh = -(images.unsqueeze(-1) - wimg[None])**2
        # xh = xh.mean(1)
        # xh = xh * beta
        # xp = xh.softmax(-1)
        # lat  = xp @ wrec
        # return lat

    def decode(self,lat,ret='tup'):
        zmax,kmax = lat
        zmax                     #(B,1,E)
        # w  = self.norm_w()[kmax]
        w  = self.w   [kmax]     #(B,G,E)
        mu = self.mu.T[kmax,:,]  #(B,G,1)
        y = (zmax @ w.transpose(2,1))[:,0] + mu
        return y

class BestCAE(nn.Module):
    '''
    Locally Linear Generative Autoencoder
    Hyper parameters make a big difference

        conf.lconf  =lconf= GMMLayerConfig(
            depth = 1,
            iter_per_layer=-1,
            n_step = 5,
            # beta = 1.001,
            beta = 0.01001,
            graph_dim = conf.dataset.graph_dim,
            embed_dim = 20,
            kernel_size = 15,
            model_name = 'LLGAE',
            p_null = 0.,
        )
        conf.learning_rate = 0.001
    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        self.G = G = config.graph_dim
        self.E = E = config.embed_dim
        self.K = K = config.kernel_size
        self.experts = nn.ModuleList([BetaVAENoNoise(device,config) for k in range(K)])



    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec')
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss

    grad_loss = loss
    def encode(self,images,ret='rec'):
        G = self.config.graph_dim
        # x = images[:,None]         #(B,1,G)
        x = images                 #(B,G)
        B = len(x)
        z = torch.zeros((B,  self.K, self.E),device=self.device)
        logp = torch.zeros((B,  self.K, ),device=self.device)
        for k,model in enumerate(self.experts):
            lat,im_shape = model.encode(x,ret=ret)
            z[:,k:k+1,:] = lat[:,None]
            negdxsq = -(x - model.decode((lat,im_shape)).reshape(x.shape)).square().mean(-1)
            logp[:,k:k+1]=negdxsq[:,None]

        kmax = logp.argmax(-1)
        #(B,)
        # zmax = z[:,kmax]
        # import pdb; pdb.set_trace()
        zmax = torch.gather(z,index=kmax[:,None,None].expand((-1,-1,self.E)),dim=1)
        #### no detach makes it easier to fit the model
        # zmax = zmax.detach()
        return zmax[:,0],kmax,im_shape


    def decode(self,lat,ret='tup'):
        zmax,kmax,im_shape = lat
        y = torch.zeros(im_shape,device=self.device)
        for k,model in enumerate(self.experts):
            flag = kmax == k
            y[flag,:] = model.decode((zmax[flag,], (flag.sum(),)+ im_shape[1:]),ret)
        return y




class LLGAE(nn.Module):
    '''
    Locally Linear Generative Autoencoder
    Hyper parameters make a big difference

        conf.lconf  =lconf= GMMLayerConfig(
            depth = 1,
            iter_per_layer=-1,
            n_step = 5,
            # beta = 1.001,
            beta = 0.01001,
            graph_dim = conf.dataset.graph_dim,
            embed_dim = 20,
            kernel_size = 15,
            model_name = 'LLGAE',
            p_null = 0.,
        )
        conf.learning_rate = 0.001

    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        self.G = G = config.graph_dim
        self.E = E = config.embed_dim
        self.K = K = config.kernel_size

        x = nn.Linear(config.graph_dim,config.kernel_size)
        self.mu = nn.Parameter(x.weight.T)
        x = nn.Linear(config.graph_dim*config.embed_dim*config.kernel_size,1)
        self.w  = nn.Parameter(x.weight.T.reshape((K,G,E)))
        self.beta = config.beta


    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec')
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss

    grad_loss = loss

    def norm_w(self,):
        return self.w / ( 0.01 + self.w.std(dim=1,keepdims=True) )

    def encode(self,images,ret='rec'):
        beta = self.beta
        G = self.config.graph_dim
        # xw = w  = self.norm_w()
        w  = self.w              #(K,G,E)
        mu = self.mu.T.unsqueeze(0)#(1,K,G)
        x = images[:,None]         #(B,1,G)
        B = len(x)
        z = torch.zeros((B,  self.K, self.E),device=self.device)
        for i in range(self.config.n_step):
            '''
            For each possible components, finds the optimal z
            '''
            # (B,K,G)
            dx = (x - mu - (z.unsqueeze(2) * w[None]).sum(-1))
            # (B,K,G,E)
            wdx = dx.unsqueeze(-1) * w.unsqueeze(0)
            z = z + beta * (wdx.sum(2) - z)

        dxsq = -dx.square().mean(-1)
        kmax = dxsq.argmax(-1)
        #(B,)
        # zmax = z[:,kmax]
        # import pdb; pdb.set_trace()
        zmax = torch.gather(z,index=kmax[:,None,None].expand((-1,-1,self.E)),dim=1)
        #### no detach makes it easier to fit the model
        # zmax = zmax.detach()
        return zmax,kmax

        # wimg = self.project.weight.T
        # wrec = self.rec.weight.T
        # xh = -(images.unsqueeze(-1) - wimg[None])**2
        # xh = xh.mean(1)
        # xh = xh * beta
        # xp = xh.softmax(-1)
        # lat  = xp @ wrec
        # return lat

    def decode(self,lat,ret='tup'):
        zmax,kmax = lat
        zmax                     #(B,1,E)
        # w  = self.norm_w()[kmax]
        w  = self.w   [kmax]     #(B,G,E)
        mu = self.mu.T[kmax,:,]  #(B,G,1)
        y = (zmax @ w.transpose(2,1))[:,0] + mu
        return y

class ConvAutoEncoder(nn.Module):
    '''
    Assume a (de)convolutional generative model, that projects hidden
    representation locally. Specifically, we ask the system to find a
    best latent that would recover the image.
    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        self.G = G = config.graph_dim
        self.E = E = config.embed_dim
        self.K = K = config.kernel_size

        self.beta = config.beta
        self.depth = config.depth

        assert config.depth>=1,config.depth
        self.conv = nn.Conv2d(1,E,kernel_size=K,padding='valid')

        if config.depth >=3:
            self.conv2 = nn.Conv2d(E,E,kernel_size=K,padding='same')
            self.conv2b = nn.Conv2d(1,E,kernel_size=K,padding='valid')

        self.deconv = nn.ConvTranspose2d(E,1,kernel_size=K,bias=False)



    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec')
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss

    grad_loss = loss

    def encode(self,images,ret='rec'):
        x = images
        x = x.reshape((len(x),1)+self.G)
        lat = self.conv(x).relu()
        # lat = self.conv(x)
        if self.depth >=3:
            lat = lat.relu() + self.conv2b(x) + self.conv2(lat)

        return lat.reshape((len(x),-1)), lat.shape


    def decode(self,lat,ret='tup'):
        z,shape = lat
        z = z.reshape(shape)
        y = self.deconv(z)
        y = y.reshape((len(y),-1))
        return y



class ConvLocalAutoEncoder(nn.Module):
    '''
    Assume a (de)convolutional generative model, that projects hidden
    representation locally. Specifically, we ask the system to find a
    best latent that would recover the image.
    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        self.G = G = config.graph_dim
        self.E = E = config.embed_dim
        self.K = K = config.kernel_size

        self.beta = config.beta
        self.depth = config.depth

        assert config.depth>=1,config.depth
        self.conv = nn.Conv2d(1,E,kernel_size=K,padding='valid')

        if config.depth >=3:
            self.conv2 = nn.Conv2d(E,E,kernel_size=K,padding='same')
            self.conv2b = nn.Conv2d(1,E,kernel_size=K,padding='valid')

        self.deconv = nn.ConvTranspose2d(E,1,kernel_size=K,bias=False)



    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec')
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss

    grad_loss = loss

    def encode(self,images,ret='rec'):
        x = images
        x = x.reshape((len(x),1)+self.G)
        lat = self.conv(x).relu()
        # lat = self.conv(x)
        if self.depth >=3:
            lat = lat.relu() + self.conv2b(x) + self.conv2(lat)
        # shape = lat.shape
        lat0 = lat
        lat = lat.reshape(lat0.shape[:2] + (-1,))


        mask = torch.zeros((len(lat),lat.shape[-1]),device=self.device)
        p = (0.0001*lat.square().mean(1)).softmax(-1)
        for i in range(5):
            idx = CGAE._rand_sample( p )
            mask = mask + torch.eye(lat.shape[-1],device=self.device)[idx,:]
        mask = mask / (i+1.)

        lat = lat0 * mask[:,None].reshape((len(lat),1,) +lat0.shape[2:])
        # lat = lat
        # _rand_sample()
        return lat


    def decode(self,lat,ret='tup'):
        z = lat
        y = self.deconv(z)
        y = y.reshape((len(y),-1))
        return y



class KMEANS(nn.Module):
    '''
    Use the nearest neighbor to generate a discrete code
    Strictly not a kmeans algorithm
    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        self.G = G = config.graph_dim
        self.E = E = config.embed_dim
        self.K = K = config.kernel_size

        x = nn.Linear(config.graph_dim,config.kernel_size)
        self.mu = nn.Parameter(x.weight.T)
        # x = nn.Linear(config.graph_dim*config.embed_dim*config.kernel_size,1)
        # self.w  = nn.Parameter(x.weight.T.reshape((K,G,E)))
        self.beta = config.beta

    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec')
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss

    # grad_loss = loss

    def grad_loss(self,item):
        images = item['images']
        lat = self.encode(images)
        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec')
        loss = (images_recon - images)**2
        # loss = -(-loss.logsumexp(-1))
        loss = loss.mean(-1)
        # logsumexp(-1))
        #.mean(-1)
        return loss


    def encode(self,images):
        beta = self.beta
        G = self.config.graph_dim
        # xw = w  = self.norm_w()
        # w  = self.w              #(K,G,E)
        mu = self.mu.T.unsqueeze(0)#(1,K,G)
        x = images[:,None]         #(B,1,G)
        B = len(x)
        # z = torch.zeros((B,  self.K, self.E),device=self.device)

        dx = (x - mu)
        dxsq = -dx.square().mean(-1)
        kmax = dxsq.argmax(-1)

        return kmax

    def decode(self,lat,ret='tup'):
        kmax = lat
        mu = self.mu.T[kmax,:,]  #(B,G,1)
        return mu

class RandomKMEANS(nn.Module):
    '''
    Use the nearest neighbor to generate a discrete code
    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        self.G = G = config.graph_dim
        self.E = E = config.embed_dim
        self.K = K = config.kernel_size

        x = nn.Linear(config.graph_dim,config.kernel_size)
        self.mu = nn.Parameter(x.weight.T)
        # x = nn.Linear(config.graph_dim*config.embed_dim*config.kernel_size,1)
        # self.w  = nn.Parameter(x.weight.T.reshape((K,G,E)))
        self.beta = config.beta

    def loss(self,item):
        images = item['images']
        lat = self.encode(images,ret='rec')
        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec')
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss

    def grad_loss(self,item):
        images = item['images']
        logp = self.encode(images,ret='tup')
        loss = -((logp).logsumexp(-1))
        #### decode the latent to an ensemble with weight
        # images_recon = self.decode(lat,ret='rec')
        # loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss

    # grad_loss = loss

    def encode(self,images,ret='tup'):
        beta = self.beta
        G = self.config.graph_dim
        # xw = w  = self.norm_w()
        # w  = self.w              #(K,G,E)
        mu = self.mu.T.unsqueeze(0)#(1,K,G)
        x = images[:,None]         #(B,1,G)
        B = len(x)
        # z = torch.zeros((B,  self.K, self.E),device=self.device)

        dx = (x - mu)
        dxsq = -(dx.square().mean(-1)) * beta
        if ret == 'tup':
            return dxsq

        kmax = dxsq.argmax(-1)
        # kmax = T._rand_sample(dxsq.softmax(-1))

        return kmax



    def decode(self,lat,ret='tup'):
        kmax = lat
        mu = self.mu.T[kmax,:,]  #(B,G,1)
        return mu


class T:
    @staticmethod
    def _rand_sample(p):
      cp = p.cumsum(-1)
      _, idx= (torch.rand(p.shape,device=p.device) <= cp).max(-1)
      # import pdb; pdb.set_trace()
      return idx

class CGAE(nn.Module):
    '''
    Convolutional GAE
    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        self.G = G = config.graph_dim   ### a tuple of (H,W)
        self.E = E = config.embed_dim    #### a tuple of (kH,kW)
        self.K,self.H = K,H = config.kernel_size  ### choose H from K
        self.Gt = Gtotal = math.prod(G)
        self.Et = Etotal = math.prod(E)
        self.T = config.n_step
        # self.T = 1

        # 初始化模型
        # 采集batch
        # self.K = 20
        # self.GR = 20

        x = nn.Linear(Etotal, K)
        self.mu = nn.Parameter(x.weight.T.reshape(E + (K,)))
#        x = nn.Linear(config.graph_dim*config.embed_dim*config.kernel_size,1)
#        self.w  = nn.Parameter(x.weight.T.reshape((K,G,E)))
        self.beta = config.beta


    @staticmethod
    def _rand_sample(p):
        return T._rand_sample(p)
      # cp = p.cumsum(-1)
      # idx,_ = (torch.rand(p.shape,device=p.device) >= cp).max(-1)
      # return idx
    @staticmethod
    def flatten(xs):
        return xs.reshape((xs.size(0),-1))

    @staticmethod
    def expand_with_dirac(xs,kv):
        '''
        Put a image/mask to different location in the image

        xs = (60,28,30)
        ks = (20,21,13)
        xs = torch.rand(xs)
        kv = torch.rand(ks)

        xconv = expand_with_dirac(xs,kv)
        print(xconv.shape)
           XY    K   X   Y
        # [840, 13, 28, 30]
        # B is lost

        #  BW    XY   K
        # [840, 840, 13]

        '''
        # (_,X,Y)=xs.shape
        (X,Y) = xs
        device = kv.device
        K = kv.shape[-1]
        xdirac = torch.eye((X*Y),device=device).reshape((X*Y,X,Y))
        xconv = F.conv2d(xdirac[:,None], kv.transpose(-1,0)[:,None], stride = (1,1),padding='same')
        ### NCHW
        # xconv = xconv.transpose(1,-1).reshape((X*Y,X*Y,-1))
        xconv = xconv.reshape((X*Y*K, X*Y))
        return xconv

    @staticmethod
    def shift_with_component(xs,kv):
        '''
        shift an image with spatial component on k different directions
        '''
        # assert xs.shape.__len__()==2,xs.shape
        xconv = CGAE.expand_with_dirac(xs,kv)
        xs = CGAE.flatten(xs)
        xshift = xs[:,:,None,None] + xconv[None]
        '''
         B    MB  XY   K
        [60, 840, 840, 13]
        '''
        return xshift


    def encode(self, images):
        '''
        Perform gibbs sampling to find the best encoding latent
        return a single sample for each image
        '''
        K = self.K  ## number of candidate components
        G = self.G
        H = self.H
        T = self.T
        Gt = self.Gt
        Et = self.Et
        beta = self.config.beta
        B = len(images)

        x = images.reshape((-1,)+G)

        #### b ijkh tuples
        z = torch.zeros((B,H),device=self.device).long()

        #(HW, HWK)
        wijk = CGAE.expand_with_dirac(self.G, self.mu).T
        x    = CGAE.flatten(x)


        # xrec = (x*0).detach()
        xrec = self.decode(z).detach()
        res = xrec-x
        res = res.reshape((B,Gt))

        for t in range(T):
          for h in range(H):
              zh = z[:,h]
              #### 卷积产生
              #### 残差
              # import pdb; pdb.set_trace()
              resm     = res - wijk.T[zh]
              xshift   = resm[:,:,None]+wijk[None,:,:]
              logp     = -beta * (xshift).square().mean(1)
        #      p = logp.softmax((-1,-2,-3)).reshape((B,I*J*K))
              xp  = logp.softmax(-1)
              idx = CGAE._rand_sample(xp) # (B,1)
              res = (resm +  wijk.T[idx])
              z[:,h] = idx
              # z = torch.scatter(z,index=h,value=idx,dim=1)

        # idx = zh
        return z

    def decode(self,lat,ret='tup'):
        # lat = zh
        wijk = CGAE.expand_with_dirac(self.G, self.mu).T
        z = lat
        rec = wijk.T[z].sum(1)
        # import pdb; pdb.set_trace()
        return rec
        #idx_k = idx % K; idx = idx // K
        #idx_j = idx % J; idx = idx // J
        #idx_i = idx % I; idx = idx // I
        # rec = torch.gather(wijk,index=zh[:,None,:],dim=2).sum(-1)
        #
        # logp = - beta * (x-rec).square().mean(1)
        # loss =  logp.softmax(0).detach()  * logp
        # loss = -loss.mean(0)

    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec')
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss

    def grad_loss(self,item):
        images = item['images']
        x = images
        lat = self.encode(x)
        #### decode the latent to an ensemble with weight
        rec = self.decode(lat,ret='rec')

        logp = - self.beta * (x-rec).square().mean(1)
        loss = len(x) * logp.softmax(0).detach()  * logp
        loss = -loss.mean(-1)
        return loss
        #
        # loss = (images_recon - images)**2
        # loss = loss.mean(-1)
        # return loss


    grad_loss = loss

    #
    # def decode(self,lat,ret='tup'):
    #     zmax,kmax = lat
    #     zmax                     #(B,1,E)
    #     # w  = self.norm_w()[kmax]
    #     w  = self.w   [kmax]     #(B,G,E)
    #     mu = self.mu.T[kmax,:,]  #(B,G,1)
    #     y = (zmax @ w.transpose(2,1))[:,0] + mu
    #     return y
    #



class GibbsGGAE(nn.Module):
    '''
    Convolutional GAE
    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        self.G = G = config.graph_dim   ### a tuple of (H,W)
        self.E = E = config.embed_dim    #### a tuple of (kH,kW)
        self.K,self.H = K,H = config.kernel_size  ### choose H from K
        self.Gt = Gtotal = math.prod(G)
        self.Et = Etotal = math.prod(E)
        self.T = config.n_step
        self.S = 10
        # self.T = 1

        # 初始化模型
        # 采集batch
        # self.K = 20
        # self.GR = 20

        x = nn.Linear(Etotal, K)
        self.mu = nn.Parameter(x.weight.T.reshape(E + (K,)))
        # x = nn.Linear(Etotal, 100)
        # self.mu2 = nn.Parameter(x.weight.T.reshape(E + (100,)))
        # self.mu3 = nn.Linear(100, K)

        # nn.Parameter(x.weight.T.reshape(100,K)
        # E + (100,)))
#        x = nn.Linear(config.graph_dim*config.embed_dim*config.kernel_size,1)
#        self.w  = nn.Parameter(x.weight.T.reshape((K,G,E)))
        self.beta = config.beta


    @staticmethod
    def _rand_sample(p):
      cp = p.cumsum(-1)
      _,idx = (torch.rand(p.shape,device=p.device) >= cp).max(-1)
      return idx
    @staticmethod
    def flatten(xs):
        return xs.reshape((xs.size(0),-1))

    @staticmethod
    def expand_with_dirac(xs,kv):
        '''
        Put a image/mask to different location in the image

        xs = (60,28,30)
        ks = (20,21,13)
        xs = torch.rand(xs)
        kv = torch.rand(ks)

        xconv = expand_with_dirac(xs,kv)
        print(xconv.shape)
           XY    K   X   Y
        # [840, 13, 28, 30]
        # B is lost

        #  BW    XY   K
        # [840, 840, 13]

        '''
        # (_,X,Y)=xs.shape
        (X,Y) = xs
        device = kv.device
        K = kv.shape[-1]
        xdirac = torch.eye((X*Y),device=device).reshape((X*Y,X,Y))
        xconv = F.conv2d(xdirac[:,None], kv.transpose(-1,0)[:,None],padding='same')
        ### NCHW
        # xconv = xconv.transpose(1,-1).reshape((X*Y,X*Y,-1))
        xconv = xconv.reshape((X*Y*K, X*Y))
        return xconv

    @staticmethod
    def shift_with_component(xs,kv):
        '''
        shift an image with spatial component on k different directions
        '''
        # assert xs.shape.__len__()==2,xs.shape
        xconv = CGAE.expand_with_dirac(xs,kv)
        xs = CGAE.flatten(xs)
        xshift = xs[:,:,None,None] + xconv[None]
        '''
         B    MB  XY   K
        [60, 840, 840, 13]
        '''
        return xshift


    def encode(self, images):
        '''
        Perform gibbs sampling to find the best encoding latent
        return a single sample for each image
        '''
        K = self.K  ## number of candidate components
        G = self.G
        H = self.H
        T = self.T
        Gt = self.Gt
        Et = self.Et
        S = self.S
        beta = self.config.beta
        B = len(images)

        x = images.reshape((-1,)+G)

        #### b h tuples
        z = torch.zeros((B*S,H),device=self.device).long()

        #(HW, HWK)
        # wijk = CGAE.expand_with_dirac(self.G, self.mu).T
        x    = CGAE.flatten(x)
        wk   = CGAE.flatten(self.mu.T)
        # wk2   = CGAE.flatten(self.mu2.T)

        # xrec = (x*0).detach()
        xrec = self.decode(z).detach()
        res = xrec-x.repeat((S,1))
        # ((S*B,-1))
        # res = res[None].expand((S,-1,-1,)).reshape((S*B,Gt))
        # res = res.reshape((B,Gt))
        # import pdb; pdb.set_trace()

        for t in range(T):
          for h in range(H):
              zh = z[:,h]
              #### 卷积产生
              #### 残差
              ### wk (K,Gt)
              # import pdb; pdb.set_trace()
              resm = res - wk[zh]
              # xshift   = x[:,:,None]-wijk[None,:,:]

              xshift   = resm[:,:,None] + wk.T[None]
              logp     = -beta * (xshift).square().mean(1)
              # logp = beta* self.mu3(( resm @ wk2.T).relu())
        #      p = logp.softmax((-1,-2,-3)).reshape((B,I*J*K))


              xp  = logp.softmax(-1)


              # idx = CGAE._rand_sample(xp) # (B,1)
              idx = xp.argmax(-1)

              res = (resm  + wk[idx])
              z[:,h] = idx
              # z = torch.scatter(z,index=h,value=idx,dim=1)

        # idx = zh
        return z

    def decode(self,lat,ret='tup'):
        # lat = zh

        wk   = CGAE.flatten(self.mu.T)
        # wijk = CGAE.expand_with_dirac(self.G, self.mu).T
        z = lat
        # import pdb; pdb.set_trace()
        rec = wk[z].sum(1)
        # import pdb; pdb.set_trace()
        return rec
        #idx_k = idx % K; idx = idx // K
        #idx_j = idx % J; idx = idx // J
        #idx_i = idx % I; idx = idx // I
        # rec = torch.gather(wijk,index=zh[:,None,:],dim=2).sum(-1)
        #
        # logp = - beta * (x-rec).square().mean(1)
        # loss =  logp.softmax(0).detach()  * logp
        # loss = -loss.mean(0)

    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec').reshape((len(images),self.S,-1)).mean(1)
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss

    def grad_loss(self,item):
        images = item['images']
        x = images
        lat = self.encode(x)
        #### decode the latent to an ensemble with weight
        rec = self.decode(lat,ret='rec')

        logp = - self.beta * (x-rec).square().mean(1)
        logp = logp.reshape((len(x),self.S,1))
        loss = (logp.softmax(1).detach() * logp).sum(1)

        # loss = len(x) * logp.softmax(0).detach()  * logp
        # loss = (logp.softmax(0).detach()  * logp ) /logp.softmax(0).detach().sum(-1,keepdims=True)
        loss = -loss.mean(-1)
        return loss
        #
        # loss = (images_recon - images)**2
        # loss = loss.mean(-1)
        # return loss


    grad_loss = loss

    #
    # def decode(self,lat,ret='tup'):
    #     zmax,kmax = lat
    #     zmax                     #(B,1,E)
    #     # w  = self.norm_w()[kmax]
    #     w  = self.w   [kmax]     #(B,G,E)
    #     mu = self.mu.T[kmax,:,]  #(B,G,1)
    #     y = (zmax @ w.transpose(2,1))[:,0] + mu
    #     return y
    #



class SOGAE(nn.Module):
    '''
    SecondOrderGenerativeAutoEncoder
    ref: CATSMILE-9015
    '''

    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        self.G = G = config.graph_dim
        self.E = E = config.embed_dim
        self.K = K = config.kernel_size

        x = nn.Linear(G,E*E)
        self.xw = nn.Parameter(x.weight.T.reshape((G,E*E)))
        self.beta = config.beta

        x = nn.Linear(1,E)
        self.z0 = nn.Parameter(x.weight.T.reshape((1,E,1)))
        # .reshape((G,E*E)))



    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec')
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss
    grad_loss = loss



    def encode(self,images):
        beta = self.beta
        G = self.config.graph_dim
        B = len(images)
        E = self.E
        xw = self.xw #(G,E*E)

        x = images[:,None]    #(B,1,G)

        z  = torch.ones((B,  self.E, 1),device=self.device) * self.z0
        y  = torch.zeros((B,1,G),device=self.device)
        for i in range(self.config.n_step):
            dz =  ((x-y)  @ xw).reshape((B,E,E)) @ (z )
            z = z + beta * dz
            dy = (xw @ ((z - self.z0) * beta * dz.transpose(2,1)).reshape((B,E*E,1))).transpose(2,1)
            dy = (xw @ ((z ) * beta * dz.transpose(2,1)).reshape((B,E*E,1))).transpose(2,1)
            y = y + dy
        return z


    def decode(self,lat,ret='tup'):
        beta = self.beta
        G = self.config.graph_dim
        E = self.E
        # B = len(images)
        B = len(lat)
        xw = self.xw #(G,E*E)

        zt  = lat.reshape((B,E,1))
        zt = zt.detach()
        z = torch.ones((B,  self.E, 1),device=self.device)  * self.z0
        y  = torch.zeros((B,1,G),device=self.device)

        dz = (zt-z) / self.config.n_step
        for i in range(self.config.n_step):
            dy = (xw @ ((z ) * beta * dz.transpose(2,1)).reshape((B,E*E,1))).transpose(2,1)
            z = z + dz
            y = y + dy
        return y.reshape((B,G))

class SOGAE2(nn.Module):
    '''
    SecondOrderGenerativeAutoEncoder
    ref: CATSMILE-9015
    '''

    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        self.G = G = config.graph_dim
        self.E = E = config.embed_dim
        self.K = K = config.kernel_size

        x = nn.Linear(G,E*E)
        self.xw = nn.Parameter(x.weight.T.reshape((G,E*E)))
        x = nn.Linear(G,E*E)
        self.xw2 = nn.Parameter(x.weight.T.reshape((G,E*E)))
        self.beta = config.beta

        x = nn.Linear(1,E)
        self.z0 = nn.Parameter(x.weight.T.reshape((1,E,1)))
        # .reshape((G,E*E)))



    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec')
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss
    grad_loss = loss



    def encode(self,images):
        beta = self.beta
        G = self.config.graph_dim
        B = len(images)
        E = self.E
        xw = self.xw #(G,E*E)

        x = images[:,None]    #(B,1,G)

        z  = torch.ones((B,  self.E, 1),device=self.device) #* self.z0
        y  = torch.zeros((B,1,G),device=self.device)
        for i in range(self.config.n_step):
            dz =  ((x)  @ xw).reshape((B,E,E)) @ (z - self.z0)
            z = 0.8 * z + beta*dz
        return z


    def decode(self,lat,ret='tup'):
        beta = self.beta
        G = self.config.graph_dim
        E = self.E
        # B = len(images)
        B = len(lat)
        xw = self.xw2 #(G,E*E)

        zt  = lat.reshape((B,E,1))
        # zt = zt.detach()
        z  = zt
        # z = torch.ones((B,  self.E, 1),device=self.device)  * self.z0
        x  = torch.zeros((B,1,G),device=self.device)

        dz = (zt-z) / self.config.n_step
        for i in range(self.config.n_step):
            dz =  ((x)  @ xw).reshape((B,E,E)) @ (z - self.z0)
            z = z + beta*dz
            dx = (xw @ ((z - self.z0) * beta * dz.transpose(2,1)).reshape((B,E*E,1))).transpose(2,1)
            x = x + dx
            # z = z + dz
            # y = y + dy
        return x.reshape((B,G))


class GLGAE(nn.Module):
    '''
    Gaussian latent Generative Autoencoder
    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        self.G = G = config.graph_dim
        self.E = E = config.embed_dim
        self.K = K = config.kernel_size

        x = nn.Linear(config.graph_dim,config.kernel_size)
        self.xa = nn.Parameter(x.weight.T)
        x = nn.Linear(config.embed_dim,config.kernel_size)
        self.xb = nn.Parameter(x.weight.T)
        self.beta = config.beta
        self.p_null = 0.0001


    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec')
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss
    # grad_loss = loss


    def grad_loss(self,item):
        images = item['images']
        lat = self.encode(images)

        #### decode the latent to an ensemble with weight
        xp, irec = self.decode(lat,ret='tup')
        ### calculate expectation of likelihood
        loss = xp * (irec.unsqueeze(0) - images.unsqueeze(1)).square().mean(-1)
        # import pdb; pdb.set_trace()
        loss = loss.sum(-1)
        return loss

    def encode(self,images):
        beta = self.beta
        G = self.config.graph_dim
        B = len(images)
        # b = self.p
        p_null = self.p_null
        # xw = w  = self.norm_w()
        xa = self.xa #(G,K)
        xb = self.xb #(E,K)
        x = images[:,None]         #(B,1,G)
        # (B,K)
        # lpik = (x - xa.T[None]).square().neg().mean(-1)
        lpik = (x - xa.T[None]).square().neg().mean(-1).log_softmax(-1)

        # pik = (x - xa.T[None]).square().neg().mean(-1)
        z = lpik.softmax(-1) @ xb.T

        # z = torch.zeros((B,  self.E),device=self.device)
        # z = torch.normal(0,1,(B,  self.E),device=self.device)
        for i in range(self.config.n_step):
            lxqik = (z[:,None] - xb.T[None]).square().neg().mean(-1)
            ## (B,E) / (B,1)
            ljp = (lpik + lxqik)
            xh = torch.cat([ljp,ljp[:,0:1]*0+math.log(p_null)],dim=1)
            xp = xh.softmax(-1)[:,:-1]
            # import pdb; pdb.set_trace()
            # (pik + xqik)
            z = xp @ xb.T
            # z = ((pik + xqik) @ xb.T )/ ( p_null + (pik * xqik).sum(-1).unsqueeze(-1))
            '''
            For each possible components, finds the optimal z
            '''
        #     print((ljp).logsumexp(-1).mean().item())
        #     # print((xqik/(p_null+xqik.sum(-1,keepdims=True)))[:3])
        #     # (B,K,G)
        # print('[done]')
        # import pdb; pdb.set_trace()

        return z

    def decode(self,lat,ret='tup'):
        z = lat
        xb =self.xb
        xa = self.xa
        xqik = (z[:,None] - xb.T[None]).square().neg().mean(-1).exp()
        xqik = xqik / (self.p_null + xqik.sum(-1,keepdims=True))
        if ret=='tup':
            return xqik,xa.T
        y = xqik @ xa.T
        return y



class GLGAEGrad(nn.Module):
    '''
    Gaussian latent Generative Autoencoder
    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        self.G = G = config.graph_dim
        self.E = E = config.embed_dim
        self.K = K = config.kernel_size

        x = nn.Linear(config.graph_dim,config.kernel_size)
        self.xa = nn.Parameter(x.weight.T)
        x = nn.Linear(config.embed_dim,config.kernel_size)
        self.xb = nn.Parameter(x.weight.T)
        self.beta = config.beta
        self.p_null = 0.0001


    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec')
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss
    # grad_loss = loss


    def grad_loss(self,item):
        images = item['images']
        lat = self.encode(images)

        #### decode the latent to an ensemble with weight
        xp, irec = self.decode(lat,ret='tup')
        ### calculate expectation of likelihood
        loss = xp * (irec.unsqueeze(0) - images.unsqueeze(1)).square().mean(-1)
        # import pdb; pdb.set_trace()
        loss = loss.sum(-1)
        return loss

    def encode(self,images):
        beta = self.beta
        G = self.config.graph_dim
        B = len(images)
        # b = self.p
        p_null = self.p_null
        # xw = w  = self.norm_w()
        xa = self.xa #(G,K)
        xb = self.xb #(E,K)
        x = images[:,None]         #(B,1,G)
        # (B,K)
        # lpik = (x - xa.T[None]).square().neg().mean(-1)
        pik = (x - xa.T[None]).square().neg().mean(-1).softmax(-1)
        # log_softmax(-1)

        debug = '--debug' in sys.argv
        debug = debug*2
        lpik = (x - xa.T[None]).square().neg().mean(-1)
        z = lpik.softmax(-1) @ xb.T
        # z = torch.zeros((B,  self.E),device=self.device)
        # z = torch.normal(0,1,(B,  self.E),device=self.device)
        for i in range(self.config.n_step):

            # xqik = (z[:,None] - xb.T[None]).square().neg().mean(-1).exp()
            # xqik = xqik / (p_null+xqik.sum(-1,keepdims=True))

            lxqik = (z[:,None] - xb.T[None]).square().neg().mean(-1)
            # lxqik = (z[:,None] - xb.T[None]).square().neg().mean(-1)
            ## (B,E) / (B,1)
            lxqik = torch.cat([lxqik,lxqik[:,0:1]*0+math.log(p_null)],dim=1).log_softmax(-1)


            ljp = (lpik + lxqik[:,:-1])
            # xh = torch.cat([ljp,ljp[:,0:1]*0+math.log(p_null)],dim=1)
            # xp = xh.softmax(-1)[:,:-1]

            xp = ljp.softmax(-1)
            # import pdb; pdb.set_trace()
            # (pik + xqik)
            # z = z +  0.01 * ( xp @ xb.T - z )

            z = z + 0.1 * (xp @ xb.T -z )
            # z =(xp @ xb.T )

            # xp = (pik * xqik) / (pik * xqik).sum(-1,keepdims=True)
            # z = z + 0.1 * (xp @ xb.T -z )

            # z = ((pik * xqik) @ xb.T )/ (0.+ (pik * xqik).sum(-1).unsqueeze(-1))
            '''
            For each possible components, finds the optimal z
            '''
            xx = (x - xa.T[None]).square().neg().mean(-1)
            xx = (xx + (z[:,None] - xb.T[None]).square().neg().mean(-1))
            if debug >= 2:
                print((xx).logsumexp(-1).mean().item())
            # print((ljp).logsumexp(-1).mean().item())
            # print((xqik/(p_null+xqik.sum(-1,keepdims=True)))[:3])
            # (B,K,G)
        if debug >=2 :print('[done]')
        # import pdb; pdb.set_trace()

        return z

    def decode(self,lat,ret='tup'):
        z = lat
        xb =self.xb
        xa = self.xa
        xqik = (z[:,None] - xb.T[None]).square().neg().mean(-1).exp()
        xqik = xqik / (self.p_null + xqik.sum(-1,keepdims=True))
        if ret=='tup':
            return xqik,xa.T
        y = xqik @ xa.T
        return y


class GLGAEGradEOL(nn.Module):
    '''
    Gaussian latent Generative Autoencoder
    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        self.G = G = config.graph_dim
        self.E = E = config.embed_dim
        self.K = K = config.kernel_size

        x = nn.Linear(config.graph_dim,config.kernel_size)
        self.xa = nn.Parameter(x.weight.T)
        x = nn.Linear(config.embed_dim,config.kernel_size)
        self.xb = nn.Parameter(x.weight.T)
        self.beta = config.beta
        # self.p_null = 0.0001
        self.p_null = config.p_null
        # 0.0000


    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec')
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss
    grad_loss = loss
    #
    #
    # def grad_loss(self,item):
    #     images = item['images']
    #     lat = self.encode(images)
    #
    #     #### decode the latent to an ensemble with weight
    #     xp, irec = self.decode(lat,ret='tup')
    #     ### calculate expectation of likelihood
    #     loss = xp * (irec.unsqueeze(0) - images.unsqueeze(1)).square().mean(-1)
    #     # import pdb; pdb.set_trace()
    #     loss = loss.sum(-1)
    #     return loss

    def encode(self,images):
        beta = self.beta
        G = self.config.graph_dim
        B = len(images)
        # b = self.p
        p_null = self.p_null
        # xw = w  = self.norm_w()
        xa = self.xa #(G,K)
        xb = self.xb #(E,K)
        x = images[:,None]         #(B,1,G)
        # (B,K)
        # lpik = (x - xa.T[None]).square().neg().mean(-1)
        pik = (x - xa.T[None]).square().neg().mean(-1).softmax(-1)
        # log_softmax(-1)

        debug = '--debug' in sys.argv
        debug = debug*2
        lpik = (x - xa.T[None]).square().neg().mean(-1)
        z = lpik.softmax(-1) @ xb.T
        # assert 0
        # z = torch.zeros((B,  self.E),device=self.device)
        # z = torch.normal(0,1,(B,  self.E),device=self.device)
        for i in range(self.config.n_step):

            xqik = (z[:,None] - xb.T[None]).square().neg().mean(-1).exp()
            xqik = xqik / (p_null+xqik.sum(-1,keepdims=True))

            # xp = xqik * lpik
            # xp = ljp.softmax(-1)
            # import pdb; pdb.set_trace()
            # (pik + xqik)
            # z = z +  0.01 * ( xp @ xb.T - z )

            # z = z + 0.1 * (xp @ xb.T -z )
            # z =(xp @ xb.T )

            xp = (pik * xqik) / (pik * xqik).sum(-1,keepdims=True)
            z =(xp @ xb.T )
            # z = z + 0.1 * (xp @ xb.T -z )

            # z = ((pik * xqik) @ xb.T )/ (0.+ (pik * xqik).sum(-1).unsqueeze(-1))
            '''
            For each possible components, finds the optimal z
            '''
            xx = (x - xa.T[None]).square().neg().mean(-1)
            xx = (xx + (z[:,None] - xb.T[None]).square().neg().mean(-1))
            if debug >= 2:
                print((xx).logsumexp(-1).mean().item())
            # print((ljp).logsumexp(-1).mean().item())
            # print((xqik/(p_null+xqik.sum(-1,keepdims=True)))[:3])
            # (B,K,G)
        if debug >=2 :print('[done]')
        # import pdb; pdb.set_trace()

        return z

    def decode(self,lat,ret='tup'):
        z = lat
        xb =self.xb
        xa = self.xa
        xqik = (z[:,None] - xb.T[None]).square().neg().mean(-1).exp()
        xqik = xqik / (self.p_null + xqik.sum(-1,keepdims=True))
        if ret=='tup':
            return xqik,xa.T
        y = xqik @ xa.T
        return y


class GaussianGenerativeAutoEncoderLEOP(nn.Module):
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()

        self.project = nn.Linear(config.graph_dim,config.kernel_size)
        self.rec = nn.Linear(config.kernel_size,config.embed_dim)
#        self.rec = nn.Linear(config.graph_dim,config.embed_dim)
        # self.beta = config.beta
        self.beta = getattr(config,'beta',0.001)
        # self.beta = 0.001

    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        images_recon = self.decode(lat)
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss
    grad_loss = loss

    def encode(self,images):
        '''
        Perform Gradient Descent or zero-gradient method to find the optimal code
        '''
        beta = self.beta
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        xh  = -(images.unsqueeze(-1) - wimg[None])**2
        xh  = xh.mean(1)
        xh  = xh * beta
        xp  = xh.softmax(-1)
        '''
        this is p_{bk}, needs to match with latent
        '''

        lat = xp @ wrec
        return lat

    def decode(self,lat,ret='rec'):
        beta = self.beta
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        # wrec = wrec - wrec.mean(0,keepdims=True)

        xh = -(lat.unsqueeze(-1) - wrec.T[None]).square()
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        # print((100*xp[:2,:170]).int())
        # print((100*xp[:2,:170]).cumsum(-1).int())
        # print((100*xp[:2,:170]).sum(-1).int())
        # print((100*xp[:2,100:200]).int())
        images_recon = xp @ wimg.T
        #tempself.project.weight
        return images_recon


class GlobalMixtureEncoderStratifiedEOL(nn.Module):
    '''
    Change loss function to expectation of probability

    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()

        self.project = nn.Linear(config.graph_dim,config.kernel_size)
        self.rec = nn.Linear(config.kernel_size,config.embed_dim//2)
        # self.img2= nn.Linear(config.kernel_size**2,config.graph_dim)
        self.img2= nn.Linear(config.kernel_size*(config.kernel_size+1),config.graph_dim)
        self.rec2= nn.Linear(config.kernel_size+1,config.embed_dim//2)
#        self.rec = nn.Linear(config.graph_dim,config.embed_dim)
        self.beta = config.beta
        self.n_step = config.n_step
        # self.beta = getattr(config,'beta',0.001)
        # self.beta = 0.001


    def loss(self,item):
        images = item['images']
        lat = self.encode(images)

        #### decode the latent to an ensemble with weight
        images_recon = self.decode(lat,ret='rec')
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss

    def grad_loss(self,item):
        images = item['images']
        lat = self.encode(images)

        #### decode the latent to an ensemble with weight
        xp, irec = self.decode(lat)
        ### calculate expectation of likelihood
        loss = xp * (irec.unsqueeze(0) - images.unsqueeze(1)).square().mean(-1)
        # import pdb; pdb.set_trace()
        loss = loss.sum(-1)
        return loss
    # grad_loss = loss

    def encode(self,images):
        beta = self.beta
        E2 = self.config.embed_dim // 2
        K =self.config.kernel_size
        G = self.config.graph_dim
        # lat = transform(images)
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        wrec2 = self.rec2.weight.T
        wimg2= self.img2.weight.T.reshape((K+1,K,G)).T
        # wrec = wrec - wrec.mean(0,keepdims=True)
        #xh = images @ self.project.weight.T

        # xd = images.unsqueeze(-1) - wimg[None]
        # xh = -beta *(xd).square().mean(1)
        # xp = xh.softmax(-1)
        # lat  = xp @ wrec

        xh = -(images.unsqueeze(-1) - wimg[None])**2
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        lat  = xp @ wrec

        # xp =
        if 1:
            xh = -(lat.unsqueeze(-1) - wrec.T[None] ).square()
            xh = xh.mean(1)
            xh = xh * beta
            xp = xh.softmax(-1)

        if 1:
            '''
            Calculate lat2 using the most likely component
            '''
            _,idx=  xp.max(dim=-1)
            # import pdb; pdb.set_trace()
            B = len(images)
            # import pdb; pdb.set_trace()
            xwimg = wimg2[:,idx,:].transpose(1,0)
            # .shape
            # xwimg = torch.gather( wimg2[None].repeat((B,1,1,1)),index=idx[:,None,None,None].repeat((1,G,1,K+1)),dim=2)[:,:,0]
            xd = images.unsqueeze(-1)  - wimg[:,idx].T[:,:,None]- xwimg
            xh2 = -xd.square().mean(1)*beta
            xp2 = xh2.softmax(-1)
            lat2 = xp2 @ wrec2
        else:
            '''
            Calculate lat2 using expected probabiliy
            '''
            xd = images.unsqueeze(-1) - wimg[None]
            xdd = xd.unsqueeze(-1) - wimg2[None]
            xhh = -xdd.square().mean(1) * beta
            xpp = xhh.softmax(-1)

            xpe = xp[:,None] @ xpp
            lat2 = xpe[:,0] @ wrec2

        # print((100*xp[:,10]).int())
        return lat,lat2
        # xp @ xpp
        # # xp3 = (xpp * xp.unsqueeze(-1))
        #
        # import pdb; pdb.set_trace()
        #
        # xp, xrec =self.decode([lat,lat*0],ret='tup')
        # xrec = xp @ xrec
        #
        # xwimg2 = torch.tensordot( xp ,wimg2,1)
        # xh2 = -(images.unsqueeze(-1) - xwimg2.transpose(2,1)).square().mean(1)*beta
        # xp2 = xh2.softmax(-1)
        # lat2 = xp @ wrec2
        # return [lat,lat2]

    def decode(self,lat,ret='tup'):
        lat1,lat2 = lat
        E2 = self.config.embed_dim // 2
        K =self.config.kernel_size
        G = self.config.graph_dim
        beta = self.beta
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        wrec2= self.rec2.weight.T
        wimg2= self.img2.weight.T.reshape((K+1,K,G)).T
        # wrec = wrec - wrec.mean(0,keepdims=True)

        xh = -(lat1.unsqueeze(-1) - wrec.T[None] ).square()
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)

        if self.n_step >=1:
            xh2 = -(lat2.unsqueeze(-1) - wrec2.T[None]).square()
            xh2 = xh2.mean(1)
            xh2 = xh2 * beta
            xp2 = xh2.softmax(-1)
            wimgg = (wimg.unsqueeze(-1)+wimg2).permute((1,2,0))
            # .transpose(0,2)
        else:
            xp2 = torch.ones((xp.shape[0],K+1),device=self.device).float()
            xp2 = xp2*0 + 1./(K+1)
            wimgg = (wimg.unsqueeze(-1)+ 0*wimg2).permute((1,2,0))
        # import pdb; pdb.set_trace()
        if 1:

            xpp = xp.unsqueeze(-1)*xp2.unsqueeze(1)
            if ret == 'rec':

                # import pdb; pdb.set_trace()
                    # xrec =
                # (torch.tensordot(xpp,wimgg,2)*1000)[0,:10].int()
                # (torch.tensordot(xp,wimg.T,1)*1000)[0,:10].int()
                if self.n_step >=-1000:
                    xrec = torch.tensordot(xpp,wimgg,2)
                else:
                    xrec = torch.tensordot(xp,wimg.T,1)
                return xrec

            elif ret=='tup':

                if self.n_step >=-1000:
                    return xpp.reshape((-1,K*(K+1))), wimgg.reshape((K*(K+1),G))
                else:
                    # v = (100*xp[0]).int()
                    # import pdb; pdb.set_trace()
                    return xp,wimg.T
                    # xp2 = xp*0 + 1./K
                    # wimgg = (wimg.unsqueeze(-1)+0*wimg2).transpose(0,2)
                # return xpp,wimgg
                # return xp, wimg.T
                # return xp*xp2, (wimg.T[None]+xwimg2)
                # .transpose(2,1)
            else:
                assert 0,ret


class GlobalMixtureEncoderEOLGrad(GlobalMixtureEncoderEOL):
    '''
    Change loss function to expectation of probability
    '''
    def grad_loss(self,item):
        images = item['images']
        lat = self.encode(images)

        #### decode the latent to an ensemble with weight
        xp, irec = self.decode(lat)
        ### calculate expectation of likelihood
        # import pdb; pdb.set_trace()
        loss = xp * (irec.unsqueeze(0) - images.unsqueeze(1)).square().mean(-1)
        # loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss


    def encode(self,images):
        beta = self.beta
        # lat = transform(images)
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        xh = -(images.unsqueeze(-1) - wimg[None])**2
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        lat  = xp @ wrec
        lat = lat*0
        # wimg
        # lat = lat*0.
        for i in range(10):
            '''
            Iterating in the direction that maximise xh
            aka pull towards the prototype that is closest to the example
            '''
            xp, irec = self.decode(lat,ret='tup')
            xhe = (xp * xh).sum(-1,keepdims=True)
            latg = ((xh -  xhe) * xp).unsqueeze(1) @ (wrec[None] - lat.unsqueeze(1))
            # xh = -(images.unsqueeze(-1) - wimg[None])**2
            # import pdb; pdb.set_trace()
            lat = lat + 0.1 * latg[:,0]
            # loss = xp * (irec.unsqueeze(0) - images.unsqueeze(1)).square().mean(-1)
            # print((100*xp[:10]).int())
        #     print( loss.mean().item())
        # print('done')
        return lat


class GlobalMixtureEncoderLEOP(GlobalMixtureEncoderEOL):
    '''
    Change loss function to expectation of probability
    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__(device,config)

        self.project = nn.Linear(config.graph_dim,config.kernel_size)
        self.rec = nn.Linear(config.kernel_size,config.embed_dim)
#        self.rec = nn.Linear(config.graph_dim,config.embed_dim)
        self.beta = config.beta


    def grad_loss(self,item):
        images = item['images']
        lat = self.encode(images)

        xp, irec = self.decode(lat)
        ### LEOP log-expecation of probability
        ### logsumexp
#        loss =
        loss = (0.001 + xp).log() + (irec.unsqueeze(0) - images.unsqueeze(1)).square().mean(-1)
        # import pdb; pdb.set_trace()
        loss = - (-loss).logsumexp(-1)
        # loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss


    def encode(self,images):
        beta = self.beta
        # lat = transform(images)
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        # wrec = wrec - wrec.mean(0,keepdims=True)
        #xh = images @ self.project.weight.T
        xh = -(images.unsqueeze(-1) - wimg[None])**2
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        lat  = xp @ wrec
        return lat

    def decode(self,lat,ret='tup'):
        beta = self.beta
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        # wrec = wrec - wrec.mean(0,keepdims=True)

        xh = -(lat.unsqueeze(-1) - wrec.T[None]).square()
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        #tempself.project.weight
        # return images_recon
        if ret == 'rec':
            images_recon = xp @ wimg.T
            return images_recon
        elif ret=='tup':
            return xp, wimg.T
        else:
            assert 0,ret

class GlobalMixtureEncoderLEOPMLP(GlobalMixtureEncoderEOL):
    '''
    Change loss function to expectation of probability
    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__(device,config)

        self.project = nn.Linear(config.graph_dim,config.kernel_size)
        self.project2 = nn.Linear(config.graph_dim,config.kernel_size)
        self.project3 = nn.Linear(config.graph_dim,config.embed_dim)
        self.rec = nn.Linear(config.kernel_size,config.embed_dim)
        self.rec2 = nn.Linear(config.kernel_size,config.embed_dim)
        self.rec3 = nn.Linear(config.embed_dim,config.embed_dim)
#        self.rec = nn.Linear(config.graph_dim,config.embed_dim)
        self.beta = config.beta


    def grad_loss(self,item):
        images = item['images']
        lat = self.encode(images)

        xp, irec = self.decode(lat)
        ### LEOP log-expecation of probability
        ### logsumexp
#        loss =
        loss = (0.001 + xp).log() + (irec.unsqueeze(0) - images.unsqueeze(1)).square().mean(-1)
        # import pdb; pdb.set_trace()
        loss = - (-loss).logsumexp(-1)
        # loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss


    def encode(self,images):
        beta = self.beta
        # lat = transform(images)
        wimg = self.project2.weight.T
        wrec = self.rec2.weight.T
        # wrec = wrec - wrec.mean(0,keepdims=True)
        #xh = images @ self.project.weight.T
        # xh = -(images.unsqueeze(-1) - wimg[None])**2
        # xh = xh.mean(1)
        xh = images @ self.project3.weight.T
        # xh = xh * beta
        # xp = xh.softmax(-1)
        # lat  = xh.relu() @ wrec
        lat = xh.relu() @ self.rec3.weight
        return lat

    def decode(self,lat,ret='tup'):
        beta = self.beta
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        # wrec = wrec - wrec.mean(0,keepdims=True)

        xh = -(lat.unsqueeze(-1) - wrec.T[None]).square()
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        #tempself.project.weight
        # return images_recon
        if ret == 'rec':
            images_recon = xp @ wimg.T
            return images_recon
        elif ret=='tup':
            return xp, wimg.T
        else:
            assert 0,ret

class GlobalGradMixtureEncoder(nn.Module):
    '''
    similar to GME, but finds optimal encoding using gradient descent
    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()

        self.project = nn.Linear(config.graph_dim,config.kernel_size)
        self.rec = nn.Linear(config.kernel_size,config.embed_dim)
#        self.rec = nn.Linear(config.graph_dim,config.embed_dim)
        # self.beta = config.beta
        self.beta = getattr(config,'beta',0.001)
        # self.beta = 0.001

    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        images_recon = self.decode(lat)
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss
    def encode_init(self,images):
        beta = self.beta
        # lat = transform(images)
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        # wrec = wrec - wrec.mean(0,keepdims=True)
        #xh = images @ self.project.weight.T
        xh = -(images.unsqueeze(-1) - wimg[None])**2
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        lat  = xp @ wrec
        return lat

    def encode(self,images):
        beta = self.beta

        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        B = images.shape[0]
        lat = self.encode_init(images)
        # lat = torch.zeros((B,self.config.embed_dim),device=self.device)
        for i in range(10):
            rec, xp = self.decode(lat,ret='tup')
            lk = (images - rec)@wimg
            lkd = lk - (lk.unsqueeze(-2)@xp.unsqueeze(-1)).squeeze(-1)
            latd = -(lat.unsqueeze(-1) - wrec.T[None])
            latg = latd @ (xp * lkd).unsqueeze(-1)
            lat = lat+beta**2*0.001*latg.squeeze(-1)
            # lat = lat+beta**2*1*latg.squeeze(-1)
            # lat = lat+0.01*latg.squeeze(-1)
            # lat = lat+0.000001*latg.squeeze(-1)
        #     print((images - rec).square().mean().item())
        # print('done')
        return lat
        # import pdb; pdb.set_trace()

        # wrec = wrec - wrec.mean(0,keepdims=True)
        #xh = images @ self.project.weight.T
        xh = -(images.unsqueeze(-1) - wimg[None])**2
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        lat  = xp @ wrec
        return lat

    def decode(self,lat,ret='rec'):
        beta = self.beta
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        # wrec = wrec - wrec.mean(0,keepdims=True)

        xh = -(lat.unsqueeze(-1) - wrec.T[None]).square()
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        images_recon = xp @ wimg.T
        if ret=='tup':
            return images_recon,xp
        #tempself.project.weight
        return images_recon


class GlobalGradMixtureDiffEncoder(nn.Module):
    '''
    similar to GME, but finds optimal encoding using gradient descent
    '''
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()

        self.project = nn.Linear(config.graph_dim,config.kernel_size)
        self.rec = nn.Linear(config.kernel_size,config.embed_dim//2)
        self.img2 = nn.Linear(config.graph_dim*config.embed_dim//2,config.kernel_size)
#        self.rec = nn.Linear(config.graph_dim,config.embed_dim)
        # self.beta = config.beta
        self.beta = getattr(config,'beta',0.001)
        # self.beta = 0.001

    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        images_recon = self.decode(lat)
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss
    def encode_init(self,images):
        beta = self.beta
        # lat = transform(images)
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        # wrec = wrec - wrec.mean(0,keepdims=True)
        #xh = images @ self.project.weight.T
        xh = -(images.unsqueeze(-1) - wimg[None])**2
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        lat  = xp @ wrec
        return lat

    def encode(self,images):
        beta = self.beta

        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        B = images.shape[0]
        E2  = self.config.embed_dim//2
        wimg2 = self.img2.weight.T
        wimg2 = (wimg2.T.reshape((self.config.kernel_size,-1,E2))).transpose(1,0)
        #.transpose(2,3)
        #.transpose(2,3)

        lat = self.encode_init(images)
        lat2 = torch.zeros((B, E2),device=self.device)
        for i in range(10):
            rec, xp = self.decode([lat,lat2],ret='tup')
            xd = (images - rec)
            lk = (images - rec)@wimg
            lkd = lk - (lk.unsqueeze(-2)@xp.unsqueeze(-1)).squeeze(-1)
            latd = -(lat.unsqueeze(-1) - wrec.T[None])
            latg = latd @ (xp * lkd).unsqueeze(-1)
            lat = lat+beta**2*0.001*latg.squeeze(-1)
            # xd @
            # lat2g =
            # import pdb; pdb.set_trace()
            xd = (images - rec).unsqueeze(-1)
            lat2g = torch.tensordot(xp.unsqueeze(1)*xd, wimg2,2)
            lat2= lat2 + beta**2 *0.1 *lat2g
            # lat = lat+beta**2*1*latg.squeeze(-1)
            # lat = lat+0.01*latg.squeeze(-1)
            # lat = lat+0.000001*latg.squeeze(-1)
        #     print((images - rec).square().mean().item())
        # print('done')
        return lat,lat2
        # import pdb; pdb.set_trace()

        # wrec = wrec - wrec.mean(0,keepdims=True)
        #xh = images @ self.project.weight.T
        xh = -(images.unsqueeze(-1) - wimg[None])**2
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        lat  = xp @ wrec
        return lat


    def decode(self,lat,ret='rec'):
        lat1,lat2 = lat
        beta = self.beta
        E2 = self.config.embed_dim//2
        K = self.config.kernel_size
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        wimg2= self.img2.weight.T
        wimg2 = (wimg2.transpose(1,0).reshape((self.config.kernel_size,-1,E2))[None])*1
        # wimg2 = wimg2 / (0.01+ wimg2.std(dim=2,keepdims=True))

        # wrec = wrec - wrec.mean(0,keepdims=True)

        xh = -(lat1.unsqueeze(-1) - wrec.T[None]).square()
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        xx = xp[:,:,None] * lat2[:,None,:]
        xd = xx.reshape((-1,1,K*E2))@wimg2.transpose(3,2).reshape((1,K*E2,-1))
        images_recon = y= xp @ wimg.T + xd[:,0]
        images_recon = xp @ wimg.T

        if ret=='tup':
            return images_recon,xp

        return images_recon


class GlobalMixtureEncoderDiffEOL(nn.Module):
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()

        self.project = nn.Linear(config.graph_dim,config.kernel_size)
        self.img2 = nn.Linear(config.graph_dim*config.embed_dim//2,config.kernel_size)
        self.rec  = nn.Linear(config.kernel_size,config.embed_dim//2)
        self.rec2 = nn.Linear(config.kernel_size,config.embed_dim//2)
#        self.rec = nn.Linear(config.graph_dim,config.embed_dim)
        self.beta = config.beta
        self.n_step = config.n_step
        # self.beta = getattr(config,'beta',0.001)
        # self.beta = 0.001

    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        images_recon = self.decode(lat)
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss

    def grad_loss(self,item):
        images = item['images']
        lat = self.encode(images)
        xp, xrec  = self.decode(lat,'tup')
        # import pdb; pdb.set_trace()
        '''
        Expectation of reconstruction loss
        '''
        loss = (xrec - images[:,:,None]).square().mean(1)
        loss = (xp * loss).sum(-1)
        # loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss
    # grad_loss = loss

    def encode(self,images):
        E  = self.config.embed_dim
        E2 = self.config.embed_dim//2
        beta = self.beta
        # lat = transform(images)
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        # wrec2= self.rec2.weight.T
        wimg2= self.img2.weight.T
        # wimg2 = wimg2/(0.01+)
        wimg2 = (wimg2.transpose(1,0).reshape((self.config.kernel_size,-1,E2))[None])*1
        # wimg2 = wimg2 / (0.01+ wimg2.std(dim=2,keepdims=True))
        # wrec1,wrec2 = wrec[:,:E//2],wrec[:,E//2:]

        # wrec = wrec - wrec.mean(0,keepdims=True)
        #xh = images @ self.project.weight.T
        xd = images.unsqueeze(-1) - wimg[None]
        xh = -(xd).square().mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        lat1  = xp @ wrec
        xe = xd.transpose(2,1).unsqueeze(2) @ wimg2
        lat2  = xe[:,:,0].transpose(2,1) @ xp.unsqueeze(-1)
        lat2  = lat2.squeeze(-1)
        return [lat1,lat2]

    def decode(self,lat,ret='rec'):
        lat1,lat2 = lat
        beta = self.beta
        E2 = self.config.embed_dim//2
        K = self.config.kernel_size
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        wimg2= self.img2.weight.T
        wimg2 = (wimg2.transpose(1,0).reshape((self.config.kernel_size,-1,E2))[None])*1
        # wimg2 = wimg2 / (0.01+ wimg2.std(dim=2,keepdims=True))

        # wrec = wrec - wrec.mean(0,keepdims=True)

        xh = -(lat1.unsqueeze(-1) - wrec.T[None]).square()
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        G = self.config.graph_dim
        # xw = wimg2.transpose(3,2).reshape((1,K,E2,G)) .permute
        xw = wimg2.reshape((K,G,E2)).transpose(2,0)

        if ret =='tup':
            '''
            expected distance to reconstruction
            '''

            # import pdb; pdb.set_trace()
            xd = torch.tensordot(lat2,xw,1) + wimg[None]
            return xp,xd
        elif ret == 'rec':
            xx = xp[:,:,None] * lat2[:,None,:]
            xd = xx.reshape((-1,1,K*E2))@wimg2.transpose(3,2).reshape((1,K*E2,-1))
            images_recon = y= xp @ wimg.T + xd[:,0]
            # if
            return images_recon
        else:
            assert 0, ret

    def _callback(epoch):
        '''
        Disable gradient in first 10 steps to initialise prototypes
        '''
        n_epoch = 10
        if epoch<n_epoch:
            conf.model.n_step = 0
        elif epoch == n_epoch:
            conf.model.n_step = conf.lconf.n_step
            # conf.model.n_step = 5
            # conf.lconf.n_step
            conf.learning_rate = 0.001
            # conf.learning_rate = 0.1
            conf.optimizer = add_optimizer(conf, conf.params)



import sys
class GlobalMixtureEncoderDiffEOLGrad(GlobalMixtureEncoderDiffEOL):

    def encode(self,images):
        beta = self.beta
        E  = self.config.embed_dim
        E2 = self.config.embed_dim//2
        K = self.config.kernel_size
        G = self.config.graph_dim
        # lat = transform(images)
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        # wrec2= self.rec2.weight.T
        wimg2= self.img2.weight.T
        # wimg2 = wimg2/(0.01+)
        wimg2 = (wimg2.transpose(1,0).reshape((self.config.kernel_size,-1,E2))[None])*1
        # wimg2 = wimg2 / (0.01+ wimg2.std(dim=2,keepdims=True))
        # wrec1,wrec2 = wrec[:,:E//2],wrec[:,E//2:]

        # wrec = wrec - wrec.mean(0,keepdims=True)
        #xh = images @ self.project.weight.T
        xd = images.unsqueeze(-1) - wimg[None]
        xh = -(xd).square().mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        lat1  = xp @ wrec
        xe = xd.transpose(2,1).unsqueeze(2) @ wimg2

        # lat2  = xe[:,:,0].transpose(2,1) @ xp.unsqueeze(-1)
        # lat2  = lat2.squeeze(-1)
        lat2 = lat1*0

        xw = wimg2.reshape((K,G,E2))
        debug = '--debug' in sys.argv
        # .transpose(2,0)
        for i in range(self.n_step):
            xp,xrec = self.decode((lat1,lat2), ret='tup')
            # import pdb; pdb.set_trace()

            # xp =( -beta*( images.unsqueeze(-1) - xrec ).square().mean(1)).softmax(-1)
            # lat1 = xp @ wrec

            # wimg[None]
            xd =  (images.unsqueeze(-1) - xrec ).transpose(2,1)
            lat2g =  xp.unsqueeze(1) @ (xw.unsqueeze(0) * xd.unsqueeze(-1)).sum(-2)

            # import pdb; pdb.set_trace()
            lat2 = lat2 + 0.01*lat2g[:,0]
            if debug>=1:
                loss = (xrec - images[:,:,None]).square().mean(1)
                loss = (xp * loss).sum(-1)
                print(loss.mean().item())
        if debug >=1:
            print((100*xp[:10]).int())
            print('done')
            import matplotlib.pyplot as plt
            xi = 8
            x = wimg[:,xi]
            # x = wimg2[:,xi,:,0]
            x= x.reshape((28,28)).detach().cpu()
            plt.close();plt.imshow(x);plt.savefig(__file__+'.temp.png')
            import pdb; pdb.set_trace()
        return [lat1,lat2]

    def decode(self,lat,ret='rec'):
        lat1,lat2 = lat
        beta = self.beta
        E2 = self.config.embed_dim//2
        K = self.config.kernel_size
        G = self.config.graph_dim

        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        wimg2= self.img2.weight.T
        wimg2 = (wimg2.transpose(1,0).reshape((self.config.kernel_size,-1,E2))[None])*1
        # wimg2 = wimg2 / (0.01+ wimg2.std(dim=2,keepdims=True))
        xw = wimg2.reshape((K,G,E2)).transpose(2,0)

        # wrec = wrec - wrec.mean(0,keepdims=True)
        # import pdb; pdb.set_trace()
        # lat2[:,None,:]
        xh = -(lat1.unsqueeze(-1) - wrec.T[None]).square()
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        # xw = wimg2.transpose(3,2).reshape((1,K,E2,G)) .permute
        if ret =='tup':
            '''
            expected distance to reconstruction
            '''

            # import pdb; pdb.set_trace()
            xd = torch.tensordot(lat2,xw,1) + wimg[None]
            return xp,xd

        elif ret == 'rec':
            xx = xp[:,:,None] * lat2[:,None,:]
            xd = xx.reshape((-1,1,K*E2))@wimg2.transpose(3,2).reshape((1,K*E2,-1))
            images_recon = y= xp @ wimg.T + xd[:,0]
            # if
            return images_recon
        else:
            assert 0, ret



class MixedDiffEncoder(nn.Module):
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()

        self.project = nn.Linear(config.graph_dim,config.kernel_size)
        self.img2 = nn.Linear(config.graph_dim*config.embed_dim//2,config.kernel_size)
        self.rec  = nn.Linear(config.kernel_size,config.embed_dim//2)
        self.rec2 = nn.Linear(config.kernel_size,config.embed_dim//2)
#        self.rec = nn.Linear(config.graph_dim,config.embed_dim)
        self.beta = config.beta
        # self.beta = getattr(config,'beta',0.001)
        # self.beta = 0.001

    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        images_recon = self.decode(lat)
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss
    grad_loss = loss

    def encode(self,images):
        E  = self.config.embed_dim
        E2 = self.config.embed_dim//2
        beta = self.beta
        # lat = transform(images)
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        # wrec2= self.rec2.weight.T
        wimg2= self.img2.weight.T
        # wimg2 = wimg2/(0.01+)
        wimg2 = (wimg2.transpose(1,0).reshape((self.config.kernel_size,-1,E2))[None])*1
        wimg2 = wimg2 / (0.01+ wimg2.std(dim=2,keepdims=True))
        # wrec1,wrec2 = wrec[:,:E//2],wrec[:,E//2:]

        # wrec = wrec - wrec.mean(0,keepdims=True)
        #xh = images @ self.project.weight.T
        xd = images.unsqueeze(-1) - wimg[None]
        xh = -(xd).square().mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        lat1  = xp @ wrec
        xe = xd.transpose(2,1).unsqueeze(2) @ wimg2
        lat2  = xe[:,:,0].transpose(2,1) @ xp.unsqueeze(-1)
        lat2  = lat2.squeeze(-1)
        return [lat1,lat2]

    def decode(self,lat):
        lat1,lat2 = lat
        beta = self.beta
        E2 = self.config.embed_dim//2
        K = self.config.kernel_size
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        wimg2= self.img2.weight.T
        wimg2 = (wimg2.transpose(1,0).reshape((self.config.kernel_size,-1,E2))[None])*1
        wimg2 = wimg2 / (0.01+ wimg2.std(dim=2,keepdims=True))

        # wrec = wrec - wrec.mean(0,keepdims=True)

        xh = -(lat1.unsqueeze(-1) - wrec.T[None]).square()
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        xx = xp[:,:,None] * lat2[:,None,:]
        xd = xx.reshape((-1,1,K*E2))@wimg2.transpose(3,2).reshape((1,K*E2,-1))
        images_recon = y= xp @ wimg.T + xd[:,0]
        # torch.tensordot(wimg2.transpose(3,2).reshape((1,K*E2,-1)),xx.unsqueeze(-1),2)
        #tempself.project.weight
        return images_recon

class Conv2DMixtureV1(nn.Module):
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        self.conv1 = Conv2DMixtureEncoder(device, config =GMMLayerConfig(
        kernel_size=60,embed_dim=20, graph_dim=49,depth=1,iter_per_layer=1,model_name='na',beta=0.001))
        self.conv2 = GlobalMixtureEncoder(device, config =GMMLayerConfig(
        kernel_size=160,embed_dim=20, graph_dim=16*self.conv1.config.embed_dim, depth=1,iter_per_layer=1,model_name='na',beta=0.001))

    def loss(self,item):
        images = item['images']

        lat = self.encode(images)
        images_recon = self.decode(lat)
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss

    def encode(self,images):
        x = self.conv1.encode(images)
        # import pdb; pdb.set_trace()
        x = self.conv2.encode(x.reshape((len(x),-1)))
        # print(x.shape)
        return x

    def decode(self,y,ret='rec'):
        y = self.conv2.decode(y).reshape((len(y),16,self.conv1.config.embed_dim))
        y = self.conv1.decode(y)
        return y


class Conv2DMixtureEncoder(nn.Module):
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()
        # g = config.graph_dim ** 0.5
        # g = int(g)
        # g = 49
        xe = config.embed_dim
        # xe = 10
        self.project = nn.Linear(config.graph_dim, config.kernel_size)
        self.rec = nn.Linear(config.kernel_size, config.embed_dim)
#        self.rec = nn.Linear(config.graph_dim,config.embed_dim)
        self.beta = config.beta
        # self.beta = 0.001

    def loss(self,item):
        images = item['images']
        lat = self.encode(images)
        images_recon = self.decode(lat)
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss

    def encode(self,images):
        beta = self.beta
        # lat = transform(images)
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        # wrec = wrec - wrec.mean(0,keepdims=True)
        B = len(images)
        x = images.reshape((B,28,28))
        x = x.reshape((B,4,7,4,7)).permute((0,1,3,2,4)).reshape((B,16,49))

        # x = x.reshape((L,28,28,25))

        #xh = images @ self.project.weight.T
        xh = -(x.unsqueeze(-1) - wimg[None,None])**2
        xh = xh.mean(-2)
        xh = xh * beta
        xp = xh.softmax(-1)
        lat  = xp @ wrec
        # import pdb; pdb.set_trace()
        return lat

    def decode(self,lat):
        beta = self.beta
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        # wrec = wrec - wrec.mean(0,keepdims=True)

        xh = -(lat.unsqueeze(-1) - wrec.T[None,None]).square()
        xh = xh.mean(-2)
        xh = xh * beta
        xp = xh.softmax(-1)
        y = xp @ wimg.T
        B,L1,L2 = y.shape[:3]

        y = y.reshape((B,4,4,7,7)).permute((0,1,3,2,4))
        xy = y.reshape((B,-1))
        return xy




class Gauss2DLayers(nn.Module):
    def __init__(self,
        device,
        graph_dim,
        embed_dim,):
        pass
