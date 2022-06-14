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

class GMMLayerConfig(object):
    pass
from dataclasses import dataclass
@dataclass
class GMMLayerConfig(object):
    depth:int
    graph_dim:int
    embed_dim:int
    iter_per_layer:int
    kernel_size:int
    model_name:str
    beta: float
    n_step: int = 1
    def to_model(self,device):
#        cls = getattr(self.model_name)
        cls = eval(self.model_name)
#        if cls is None:
#            cls = AddModelWithAttentionStacked
        return cls(device,self,None)
#SASConfig = LayerConfig

from sklearn.decomposition import PCA

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

class AutoEncoder(EncoderBase):
    def __init__(self,
        device,
        config, _=None):
        self.device = device
        self.config = config
        super().__init__()

        self.project = nn.Linear(config.graph_dim,config.embed_dim)
        self.rec = nn.Linear(config.graph_dim,config.embed_dim)

    def loss(self,item):
        images = item['images']
        # lat = transform(images)
        # images_recon = lat @ self.rec.weight
        images_recon = self.decode(self.encode(images))
        loss = (images_recon - images)**2
        loss = loss.mean(-1)
        return loss
    def encode(self,images):
        lat = images @ self.project.weight.T
        return lat
    def decode(self,lat):
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
        import pdb; pdb.set_trace()
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
    #
    # def grad_loss(self,item):
    #     images = item['images']
    #     lat = self.encode(images)
    #
    #     #### decode the latent to an ensemble with weight
    #     xp, irec = self.decode(lat)
    #     ### calculate expectation of likelihood
    #     loss = xp * (irec.unsqueeze(0) - images.unsqueeze(1)).square().mean(-1)
    #     loss = loss.sum(-1)
    #     return loss


    def norm_w(self,):
        return self.w / ( 0.01 + self.w.std(dim=1,keepdims=True) )
    def encode(self,images):
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
            # import pdb; pdb.set_trace()

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

        return zmax,kmax
        wimg = self.project.weight.T
        wrec = self.rec.weight.T
        xh = -(images.unsqueeze(-1) - wimg[None])**2
        xh = xh.mean(1)
        xh = xh * beta
        xp = xh.softmax(-1)
        lat  = xp @ wrec
        return lat

    def decode(self,lat,ret='tup'):
        zmax,kmax = lat
        zmax                     #(B,1,E)
        # w  = self.norm_w()[kmax]
        w  = self.w   [kmax]     #(B,G,E)
        mu = self.mu.T[kmax,:,]  #(B,G,1)
        y = (zmax @ w.transpose(2,1))[:,0] + mu
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
