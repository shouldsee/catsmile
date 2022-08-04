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

class LanguageModelPrototype(nn.Module):
    INF = 1E15
    meta = {}
    def log_param(self,buf,plt):
        return

class TranslationModelPrototype(nn.Module):
    meta = {}
    def log_param(self,buf,plt):
        return


from markov_lm.Model_NLP import NLPLayerConfig

@dataclass
class NLMLayerConfig(NLPLayerConfig):
    pass

    def to_model(self,device):
        cls = eval(self.model_name)
        return cls(device,self,None)
import itertools
def get_kperm_tensor(k,device=None,dtype=torch.long):
    return torch.tensor( list(itertools.permutations(range(k))),device=device,dtype=dtype)

class DLMPrototype(LanguageModelPrototype):
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
        # self.loss_name = config.loss_name
        # self.grad_loss_name = config.grad_loss_name
        self.loss_name = 'KLD'
        self.grad_loss_name = 'KLD'
        '''
        window_size should be attribute of dataset, not model
        '''
        self.window_size = window_size = config.window_size
        assert config.window_size>=1, config

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)
        self.layers      = nn.ModuleList([
            nn.Linear(embed_dim*window_size, embed_dim*window_size).to(self.device)
            for _ in range(self.config.depth-1)
            ])
        self.final_layer      = nn.Linear(embed_dim*window_size,  embed_dim).to(self.device)


    def unembed(self,x):
        y = x.matmul(self.embed.weight.T)
        return y

    def grad_loss(self,item):
        return self._loss(item,self.grad_loss_name)

    def forward(self,item):
        return self._loss(item,'forward')

    def loss(self,item):
        return self._loss(item, 'loss')

    def _loss(self,item, ret):
        CLS_NAME = self.__class__.__name__

        source = item['source'] ### token sequence
        target = item['target'] ### token seq

        source_embed = self.embed(source)

        target_embed = self.embed(target)
        # S = source.size(1)
        # source = None; source_embed=None
        T = target.size(1)
        B = source.size(0)
        K = self.config.kernel_size
        target_len = item['target_len']
        target_notnull = torch.arange(target.size(1),device=self.device)[None,:]<target_len[:,None]
        # import pdb; pdb.set_trace()
        if item['has_start_token']==1:
            target_notnull[:,0] = False

        source_len = item['source_len']
        source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]

        WS = self.window_size
        logp_sum = 0.
        E = self.embed_dim
        target_embed = self.embed(target)
        B = len(target)
        i = 0

        att = torch.ones((B,1,1),device=self.device)
        # (P,W)
        if CLS_NAME  in  'DLM2 DLM5'.split():
            '# Calc loss with mixture model directrly'
            '(B,T,E)'
            yp = self.unembed( self.final_layer( target_embed ) ).log_softmax(-1)
            logp = torch.gather( yp,index=target[:,None,:].repeat((1,yp.size(1),1)),dim=2)

            prior = 0.
            if CLS_NAME == 'DLM5':
                prior = prior + self.shared_log_align[None]
            elif CLS_NAME =='DLM2':
                pass
            else:
                raise NotImplementedError(CLS_NAME)
            xt = torch.arange(T,device=self.device)
            prior = prior + -self.INF * (xt[None,:,None] >= xt[None,None,:])
            prior = prior.log_softmax(dim=1)
            logp = (logp + prior)
            att  = logp.softmax(dim=1)
            logp = logp.logsumexp(dim=1)
            logp_sum = (logp * target_notnull).sum(dim=1)

        elif CLS_NAME=='DLM7':
            ### padding one
            target_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            yp = self.unembed(self.final_layer(target_embed_parent)).log_softmax(-1)
            model_w = torch.gather(yp, index=target[:,:,None],dim=-1)[:,:]
            model_q =  torch.gather(yp[:,0], index=target[:,:],dim=-1)[:,:,None]
            # model_w[:,0:1,:].repeat((1,T,1))
            prior = self.shared_log_align.log_softmax(-1)[None]
            lp = torch.cat([model_q,model_w],dim=-1) + prior
            # import pdb; pdb.set_trace()
            att = lp.softmax(dim=2).transpose(2,1)
            lp = lp.logsumexp(-1)
            logp_sum = (lp * target_notnull).sum(dim=1)

        elif CLS_NAME=='DLM8':
            ### padding one
            target_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            yp = self.unembed(self.final_layer(target_embed_parent).reshape((B,T,K,E))).log_softmax(-1)
            model_w =  torch.gather(yp, index=target[:,:,None,None].repeat((1,1,K,1)),dim=-1).squeeze(-1)
            model_q =  torch.gather(yp[:,0,0], index=target[:,:],dim=-1)[:,:,None]
            # model_w[:,0:1,:].repeat((1,T,1))
            prior = self.shared_log_align.log_softmax(-1)[None]
            lp = torch.cat([model_q,model_w],dim=-1) + prior
            # import pdb; pdb.set_trace()
            ## (B, T, K+1)
            att = lp.softmax(dim=2).transpose(2,1)
            lp = lp.logsumexp(-1)
            logp_sum = (lp * target_notnull).sum(dim=1)

        elif CLS_NAME=='DLM9':
            ### padding one
            target_embed_parent = torch.ones((K,E),device=self.device)
            ## (K, C)
            yp = self.unembed(target_embed_parent).log_softmax(-1)
            ## (B,T, K)
            model_w = yp.T[target]
            ## (1, 1, K)
            prior = self.shared_log_align.log_softmax(-1)[None]
            lp = model_w + prior

            att = lp.softmax(dim=2).transpose(2,1)
            lp  = lp.logsumexp(-1)
            logp_sum = (lp * target_notnull).sum(dim=1)

        elif CLS_NAME=='DLM10':
            ### padding one
            ## (K, E) -> (K, C)
            yp = self.unembed( self.k_vector).log_softmax(-1)
            ## (B, T, K)
            model_w = yp.T[target]
            ## (B, T, E)
            target_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            ## (B, T, K)
            cond = self.embed_to_logp( target_embed_parent ).log_softmax(-1)

            lp = model_w + cond

            att = lp.softmax(dim=2).transpose(2,1)
            lp  = lp.logsumexp(-1)
            logp_sum = (lp * target_notnull).sum(dim=1)

        elif CLS_NAME == 'DLM11':
            target_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            h0 = torch.ones([1,B,E],device=self.device)

            # (B, T, E)
            yp,h1 = self.rnn(target_embed_parent,h0)

            # (B, T, C)
            lyp = self.unembed(yp).log_softmax(-1)
            lp =  torch.gather(lyp [:,:,:], index=target[:,:,None],dim=-1)[:,:,:]

            att = lp.softmax(dim=2).transpose(2,1)
            lp  = lp.logsumexp(-1)
            logp_sum = (lp * target_notnull).sum(dim=1)

        elif CLS_NAME == 'DLM12':
            target_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            h0 = torch.ones([1,B,E],device=self.device)
            c0 = torch.ones([1,B,E],device=self.device)

            # (B, T, E)
            yp,(h1,c1) = self.rnn(target_embed_parent,(h0,c0))

            # (B, T, C)
            lyp = self.unembed(yp).log_softmax(-1)
            lp =  torch.gather(lyp [:,:,:], index=target[:,:,None],dim=-1)[:,:,:]

            att = lp.softmax(dim=2).transpose(2,1)
            lp  = lp.logsumexp(-1)
            logp_sum = (lp * target_notnull).sum(dim=1)


        elif CLS_NAME == 'DLM13':
            ### P(z_{i+1}|s(a_{i+1}))
            ### (B, T, K)
            z_by_sa = self.embed_to_logp(target_embed).log_softmax(-1)
            ### (K, E)
            yp      = self.unembed( self.k_vector).log_softmax(-1)
            ## (K,C) (C,K) (B, T, K) -> (B, K, T)
            yp      = yp.T[target].transpose(2,1)

            ## (B, T, T)
            btt = ( z_by_sa.unsqueeze(3) + yp.unsqueeze(1) ).logsumexp(2)

            # ### (B, T, T)
            xt = torch.arange(T,device=self.device)
            prior = self.shared_log_align.log_softmax(-1)[None]
            prior = prior + -self.INF * (xt[None,:,None] >= xt[None,None,:])
            a_prior =  prior.log_softmax(dim=1)

            lp = btt + a_prior


            att = lp.softmax(dim=1)
            # .transpose(2,1)
            lp  = lp.logsumexp(1)
            logp_sum = (lp * target_notnull).sum(dim=1)
        elif CLS_NAME == 'DLM14':
            '''
            Needs E,K to 2E,K

            T,1,E
            1,T,E
            T,T,2E
            B,T,T,K
            B,1,1,K,T
            B,T,T,T
            1,T,T,T
            B,1,1,T

            20*20*20
            '''
            ### P(z_{i+1}|s(a_{i+1}))
            ### (B, T, T, K)
            z_by_sa = self.embed_to_logp(target_embed)[:,:,None,:] + self.embed_to_logp2(target_embed)[:,None,:,:]
            z_by_sa = z_by_sa.log_softmax(-1)

            ### (K, E)
            yp      = self.unembed( self.k_vector).log_softmax(-1)
            ## (K,C) (C,K) (B, T, K) -> (B, K, T)
            yp      = yp.T[target].transpose(2,1)

            ## (B, T, T, T)
            btt = ( z_by_sa.unsqueeze(4) + yp[:,None,None] ).logsumexp(dim=3)

            # ### (1, T, T, T)
            xt = torch.arange(T,device=self.device)
            prior   = self.shared_log_align.log_softmax(-1)[None,:,:,:]
            prior   = prior + -self.INF * ( (xt[:,None,None] >= xt[None,None,:]) | (xt[None,:,None] >= xt[None,None,:]))[None]
            a_prior =  prior.reshape((T*T,T)).log_softmax(dim=0).reshape((1,T,T,T))

            lp = btt + a_prior


            # att = lp.softmax(dim=1)
            att = lp.logsumexp(1).softmax(dim=1)
            # .transpose(2,1)
            lp  = lp.logsumexp(dim=(1,2))
            logp_sum = (lp * target_notnull).sum(dim=1)


        elif CLS_NAME == 'DLM16':
            '''
            Use last two words to compute conditional
            '''
            yp = self.unembed( self.k_vector).log_softmax(-1)
            ## (B, T, K)
            model_w = yp.T[target]

            # target_embed_parent_1 = torch.cat([torch.ones((B,2,E),device=self.device),target_embed],dim=1)[:,:-2]
            target_embed_parent_2 = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            '1 -> 2 -> 3'
            h0 = torch.ones([1,1,K,1],device=self.device).log_softmax(2)
            t1 = self.embed_to_logp(target_embed_parent_2).reshape((B,T,K,K))

            ## (B, T, K)
            h1 = (h0 + t1).logsumexp(2).log_softmax(-1)


            lp = model_w + h1

            att = lp.softmax(dim=2).transpose(2,1)
            lp  = lp.logsumexp(-1)
            logp_sum = (lp * target_notnull).sum(dim=1)

        elif CLS_NAME == 'DLM17':
            '''
            Use last two words to compute conditional
            '''
            yp = self.unembed( self.k_vector).log_softmax(-1)
            ## (B, T, K)
            model_w = yp.T[target]

            target_embed_parent_1 = torch.cat([torch.ones((B,2,E),device=self.device),target_embed],dim=1)[:,:-2]
            target_embed_parent_2 = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            '1 -> 2 -> 3'
            h0 = torch.ones([1,1,K],device=self.device).log_softmax(2).unsqueeze(-1)
            t1 = self.embed_to_logp(target_embed_parent_1).reshape((B,T,K,K))

            ## (B, T, K)
            h1 = (h0 + t1).logsumexp(2).log_softmax(-1).unsqueeze(-1)

            t2 = self.embed_to_logp(target_embed_parent_2).reshape((B,T,K,K))
            h3 = (h1 + t2).logsumexp(2).log_softmax(-1)


            lp = model_w + h3

            att = lp.softmax(dim=2).transpose(2,1)
            lp  = lp.logsumexp(-1)
            logp_sum = (lp * target_notnull).sum(dim=1)

        elif CLS_NAME=='DLM18':
            ### padding one
            ## (K, E) -> (K, C)
            yp = self.unembed(self.k_vector).log_softmax(-1)
            ## (B, T, K)
            model_q = yp.T[target]
            ## (B, T, E)
            target_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]

            ## (B, T, C)
            model_w = self.unembed(self.final_layer(target_embed_parent)).log_softmax(-1)
            model_w = torch.gather(model_w, index= target[:,:,None],dim=-1)

            # a_prior =  torch.ones((1,1,K+1),device=self.device).log_softmax(-1)
            a_prior = self.embed_to_logp(target_embed_parent).log_softmax(-1)
             # torch.ones((1,1,K+1),device=self.device).log_softmax(-1)

            ## (B, T, K)
            # cond = self.embed_to_logp( target_embed_parent ).log_softmax(-1)

            lp = torch.cat([model_w , model_q ],dim=-1) + a_prior
            # lp = model_w + cond

            att = lp.softmax(dim=2).transpose(2,1)
            lp  = lp.logsumexp(-1)
            logp_sum = (lp * target_notnull).sum(dim=1)

        elif CLS_NAME in 'DLM1 DLM3 DLM4'.split():
            kperm = get_kperm_tensor(WS,device=self.device)
            P = len(kperm)
            att = torch.zeros((B,T,T),device=self.device)
            for i in range(T):
                ### (B,W,E)
                xx = target_embed[:,i-WS-1:i-1]
                if xx.size(1)<WS:
                    xx = torch.cat([torch.ones((B,WS-xx.size(1),E),device=self.device),xx,],dim=1)

                '''Packing'''

                if CLS_NAME  == 'DLM3':
                    xx = xx[:,kperm].reshape((B,P,-1))
                    xx = self.final_layer(xx)
                    prior = torch.ones((B,P,1),device=self.device)
                elif CLS_NAME == 'DLM1':
                    ### (B,1)
                    xx = xx.reshape((B, 1,-1))
                    xx = self.final_layer(xx)
                    prior = torch.ones((B,1,1),device=self.device)
                elif CLS_NAME == 'DLM4':
                    xx = xx.reshape((B, 1,-1))
                    xx = self.final_layer(xx).reshape((B,K,E))
                    prior = torch.ones((B,K,1),device=self.device)
                else:
                    raise NotImplementedError(f'Packing for {CLS_NAME}')
                prior = prior.log_softmax(dim=1)
                yp = self.unembed(xx).log_softmax(-1)

                '''
                Unpacking
                '''
                y = target[:,i]
                logp = torch.gather(yp,index=y[:,None,None].repeat((1,yp.size(1),1)),dim=2).squeeze(-1)
                logp = prior.squeeze(-1)+logp
                logp = logp.logsumexp(1)


                logp = logp * target_notnull[:,i]  ## zero out null pos
                logp_sum += logp
        else:
            raise NotImplementedError(CLS_NAME)

        ### normalise by average token count, but sum by batchsize
        # lossVal = lossVal/ (source_notnull.sum() / B)
        logp_per_token = logp_sum/ (target_notnull.sum(dim=1))
        #* 1000
        loss = -logp_per_token

        if ret=='forward':
            return loss,att
        return loss

class DLM1(DLMPrototype):
    pass

class DLM3(DLMPrototype):
    pass

class DLM2(DLMPrototype):
    pass
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        self.final_layer      = nn.Linear(config.embed_dim,  config.embed_dim).to(self.device)

class DLM5(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        self.final_layer      = nn.Linear(config.embed_dim,  config.embed_dim).to(self.device)

        S = self.config.n_step
        x = nn.Linear(S,S)
        self.shared_log_align = nn.Parameter(x.weight)

    def log_param(self,buf,plt):
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        key = 'self.shared_log_align '
        mat = self.shared_log_align.log_softmax(0).cpu().detach()

        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat.T,vmin=-4,vmax=0.,origin='lower')
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        ax.set_title(f'[log_param]{key}\nModel_name={self.config.model_name}\nEpoch:{epoch}')
        # ax.set_xlabel('$a_i$')
        # ax.set_ylabel('$a_{i+1}$')
        plt.suptitle('')
        buf.write(write_png_tag(fig))
        pass

class DLM4(DLMPrototype):
    pass
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        self.final_layer      = nn.Linear(config.embed_dim* config.window_size,  config.kernel_size*config.embed_dim).to(self.device)


class DLM7(DLM5):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        self.final_layer      = nn.Linear(config.embed_dim,  config.embed_dim).to(self.device)

        S = self.config.n_step
        x = nn.Linear(1,2)
        self.shared_log_align = nn.Parameter(x.weight.T)

class DLM8(DLM5):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)
        self.final_layer      = nn.Linear(config.embed_dim,  K*config.embed_dim).to(self.device)

        # S = self.config.n_step
        x = nn.Linear(1,K+1)
        self.shared_log_align = nn.Parameter(x.weight.T)

    def log_param(self,buf,plt):
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        key = 'self.shared_log_align '
        mat = self.shared_log_align.log_softmax(1).cpu().detach()

        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat.T,vmin=-4,vmax=0.,origin='lower')
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        ax.set_title(f'[log_param]{key}\nModel_name={self.config.model_name}\nEpoch:{epoch}')
        # ax.set_xlabel('$a_i$')
        # ax.set_ylabel('$a_{i+1}$')
        plt.suptitle('')
        buf.write(write_png_tag(fig))
        pass

class DLM9(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)
        # self.final_layer      = nn.Linear(config.embed_dim,  K*config.embed_dim).to(self.device)

        x = nn.Linear(1,K)
        self.shared_log_align = nn.Parameter(x.weight.T)

    def log_param(self,buf,plt):
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        key = 'self.shared_log_align '
        mat = self.shared_log_align.log_softmax(1).cpu().detach()

        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat.T,vmin=-4,vmax=0.,origin='lower')
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        ax.set_title(f'[log_param]{key}\nModel_name={self.config.model_name}\nEpoch:{epoch}')
        # ax.set_xlabel('$a_i$')
        # ax.set_ylabel('$a_{i+1}$')
        plt.suptitle('')
        buf.write(write_png_tag(fig))
        pass

class DLM10(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        # x = nn.Linear(1,K)
        # self.shared_log_align = nn.Parameter(x.weight.T)
        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp = nn.Linear(config.embed_dim, K+1).to(self.device)
        # self.embed_to_logp = nn.Parameter(x.weight.T)

    def log_param(self,buf,plt):
        return
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        key = 'self.shared_log_align '
        mat = self.shared_log_align.log_softmax(1).cpu().detach()

        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat.T,vmin=-4,vmax=0.,origin='lower')
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        ax.set_title(f'[log_param]{key}\nModel_name={self.config.model_name}\nEpoch:{epoch}')
        # ax.set_xlabel('$a_i$')
        # ax.set_ylabel('$a_{i+1}$')
        plt.suptitle('')
        buf.write(write_png_tag(fig))
        pass



class DLM16(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        # x = nn.Linear(1,K)
        # self.shared_log_align = nn.Parameter(x.weight.T)
        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp = nn.Linear(config.embed_dim, K*K).to(self.device)
        # self.embed_to_logp = nn.Parameter(x.weight.T)

    def log_param(self,buf,plt):
        return
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        key = 'self.shared_log_align '
        mat = self.shared_log_align.log_softmax(1).cpu().detach()

        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat.T,vmin=-4,vmax=0.,origin='lower')
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        ax.set_title(f'[log_param]{key}\nModel_name={self.config.model_name}\nEpoch:{epoch}')
        # ax.set_xlabel('$a_i$')
        # ax.set_ylabel('$a_{i+1}$')
        plt.suptitle('')
        buf.write(write_png_tag(fig))


class DLM17(DLM16):
    pass


class DLM18(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        # x = nn.Linear(1,K)
        # self.shared_log_align = nn.Parameter(x.weight.T)
        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector        =  nn.Parameter(x.weight.T)
        self.final_layer      = nn.Linear(config.embed_dim,  1*config.embed_dim).to(self.device)
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp = nn.Linear(config.embed_dim, K+1).to(self.device)
        # self.embed_to_logp = nn.Parameter(x.weight.T)

    def log_param(self,buf,plt):
        return
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        key = 'self.shared_log_align '
        mat = self.shared_log_align.log_softmax(1).cpu().detach()

        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat.T,vmin=-4,vmax=0.,origin='lower')
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        ax.set_title(f'[log_param]{key}\nModel_name={self.config.model_name}\nEpoch:{epoch}')
        # ax.set_xlabel('$a_i$')
        # ax.set_ylabel('$a_{i+1}$')
        plt.suptitle('')
        buf.write(write_png_tag(fig))


class DLM11(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        self.rnn = nn.RNN(input_size = E,hidden_size=E, batch_first= True, num_layers=1, nonlinearity='tanh')
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)



class DLM12(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        self.rnn = nn.LSTM(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)


class DLM13(DLM5):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        T = config.n_step

        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        self.embed_to_logp = nn.Linear(config.embed_dim, K).to(self.device)
        x = nn.Linear(T,  T).to(self.device)


        self.shared_log_align = nn.Parameter(x.weight)



    def log_param(self,buf,plt):
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        key = 'self.shared_log_align   $\log P(a_i=b)$'

        T = self.config.n_step
        xt = torch.arange(T,device=self.device)
        prior =  self.shared_log_align
        prior = prior + -self.INF * (xt[None,:,None] >= xt[None,None,:])[0]

        mat = prior.log_softmax(0).cpu().detach()

        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat.T,vmin=-4,vmax=0.,origin='lower')
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        loss = self.meta['test_losses'][-1]
        ax.set_title(f'[log_param]{key}\nModel_name={self.config.model_name}\nEpoch:{epoch}, TestLoss:{self.meta["test_losses"][-1]:.4f}')
        ax.set_xlabel('$b$')
        ax.set_ylabel('$i$')
        plt.suptitle('')
        buf.write(write_png_tag(fig))
        pass


class DLM14(DLM5):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        T = config.n_step

        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        self.embed_to_logp = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2 = nn.Linear(config.embed_dim, K).to(self.device)


        x = nn.Linear(T, T*T).to(self.device)
        self.shared_log_align = nn.Parameter(x.weight.reshape((T,T,T)))



    def log_param(self,buf,plt):
        # bigmat = prior.log_softmax(0).cpu().detach()
        T = self.config.n_step
        xt = torch.arange(T,device=self.device)
        prior   = self.shared_log_align.log_softmax(-1)[None,:,:,:]
        prior   = prior + -self.INF * ( (xt[:,None,None] >= xt[None,None,:]) | (xt[None,:,None] >= xt[None,None,:]))[None]
        a_prior =  prior.reshape((T*T,T)).log_softmax(dim=0).reshape((T,T,T)).cpu().detach()
        for i in range(10):
            mat = a_prior[:,:,i]


            fig,ax = plt.subplots(1,1,figsize=[10,10])
            key = 'self.shared_log_align   $\log P(a_i=b)$'



            # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
            im = ax.imshow(mat.T,vmin=-4,vmax=0.,origin='lower')
            # im = ax.imshow(mat,)
            plt.sca(ax)
            plt.colorbar(im)
            epoch = self.meta['epoch']
            loss = self.meta['test_losses'][-1]
            ax.set_title(f'[log_param]{key}\nModel_name={self.config.model_name}\nEpoch:{epoch}, TestLoss:{self.meta["test_losses"][-1]:.4f}')
            ax.set_xlabel('$b$')
            ax.set_ylabel('$i$')
            plt.suptitle('')
            buf.write(write_png_tag(fig))
        pass


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



SAM1 = SoftAlignmentModelSimpleMean
SAM2 = SoftAlignmentModelAllowSourcePad
SAM3 = SoftAlignmentModel
SAM4 = GaussianSoftAlignmentModel
SAM5 = SharedSoftAlignmentModel


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
