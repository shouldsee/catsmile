import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from dataclasses import dataclass
from markov_lm.Model_gmm import AbstractLayerConfig
# from transformers.models.bert.modeling_bert import BertLayer,BertConfig
from markov_lm.nlp.model_seq2seq import Seq2SeqWithAttention
from markov_lm.Model_rnn import GRUMinimal,RNNConfig


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
    window_size:int =-1

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
    @staticmethod
    def std_norm(v, dim,keepdims=True):
        v = v/(1E-10+v.std(dim=dim,keepdims=keepdims))
        return v

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
        # self.grad_loss_name = 'KLD'
        '''
        window_size should be attribute of dataset, not model
        '''
        self.window_size = window_size = config.window_size
        # assert config.window_size>=1, config

        #### share embed usually works
        # self.vocab      = nn.Linear(embed_dim,graph_dim,).to(self.device)
        self.embed      = nn.Embedding(graph_dim,embed_dim,).to(self.device)

    def unembed(self,x):
        y = x.matmul(self.embed.weight.T)
        return y

    def grad_loss(self,item):
        return self._loss(item, 'grad_loss')

    def forward(self,item):
        return self._loss(item,'forward')

    def loss(self,item):
        return self._loss(item, 'loss')

    @staticmethod
    def sample_logp(lp,dim,return_logp=False):
        xp = lp.softmax(dim)
        # p = lp.log_softmax(dim).exp()
        '''
        Sampling bug....
        '''
        # _,idx = (torch.rand( xp.shape, device=xp.device) < xp.cumsum(dim)).max(dim) ### [A critical bug!!!]

        _,idx = (torch.rand( xp.shape[:-1] + (1,), device=xp.device) < xp.cumsum(dim)).max(dim)
        lp = torch.gather( lp, index=idx.unsqueeze(dim),dim=dim).squeeze(dim)
        rand_sampled_output = idx #.clip(dataset.tgt_vocab.offset,None)
        # import pdb; pdb.set_trace()
        if return_logp:
            return lp,rand_sampled_output
        else:
            return rand_sampled_output


    def get_default_init(self,h0,B):
        if h0 is None:
            E = self.config.embed_dim
            # B = len(h0)
            h0  = torch.ones([1,B,E],device=self.device)
        return h0


    def _loss(self,item, ret):
        CLS_NAME = self.__class__.__name__

        source = item['source'] ### token sequence
        target = item['target'] ### token seq

        # source_embed = self.embed(source)
        # source_len = item['source_len']
        # source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]

        target_embed = self.embed(target)
        # S = source.size(1)
        # source = None; source_embed=None
        T = target.size(1)
        B = target.size(0)
        N = W = self.config.window_size
        K = self.config.kernel_size
        target_len = item['target_len']
        # target_notnull = torch.arange(target.size(1),device=self.device)[None,:]<target_len[:,None]
        target_notnull = torch.arange(target.size(1),device=self.device)[None,:]<target_len[:,None]
        # import pdb; pdb.set_trace()
        if (item['has_start_token'].__class__==int and item['has_start_token']==1) or  item['has_start_token'].ravel()[0]==1:
            target_notnull[:,0] = False


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

        elif CLS_NAME in 'DLM67'.split():
            '''
            This branch predicts a discrete mixer instead of a vector to speeds up final softmax
            '''

            target_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            target_embed_parent = target_embed_parent/target_embed_parent.std(dim=1,keepdims=True)

            h0 = torch.ones([1,B,E],device=self.device)
            c0 = torch.ones([1,B,E],device=self.device)

            prior = 0.
            prior_sum = 0.

            if 1:

                '''
                parse to get a state distrib
                '''
                h1, _ = self.embed_to_latent(target_embed)

                ### (1,T,E,K)
                '''
                Sample a sequence (w_i,l_i) from parser
                '''

                ### (B,T,W)
                z_lp = (self.std_norm(h1,-1)@self.h_to_z).log_softmax(-1)

                idx     = self.sample_logp(z_lp, return_logp=False, dim=-1)
                '''
                calculate score as log_p_decode(sampled_code) - log_p_encode(sampled_code) + log_p_prior(sampled_code)
                '''

                post_lp = torch.gather(z_lp,index=idx.unsqueeze(-1),dim=-1).squeeze(-1)

                ## (1, T, W)
                z_lp_p  = self.z_prior.log_softmax(-1)[None]
                prior_lp= torch.gather(z_lp_p.repeat((B,1,1)), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)

                # prior = prior + prior_lp - post_lp

                kl = (z_lp.exp() * (z_lp_p - z_lp)).sum(-1)
                prior = prior + kl

                # h1r = self.z_vector[idx]
                # h1r = self.std_norm(self.z_vector,-1)[idx]
                yp = self.latent_to_emittor(h1r)

                self._temp = dict(h1r=h1r,idx=idx,z_lp=z_lp,)
                if ret=='encode': return z_lp.softmax(-1)


            lp = self._hidden_to_cats(yp,ret='target',target=target)


            att = lp.softmax(dim=2).transpose(2,1)
            lp  = lp.logsumexp(-1)

            # logp_sum = (lp * target_notnull).sum(dim=1)
            logp_sum = ((lp + prior) * target_notnull).sum(dim=1)
            logp_sum = logp_sum + prior_sum
            # logp_sum = logp_sum + prior



        elif CLS_NAME in 'DLM23 DLM26 DLM29 DLM43 DLM44 DLM46 DLM47 DLM50 DLM51 DLM52 DLM53 DLM54 DLM55 DLM56 DLM57 DLM58 DLM59 DLM60 DLM61 DLM62 DLM63 DLM64 DLM65 DLM66'.split():
            '''
            This branch predicts a discrete mixer instead of a vector to speeds up final softmax
            '''

            target_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            target_embed_parent = target_embed_parent/target_embed_parent.std(dim=1,keepdims=True)

            h0 = torch.ones([1,B,E],device=self.device)
            c0 = torch.ones([1,B,E],device=self.device)

            prior = 0.
            prior_sum = 0.
            # (B, T, E)
            if CLS_NAME in 'DLM23 DLM26'.split():

                # target_embed_parent[:,0:1] = h0.transpose(1,0)

                # _ , (h0,c0) = self.rnn(  h0.transpose(1,0),(h0,c0))
                target_embed_parent[:,0:1] = h0.transpose(1,0)
                yp, (h1,c1) = self.rnn(target_embed_parent,(h0,c0))
                yp = yp / yp.std(dim=-1, keepdims=True)
                if ret=='encode': return yp


            elif CLS_NAME in 'DLM29 DLM43 DLM44 DLM46 DLM47'.split():
                # h0 = torch.ones([,E],device=self.device)
                h0 = self.w_init_vector[None].repeat((B,1,1)).reshape((1,B*W,E))
                yp, h1      = self.rnn(target_embed_parent[:,None].repeat((1,W,1,1)).reshape((B*W,T,E)), h0)
                # import pdb; pdb.set_trace()
                # yp = yp.reshape((BW,T,E))
                yp = yp / yp.std(dim=-1, keepdims=True)

                target_notnull = target_notnull[:,None].repeat((1,W,1)).reshape((B*W,T))
                target = target[:,None].repeat((1,W,1)).reshape((B*W,T))
                if ret=='encode': return yp.reshape((B,W,T,E))[:,0]

            elif CLS_NAME in 'DLM50 DLM51'.split():
                _ , h1      = self.rnn_enc(target_embed[:,1:].flip([1,]) , h0)

                '''
                Needs to sample from h1
                '''

                # mu_p   = self.h_prior[None,:,0:1].transpose(2,1)
                mu_p     = self.h_prior[0,None,None,:,]
                beta_p   = self.h_prior[1,None,None,:,].exp()
                # beta_p = self.h_prior[None,:,1:2].exp().transpose(2,1)
                #.clip(0,100)

                mu     = h1@self.h_to_mu
                beta   = self.h_post[1,None,None,:,].exp()
                # beta_p = self.h_prior[None,:,1:2].exp().transpose(2,1)
                # beta = beta_p
                # beta = h1@self.h_to_beta

                h1r  = torch.normal(0,1,mu.shape, device=self.device) / beta + mu
                # h1r = h1r.tanh()

                yp, h2      = self.rnn_dec(target_embed_parent, h1r)
                # yp = yp + h1r[0,:,None]
                yp = yp / yp.std(dim=-1, keepdims=True)

                # lp1 = -0.5*((h1r + 1)*beta_p).square() + beta_p.log() - 0.5 * math.log(2*math.pi)
                # lp2 = -0.5*((h1r - 1)*beta_p).square() + beta_p.log() - 0.5 * math.log(2*math.pi)
                # # prior = (torch.stack([lp1,lp2],dim=0) + math.log(0.5)).logsumexp(0)
                # lp3 = torch.stack([mu_p.sigmoid(),1-mu_p.sigmoid()],dim=0).log()
                # prior = (torch.stack([lp1,lp2],dim=0) + lp3 ).logsumexp(0)

                prior = prior +( -0.5*((h1r - mu_p)*beta_p).square() + beta_p.log() - 0.5 * math.log(2*math.pi) )
                prior = prior -( -0.5*((h1r - mu)*beta).square() + beta.log() - 0.5 * math.log(2*math.pi) )
                prior_sum = prior[0].sum(1)
                prior = 0.
                self._temp = dict(mu=mu,beta=beta,h1r=h1r,prior=prior)
                if ret=='encode': return mu[0,:,None,:].transpose(2,1)
                # print(prior.mean())

            elif CLS_NAME in 'DLM57 DLM58 DLM59 DLM60 DLM61 DLM62'.split():

                '''
                Encode right to left, decode left to right
                '''
                h1, _ = self.embed_to_latent(target_embed)

                '''
                Needs to sample from h1
                '''

                mu_p   = self.h_prior[0:1]
                beta_p = self.h_prior[1:2].exp()
                # if self.meta.get('epoch',0)<=5:
                #     prior_w = 0.
                # else:
                prior_w = 1.
                mu   = h1 @self.h_to_mu
                # mu   = h1
                beta = self.h_post[1:2].exp()

                '''
                Resampling
                '''
                h1r = torch.normal( 0, 1, mu.shape, device=self.device) / beta + mu
                yp = self.latent_to_emittor(h1r)


                # import pdb; pdb.set_trace()
                '''
                Calculate extra terms due to posterior correction from prior dist
                '''
                prior = prior +( -0.5*((h1r - mu_p)*beta_p).square() + beta_p.log() - 0.5 * math.log(2*math.pi) )
                prior = prior -( -0.5*((h1r - mu)*beta).square() + beta.log() - 0.5 * math.log(2*math.pi) )
                prior = prior.sum((2))
                prior = prior*prior_w
                self._temp = dict(mu=mu,beta=beta,h1r=h1r,prior=prior)


                if ret=='encode': return mu



            elif CLS_NAME in 'DLM66'.split():

                '''
                Encode right to left, decode left to right
                '''
                # h1, _ = self.embed_to_latent(target_embed)

                '''
                Needs to sample from h1
                '''

                mu_p   = self.h_prior[0:1]
                beta_p = self.h_prior[1:2].exp()
                # if self.meta.get('epoch',0)<=5:
                #     prior_w = 0.
                # else:
                prior_w = 1.
                # mu   = h1 @self.h_to_mu
                # mu   = h1
                mu   = self.embed(target)
                beta = self.h_post[1:2].exp()

                '''
                Resampling
                '''
                h1r = torch.normal( 0, 1, mu.shape, device=self.device) / beta + mu
                yp = self.latent_to_emittor(h1r)


                # import pdb; pdb.set_trace()
                '''
                Calculate extra terms due to posterior correction from prior dist
                '''
                prior = prior +( -0.5*((h1r - mu_p)*beta_p).square() + beta_p.log() - 0.5 * math.log(2*math.pi) )
                prior = prior -( -0.5*((h1r - mu)*beta).square() + beta.log() - 0.5 * math.log(2*math.pi) )
                prior = prior.sum((2))
                prior = prior*prior_w
                self._temp = dict(mu=mu,beta=beta,h1r=h1r,prior=prior)


                if ret=='encode': return mu


            elif CLS_NAME in 'DLM65'.split():
                '''
                This diverges
                '''

                '''
                Encode right to left, decode left to right
                '''
                h1, _ = self.embed_to_latent(target_embed)

                '''
                Needs to sample from h1
                '''

                mu_p   = self.h_prior[0:1]
                beta_p = self.h_prior[1:2].exp()

                # mu   = h1 @self.h_to_mu
                # beta = self.h_post[1:2].exp()

                '''
                Resampling
                '''
                h1r = mu = h1
                yp = self.latent_to_emittor(h1r)


                # import pdb; pdb.set_trace()
                '''
                Calculate extra terms due to posterior correction from prior dist
                '''
                prior = prior +( -0.5*((h1r - mu_p)*beta_p).square() + beta_p.log() - 0.5 * math.log(2*math.pi) )
                # prior = prior -( -0.5*((h1r - mu)*beta).square() + beta.log() - 0.5 * math.log(2*math.pi) )
                prior = prior.sum((2))
                self._temp = dict(h1r=h1r,prior=prior)


                if ret=='encode': return mu





            elif CLS_NAME in 'DLM63'.split():

                '''
                Encode right to left, decode left to right
                '''
                h1, _ = self.embed_to_latent(target_embed)

                '''
                Needs to sample from h1
                '''

                ### (1,T,E,K)
                mu_p   = self.h_prior[0:1]
                beta_p = self.h_prior[1:2].exp()
                k_p    = self.h_prior[2:3].log_softmax(-1)
                ## ( T, E, K)
                # k_p  = self.k_prior[None,:,None]

                mu   = h1 @self.h_to_mu
                beta = self.h_post[1:2].exp()

                '''
                Resampling
                '''
                h1r = torch.normal( 0, 1, mu.shape, device=self.device) / beta + mu
                yp = self.latent_to_emittor(h1r)


                # import pdb; pdb.set_trace()
                '''
                Calculate extra terms due to posterior correction from prior dist
                '''
                prior = prior +(( -0.5*((h1r.unsqueeze(-1) - mu_p)*beta_p).square() + beta_p.log() - 0.5 * math.log(2*math.pi) ) + k_p).logsumexp(-1)
                prior = prior -( -0.5*((h1r - mu)*beta).square() + beta.log() - 0.5 * math.log(2*math.pi) )
                prior = prior.sum((2))
                self._temp = dict(mu=mu,beta=beta,h1r=h1r,prior=prior)

                if ret=='encode': return mu


            elif CLS_NAME in 'DLM64'.split():

                '''
                Encode right to left, decode left to right
                '''
                h1, _ = self.embed_to_latent(target_embed)

                '''
                Needs to sample from h1
                '''

                ### (1,T,E,K)

                '''
                Sampling discrete variables
                '''
                ### (B,T,W)
                z_lp = (self.std_norm(h1,-1)@self.h_to_z).log_softmax(-1)
                # z_lp = (h1@self.h_to_z).log_softmax(-1)


                idx     = self.sample_logp(z_lp, return_logp=False, dim=-1)
                # import pdb; pdb.set_trace()
                '''
                Calculate extra terms due to posterior correction from prior dist
                '''

                post_lp = torch.gather(z_lp,index=idx.unsqueeze(-1),dim=-1).squeeze(-1)

                ## (1, T, W)
                z_lp_p  = self.z_prior.log_softmax(-1)[None]
                prior_lp= torch.gather(z_lp_p.repeat((B,1,1)), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)


                # prior = prior + prior_lp - post_lp

                kl = (z_lp.exp() * (z_lp_p - z_lp)).sum(-1)
                prior = prior + kl

                # h1r = self.z_vector[idx]
                h1r = self.std_norm(self.z_vector,-1)[idx]
                yp = self.latent_to_emittor(h1r)


                self._temp = dict(h1r=h1r,idx=idx,z_lp=z_lp,)
                #mu=mu,beta=beta,h1r=h1r,prior=prior)

                if ret=='encode': return z_lp.softmax(-1)


            elif CLS_NAME in 'DLM53 DLM54'.split():
                _ , h1      = self.rnn_enc(target_embed[:,:].flip([1,]), h0)
                '''
                no need to sample from h1
                '''
                h1 = h1[0]
                yp, h2      = self.rnn_dec(target_embed_parent, h1[None])
                # yp = yp + h1r[0,:,None]
                yp = yp / yp.std(dim=-1, keepdims=True)

                self._temp = dict(h1=h1)
                if ret=='encode': return h1


            elif CLS_NAME in 'DLM55 DLM56'.split():
                # if item.get('is_test',[0])[0]==1:
                #     target = torch.randint(target.min(),target.max()-1,size=target.shape,device=self.device)
                #     target_embed = self.embed(target)
                h1 = self.enc(target_embed)

                mu_p   = self.h_prior[None,:,0:1].transpose(2,1)[0]
                beta_p = self.h_prior[None,:,1:2].exp().transpose(2,1).clip(0.01,100)[0]

                mu   = h1@self.h_to_mu + h1
                beta = beta_p
                # beta = h1@self.h_to_beta
                # beta = beta.exp().clip(0.01,100)
                h1r  = torch.normal(0,1, mu.shape, device=self.device) / beta + mu
                # import pdb; pdb.set_trace()
                prior = prior +( -0.5*((h1r - mu_p)*beta_p).square() + beta_p.log() - 0.5 * math.log(2*math.pi) )
                prior = prior -( -0.5*((h1r - mu)*beta).square() + beta.log() - 0.5 * math.log(2*math.pi) )
                prior = prior.sum(1)
                # prior = 0.
                self._temp = dict(mu=mu,beta=beta,h1r=h1r,prior=prior)

                # if torch.isnan(prior).any():
                #     import pdb; pdb.set_trace()

                yp = self.dec(h1r)

                yp = yp /(yp.std(dim=-1, keepdims=True))

                self._temp = dict(h1=h1)
                if ret=='encode': return h1


            elif CLS_NAME in 'DLM52'.split():
                _ , h1      = self.rnn_enc(target_embed[:,1:].flip([1,]) , h0)

                '''
                Needs to sample from h1
                '''

                # mu_p   = self.h_prior[None,:,0:1].transpose(2,1)
                # beta_p = self.h_prior[None,:,1:2].exp().transpose(2,1)

                mup   = (h1@self.h_to_mu).reshape((1,B,E,3)).softmax(dim=-1)
                # beta_p = self.h_prior[None,:,1:2].exp().transpose(2,1)
                # beta = beta_p
                # beta = h1@self.h_to_beta
                # beta = beta.exp()
                # import pdb; pdb.set_trace()
                mupc = mup.cumsum(-1)
                xr = torch.rand(mup.shape[:-1],device=self.device)
                _ , idx = (xr.unsqueeze(-1)<mupc).max(-1)
                h1r = torch.tensor((-1.,0.,1.),device=self.device)[idx]
                # import pdb; pdb.set_trace()
                # h1r  = torch.normal(0,1,mu.shape, device=self.device) / beta + mu
                # h1r = h1r.tanh()


                yp, h2      = self.rnn_dec(target_embed_parent, h1r)
                yp = yp / yp.std(dim=-1, keepdims=True)
                self._temp = dict(mup=mup,h1r=h1r)

                mup_prior_log   = self.h_mu_prior.reshape((1,1, E,3)).log_softmax(-1)
                # [None,:,0:1].transpose(2,1)
                # mup_p = self.h_prior
                prior = prior + torch.gather( mup_prior_log - mup.log(),dim=-1,index=idx.unsqueeze(-1)).squeeze(-1)
                # [idx]
                prior = prior[0].sum(1)
                # print(prior.mean())



            else:
                raise NotImplementedError( CLS_NAME )



            lp = self._hidden_to_cats(yp,ret='target',target=target)


            att = lp.softmax(dim=2).transpose(2,1)
            lp  = lp.logsumexp(-1)

            # logp_sum = (lp * target_notnull).sum(dim=1)
            logp_sum = ((lp + prior) * target_notnull).sum(dim=1)
            logp_sum = logp_sum + prior_sum
            # logp_sum = logp_sum + prior


            if CLS_NAME in 'DLM29 DLM43 DLM44 DLM46 DLM47'.split():
                wp = self.w_init_logp.log_softmax(0)[None].repeat((B,1,1)).reshape((B*W,))
                ## (BW, 1)
                logp_sum = logp_sum + wp
                logp_sum = logp_sum.reshape((B,W)).logsumexp(-1)
                target_notnull = target_notnull.reshape((B,W,T))[:,0]




        elif CLS_NAME in 'DLM40'.split():
            # torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            target_parent = torch.cat([torch.zeros((B,1),device=self.device,dtype=torch.long),target[:,:-1]],dim=1)


            logit = self.submodel.forward(target_parent)[0]
            logit = logit.log_softmax(-1)
            lp =  torch.gather( logit[:,:,:], index=target[:,:,None],dim=-1)[:,:,:]
            # lp = log
            # import pdb; pdb.set_trace()

            att = lp.softmax(dim=2)
            lp  = lp.logsumexp(2)
            logp_sum = (lp * target_notnull).sum(dim=1)

        elif CLS_NAME in 'DLM41'.split():
            # torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            target_parent = torch.cat([torch.zeros((B,1),device=self.device,dtype=torch.long),target[:,:-1]],dim=1)


            logit = self.submodel.forward(target_parent,ret='embed')

            target_embed_parent = self.embed(target_parent)
            target_embed_parent = target_embed_parent/target_embed_parent.std(dim=1,keepdims=True)
            yp = self.unembed( self.std_norm(self.k_vector,-1))
            yp = (torch.exp(self.k_scale)* yp ).log_softmax(-1)
            model_w = yp.T[target]
            kp = self.embed_to_logp(logit).log_softmax(-1)

            lp = (model_w + kp)


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



        elif CLS_NAME == 'DLM25':
            ### P(z_{i+1}|s(a_{i+1}))
            ### (B, T, K)
            z_by_sa = self.embed_to_logp(target_embed).log_softmax(-1)
            ### (K, E)
            yp      = self.unembed( self.k_vector/self.k_vector.std(-1,keepdims=True)).log_softmax(-1)
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

        elif CLS_NAME == 'DLM27':
            '''
            Use last two words to compute conditional
            '''
            yp = self.unembed( self.std_norm(self.k_vector,-1) ).log_softmax(-1)
            ## (B, T, K)
            model_w = yp.T[target]

            target_embed_parent_1 = torch.cat([torch.ones((B,2,E),device=self.device),target_embed],dim=1)[:,:-2]
            target_embed_parent_2 = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            target_embed_parent_1 = self.std_norm(target_embed_parent_1, -1 )
            target_embed_parent_2 = self.std_norm(target_embed_parent_2, -1 )
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

        elif CLS_NAME=='DLM28':
            ### padding one
            ## (K, E) -> (K, C)
            yp = self.unembed(self.std_norm(self.k_vector,-1)).log_softmax(-1)
            ## (B, T, K)
            model_q = yp.T[target]
            ## (B, T, E)
            target_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            target_embed_parent = self.std_norm(target_embed_parent,-1)

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

        elif CLS_NAME=='DLM19':
            ### padding one
            ## (KS, K, E) -> (KS, KP, C)
            # KS = 2
            KS = self.KS
            KP = self.KP
            yp = self.unembed( self.k_vector.reshape((KS,KP,E))).log_softmax(-1)

            ## (KS, K, B, T)
            ## (B, T, KS, KP)
            model_w = yp[:,:,target].permute((2,3,0,1))

            ## (B, T, E)
            target_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]

            ## (B, T, KS, KP)
            cond = self.embed_to_logp( target_embed_parent ).reshape((B,T,KS,KP)).log_softmax(-1)

            ## (B, T, KS, KP)
            if 1:
            # if ret in 'grad_loss forward'.split():
                lp = model_w + cond
                att = lp.softmax(dim=3).reshape((B,T,KS*KP)).transpose(2,1)
                lp  = lp.logsumexp(-1).sum(-1)
                logp_sum = (lp * target_notnull).sum(dim=1)
            elif ret =='loss' :
                lp = cond.unsqueeze(-1) + yp[None,None]
                lp = lp.logsumexp(3).sum(2).log_softmax(-1)
                lp = torch.gather(lp,index=target.unsqueeze(-1),dim=-1).squeeze(-1)
                # att = (model_w + cond).softmax(dim=3).reshape((B,T,KS*KP)).transpose(2,1)
                logp_sum = (lp * target_notnull).sum(dim=1)
            else:
                raise NotImplementedError(loss_name)

        elif CLS_NAME=='DLM20':
            ### padding one
            ## (K, E) -> (K, C)
            # target_embed = target_embed/target_embed.std(dim=1,keepdims=True)
            # yp = self.unembed( self.k_vector/self.k_vector.std(dim=1,keepdims=True))
            # yp = (torch.exp(self.k_scale)* yp ).log_softmax(-1)
            yp = self.unembed( self.k_vector).log_softmax(-1)
            ## (B, T, K)
            model_w = yp.T[target]
            ## (B, T, E)
            target_embed_parent   = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            target_embed_parent_2 = torch.cat([torch.ones((B,2,E),device=self.device),target_embed],dim=1)[:,:-2]
            target_embed_parent_3 = torch.cat([torch.ones((B,3,E),device=self.device),target_embed],dim=1)[:,:-3]
            target_embed_parent_4 = torch.cat([torch.ones((B,4,E),device=self.device),target_embed],dim=1)[:,:-4]

            ## (B, T, K)
            cond = (
            0.5* self.embed_to_logp( target_embed_parent )
            +0.25*self.embed_to_logp2(target_embed_parent_2)
            +0.125* self.embed_to_logp3(target_embed_parent_3)
            +0.0625* self.embed_to_logp4(target_embed_parent_4)
            ).log_softmax(-1)

            lp = model_w + cond

            att = lp.softmax(dim=2).transpose(2,1)
            lp  = lp.logsumexp(-1)
            logp_sum = (lp * target_notnull).sum(dim=1)

        elif CLS_NAME=='DLM21':
            ### padding one
            ## (K, E) -> (K, C)
            yp = self.unembed_weight.log_softmax(-1)
            ## (B, T, K)
            model_w = yp.T[target]
            ## (B, T, E)
            target_embed_parent   = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            target_embed_parent_2 = torch.cat([torch.ones((B,2,E),device=self.device),target_embed],dim=1)[:,:-2]
            target_embed_parent_3 = torch.cat([torch.ones((B,3,E),device=self.device),target_embed],dim=1)[:,:-3]
            target_embed_parent_4 = torch.cat([torch.ones((B,4,E),device=self.device),target_embed],dim=1)[:,:-4]

            ## (B, T, K)
            cond = (
            0.5* self.embed_to_logp( target_embed_parent )
            +0.25*self.embed_to_logp2(target_embed_parent_2)
            +0.125* self.embed_to_logp3(target_embed_parent_3)
            +0.0625* self.embed_to_logp4(target_embed_parent_4)
            ).log_softmax(-1)

            lp = model_w + cond

            att = lp.softmax(dim=2).transpose(2,1)
            lp  = lp.logsumexp(-1)
            logp_sum = (lp * target_notnull).sum(dim=1)

        elif CLS_NAME=='DLM22':
            ### padding one
            ## (K, E) -> (K, C)
            target_embed = target_embed/target_embed.std(dim=1,keepdims=True)
            yp = self.unembed( self.k_vector/self.k_vector.std(dim=1,keepdims=True))
            yp = (torch.exp(self.k_scale)* yp ).log_softmax(-1)
            # yp = self.unembed( self.k_vector).log_softmax(-1)
            ## (B, T, K)
            model_w = yp.T[target]
            ## (B, T, E)
            target_embed_parent   = torch.cat([torch.ones((B,1,E),device=self.device) /E**0.5,target_embed],dim=1)[:,:-1]
            target_embed_parent_2 = torch.cat([torch.ones((B,2,E),device=self.device) /E**0.5,target_embed],dim=1)[:,:-2]
            target_embed_parent_3 = torch.cat([torch.ones((B,3,E),device=self.device) /E**0.5,target_embed],dim=1)[:,:-3]
            target_embed_parent_4 = torch.cat([torch.ones((B,4,E),device=self.device) /E**0.5,target_embed],dim=1)[:,:-4]

            ## (B, T, K)
            cond = (
            0.5* self.embed_to_logp( target_embed_parent )
            +0.25*self.embed_to_logp2(target_embed_parent_2)
            +0.125* self.embed_to_logp3(target_embed_parent_3)
            +0.0625* self.embed_to_logp4(target_embed_parent_4)
            ).log_softmax(-1)

            lp = model_w + cond

            att = lp.softmax(dim=2).transpose(2,1)
            lp  = lp.logsumexp(-1)
            logp_sum = (lp * target_notnull).sum(dim=1)


        elif CLS_NAME=='DLM24':
            ### padding one
            ## (K, E) -> (K, C)
            yp = self.unembed( self.k_vector)
            yp = (torch.exp(self.k_scale)* yp ).log_softmax(-1)
            # yp = self.unembed_weight.log_softmax(-1)
            ## (B, T, K)
            model_w = yp.T[target]
            ## (B, T, E)
            target_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            target_embed_parent_2 = torch.cat([torch.ones((B,2,E),device=self.device),target_embed],dim=1)[:,:-2]
            target_embed_parent_3 = torch.cat([torch.ones((B,3,E),device=self.device),target_embed],dim=1)[:,:-3]
            target_embed_parent_4 = torch.cat([torch.ones((B,4,E),device=self.device),target_embed],dim=1)[:,:-4]

            ## (B, T, K)
            cond = (
            0.5* self.embed_to_logp( target_embed_parent )
            +0.25*self.embed_to_logp2(target_embed_parent_2)
            +0.125* self.embed_to_logp3(target_embed_parent_3)
            +0.0625* self.embed_to_logp4(target_embed_parent_4)
            ).log_softmax(-1)

            lp = model_w + cond

            att = lp.softmax(dim=2).transpose(2,1)
            lp  = lp.logsumexp(-1)
            logp_sum = (lp * target_notnull).sum(dim=1)



        elif CLS_NAME=='DLM30':
            '''
            construct a mapping variable to reorder input.

            then use the reorder variable to calculate the expected input, which is then converted to the logit.

            the hope is that using mean-field for exponential model would not cause too much harm.
            '''

            ### padding one
            ## (K, E) -> (K, C)
            target_embed = self.std_norm(target_embed,-1)
            yp = self.unembed( self.std_norm(self.k_vector,-1))
            yp = (torch.exp(self.k_scale)* yp ).log_softmax(-1)
            model_w = yp.T[target]

            'Calculate how likely this vector is of node N. E->N'
            ## (B, T, N)
            isnode = self.embed_to_logp_isnode(target_embed)
            '''
            For each node, determine the most-likely token by expectation
            '''

            #### (B, Ts, Tt)
            xt = torch.arange(T,device=self.device)
            prior =  self.shared_log_align
            prior = prior + -self.INF * (xt[None,:,None] >= xt[None,None,:])


            ### (B, Ts, N, Tt)
            isnode = isnode[:,:,:,None] + prior.transpose(1,0)[None,:,:,:]
            att = isnode.softmax(dim=1)
            ### target embed  (B, Ts, E)
            nodes = torch.einsum('bxny,bxe->byne',att, target_embed)
            cond    = self.embed_to_logp( nodes.reshape((B,T,-1)) ).log_softmax(-1)

            ### (B, Tt, K)
            lp = model_w + cond
            att = att[:,:,0]

            # att = lp.softmax(dim=2).transpose(2,1)
            lp  = lp.logsumexp(-1)
            logp_sum = (lp * target_notnull).sum(dim=1)


        elif CLS_NAME in 'DLM31 DLM32 DLM33 DLM34 DLM35'.split():
            '''
            construct a mapping variable to reorder input.
            then use the reorder variable to calculate the expected input, which is then converted to the logit.
            the hope is that using mean-field for exponential model would not cause too much harm.
            '''


            ### padding one
            ## (K, E) -> (K, C)
            target_embed = self.std_norm(target_embed,-1)
            yp = self.unembed( self.std_norm(self.k_vector,-1))
            yp = (torch.exp(self.k_scale)* yp ).log_softmax(-1)
            model_w = yp.T[target]

            ### (B, T,K)


            '''
            Get start of chain distrib

            (B, Ts, Tt)  log_softmax(dim=1)
            '''

            '''
            (B, T, E) -> (B, Ts, 1)
            + (B, Ts, Tt)  prior
            '''


            #### (B, Ts, Tt)
            xt = torch.arange(T,device=self.device)
            prior =  self.shared_log_align[0:1]
            prior = prior + -self.INF * (xt[None,:,None] >= xt[None,None,:])


            'Calculate how likely this vector is of node N. E->N'
            ## (B, T, N)
            isnode = self.embed_to_logp_isnode(target_embed)[:,:,0:1]

            #### (B, Ts, Tt)
            lp1 = isnode + prior
            # att1 = lp1.softmax(dim=1)
            lp1 = lp1.log_softmax(dim=1)

            ### (B, Ts1, Ts2, 1)
            def aff_sublayer( transition, target_embed):
                if CLS_NAME in 'DLM31 DLM32 DLM35'.split():
                    aff = self.std_norm( transition(target_embed), -1) @ target_embed.transpose(2,1)
                    if CLS_NAME=='DLM32': aff = aff.log_softmax(dim=2)
                elif CLS_NAME in 'DLM33 DLM34'.split():
                    aff = self.embed_to_logp_trans(self.std_norm( transition(target_embed),-1)).log_softmax(-1)
                    aff = (aff.unsqueeze(-1) + model_w.transpose(2,1)[:,None]).logsumexp(dim=2)
                else:
                    assert 0,CLS_NAME
                return aff
            # aff =

            aff = aff_sublayer(self.transition1, target_embed)
            xt = torch.arange(T,device=self.device)
            # prior = 0.
            prior = self.shared_log_align[1:2,:,:,None,]
            prior = prior + -self.INF * (xt[None,:,None,None] >= xt[None,None,None,:])
            prior = prior + -self.INF * (xt[None,None,:,None] >= xt[None,None,None,:])
            lp2 = lp1[:,:,None] + (aff.unsqueeze(-1) + prior).log_softmax(2)
            # lp2 = aff.unsqueeze(-1) + prior + lp1[:,:,None]
            # att2 = lp2.softmax(dim=1)
            lp2 = lp2.logsumexp(dim=1)

            ### (B, Ts1, Ts2, 1)
            aff = aff_sublayer(self.transition2, target_embed)


            xt = torch.arange(T,device=self.device)
            prior = 0.
            prior = self.shared_log_align[2:3,:,:,None,]
            prior = prior + -self.INF * (xt[None,:,None,None] >= xt[None,None,None,:])
            prior = prior + -self.INF * (xt[None,None,:,None] >= xt[None,None,None,:])
            lp3 = lp2[:,:,None] + (aff.unsqueeze(-1) + prior).log_softmax(2)
            # att3 = lp3.softmax(dim=1)
            lp3 = lp3.logsumexp(dim=1)


            ### (B, Ts1, Ts2, 1)
            aff = aff_sublayer(self.transition3, target_embed)

            xt = torch.arange(T,device=self.device)
            prior = 0.
            prior = self.shared_log_align[3:4,:,:,None,]
            prior = prior + -self.INF * (xt[None,:,None,None] >= xt[None,None,None,:])
            prior = prior + -self.INF * (xt[None,None,:,None] >= xt[None,None,None,:])
            lp4  = lp3[:,:,None] + (aff.unsqueeze(-1) + prior).log_softmax(2)
            # att4 = lp4.softmax(dim=1)
            lp4  = lp4.logsumexp(dim=1)


            ### (B, Ts, N, Tt)
            isnode = torch.stack([lp1,lp2,lp3,lp4],dim=2)
            att    = isnode.softmax(dim=1)

            ### target embed  (B, Tt, N, E)
            nodes = torch.einsum('bxny,bxe->byne',att, target_embed)

            att = att[:,:,0:2]
            att = att.reshape((B,-1,T))
            if CLS_NAME in 'DLM30 DLM31 DLM32 DLM33'.split():
                cond    = self.embed_to_logp( nodes.reshape((B,T,-1)) ).log_softmax(-1)

                ### (B, Tt, K)
                lp = model_w + cond
            elif CLS_NAME in 'DLM34 DLM35'.split():
                h0 = torch.ones([1,B*T,E],device=self.device)
                c0 = torch.ones([1,B*T,E],device=self.device)

                x = nodes.reshape((B*T, N, E))
                yp, (h1,c1) = self.rnn(x,(h0,c0))
                ypp = self.std_norm( yp[:,-1,:].reshape((B,T,E)),dim=-1)
                kp = self.embed_to_logp_rnn(ypp).log_softmax(-1)


                lp = model_w + kp

                # self.embed_to_logp
            else:
                raise NotImplementedError(CLS_NAME)



            # att = lp.softmax(dim=2).transpose(2,1)
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
        # print('[mean]',target_notnull.sum(1).float().mean())


        logp_per_token = logp_sum/ target_notnull.sum(dim=1)
        # .clip(1,None)
        #* 1000
        loss = -logp_per_token

        if ret=='forward':
            return loss,att
        elif ret in 'loss grad_loss'.split():
            return loss
        # elif ret=='grad_loss':
        #     return loss
        elif ret=='loss_per_loc':
            return -lp,target_notnull
        else:
            raise NotImplementedError(f'''{ret} for ._loss()''')

    def _hidden_to_cats(self, yp, target, ret):
        '''
        Convert hidden vector to explicit categorical distribution
        '''
        CLS_NAME = self.__class__.__name__
        if CLS_NAME in 'DLM23 DLM43'.split():
            # (B, T, C)
            lyp = self.unembed(yp).log_softmax(-1)
            lp =  torch.gather(lyp [:,:,:], index=target[:,:,None],dim=-1)[:,:,:]
            if ret=='full':
                raise NotImplementedError
                # lp = (yp[None,None] + kp.unsqueeze(-1)).logsumexp(2)
                # return lp
                # pass
            elif ret =='target':
                return lp
            else:
                raise NotImplementedError(f'{ret}')

        elif CLS_NAME in 'DLM26 DLM29 DLM44 DLM46 DLM47 DLM50 DLM51 DLM52 DLM53 DLM54 DLM55 DLM56 DLM57 DLM58 DLM59 DLM60 DLM61 DLM62 DLM63 DLM64 DLM65 DLM66'.split():
            ## (B, T ,K )
            kp = self.embed_to_logp(yp).log_softmax(-1)
            yp = self.unembed( self.std_norm(self.k_vector,-1))
            yp = (torch.exp(self.k_scale)* yp ).log_softmax(-1)

            # if class
            if ret=='full':
                lp = (yp[None,None] + kp.unsqueeze(-1)).logsumexp(2)
                return lp

            elif ret =='target':
                # if CLS_NAME in 'DLM29 DLM43 DLM44 DLM46 DLM47'.split():
                model_w = yp.T[target]

                ## yp (K, C)
                ## kp (B,T,K)
                # if target is not None and 23 in target[7].detach().cpu().numpy().tolist():
                #     pass
                    # import pdb; pdb.set_trace()
                lp = (model_w + kp)
                return lp
            else:
                raise NotImplementedError( f'{ret!r} for {CLS_NAME!r}' )
        else:
            raise NotImplementedError( CLS_NAME )
        # if ret==''


class DLM1(DLMPrototype):
    # pass
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        W = min(self.config.window_size,0 )
        embed_dim       = self.embed_dim
        self.layers     = nn.ModuleList([
            nn.Linear(embed_dim*W, embed_dim*W).to(self.device)
            for _ in range(self.config.depth-1)
            ])
        self.final_layer      = nn.Linear(embed_dim*W,  embed_dim).to(self.device)

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
        self.embed_to_logp = nn.Linear(config.embed_dim, K).to(self.device)
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



class DLM20(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        # x = nn.Linear(1,K)
        # self.shared_log_align = nn.Parameter(x.weight.T)
        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        # x = nn.Linear(K,  1).to(self.device)
        # self.k_scale   =  nn.Parameter(x.weight.T)
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2 = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp3 = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp4 = nn.Linear(config.embed_dim, K).to(self.device)
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

class DLM22(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        # x = nn.Linear(1,K)
        # self.shared_log_align = nn.Parameter(x.weight.T)
        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2 = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp3 = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp4 = nn.Linear(config.embed_dim, K).to(self.device)
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

    # def unembed(self,x):
    #     # import pdb; pdb.set_trace()
    #     # assert 0
    #     xdy = x[:,None,:] - self.embed.weight[None,:,:]
    #     xdy = -xdy.square().mean(-1)
    #     # y = x.matmul(self.embed.weight.T)
    #     return xdy




class DLM30(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)
        T = config.n_step
        N = config.window_size
        E = config.embed_dim
        x = nn.Linear(T,N*T)
        self.shared_log_align = nn.Parameter(x.weight.T.reshape((N,T,T)))

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale    =  nn.Parameter(x.weight.T)
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(N*E, K).to(self.device)
        self.embed_to_logp_isnode =  nn.Linear(E, N).to(self.device)
        # self.embed_to_logp2 = nn.Linear(config.embed_dim, K).to(self.device)
        # self.embed_to_logp3 = nn.Linear(config.embed_dim, K).to(self.device)
        # self.embed_to_logp4 = nn.Linear(config.embed_dim, K).to(self.device)
        # self.embed_to_logp = nn.Parameter(x.weight.T)

    def log_param(self,buf,plt):
        # return
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        key = 'self.shared_log_align'
        mat = self.shared_log_align.log_softmax(1).reshape((-1,self.config.n_step)).cpu().detach()

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



class DLM31(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)
        T = config.n_step
        N = config.window_size
        E = config.embed_dim
        x = nn.Linear(T,N*T)
        self.shared_log_align = nn.Parameter(x.weight.T.reshape((N,T,T)))

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale    =  nn.Parameter(x.weight.T)
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp_trans  = nn.Linear(E, K).to(self.device)
        self.embed_to_logp  = nn.Linear(N*E, K).to(self.device)
        self.embed_to_logp_isnode =  nn.Linear(E, 1).to(self.device)
        self.transition1  = nn.Linear(E, E).to(self.device)
        self.transition2  = nn.Linear(E, E).to(self.device)
        self.transition3  = nn.Linear(E, E).to(self.device)
        # self.embed_to_logp2 = nn.Linear(config.embed_dim, K).to(self.device)
        # self.embed_to_logp3 = nn.Linear(config.embed_dim, K).to(self.device)
        # self.embed_to_logp4 = nn.Linear(config.embed_dim, K).to(self.device)
        # self.embed_to_logp = nn.Parameter(x.weight.T)

    def log_param(self,buf,plt):
        # return
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        key = 'self.shared_log_align'
        mat = self.shared_log_align.log_softmax(1).reshape((-1,self.config.n_step)).cpu().detach()

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


    # def unembed(self,x):
    #     # import pdb; pdb.set_trace()
    #     # assert 0
    #     xdy = x[:,None,:] - self.embed.weight[None,:,:]
    #     xdy = -xdy.square().mean(-1)
    #     # y = x.matmul(self.embed.weight.T)
    #     return xdy
class DLM32(DLM31):
    pass
class DLM33(DLM31):
    pass

class DLM34(DLM31):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)
        T = config.n_step
        N = config.window_size
        E = config.embed_dim
        x = nn.Linear(T,N*T)
        self.embed_to_logp_rnn  = nn.Linear(E, K).to(self.device)
        self.rnn = nn.LSTM(input_size = E,hidden_size=E, batch_first= True, num_layers=1)


class DLM35(DLM34):
    pass

class DLM24(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        # x = nn.Linear(1,K)
        # self.shared_log_align = nn.Parameter(x.weight.T)
        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2 = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp3 = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp4 = nn.Linear(config.embed_dim, K).to(self.device)
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

    # def unembed(self,x):
    #     # import pdb; pdb.set_trace()
    #     # assert 0
    #     xdy = x[:,None,:] - self.embed.weight[None,:,:]
    #     xdy = -xdy.square().mean(-1)
    #     # y = x.matmul(self.embed.weight.T)
    #     return xdy


class DLM21(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        self.unembed_weight= nn.Parameter(nn.Linear(K, config.graph_dim).to(self.device).weight.T)

        # x = nn.Linear(1,K)
        # self.shared_log_align = nn.Parameter(x.weight.T)
        # x = nn.Linear(K,  config.embed_dim).to(self.device)
        # self.k_vector   =  nn.Parameter(x.weight.T)
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2 = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp3 = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp4 = nn.Linear(config.embed_dim, K).to(self.device)
        # self.embed_to_logp = nn.Parameter(x.weight.T)

    # def unembed(self,x):
    #     # import pdb; pdb.set_trace()
    #     # assert 0
    #     xdy = x[:,None,:] - self.embed.weight[None,:,:]
    #     xdy = -xdy.square().mean(-1)
    #     # y = x.matmul(self.embed.weight.T)
    #     return xdy

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


class DLM19(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        KP= 16
        KS = K//KP
        assert K % 16 ==0 , K
        # K = KK*10
        # config.kernel_size = K
        self.KS= KS
        self.KP = KP

        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        # x = nn.Linear(1,K)
        # self.shared_log_align = nn.Parameter(x.weight.T)
        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp = nn.Linear(config.embed_dim, K).to(self.device)
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


class DLM27(DLM16):
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

class DLM28(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector        =  nn.Parameter(x.weight.T)
        self.final_layer      = nn.Linear(config.embed_dim,  1*config.embed_dim).to(self.device)
        self.embed_to_logp = nn.Linear(config.embed_dim, K+1).to(self.device)


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



class DLM29(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        self.rnn = nn.RNN(input_size = E,hidden_size=E, batch_first= True, num_layers=1, nonlinearity='tanh')
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)

        W = config.window_size
        assert W >=1, config

        x = nn.Linear(W,  config.embed_dim).to(self.device)
        self.w_init_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(W,  1).to(self.device)
        self.w_init_logp     =  nn.Parameter(x.weight.T)


class DLM50(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        E            = config.embed_dim
        self.rnn_enc = nn.RNN(input_size = E, hidden_size=E, batch_first= True, num_layers=1, nonlinearity='tanh')
        self.rnn_dec = nn.RNN(input_size = E, hidden_size=E, batch_first= True, num_layers=1, nonlinearity='tanh')
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)

        x = nn.Linear(E,  2).to(self.device)
        self.h_prior   =  nn.Parameter(x.weight.T)


        x = nn.Linear(E,  E).to(self.device)
        self.h_to_mu   =  nn.Parameter(x.weight.T)
        x = nn.Linear(E,  E).to(self.device)
        self.h_to_beta   =  nn.Parameter(x.weight.T)


class DLM51(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        E            = config.embed_dim
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2  = nn.Linear(config.embed_dim, K).to(self.device)

        x = nn.Linear(2, E ).to(self.device)
        self.h_prior   =  nn.Parameter(x.weight.T)

        x = nn.Linear(2, E ).to(self.device)
        self.h_post    =  nn.Parameter(x.weight.T)

        x = nn.Linear(E,  E).to(self.device)
        self.h_to_mu   =  nn.Parameter(x.weight.T)
        x = nn.Linear(E,  E).to(self.device)
        self.h_to_beta   =  nn.Parameter(x.weight.T)

    # def dec(self,h1):
    #     T = self.config.n_step
    #     E = self.config.embed_dim
    #     return h1.reshape((len(h1),-1,E))
    #     # return self.rnn_dec(h1[None]).reshape((-1,T,E))



class DLM57(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        T = config.n_step
        # data_dim
        E            = config.embed_dim
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2  = nn.Linear(config.embed_dim, K).to(self.device)

        x = nn.Linear(T*E,  2).to(self.device)
        self.h_prior   =  nn.Parameter(x.weight.T.reshape((2,T,E)))

        x = nn.Linear(T*E,  2).to(self.device)
        self.h_post    =  nn.Parameter(x.weight.T.reshape((2,T,E)))


        x = nn.Linear(E,  E).to(self.device)
        self.h_to_mu   =  nn.Parameter(x.weight.T)
        x = nn.Linear(E,  E).to(self.device)
        self.h_to_beta   =  nn.Parameter(x.weight.T)

    def sample_token(self,B,T,prompt=None):
        E = self.config.embed_dim
        T = self.config.n_step
        mu_p   = self.h_prior[0:1]
        beta_p = self.h_prior[1:2].exp()
        h1r = torch.normal( 0, 1, (B,T,E), device=self.device) / beta_p + mu_p
        return self.sample_token_from_latent(h1r)

    def sample_token_from_latent(self, h1r,return_logp=False):

        yp = self.latent_to_emittor(h1r)
        ## (B,T,C) tensor
        logp = self._hidden_to_cats(yp,ret='full',target=None)
        return self.sample_logp(logp,-1,return_logp)
        # return lp,idx

    def latent_to_emittor(self, h1r,h0=None):
        h0 = self.get_default_init(h0,len(h1r))
        # h1r = h1r.tanh()
        # h1r = self.std_norm(h1r, dim=-1)
        # h0  = torch.ones([1,B,E],device=self.device)
        yp, h2      = self.rnn_dec(h1r, h0)
        yp = yp / yp.std(dim=-1, keepdims=True)
        return yp

    def embed_to_latent(self, target_embed, h0=None):
        h0 = self.get_default_init(h0,len(target_embed))
        h1 , h1blah      = self.rnn_enc(target_embed[:,:].flip([1,]) , h0)
        h1 = h1.flip([1,])
        return h1,h1blah

class DLM65(DLM57):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        T = config.n_step
        # data_dim
        E            = config.embed_dim
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2  = nn.Linear(config.embed_dim, K).to(self.device)

        x = nn.Linear(T*E,  2).to(self.device)
        self.h_prior   =  nn.Parameter(x.weight.T.reshape((2,T,E)))

        x = nn.Linear(T*E,  2).to(self.device)
        self.h_post    =  nn.Parameter(x.weight.T.reshape((2,T,E)))


        x = nn.Linear(E,  E).to(self.device)
        self.h_to_mu   =  nn.Parameter(x.weight.T)
        x = nn.Linear(E,  E).to(self.device)
        self.h_to_beta   =  nn.Parameter(x.weight.T)

    def sample_token(self,B,T,prompt=None):
        E = self.config.embed_dim
        T = self.config.n_step
        mu_p   = self.h_prior[0:1]
        beta_p = self.h_prior[1:2].exp()
        h1r = torch.normal( 0, 1, (B,T,E), device=self.device) / beta_p + mu_p
        return self.sample_token_from_latent(h1r)

    def sample_token_from_latent(self, h1r,return_logp=False):

        yp = self.latent_to_emittor(h1r)
        ## (B,T,C) tensor
        logp = self._hidden_to_cats(yp,ret='full',target=None)
        return self.sample_logp(logp,-1,return_logp)
        # return lp,idx

    def latent_to_emittor(self, h1r,h0=None):
        h0 = self.get_default_init(h0,len(h1r))
        # h1r = h1r.tanh()
        # h1r = self.std_norm(h1r, dim=-1)
        # h0  = torch.ones([1,B,E],device=self.device)
        yp, h2      = self.rnn_dec(h1r, h0)
        yp = yp / yp.std(dim=-1, keepdims=True)
        return yp

    def embed_to_latent(self, target_embed, h0=None):
        h0 = self.get_default_init(h0,len(target_embed))
        h1 , h1blah      = self.rnn_enc(target_embed[:,:].flip([1,]) , h0)
        h1 = h1.flip([1,])
        return h1,h1blah



class DLM63(DLM57):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        T = config.n_step
        # data_dim
        E            = config.embed_dim
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2  = nn.Linear(config.embed_dim, K).to(self.device)

        W = config.window_size
        x = nn.Linear(T*E*W,  3).to(self.device)
        self.h_prior   =  nn.Parameter(x.weight.T.reshape((3,T,E,W)))

        x = nn.Linear(T*E,  2).to(self.device)
        self.h_post    =  nn.Parameter(x.weight.T.reshape((2,T,E)))


        x = nn.Linear(E,  E).to(self.device)
        self.h_to_mu   =  nn.Parameter(x.weight.T)
        x = nn.Linear(E,  E).to(self.device)
        self.h_to_beta   =  nn.Parameter(x.weight.T)


    def latent_to_emittor(self, h1r,h0=None):
        h0 = self.get_default_init(h0,len(h1r))
        yp, h2      = self.rnn_dec(h1r, h0)
        yp = yp / yp.std(dim=-1, keepdims=True)
        return yp

    def embed_to_latent(self, target_embed, h0=None):
        h0 = self.get_default_init(h0,len(target_embed))
        h1 , h1blah      = self.rnn_enc(target_embed[:,:].flip([1,]) , h0)
        h1 = h1.flip([1,])
        return h1,h1blah
    def sample_token(self,B,T,prompt=None):
        return None



class DLM64(DLM57):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        T = config.n_step
        # data_dim
        E            = config.embed_dim
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2  = nn.Linear(config.embed_dim, K).to(self.device)

        W = config.window_size
        x = nn.Linear(T,  W).to(self.device)
        self.z_prior   =  nn.Parameter(x.weight.T)
        # .reshape((3,T,E,W)))

        x = nn.Linear(W, E).to(self.device)
        self.z_vector   =  nn.Parameter(x.weight.T)

        x = nn.Linear(E,  W).to(self.device)
        self.h_to_z   =  nn.Parameter(x.weight.T)
        # x = nn.Linear(E,  E).to(self.device)
        # self.h_to_beta   =  nn.Parameter(x.weight.T)


    def latent_to_emittor(self, h1r,h0=None):
        h0 = self.get_default_init(h0,len(h1r))
        yp, h2      = self.rnn_dec(h1r, h0)
        yp = yp / yp.std(dim=-1, keepdims=True)
        return yp

    def embed_to_latent(self, target_embed, h0=None):
        h0 = self.get_default_init(h0,len(target_embed))
        h1 , h1blah      = self.rnn_enc(target_embed[:,:].flip([1,]) , h0)
        h1 = h1.flip([1,])
        return h1,h1blah
    def sample_token(self,B,T,prompt=None):
        return None
    def sample_token_from_latent(self,h1):
        return None



class DLM67(DLM57):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        T = config.n_step
        # data_dim
        E            = config.embed_dim
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2  = nn.Linear(config.embed_dim, K).to(self.device)

        W = config.window_size
        x = nn.Linear(T,  W).to(self.device)
        self.z_prior   =  nn.Parameter(x.weight.T)
        # .reshape((3,T,E,W)))

        x = nn.Linear(W, E).to(self.device)
        self.z_vector   =  nn.Parameter(x.weight.T)

        x = nn.Linear(E,  W).to(self.device)
        self.h_to_z   =  nn.Parameter(x.weight.T)
        # x = nn.Linear(E,  E).to(self.device)
        # self.h_to_beta   =  nn.Parameter(x.weight.T)


    def latent_to_emittor(self, h1r,h0=None):
        h0 = self.get_default_init(h0,len(h1r))
        yp, h2      = self.rnn_dec(h1r, h0)
        yp = yp / yp.std(dim=-1, keepdims=True)
        return yp

    def embed_to_latent(self, target_embed, h0=None):
        h0 = self.get_default_init(h0,len(target_embed))
        h1 , h1blah      = self.rnn_enc(target_embed[:,:].flip([1,]) , h0)
        h1 = h1.flip([1,])
        return h1,h1blah
    def sample_token(self,B,T,prompt=None):
        return None
    def sample_token_from_latent(self,h1):
        return None

    def _hidden_to_cats(self,yp,target,ret):
        CLS_NAME = self.__class__.__name__

        ## (B, T ,K )
        kp = self.embed_to_logp(yp).log_softmax(-1)
        yp = self.unembed( self.std_norm(self.k_vector,-1))
        yp = (torch.exp(self.k_scale)* yp ).log_softmax(-1)

        # if class
        if ret=='full':
            lp = (yp[None,None] + kp.unsqueeze(-1)).logsumexp(2)
            return lp

        elif ret =='target':
            # if CLS_NAME in 'DLM29 DLM43 DLM44 DLM46 DLM47'.split():
            model_w = yp.T[target]

            ## yp (K, C)
            ## kp (B,T,K)
            # if target is not None and 23 in target[7].detach().cpu().numpy().tolist():
            #     pass
                # import pdb; pdb.set_trace()
            lp = (model_w + kp)
            return lp
        else:
            raise NotImplementedError( f'{ret!r} for {CLS_NAME!r}' )

class DLM66(DLM57):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        T = config.n_step
        # data_dim
        E            = config.embed_dim
        # self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec_2 = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2  = nn.Linear(config.embed_dim, K).to(self.device)

        # W = config.window_size
        # x = nn.Linear(T,  W).to(self.device)
        # self.z_prior   =  nn.Parameter(x.weight.T)
        # # .reshape((3,T,E,W)))
        #
        # x = nn.Linear(W, E).to(self.device)
        # self.z_vector   =  nn.Parameter(x.weight.T)
        #
        # x = nn.Linear(E,  W).to(self.device)
        # self.h_to_z   =  nn.Parameter(x.weight.T)
        # # x = nn.Linear(E,  E).to(self.device)
        # self.h_to_beta   =  nn.Parameter(x.weight.T)


    def latent_to_emittor(self, h1r,h0=None):
        h0 = self.get_default_init(h0,len(h1r))
        yp = h1r
        for i in range(9):
            ypo = yp
            yp, h0      = self.rnn_dec(yp, h0)
            yp = 0.5*(yp+ypo)
            yp = yp.flip([1,])

        yp = yp / yp.std(dim=-1, keepdims=True)
        yp = yp.flip([1,])
        return yp

    def embed_to_latent(self, target_embed, h0=None):
        h0 = self.get_default_init(h0,len(target_embed))
        h1 , h1blah      = self.rnn_enc(target_embed[:,:].flip([1,]) , h0)
        h1 = h1.flip([1,])
        return h1,h1blah

    def sample_token(self,B,T,prompt=None):
        return None



class DLM62(DLM57):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        T = config.n_step
        # data_dim
        E            = config.embed_dim
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2  = nn.Linear(config.embed_dim, K).to(self.device)

        x = nn.Linear(T*E,  2).to(self.device)
        self.h_prior   =  nn.Parameter(x.weight.T.reshape((2,T,E)))
        self.h_post    =  nn.Parameter(x.weight.T.reshape((2,T,E)))


        x = nn.Linear(E,  E).to(self.device)
        self.h_to_mu   =  nn.Parameter(x.weight.T)
        x = nn.Linear(E,  E).to(self.device)
        self.h_to_beta   =  nn.Parameter(x.weight.T)

        submodel, self.embed = DLM40.get_mingpt_model(config)
        self.transformer = submodel.transformer

        x = nn.Linear(E,  E).to(self.device)
        self.rescale = x


    def _ref_mingpt_foward(self):
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)


    def latent_to_emittor(self, h1r,h0=None):
        h0 = self.get_default_init(h0,len(h1r))
        # h0  = torch.ones([1,B,E],device=self.device)
        x = h1r
        D = len(self.transformer.h)//2
        for block in self.transformer.h[D:D*2]:
            x = block(x)

        yp = x
        yp = yp / yp.std(dim=-1, keepdims=True)
        return yp

    def embed_to_latent(self, target_embed, h0=None):
        # tfm = self.transformer

        T = target_embed.shape[1]
        pos = torch.arange(0, T, dtype=torch.long, device=self.device).unsqueeze(0) # shape (1, t)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = target_embed
        x = self.transformer.drop(tok_emb + pos_emb)
        D = len(self.transformer.h)//2
        for block in self.transformer.h[0:D]:
            x = block(x)
        x = self.rescale(x)


        return x,None

        # h0 = self.get_default_init(h0,len(target_embed))
        # x = target_embed
        # #(B,T,E)
        # # h1 , h1blah      = self.rnn_enc(target_embed[:,:].flip([1,]) , h0)
        # # h1 = h1.flip([1,])
        # return h1,h1blah
        #

class DLM59(DLM57):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        T = config.n_step
        # data_dim
        E            = config.embed_dim
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_enc_2 = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec   = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec_2 = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2  = nn.Linear(config.embed_dim, K).to(self.device)

        x = nn.Linear(T*E,  2).to(self.device)
        self.h_prior   =  nn.Parameter(x.weight.T.reshape((2,T,E)))
        self.h_post    =  nn.Parameter(x.weight.T.reshape((2,T,E)))


        x = nn.Linear(E,  E).to(self.device)
        self.h_to_mu   =  nn.Parameter(x.weight.T)
        x = nn.Linear(E,  E).to(self.device)
        self.h_to_beta   =  nn.Parameter(x.weight.T)



    def latent_to_emittor(self, h1r,h0=None):
        h0 = self.get_default_init(h0,len(h1r))
        # h0  = torch.ones([1,B,E],device=self.device)
        # h1r = h1r.flip([1])
        yp, h2      = self.rnn_dec(h1r, h0)
        yp = yp.flip([1])

        yp, h3      = self.rnn_dec_2(yp,h2)
        yp = yp.flip([1])

        yp = yp / yp.std(dim=-1, keepdims=True)
        return yp


    def embed_to_latent(self, target_embed, h0=None):
        h0 = self.get_default_init(h0,len(target_embed))
        # target_embed = target_embed.flip([1,])
        ypo = target_embed
        yp , h2      = self.rnn_enc(target_embed[:,:] , h0)
        yp = yp.flip([1,])

        ypo = yp
        yp , h3      = self.rnn_enc_2(yp,h2)
        yp = yp.flip([1,])
        # target_embed[:,:].flip([1,]) , h0)
        return yp, h3


class DLM60(DLM57):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        T = config.n_step
        # data_dim
        E            = config.embed_dim
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_enc_2 = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec   = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec_2 = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2  = nn.Linear(config.embed_dim, K).to(self.device)

        x = nn.Linear(T*E,  2).to(self.device)
        self.h_prior   =  nn.Parameter(x.weight.T.reshape((2,T,E)))
        self.h_post    =  nn.Parameter(x.weight.T.reshape((2,T,E)))


        x = nn.Linear(E,  E).to(self.device)
        self.h_to_mu   =  nn.Parameter(x.weight.T)
        x = nn.Linear(E,  E).to(self.device)
        self.h_to_beta   =  nn.Parameter(x.weight.T)
    #

    def latent_to_emittor(self, h1r,h0=None):
        h0 = self.get_default_init(h0,len(h1r))
        # h0  = torch.ones([1,B,E],device=self.device)
        # h1r = h1r.flip([1])
        c = 1./2
        ypo = h1r
        yp, h2      = self.rnn_dec(h1r, h0)
        yp = (ypo + yp)*c
        yp = yp.flip([1])

        ypo = yp
        yp, h3      = self.rnn_dec_2(yp,h2)
        yp = (ypo + yp)*c
        yp = yp.flip([1])

        yp = yp / yp.std(dim=-1, keepdims=True)
        return yp


    def embed_to_latent(self, target_embed, h0=None):
        c = 1./2

        h0 = self.get_default_init(h0,len(target_embed))
        # target_embed = target_embed.flip([1,])
        ypo = target_embed
        yp , h2      = self.rnn_enc(target_embed[:,:] , h0)
        yp = (ypo + yp)*c
        yp = yp.flip([1,])

        ypo = yp
        yp , h3      = self.rnn_enc_2(yp,h2)
        yp = (ypo + yp)*c
        yp = yp.flip([1,])
        # target_embed[:,:].flip([1,]) , h0)
        return yp, h3

class DLM61(DLM57):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        T = config.n_step
        # data_dim
        E            = config.embed_dim
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_enc_2 = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_enc_3 = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_enc_4 = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec   = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec_2 = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec_3 = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec_4 = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2  = nn.Linear(config.embed_dim, K).to(self.device)

        x = nn.Linear(T*E,  2).to(self.device)
        self.h_prior   =  nn.Parameter(x.weight.T.reshape((2,T,E)))
        self.h_post    =  nn.Parameter(x.weight.T.reshape((2,T,E)))


        x = nn.Linear(E,  E).to(self.device)
        self.h_to_mu   =  nn.Parameter(x.weight.T)
        x = nn.Linear(E,  E).to(self.device)
        self.h_to_beta   =  nn.Parameter(x.weight.T)


    def latent_to_emittor(self, h1r,h0=None):
        h0 = self.get_default_init(h0,len(h1r))
        # h0  = torch.ones([1,B,E],device=self.device)
        # h1r = h1r.flip([1])
        c = 1./2
        ypo = h1r
        yp, h2      = self.rnn_dec(h1r, h0)
        yp = (ypo + yp)*c
        yp = yp.flip([1])

        ypo = yp
        yp, h2      = self.rnn_dec_2(yp,h2)
        yp = (ypo + yp)*c
        yp = yp.flip([1])

        ypo = yp
        yp, h2      = self.rnn_dec_3(yp,h2)
        yp = (ypo + yp)*c
        yp = yp.flip([1])

        ypo = yp
        yp, h2      = self.rnn_dec_4(yp,h2)
        yp = (ypo + yp)*c
        yp = yp.flip([1])

        yp = yp / yp.std(dim=-1, keepdims=True)
        return yp


    def embed_to_latent(self, target_embed, h0=None):
        c = 1./2

        h0 = self.get_default_init(h0,len(target_embed))
        # target_embed = target_embed.flip([1,])
        ypo = target_embed
        yp , h2      = self.rnn_enc(target_embed[:,:] , h0)
        yp = (ypo + yp)*c
        yp = yp.flip([1,])

        ypo = yp
        yp , h3      = self.rnn_enc_2(yp,h2)
        yp = (ypo + yp)*c
        yp = yp.flip([1,])

        ypo = yp
        yp , h3      = self.rnn_enc_3(yp,h3)
        yp = (ypo + yp)*c
        yp = yp.flip([1,])

        ypo = yp
        yp , h3      = self.rnn_enc_4(yp,h3)
        yp = (ypo + yp)*c
        yp = yp.flip([1,])

        # target_embed[:,:].flip([1,]) , h0)
        return yp, h3


class DLM58(DLM57):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        T = config.n_step
        E            = config.embed_dim
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        self.embed_to_logp2  = nn.Linear(config.embed_dim, K).to(self.device)

        x = nn.Linear(T*E,  2).to(self.device)
        self.h_prior   =  nn.Parameter(x.weight.T.reshape((2,T,E)))
        self.h_post    =  nn.Parameter(x.weight.T.reshape((2,T,E)))

        x = nn.Linear(E,  E).to(self.device)
        self.h_to_mu   =  nn.Parameter(x.weight.T)
        x = nn.Linear(E,  E).to(self.device)
        self.h_to_beta   =  nn.Parameter(x.weight.T)

    def latent_to_emittor(self, h1r,h0=None):
        h0 = self.get_default_init(h0,len(h1r))
        # h0  = torch.ones([1,B,E],device=self.device)
        h1run = h1r*0
        hc = h1run[:,0:1]*0
        T = h1r.shape[1]
        for t in range(T):
            hc = hc*0.5 + 0.5*h1r[:,t:t+1,:]
            h1run[:,t:t+1] = hc

        yp, h2      = self.rnn_dec(h1run, h0)
        # yp, h2      = self.rnn_dec(h1r.cumsum(1), h0)
        yp = yp / yp.std(dim=-1, keepdims=True)
        return yp


class DLM52(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        E            = config.embed_dim
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        # self.embed_to_logp2  = nn.Linear(config.embed_dim, K).to(self.device)


        # x = nn.Linear(E,  2).to(self.device)
        # self.h_prior   =  nn.Parameter(x.weight.T)
        x = nn.Linear(E,  3).to(self.device)
        self.h_mu_prior   =  nn.Parameter(x.weight.T)

        x = nn.Linear(E,  E*3).to(self.device)
        self.h_to_mu   =  nn.Parameter(x.weight.T)
        # x = nn.Linear(E,  E).to(self.device)
        # self.h_to_beta   =  nn.Parameter(x.weight.T)


class DLM53(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        E            = config.embed_dim
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)

        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)

        x = nn.Linear(E,  2).to(self.device)
        self.h_prior   =  nn.Parameter(x.weight.T)
        x = nn.Linear(E,  E).to(self.device)
        self.h_to_mu   =  nn.Parameter(x.weight.T)
        x = nn.Linear(E,  E).to(self.device)
        self.h_to_beta   =  nn.Parameter(x.weight.T)



class DLM54(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        E            = config.embed_dim
        self.rnn_enc = nn.RNN(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec = nn.RNN(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)




class DLM55(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        T = self.config.n_step
        E = self.config.embed_dim
        self.encoder = nn.Linear(T*E,E)
        self.decoder = nn.Linear(E,T*E)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)


        x = nn.Linear(E,  2).to(self.device)
        self.h_prior   =  nn.Parameter(x.weight.T)
        x = nn.Linear(E,  E).to(self.device)
        self.h_to_mu   =  nn.Parameter(x.weight.T)
        x = nn.Linear(E,  E).to(self.device)
        self.h_to_beta   =  nn.Parameter(x.weight.T)


    def enc(self,x):
        return self.encoder(x.reshape((len(x),-1)))
    def dec(self,h1):
        T = self.config.n_step
        E = self.config.embed_dim
        return self.decoder(h1).reshape((-1,T,E))



class DLM56(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        T = self.config.n_step
        E = self.config.embed_dim
        E1 = E*10
        self.encoder1 = nn.Linear(T*E,E1)
        self.encoder2 = nn.Linear(E1,E)

        self.decoder1 = nn.Linear(E,E1)
        self.decoder2 = nn.Linear(E1,T*E)
        # T*E)

        K = config.kernel_size
        assert K >=1,config

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)

    def enc(self,x):
        x = self.encoder1(x.reshape((len(x),-1)))
        x = self.encoder2(x.relu())
        return x.relu()

    def dec(self,h1):
        T = self.config.n_step
        E = self.config.embed_dim
        z = h1
        z = self.decoder1(z).relu()
        y = self.decoder2(z)
        y = y.reshape((-1,T,E))
        return y



class DLM12(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        self.rnn = nn.LSTM(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)


class DLM23(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        self.rnn = nn.LSTM(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)


class DLM44(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        self.rnn = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        # x = nn.Linear(1,K)
        # self.shared_log_align = nn.Parameter(x.weight.T)
        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        W = config.window_size
        assert W >=1, config

        x = nn.Linear(W,  config.embed_dim).to(self.device)
        self.w_init_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(W,  1).to(self.device)
        self.w_init_logp     =  nn.Parameter(x.weight.T)


class DLM45(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        self.rnn = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        # x = nn.Linear(1,K)
        # self.shared_log_align = nn.Parameter(x.weight.T)
        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)


class DLM46(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        self.rnn = GRUMinimal(device, RNNConfig(input_size=E,hidden_size=E,num_layers=1))
        # if config.window_size != 0
        # self.rnn = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        K = config.kernel_size
        assert K >=1,config
        # if self.window_size!=-1:
        #     '''
        #     Ablation studies
        #     '''
        #     self.rnn.mutate(self.window_size)
        if ',' in config.loss_name:
            for k in config.loss_name.split(','):
                self.rnn.mutate(int(k))


        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        # x = nn.Linear(1,K)
        # self.shared_log_align = nn.Parameter(x.weight.T)
        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        W = config.window_size
        assert W >=1, config

        x = nn.Linear(W,  config.embed_dim).to(self.device)
        self.w_init_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(W,  1).to(self.device)
        self.w_init_logp     =  nn.Parameter(x.weight.T)


from markov_lm.Model_rnn import MGRU,RNNConfig,MGRUWithAttention
class DLM47(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        if '10' in config.loss_name.split(',') or '11' in config.loss_name:
            self.rnn = MGRUWithAttention(device, RNNConfig(input_size=E,hidden_size=E,num_layers=1,max_step=config.n_step,head_size=config.window_size))
            # import pdb; pdb.set_trace()
        else:
            self.rnn = MGRU(device, RNNConfig(input_size=E,hidden_size=E,num_layers=1,max_step=config.n_step,head_size=config.window_size))
        # if config.window_size != 0
        # self.rnn = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        K = config.kernel_size
        assert K >=1,config
        if ',' in config.loss_name:
            for k in config.loss_name.split(','):
                self.rnn.mutate(int(k))

        # import pdb; pdb.set_trace()

        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        # x = nn.Linear(1,K)
        # self.shared_log_align = nn.Parameter(x.weight.T)
        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)




class DLM43(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        self.rnn = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        # # x = nn.Linear(1,K)
        # # self.shared_log_align = nn.Parameter(x.weight.T)
        # x = nn.Linear(K,  config.embed_dim).to(self.device)
        # self.k_vector   =  nn.Parameter(x.weight.T)
        # x = nn.Linear(K,  1).to(self.device)
        # self.k_scale   =  nn.Parameter(x.weight.T)
        # # self.embed_to_logp = nn.Parameter(x.weight.T)
        # self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        # # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)



class DLM26(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        self.rnn = nn.LSTM(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        # x = nn.Linear(1,K)
        # self.shared_log_align = nn.Parameter(x.weight.T)
        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

    def sample_token(self,B,T,return_logp=False, prompt=None):

        h0 = self.get_default_init(None,B)
        c0 = self.get_default_init(None,B)
        E = self.config.embed_dim
        T = self.config.n_step

        if prompt is not None:
            # target_embed = self.embed(prompt)
            # target_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            # target_embed_parent = target_embed_parent/target_embed_parent.std(dim=1,keepdims=True)
            #
            # h0 = torch.ones([1,B,E],device=self.device)
            # c0 = torch.ones([1,B,E],device=self.device)
            #
            # prior = 0.
            # prior_sum = 0.
            # # (B, T, E)
            # if CLS_NAME in 'DLM23 DLM26'.split():
            #
            #     # target_embed_parent[:,0:1] = h0.transpose(1,0)
            #
            #     # _ , (h0,c0) = self.rnn(  h0.transpose(1,0),(h0,c0))
                # target_embed_parent[:,0:1] = h0.transpose(1,0)
                # yp, (h1,c1) = self.rnn(target_embed_parent,(h0,c0))
                # yp = yp / yp.std(dim=-1, keepdims=True)

            # for i in range(T):

            prompt_embed = self.embed(prompt)
            prompt_embed = torch.cat([torch.ones((B,1,E),device=self.device), prompt_embed],dim=1)
            # [:,:-1]

            prompt_embed = self.std_norm(prompt_embed,-1)

            offset = prompt.shape[1]

            ypp, (h0,c0) = self.rnn(prompt_embed,(h0,c0))
            ypp = ypp[:,-1:]

            if 0:
                ypp, (h0,c0) = self.rnn(prompt_embed[:,:-1],(h0,c0))
                ypp = ypp[:,-1:]

            # [DEBUG]
            if 0:
                ypp, (h0,c0) = self.rnn(prompt_embed[:,:-1],(h0,c0))


                ii = 7
                seq2sent = lambda x,self=self:[self.dataset.tgt_wordize(vv) for vv in x]
                xprompt_ypp = ypp
                expected = 23
                old_rnn_state = (ypp,(h0,c0))

                ypo,(h1,h1) = self.rnn(prompt_embed[:,-1:],(h0,c0))
                ypp = ypp[:,-1:]
        else:
            tok_embed = h0.transpose(0,1)
            tok_embed = self.std_norm(tok_embed,-1)
            ypp, (h0,c0) = self.rnn(  tok_embed,(h0,c0))
            offset = 0

        sampled = torch.zeros((B,T),dtype=torch.long,device=self.device)
        lps     = torch.zeros((B,T),dtype=torch.float,device=self.device)

        for i in range(T):
            if i<offset:
                tok = prompt[:,i:i+1]
                lp = tok*0
            else:
                # import pdb; pdb.set_trace()

                # logp = self._hidden_to_cats(ypp,ret='full',target=None).log_softmax(-1)
                # if prompt
                if 0:
                    'DEBUG'
                    ypp = xprompt_ypp[:,-1:]
                    ypp,(h0,c0) = old_rnn_state; ypp = ypp[:,-2:-1]
                    lp, tok = self.sample_token_from_latent(ypp, return_logp=True); xp=self._hidden_to_cats(ypp,ret='full',target=None).exp(); tok[7][0],lp[7],xp[7][0][23]
                    '''
                    (tensor(23), tensor([-0.9267], grad_fn=<SelectBackward0>), tensor(0.3958, grad_fn=<SelectBackward0>))

                    From loss evaluation
                    tensor(-0.0102, grad_fn=<SelectBackward0>)
                    tensor([ 0.1841, -0.6415,  0.0125,  1.4693, -0.0258], grad_fn=<SliceBackward0>)
                    '''

                    tok_embed = self.embed(tok)
                    tok_embed = self.std_norm(tok_embed,-1)

                    h0,c0 = old_rnn_state[1]
                    ypp, (h0,c0) = self.rnn(tok_embed,(h0,c0))


                    ypo[7,0,:3],ypp[7,0,:3]


                ypp = self.std_norm(ypp,-1)
                logp = self._hidden_to_cats(ypp,ret='full',target=None).log_softmax(-1)
                lp,tok =  self.sample_logp(logp,-1,return_logp=True)

                # if prompt is not None and prompt[7][9]==23 and i-offset==0:
                #     print(logp[7,0,23])
                # import pdb; pdb.set_trace()

#
                # lp, tok = self.sample_token_from_latent(ypp, return_logp=True);
                tok_embed = self.embed(tok)
                tok_embed = self.std_norm(tok_embed,-1)
                ypp, (h0,c0) = self.rnn(tok_embed,(h0,c0))
                # lp, tok = self.sample_logp

                # tok[7][0],lp[7]
                # ypp = xprompt_ypp[:,-2:-1]
                # import pdb; pdb.set_trace()

                # else:?
            sampled[:,i:i+1] = tok
            lps[:,i:i+1] = lp

        if return_logp:
            return lps,sampled
        else:
            return sampled

    def sample_token_from_latent(self, h1r,return_logp=False,prompt=None):
        yp = h1r

        ## (B,T,C) tensor
        logp = self._hidden_to_cats(yp,ret='full',target=None).log_softmax(-1)
        return self.sample_logp(logp,-1,return_logp)

class DLM40(DLMPrototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        self.submodel, self.embed__ = self.get_mingpt_model(config)
    @property
    def embed(self):
        return self.submodel.transformer.wte

    @staticmethod
    def get_mingpt_model(config):
        from markov_lm.mingpt.model import GPT

        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        W = config.window_size
        # ?kernel_size
        D = config.depth
        G = config.graph_dim

        # assert K >=1,config

        model_config = GPT.get_default_config()
        model_config.model_type = None
        model_config.vocab_size = G # openai's model vocabulary
        model_config.block_size = config.n_step  # openai's model block_size (i.e. input context length)
        # n_layer=12, n_head=12, n_embd=768
        model_config.n_layer = D
        model_config.n_head = W
        model_config.n_embd = E
        submodel = GPT(model_config)
        embed = submodel.transformer.wte
        return submodel,embed
        # self.embed.weight.data.normal_(mean=0.0, std=1E-5)

class DLM41(DLM40):
    # Prototype):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = config.embed_dim
        K = config.kernel_size
        assert K >=1,config
        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)

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


class DLM25(DLMPrototype):
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
    loss =  (val * target_notnull ).sum(-1) / target_notnull.sum(-1).clip(1,None)
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
