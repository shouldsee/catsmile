import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from dataclasses import dataclass
from markov_lm.Model_gmm import AbstractLayerConfig
# from transformers.models.bert.modeling_bert import BertLayer,BertConfig
from markov_lm.nlp.model_seq2seq import Seq2SeqWithAttention
from markov_lm.Model_rnn import GRUMinimal,RNNConfig
from markov_lm.util_html import register_object_method

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
    @staticmethod
    def callback_checkpoint(conf, model, target):
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
    _custom_hidden_to_cats = 0
    @staticmethod
    def std_norm(v, dim,keepdims=True):
        v = v/(1E-10+v.std(dim=dim,keepdims=keepdims))
        return v

    # def callback_checkpoint

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
        self._init_attach_method()

    def _init_attach_method(self):
        '''
        Dynamically attaches method to new object to allows easier code sharing
        '''
        # self._hidden_to_cats()
        CLS_NAME = self.__class__.__name__
        '''
        Setting _hidden_to_cats()
        '''
        if CLS_NAME in 'DLM23 DLM43'.split():
            @register_object_method(self)
            def _hidden_to_cats(self, yp, target, ret):
                '''
                Convert hidden vector to explicit categorical distribution
                '''
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

        elif CLS_NAME in ('DLM26 DLM28 DLM29 DLM44 DLM46 DLM47 DLM50 DLM51 DLM52 DLM53 DLM54 DLM55 DLM56 DLM57 DLM58 DLM59 DLM60 DLM61 DLM62 DLM63 '
            'DLM64 DLM65 DLM66 DLM67 DLM68 DLM69 DLM70 DLM71 DLM72').split():
            @register_object_method(self)
            def _hidden_to_cats(self, yp, target, ret):
                # CLS_NAME = self.__class__.__name__
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
                    lp = (model_w + kp)
                    return lp
                else:
                    raise NotImplementedError( f'{ret!r} for {CLS_NAME!r}' )
        elif self._custom_hidden_to_cats:
            pass
        else:
            raise NotImplementedError( CLS_NAME )

    def unembed(self,x):
        y = x.matmul(self.embed.weight.T)
        return y

    def grad_loss(self,item):
        return self._loss(item, 'grad_loss')

    def forward(self,item):
        return self._loss(item,'forward')

    def loss(self,item,rng=None):
        return self._loss(item, 'loss',rng)

    @staticmethod
    def sample_logp(lp,dim,return_logp=False, is_log=True,n = 1):
        # assert dim==-1,"Not impl for dim=%d"%dim
        if is_log:
            xp = lp.softmax(dim)
        else:
            xp = lp
        # p = lp.log_softmax(dim).exp()
        '''
        Sampling bug....
        '''
        # _,idx = (torch.rand( xp.shape, device=xp.device) < xp.cumsum(dim)).max(dim) ### [A critical bug!!!]

        _,idx = (torch.rand( xp.shape[:-1] + (1,n), device=xp.device) < xp.cumsum(-1).unsqueeze(-1)).max(-2)
        lp = torch.gather( lp.unsqueeze(-1).expand( *((-1,)*len(lp.shape)+(n,)) ), index=idx.unsqueeze(-2),dim=-2).squeeze(-2)

        if n == 1:
            lp = lp.squeeze(-1)
            idx = idx.squeeze(-1)
        rand_sampled_output = idx #.clip(dataset.tgt_vocab.offset,None)
        # import pdb; pdb.set_trace()
        if return_logp:
            return lp, rand_sampled_output
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

            def s(xx):
                if xx.isnan().any():
                    import pdb; pdb.set_trace()

            if 1:

                '''
                parse to get a state distrib
                '''
                ### (B,T,E)
                encd, h1 = self.encode(target_embed,h0)

                ### (1,T,E,K)
                '''
                Sample a sequence (w_i,l_i) from parser
                First sample whether to skip.
                Then for the none-null postion, sample the actual state (maybe just uses the projected context vector?).
                '''
                self.encode
                self.e_to_null
                self.e_to_token
                self.w_embed
                self.w_unembed
                self.prior_logp
                self.H
                self.h_matrix
                # (H, E, E)



                ### (B,T,W)

                ### two state gate.. controls whether the vector is passed into the prior lstm.
                ### it seeems the prior lstm does not needs to emit discrete sequence?
                ### to be safe, uses discrete seq to model prior seq for now.

                encd = self.std_norm(encd,-1)

                lp_is_skip = self.e_to_null(encd).log_softmax(-1)
                lp_token   = self.e_to_token(encd).log_softmax(-1)

                lp_is_skip_s, idx_is_skip = self.sample_logp( lp_is_skip, return_logp=True, dim=-1)
                lp_token_s,   idx_token = self.sample_logp( lp_token[:,:,:-1], return_logp=True, dim=-1)
                idx_token = idx_token + 1

                idx_is_token = (1-idx_is_skip).bool()
                idx_emit     = idx_is_token * idx_token
                ## (B,T,)

                ### if no token emitted, then omit the logp required to encode token
                lp_emit_s = idx_is_token * ( (1E-10+1-lp_is_skip_s.exp()).log() + lp_token_s ) + (~idx_is_token) * lp_is_skip_s
                lp_emit_s, idx_emit
                ### ends sample generation


                lp_emit_s_embed = self.w_embed(idx_emit)

                # target_embed_parent =
                _E = lp_emit_s_embed.shape[2]
                lp_emit_s_embed_parent =  torch.cat([torch.ones((B,1,_E),device=self.device), lp_emit_s_embed],dim=1)[:,:-1]
                lp_emit_s_pred, _ = self.prior_logp(self.std_norm(lp_emit_s_embed_parent  ,-1), h0 )
                # lp_emit_s_pred, _ = self.prior_logp(self.std_norm(lp_emit_s_embed_parent * idx_is_token.unsqueeze(-1) + ~idx_is_token.unsqueeze(-1) * torch.ones_like(lp_emit_s_embed_parent,device=self.device) ,-1), h0 )
                # lp_emit_s_pred, _ = self.prior_logp(lp_emit_s_embed_parent, h0 )

                s(lp_emit_s_embed_parent)
                # s(lp_emi)
                # s(lp_emit_s_embed_parent)
                ## (P, E)
                _x  =  lp_emit_s_pred[idx_is_token,:]
                unb = self.w_unembed(self.std_norm(_x,-1)).log_softmax(-1)

                # unb = self.std_norm(_x,-1) @ (self.std_norm(self.w_embed.weight,-1).T)
                # unb =unb.log_softmax(-1)
                # s(unb)
                ## (P,)
                prior_lp = torch.gather(unb,index=idx_emit[idx_is_token][:,None],dim=-1).squeeze(-1)
                # import pdb; pdb.set_trace()
                _x = torch.zeros_like(idx_is_token,device=self.device,dtype=torch.float)
                lp_prior = _x.masked_scatter_(idx_is_token,prior_lp)
                # xx = lp_prior
                # lp_prior = prior_lp

                'Calculate decode proba given a sequence and a convolution expansion'
                '''
                needs to calculates w,delta for each character position...
                seems we need a for-loop here.. needs fast impl.

                forces to backfill from a token pos?

                if is_token, then memorize token, empty counter,
                print token
                print counter
                inc counter
                step
                '''

                '''
                H: maximum word length
                '''
                H = self.H
                # H = self.config.H
                tok= torch.ones((B,1),device=self.device,dtype=torch.long)
                ct = torch.zeros((B,1),device=self.device, dtype=torch.long) - 1

                tok_arr= torch.zeros((B,T),device=self.device,dtype=torch.long)
                ct_arr = torch.zeros((B,T),device=self.device,dtype=torch.long)
                for t in range(T):
                    t= T-1-t
                    is_tok = idx_is_token[:,t:t+1]
                    tok = is_tok * idx_token[:,t:t+1] + (~is_tok) * tok
                    ct  = is_tok * 0 + (~is_tok)*(ct+1)
                    tok_arr[:,t:t+1] = tok
                    ct_arr[:,t:t+1] = ct
                ### then gets
                ct_arr = ct_arr.clip(None, H - 1)

                ### uses h matrix to score between char and word
                # (H, E, E)
                # (B, T, E, E)
                # import pdb; pdb.set_trace()
                head_left = self.h_matrix[ct_arr] @ self.std_norm(self.w_embed(tok_arr),-1).unsqueeze(-1)
                lps = self.unembed( head_left.squeeze(-1) ).log_softmax(-1)
                lps = torch.gather(lps,index=target.unsqueeze(-1),dim=-1).squeeze(-1)
                lp_decode = lps

                # lp = lp_decode + lp_prior - lp_emit_s
                lp = lp_decode + 1* lp_prior - 0 * lp_emit_s
                # lp = lp_decode + 1* lp_prior + 1 * lp_emit_s
                self.last = dict(lp_decode=lp_decode,lp_prior=lp_prior,lp_emit_s = lp_emit_s)


                # if ret=='encode': return encd
                # if ret=='encode': return tok_arr
                if ret=='encode': return lp_emit_s_embed


            att = torch.stack([lp_decode,lp_prior,lp_emit_s],dim=-1)

            # print(att.mean(dim=(0,1)))
            # # if att.mean(dim=(0,1)).min()<-20:
            # if att.mean(dim=(0,1))[-1]<-20:
            #     import pdb; pdb.set_trace()

            lp = lp

            # lp = self._hidden_to_cats(yp,ret='target',target=target)
            # att = lp.softmax(dim=2).transpose(2,1)
            # lp  = lp.logsumexp(-1)
            # logp_sum = (lp * target_notnull).sum(dim=1)
            logp_sum = ((lp + prior) * target_notnull).sum(dim=1)
            logp_sum = logp_sum + prior_sum
            # logp_sum = logp_sum + prior


        elif CLS_NAME in 'DLM68 DLM69'.split():
            '''
            This branch predicts a discrete mixer instead of a vector to speeds up final softmax
            '''

            target_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            target_embed_parent = target_embed_parent/target_embed_parent.std(dim=1,keepdims=True)

            h0 = torch.ones([1,B,E],device=self.device)
            c0 = torch.ones([1,B,E],device=self.device)

            prior = 0.
            prior_sum = 0.

            def s(xx):
                if xx.isnan().any():
                    import pdb; pdb.set_trace()

            if 1:

                '''
                parse to get a state distrib
                '''
                ### (B,T,E)
                encd, h1 = self.encode(target_embed,h0)

                ### (1,T,E,K)
                '''
                Sample a sequence (w_i,l_i) from parser
                First sample whether to skip.
                Then for the none-null postion, sample the actual state (maybe just uses the projected context vector?).
                '''
                self.encode
                self.e_to_null
                self.e_to_token
                self.w_embed
                self.w_unembed
                self.prior_logp
                self.H
                self.h_matrix
                # (H, E, E)
                ### (B,T,W)

                ### two state gate.. controls whether the vector is passed into the prior lstm.
                ### it seeems the prior lstm does not needs to emit discrete sequence?
                ### to be safe, uses discrete seq to model prior seq for now.

                encd = self.std_norm(encd,-1)

                lp_is_skip = self.e_to_null(encd).log_softmax(-1)
                lp_token   = self.e_to_token(encd).log_softmax(-1)

                lp_is_skip_s, idx_is_skip = self.sample_logp( lp_is_skip, return_logp=True, dim=-1)
                lp_token_s,   idx_token = self.sample_logp( lp_token[:,:,:-1], return_logp=True, dim=-1)
                idx_token = idx_token + 1

                idx_is_token = (1-idx_is_skip).bool()
                idx_emit     = idx_is_token * idx_token
                ## (B,T,)

                ### if no token emitted, then omit the logp required to encode token
                lp_emit_s = idx_is_token * ( (1E-10+1-lp_is_skip_s.exp()).log() + lp_token_s ) + (~idx_is_token) * lp_is_skip_s
                lp_emit_s, idx_emit
                ### ends sample generation


                lp_emit_s_embed = self.w_embed(idx_emit)

                # target_embed_parent =
                _E = lp_emit_s_embed.shape[2]
                lp_emit_s_embed_parent =  torch.cat([torch.ones((B,1,_E),device=self.device), lp_emit_s_embed],dim=1)[:,:-1]
                lp_emit_s_pred, _ = self.prior_logp(self.std_norm(lp_emit_s_embed_parent  ,-1), h0 )
                # lp_emit_s_pred, _ = self.prior_logp(self.std_norm(lp_emit_s_embed_parent * idx_is_token.unsqueeze(-1) + ~idx_is_token.unsqueeze(-1) * torch.ones_like(lp_emit_s_embed_parent,device=self.device) ,-1), h0 )
                # lp_emit_s_pred, _ = self.prior_logp(lp_emit_s_embed_parent, h0 )

                s(lp_emit_s_embed_parent)
                # s(lp_emi)
                # s(lp_emit_s_embed_parent)
                ## (P, E)
                _x  =  lp_emit_s_pred[idx_is_token,:]
                unb = self.w_unembed(self.std_norm(_x,-1)).log_softmax(-1)

                # unb = self.std_norm(_x,-1) @ (self.std_norm(self.w_embed.weight,-1).T)
                # unb =unb.log_softmax(-1)
                # s(unb)
                ## (P,)
                prior_lp = torch.gather(unb,index=idx_emit[idx_is_token][:,None],dim=-1).squeeze(-1)
                # import pdb; pdb.set_trace()
                _x = torch.zeros_like(idx_is_token,device=self.device,dtype=torch.float)
                lp_prior = _x.masked_scatter_(idx_is_token,prior_lp)
                # xx = lp_prior
                # lp_prior = prior_lp

                'Calculate decode proba given a sequence and a convolution expansion'
                '''
                needs to calculates w,delta for each character position...
                seems we need a for-loop here.. needs fast impl.

                forces to backfill from a token pos?

                if is_token, then memorize token, empty counter,
                print token
                print counter
                inc counter
                step
                '''

                '''
                H: maximum word length
                '''
                H = self.H
                # H = self.config.H
                tok= torch.ones((B,1),device=self.device,dtype=torch.long)
                ct = torch.zeros((B,1),device=self.device, dtype=torch.long) - 1

                tok_arr= torch.zeros((B,T),device=self.device,dtype=torch.long)
                ct_arr = torch.zeros((B,T),device=self.device,dtype=torch.long)
                for t in range(T):
                    t= T-1-t
                    is_tok = idx_is_token[:,t:t+1]
                    tok = is_tok * idx_token[:,t:t+1] + (~is_tok) * tok
                    ct  = is_tok * 0 + (~is_tok)*(ct+1)
                    tok_arr[:,t:t+1] = tok
                    ct_arr[:,t:t+1] = ct
                ### then gets
                ct_arr = ct_arr.clip(None, H - 1)

                ### uses h matrix to score between char and word
                # (H, E, E)
                # (B, T, E, E)
                # import pdb; pdb.set_trace()
                head_left = self.h_matrix[ct_arr] @ self.std_norm(self.w_embed(tok_arr),-1).unsqueeze(-1)
                lps = self.unembed( head_left.squeeze(-1) ).log_softmax(-1)
                lps = torch.gather(lps,index=target.unsqueeze(-1),dim=-1).squeeze(-1)
                lp_decode = lps


                # if ret=='encode': return encd
                # if ret=='encode': return tok_arr
                if ret=='encode': return lp_emit_s_embed


            lp_decode = lp_decode * target_notnull

            att = torch.stack([lp_decode,lp_prior,lp_emit_s],dim=-1)

            'branch'
            if CLS_NAME in 'DLM68'.split():
                lpv = lp_decode + 1* lp_prior - 1 * lp_emit_s
                grad_lp  =  lpv + lpv.detach() * lp_emit_s
                v_lp = lpv

                self.last = dict(lp_decode=lp_decode,lp_prior=lp_prior,lp_emit_s = lp_emit_s)


                logp_sum = ((v_lp + prior) * target_notnull).sum(dim=1)
                logp_per_token = logp_sum/ target_notnull.sum(dim=1)
                v_loss = -logp_per_token

                logp_sum = ((grad_lp + prior) * target_notnull).sum(dim=1)
                logp_per_token = logp_sum/ target_notnull.sum(dim=1)
                grad_loss = -logp_per_token

                if ret=='forward':
                    return v_loss,att
                elif ret in 'loss'.split():
                    return v_loss
                elif ret in 'grad_loss'.split():
                    return grad_loss
                elif ret=='loss_per_loc':
                    return -v_lp,target_notnull
                else:
                    raise NotImplementedError(f'''{ret} for ._loss()''')

            elif CLS_NAME in 'DLM69'.split():
                '''
                Calculation must be done at sequence level
                '''
                lpv = lp_decode * target_notnull + 0* lp_prior - lp_emit_s
                lpv_sum = lpv.sum(1)
                lps_sum = lp_emit_s.sum(1)
                v_lp = lpv_sum[:,None]
                grad_lp = (lpv_sum + lpv_sum.detach()*lps_sum)[:,None]


                self.last = dict(lp_decode=lp_decode,lp_prior=lp_prior,lp_emit_s = lp_emit_s)

                v_loss = - v_lp.squeeze(-1) / T
                grad_loss = - grad_lp.squeeze(-1) / T
                print(att.mean((0,1)))

                # logp_sum = ((v_lp + prior) * target_notnull).sum(dim=1)
                # logp_per_token = logp_sum/ target_notnull.sum(dim=1)
                # v_loss = -logp_per_token
                #
                # logp_sum = ((grad_lp + prior) * target_notnull).sum(dim=1)
                # logp_per_token = logp_sum/ target_notnull.sum(dim=1)
                # grad_loss = -logp_per_token

                if ret=='forward':
                    return v_loss,att
                elif ret in 'loss'.split():
                    return v_loss
                elif ret in 'grad_loss'.split():
                    return grad_loss
                elif ret=='loss_per_loc':
                    return -v_lp,target_notnull
                else:
                    raise NotImplementedError(f'''{ret} for ._loss()''')
            else:
                raise NotImplementedError(f'''{CLS_NAME}''')

        elif CLS_NAME in 'DLM70 DLM71 DLM72'.split():
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
            if 1:

                '''
                Encode right to left, decode left to right
                '''
                h1, _ = self.embed_to_latent(target_embed)

                '''
                Needs to sample from h1
                '''

                mu_p   = self.h_prior[0:1]
                beta_p = self.h_prior[1:2].exp() * 10
                # if self.meta.get('epoch',0)<=5:
                #     prior_w = 0.
                # else:
                prior_w = 1.
                mu   = h1 @self.h_to_mu
                # mu   = h1
                beta = self.h_post[1:2].exp() * 10

                '''
                Resampling
                '''
                h1r = torch.normal( 0, 1, mu.shape, device=self.device) / beta + mu
                yp = self.latent_to_emittor(self.gain(h1r))


                # import pdb; pdb.set_trace()
                '''
                Calculate extra terms due to posterior correction from prior dist
                '''
                if CLS_NAME in 'DLM70'.split():
                    lp_prior  = ( -0.5*((h1r - mu_p)*beta_p).square() + beta_p.log() - 0.5 * math.log(2*math.pi) ).sum(-1)
                elif CLS_NAME in 'DLM71'.split():

                    h1rp = torch.cat([torch.ones((B,1,E),device=self.device),h1r[:,:-1]],dim=1)
                    h1rp = self.std_norm(h1rp,-1)
                    h1ro,_ = self.rnn_prior(h1rp,h0)
                    mu_p = h1ro

                    # h1rp = torch.cat([mu_p[:,0:1].repeat((B,1,1)),h1r[:,:-1]],dim=1)
                    # mu_p= h1rp
                    # beta_p = beta_p

                    lp_prior  = ( -0.5*((h1r - mu_p)*beta_p).square() + beta_p.log() - 0.5 * math.log(2*math.pi) ).sum(-1)
                elif CLS_NAME in 'DLM72'.split():

                    h1rp = torch.cat([torch.ones((B,1,E),device=self.device),h1r[:,:-1]],dim=1)
                    ### (B, T, E, W)
                    h1rph = torch.tensordot( h1rp,self.e_to_we,1)
                    # h1rph

                    # h1rp = self.std_norm(h1rp,-1)
                    # h1ro,_ = self.rnn_prior(h1rp,h0)
                    # mu_p = h1ro
                    W = self.config.window_size
                    beta_p = beta_p.ravel()[0]
                    # lp_prior  = ( -0.5*((h1r.unsqueeze(-1) - h1rph )*beta_p.unsqueeze(-1)).square() + beta_p.log() - 0.5 * math.log(2*math.pi) ).sum(-1)
                    lp_prior  = ( -0.5*((h1r.unsqueeze(-1) - h1rph )*beta_p).square() + beta_p.log() - 0.5 * math.log(2*math.pi))
                    # att = lp_prior.
                    lp_prior =  (math.log(1./W) + lp_prior.sum(-2))
                    att = lp_prior.softmax(-1)
                    lp_prior = lp_prior.logsumexp(-1)

                    # if ret=='encode': return att
                    # if ret=='encode': return
                    if ret=='encode': return h1r * target_notnull.unsqueeze(-1)

                else:
                    raise NotImplementedError
                    # -0.5 * (h1ro - h1r

                lp_encode = ( -0.5*((h1r - mu)*beta).square() + beta.log() - 0.5 * math.log(2*math.pi) ).sum(-1)

                # if ret=='encode': return mu * target_notnull.unsqueeze(-1)
                if ret=='encode': return h1r * target_notnull.unsqueeze(-1)

            lp_decode = self._hidden_to_cats(yp,ret='target',target=target)
            lp_decode = torch.gather(lp_decode,index=target.unsqueeze(-1),dim=-1).squeeze(-1)
            # lps = lp_decode * target_notnull + lp_prior - lp_encode
            '''
            Strictly speaking the nodes are not conditionally independent in rnn prior,
            '''


            lpv = lp_decode * target_notnull + 1* lp_prior - 1 * lp_encode
            grad_lp  =  lpv + lpv.detach() * lp_encode
            v_lp = lpv

            att = torch.stack([lp_decode*target_notnull,lp_prior,lp_encode],dim=-1)
            att = att *target_notnull.unsqueeze(-1)
            if getattr(self,'debug',0)>=1:
                print(f'''{beta.mean():.3f} {beta_p.mean():.3f}''' )
                print(att.mean((0,1)))


            self.last = dict(lp_decode=lp_decode,lp_prior=lp_prior,lp_encode = lp_encode)

            logp_sum = ((v_lp + prior) * target_notnull).sum(dim=1)
            logp_per_token = logp_sum/ target_notnull.sum(dim=1)
            v_loss = -logp_per_token

            logp_sum = ((grad_lp + prior) * target_notnull).sum(dim=1)
            logp_per_token = logp_sum/ target_notnull.sum(dim=1)
            grad_loss = -logp_per_token

            if ret=='forward':
                return v_loss,att
            elif ret in 'loss'.split():
                return v_loss
            elif ret in 'grad_loss'.split():
                return grad_loss
            elif ret=='loss_per_loc':
                return -v_lp,target_notnull
            else:
                raise NotImplementedError(f'''{ret} for ._loss()''')



        elif CLS_NAME in 'DLM28'.split():
            '''
            This branch predicts a discrete mixer instead of a vector to speeds up final softmax
            '''

            target_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            target_embed_parent = target_embed_parent/target_embed_parent.std(dim=1,keepdims=True)

            h0 = torch.ones([1,B,E],device=self.device)
            c0 = torch.ones([1,B,E],device=self.device)

            prior = 0.
            prior_sum = 0.

            yp, h1 = self.rnn(target_embed_parent, h0)
            yp = yp / yp.std(dim=-1, keepdims=True)
            if ret=='encode': return yp



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

        elif CLS_NAME=='DLM73':
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
            # return -logp_sum
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

class DLM73(DLMPrototype):
    '''
    Choosing between two models.
    Renamed from DLM28
    '''
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


class DLM70(DLM57):
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
        # self.gain = nn.Identity()
        self.gain = nn.Linear(E,E).to(self.device)

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
        return None
    def sample_token_from_latent(self,h1):
        return None


class DLM71(DLM57):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # K = config.kernel_size
        # assert K >=1,config
        T = config.n_step
        # data_dim
        E            = config.embed_dim
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_prior = nn.RNN(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)
        self.gain = nn.Linear(E,E).to(self.device)

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
        return None
    def sample_token_from_latent(self,h1):
        return None




class DLM72(DLM57):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        K = config.kernel_size
        assert K >=1,config
        T = config.n_step
        E            = config.embed_dim
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.rnn_prior = nn.RNN(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)
        self.gain = nn.Linear(E,E).to(self.device)

        K = config.kernel_size
        assert K >=1,config
        W = config.window_size
        # x
        x = nn.Linear(E,W*E).to(self.device)
        self.e_to_we   =  nn.Parameter(x.weight.reshape((E,E,W)))

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
        return None
    def sample_token_from_latent(self,h1):
        return None



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
        K = config.kernel_size
        assert K >=1,config

        W = config.window_size
        assert W>=1 ,config

        # self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.encode = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        # self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)


        x = nn.Linear(E,  2).to(self.device)
        self.e_to_null   =  x
         # nn.Parameter(x.weight.T)


        x = nn.Linear(E,  W).to(self.device)
        self.e_to_token   =  x
        # nn.Parameter(x.weight.T)

        # x = nn.Linear(W,  E,bias=False).to(self.device)
        x = nn.Embedding(W,  E).to(self.device)
        self.w_embed   =  x
        self.w_unembed = lambda x: (x @ self.w_embed.weight.T) #.log_softmax(-1)


        self.prior_logp = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )

        self.H = H= 30

        x = nn.Linear(H,  E*E).to(self.device)
        self.h_matrix   =  nn.Parameter(x.weight.T.reshape((H,E,E)))

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)

    def sample_token(self,B,T,prompt=None):
        return None
    def sample_token_from_latent(self,h1):
        return None

class DLM68(DLM57):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        T = config.n_step
        E            = config.embed_dim
        K = config.kernel_size
        assert K >=1,config
        W = config.window_size
        assert W>=1 ,config

        self.encode = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)


        x = nn.Linear(E,  2).to(self.device)
        self.e_to_null   =  x
        x = nn.Linear(E,  W).to(self.device)
        self.e_to_token   =  x

        x = nn.Embedding(W,  E).to(self.device)
        self.w_embed   =  x
        self.w_unembed = lambda x: (x @ self.w_embed.weight.T) #.log_softmax(-1)

        self.prior_logp = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )

        self.H = H= 30
        x = nn.Linear(H,  E*E).to(self.device)
        self.h_matrix   =  nn.Parameter(x.weight.T.reshape((H,E,E)))

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)

    def sample_token(self,B,T,prompt=None):
        return None
    def sample_token_from_latent(self,h1):
        return None

class DLM69(DLM57):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        T = config.n_step
        E            = config.embed_dim
        K = config.kernel_size
        assert K >=1,config
        W = config.window_size
        assert W>=1 ,config

        self.encode = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )
        self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)


        x = nn.Linear(E,  2).to(self.device)
        self.e_to_null   =  x
        x = nn.Linear(E,  W).to(self.device)
        self.e_to_token   =  x

        x = nn.Embedding(W,  E).to(self.device)
        self.w_embed   =  x
        self.w_unembed = lambda x: (x @ self.w_embed.weight.T) #.log_softmax(-1)

        self.prior_logp = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=1, )

        self.H = H= 30
        x = nn.Linear(H,  E*E).to(self.device)
        self.h_matrix   =  nn.Parameter(x.weight.T.reshape((H,E,E)))

        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)

    def sample_token(self,B,T,prompt=None):
        return None
    def sample_token_from_latent(self,h1):
        return None


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




class DLM28(DLMPrototype):
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





class DLM100(DLMPrototype):
    _custom_hidden_to_cats = 1

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


    _hidden_to_cats = None
    def _hidden_to_cats(self, yp, target, ret):
        # assert self._custom_hidden_to_cats == 1
        # CLS_NAME = self.__class__.__name__
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
            lp = (model_w + kp)
            return lp
        else:
            raise NotImplementedError( f'{ret!r} for {CLS_NAME!r}' )


    def _loss(self,item, ret):
        CLS_NAME = self.__class__.__name__

        source = item['source'] ### token sequence
        target = item['target'] ### token seq

        # source_embed = self.embed(source)
        # source_len = item['source_len']
        # source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]
        target_embed = self.embed(target)
        # target_isblank = item['target_isblank']
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

        if 1:


            '''
            This branch predicts a discrete mixer instead of a vector to speeds up final softmax
            '''

            target_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device),target_embed],dim=1)[:,:-1]
            # target_embed_parent = target_embed_parent/target_embed_parent.std(dim=1,keepdims=True)
            # target_embed_parent = target_embed_parent/(0.001 + target_embed_parent.std(dim=-1,keepdims=True))

            h0 = torch.ones([1,B,E],device=self.device)
            c0 = torch.ones([1,B,E],device=self.device)

            prior = 0.
            prior_sum = 0.

            yp, h1 = self.rnn(target_embed_parent, h0)
            yp = yp / yp.std(dim=-1, keepdims=True)
            if ret=='encode': return yp



            lp = self._hidden_to_cats(yp,ret='target',target=target)

            att = lp.softmax(dim=2).transpose(2,1)
            lp  = lp.logsumexp(-1)
            # logp_sum = (lp * target_notnull).sum(dim=1)
            logp_sum = ((lp + prior) * target_notnull).sum(dim=1)
            logp_sum = logp_sum + prior_sum
            # logp_sum = logp_sum + prior

        ### normalise by average token count, but sum by batchsize
        # lossVal = lossVal/ (source_notnull.sum() / B)
        # print('[mean]',target_notnull.sum(1).float().mean())

        # logp_per_token = logp_sum/ target_notnull.sum(dim=1)
        # loss = - logp_sum.sum(dim=0) / target_notnull.sum()
        loss = - logp_sum
        # [None]
        # / target_notnull.sum()
        # loss = -logp_per_token

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





class DLM101(DLMPrototype):
    _custom_hidden_to_cats = 1

    def __init__(self,device,config,_=None):
        super().__init__(device,config)

        G = config.graph_dim +1
        self.G = G
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        self.embed      = nn.Embedding(G, E).to(self.device)
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
        self.S = 31
        self.W = W = config.window_size
        assert W >=1,config.window_size
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)

        x = nn.Linear(W,K*E).to(self.device)
        self.kv_k_vector  = nn.Parameter(x.weight.reshape(W,K,E))

        x = nn.Linear(1,K*E).to(self.device)
        self.kv_v_vector  = nn.Parameter(x.weight.reshape(K,E))


        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)


    _hidden_to_cats = None
    def _hidden_to_cats(self, yp, target, ret):
        # assert self._custom_hidden_to_cats == 1
        # CLS_NAME = self.__class__.__name__
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
            lp = (model_w + kp)
            return lp
        else:
            raise NotImplementedError( f'{ret!r} for {CLS_NAME!r}' )


    def _loss(self,item, ret):
        CLS_NAME = self.__class__.__name__

        source = item['source'] ### token sequence
        target = item['target'] ### token seq

        # source_embed = self.embed(source)
        # source_len = item['source_len']
        # source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]
        target_embed = self.embed(target)


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
        G = self.G
        E = self.embed_dim
        target_embed = self.embed(target)
        B = len(target)
        i = 0
        att = torch.ones((B,1,1),device=self.device)


        def s(v):
            if torch.isnan(v).any():
                import pdb; pdb.set_trace()

        if 1:

            target_isblank = item['target_isblank'] | ~target_notnull

            ## (B, T)
            '''
            Segmenting sequences into sub-sequences

            Restruct the input by cutting at blank position.

            bound maximum segment count
            '''



            idx_list = torch.zeros((B,T),dtype=torch.long,device=self.device)
            idx      = torch.zeros((B,1),dtype=torch.long,device=self.device)
            seg_list = torch.zeros((B,T),dtype=torch.long,device=self.device)
            seg      = torch.ones((B,1),dtype=torch.long,device=self.device)
            # seg_list = torch.zeros()
            for i in range(T):
                idx_list[:,i:i+1] = idx
                seg_list[:,i:i+1] = seg * target_notnull[:,i:i+1] + 0 * ~target_notnull[:,i:i+1]
                idx = target_isblank[:,i:i+1] * 0 + (idx+1) * ~target_isblank[:,i:i+1]
                new_segment = target_isblank[:,i:i+1] & target_notnull[:,i:i+1]
                seg = new_segment * (seg+1) + ~new_segment * seg
            # torch.scatter()

            S = self.S
            # W = self.W
            seg_list = seg_list.clip(0,S)
            pos_list = seg_list * W + idx_list
            segments = (G - 1)*torch.ones((B,S*W),dtype=torch.long,device=self.device)
            segments = segments.scatter_(src=target,index=pos_list,dim=1).reshape((B,S,W))
            segments[:,0]=0


            # print(idx_list[0])
            # print(seg_list[0])
            # import pdb; pdb.set_trace()
            ## (B, S, W)

            ## (B, S, W, E ) x (W, E, K) -> (B, S, K)
            # torch.gather()
            # torch.eye()

            '''
            Encoding the segments
            with KV attention
            '''
            v = torch.tensordot( self.embed(segments),self.kv_k_vector.transpose(2,1),2)
            ## (B, S, K) @ (K,E) = (B,S,E)
            segment_embed = v.softmax(-1) @ self.kv_v_vector


            '''
            Run RNN on encoded segments
            '''
            segment_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device), segment_embed],dim=1)[:,:-1]
            # segment_embed_parent = segment_embed_parent/segment_embed_parent.std(dim=-1,keepdims=True)
            h0 = torch.ones([1,B,E],device=self.device)
            c0 = torch.ones([1,B,E],device=self.device)

            prior = 0.
            prior_sum = 0.

            yp, h1 = self.rnn(segment_embed_parent, h0)
            # yp = yp /(1E-3 + yp.std(dim=-1, keepdims=True))
            yp = yp /(yp.std(dim=-1, keepdims=True))
            # s(yup)


            if ret=='encode': return yp

            '''
            Decoding the predicted segments into predicted words
            '''

            ### (B, S, K)
            kp = (yp @ self.kv_v_vector.T).log_softmax(-1)

            ### (W, K, G)
            # config.graph_dim
            # C = self.embed.weight.shapekv_v_vector.shape[0]
            yup = self.unembed( self.kv_k_vector ).log_softmax(-1)

            yup = yup.transpose(0,1).reshape((K,W*G))
            sel = segments + torch.arange(W,device=self.device)[None,None,:]*G
            ## (WG, K) -> (B,S,W,K)
            yupp  = yup.T[sel]
            lp  = kp.unsqueeze(2) + yupp


            ### (B,S,W,K,C)
            # lp = (kp[:,:,None,:,None] + yup[None,None,:,:,:])
            # lp = torch.gather(lp,index=segments.unsqueeze(-1).unsqueeze(-1).repeat((1,1,1,K,1)),dim=-1).squeeze(-1)

            ### (B,S,W,K)->(B,S,W)
            lp = lp.logsumexp(-1)

            '''
            Concat segments back into seqs
            '''

            lp = torch.gather(lp.reshape((B,S*W)),index=pos_list,dim=-1)
            loss_per_loc = lp
            # logp_sum = (lp * target_notnull_segment).sum(dim=(1,2))
            logp_sum = (lp * target_notnull).sum(dim=1)


            # att = lp.softmax(dim=2).transpose(2,1)
            # lp  = lp.logsumexp(-1)
            # # logp_sum = (lp * target_notnull).sum(dim=1)
            # logp_sum = ((lp + prior) * target_notnull).sum(dim=1)
            # logp_sum = logp_sum + prior_sum
            # logp_sum = logp_sum + prior

        ### normalise by average token count, but sum by batchsize
        # lossVal = lossVal/ (source_notnull.sum() / B)
        # print('[mean]',target_notnull.sum(1).float().mean())

        logp_per_token = logp_sum/ target_notnull.sum(dim=1)
        # loss = -logp_per_token
        loss = -logp_sum

        if ret=='forward':
            return loss,att
        elif ret in 'loss grad_loss'.split():
            return loss
        # elif ret=='grad_loss':
        #     return loss
        elif ret=='loss_per_loc':
            return -loss_per_loc,target_notnull
        else:
            raise NotImplementedError(f'''{ret} for ._loss()''')



class DLM102(DLMPrototype):
    _custom_hidden_to_cats = 1

    def __init__(self,device,config,_=None):
        super().__init__(device,config)

        G = config.graph_dim +1
        self.G = G
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        self.embed      = nn.Embedding(G, E).to(self.device)
        self.rnn      = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        self.rnn_word_enc = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        self.rnn_word_dec = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

        # x = nn.Linear(1,K)
        # self.shared_log_align = nn.Parameter(x.weight.T)
        x = nn.Linear(K,  config.embed_dim).to(self.device)
        self.k_vector   =  nn.Parameter(x.weight.T)
        x = nn.Linear(K,  1).to(self.device)
        self.k_scale   =  nn.Parameter(x.weight.T)
        self.S = 31
        self.W = W = config.window_size
        assert W >=1,config.window_size
        # self.embed_to_logp = nn.Parameter(x.weight.T)
        self.embed_to_logp  = nn.Linear(config.embed_dim, K).to(self.device)

        x = nn.Linear(W,K*E).to(self.device)
        self.kv_k_vector  = nn.Parameter(x.weight.reshape(W,K,E))

        x = nn.Linear(1,K*E).to(self.device)
        self.kv_v_vector  = nn.Parameter(x.weight.reshape(K,E))


        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)


    _hidden_to_cats = None
    def _hidden_to_cats(self, yp, target, ret):
        # assert self._custom_hidden_to_cats == 1
        # CLS_NAME = self.__class__.__name__
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
            lp = (model_w + kp)
            return lp
        else:
            raise NotImplementedError( f'{ret!r} for {CLS_NAME!r}' )


    def _loss(self,item, ret):
        CLS_NAME = self.__class__.__name__

        source = item['source'] ### token sequence
        target = item['target'] ### token seq

        # source_embed = self.embed(source)
        # source_len = item['source_len']
        # source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]
        target_embed = self.embed(target)


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
        G = self.G
        E = self.embed_dim
        target_embed = self.embed(target)
        B = len(target)
        i = 0
        att = torch.ones((B,1,1),device=self.device)


        def s(v):
            if torch.isnan(v).any():
                import pdb; pdb.set_trace()

        if 1:

            target_isblank = item['target_isblank'] | ~target_notnull

            ## (B, T)
            '''
            Segmenting sequences into sub-sequences

            Restruct the input by cutting at blank position.

            bound maximum segment count
            '''



            idx_list = torch.zeros((B,T),dtype=torch.long,device=self.device)
            idx      = torch.zeros((B,1),dtype=torch.long,device=self.device)
            seg_list = torch.zeros((B,T),dtype=torch.long,device=self.device)
            seg      = torch.ones((B,1),dtype=torch.long,device=self.device)
            # seg_list = torch.zeros()
            for i in range(T):
                idx_list[:,i:i+1] = idx
                seg_list[:,i:i+1] = seg * target_notnull[:,i:i+1] + 0 * ~target_notnull[:,i:i+1]
                idx = target_isblank[:,i:i+1] * 0 + (idx+1) * ~target_isblank[:,i:i+1]
                new_segment = target_isblank[:,i:i+1] & target_notnull[:,i:i+1]
                seg = new_segment * (seg+1) + ~new_segment * seg
            # torch.scatter()

            S = self.S
            # W = self.W
            seg_list = seg_list.clip(0,S)
            pos_list = seg_list * W + idx_list
            segments = (G - 1)*torch.ones((B,S*W),dtype=torch.long,device=self.device)
            segments = segments.scatter_(src=target,index=pos_list,dim=1).reshape((B,S,W))
            segments[:,0]=0


            # print(idx_list[0])
            # print(seg_list[0])
            # import pdb; pdb.set_trace()

            ## (B, S, W)
            '''
            Encoding the segments
            with lower-level RNN
            with KV attention
            '''
            ## (B, S, W, E ) x (W, E, K) -> (B, S, K)
            segments_embed_seq = self.embed(segments)
            segments_embed_seq = segments_embed_seq.reshape((B*S,W,E))

            h0 = torch.ones([1,B*S,E],device=self.device)
            v1 , h1 = self.rnn_word_enc(segments_embed_seq,h0)
            segment_embed = h1.reshape((B,S,E))


            '''
            Run RNN on encoded segments

            Make sure the encoding is calculated from left segments
            '''
            segment_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device), segment_embed],dim=1)[:,:-1]
            # segment_embed_parent = segment_embed_parent/segment_embed_parent.std(dim=-1,keepdims=True)
            h0 = torch.ones([1,B,E],device=self.device)
            c0 = torch.ones([1,B,E],device=self.device)

            prior = 0.
            prior_sum = 0.

            yp, h1 = self.rnn(segment_embed_parent, h0)
            # yp = yp /(1E-3 + yp.std(dim=-1, keepdims=True))
            yp = yp /(yp.std(dim=-1, keepdims=True))

            ## (B,S,E)
            segment_decoded = yp

            if ret=='encode': return yp

            '''
            Decoding the predicted segments into predicted words

            make sure the decoder does not see the required character
            '''

            h0 = torch.ones([1,B*S,E],device=self.device,dtype=torch.float)
            # h0 = h0 /(h0.std(dim=-1, keepdims=True))


            if 0:
                # h0 = h0 /(h0.std(dim=-1, keepdims=True))
                s1,_,s3 =  segments_embed_seq.shape
                segments_embed_seq_parent = torch.cat([segment_decoded.reshape((B*S,1,E)), segments_embed_seq],dim=1)[:,:-1]
                # segments_embed_seq_parent = self.std_norm(segments_embed_seq_parent,-1)
                # segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq_parent , h0 )
                segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq_parent , segment_decoded.reshape((1,B*S,E)) )
                # segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq    _parent , h0)


            if 0:
                # h0 = h0 /(h0.std(dim=-1, keepdims=True))
                s1,_,s3 =  segments_embed_seq.shape
                segments_embed_seq_parent = torch.cat([torch.ones((s1,1,s3),device=self.device), segments_embed_seq],dim=1)[:,:-1]
                segments_embed_seq_parent = (segments_embed_seq_parent + segment_decoded.reshape((B*S,1,E))) * 0.5
                # segments_embed_seq_parent = torch.cat([ segments_embed_seq_parent[:,:,:E//2], segment_decoded.reshape((B*S,1,E)).repeat((1,W,1))[:,:,:E//2]],dim=-1)

                # segments_embed_seq_parent = self.std_norm(segments_embed_seq_parent,-1)
                segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq_parent , h0 )
                # segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq    _parent , h0)

            if 0:
                s1,_,s3 =  segments_embed_seq.shape
                segments_embed_seq_parent = torch.cat([torch.ones((s1,1,s3),device=self.device), segments_embed_seq],dim=1)[:,:-1]
                # segments_embed_seq_parent = self.std_norm(segments_embed_seq_parent,-1)
                segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq_parent , segment_decoded.reshape((1,B*S,E)) )
                # segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq    _parent , h0)

            if 1:
                '''
                Best so far
                '''
                s1,_,s3 =  segments_embed_seq.shape
                # segments_embed_seq_parent = torch.cat([torch.ones((s1,1,s3),device=self.device), segments_embed_seq],dim=1)[:,:-1]
                segments_embed_seq_parent = torch.cat([segment_decoded.reshape((B*S,1,E)), segments_embed_seq],dim=1)[:,:-1]
                # segments_embed_seq_parent = self.std_norm(segments_embed_seq_parent,-1)
                # segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq_parent , segment_decoded.reshape((1,B*S,E)) )
                segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq_parent , h0)
                # segment_decoded.reshape((1,B*S,E)) )
                # segments_embed_seq_decoded = self.std_norm(segments_embed_seq_decoded,-1)

            ## (B,S,W,G)
            lpc = self.unembed(segments_embed_seq_decoded).log_softmax(-1).reshape((B,S,W,G))
            lp = torch.gather(lpc,index=segments.unsqueeze(-1),dim=-1).squeeze(-1)


            '''
            Concat segments back into seqs
            '''

            lp = torch.gather(lp.reshape((B,S*W)),index=pos_list,dim=-1)
            loss_per_loc = lp
            # logp_sum = (lp * target_notnull_segment).sum(dim=(1,2))
            logp_sum = (lp * target_notnull).sum(dim=1)


        ### normalise by average token count, but sum by batchsize
        # lossVal = lossVal/ (source_notnull.sum() / B)
        # print('[mean]',target_notnull.sum(1).float().mean())

        logp_per_token = logp_sum/ target_notnull.sum(dim=1)
        # loss = -logp_per_token
        loss = -logp_sum

        if ret=='forward':
            return loss,att
        elif ret in 'loss grad_loss'.split():
            return loss
        # elif ret=='grad_loss':
        #     return loss
        elif ret=='loss_per_loc':
            return -loss_per_loc,target_notnull
        else:
            raise NotImplementedError(f'''{ret} for ._loss()''')


class DLM103(DLMPrototype):
    _custom_hidden_to_cats = 1

    def __init__(self,device,config,_=None):
        super().__init__(device,config)

        G = config.graph_dim +1
        self.G = G
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        self.S = 31
        self.W = W = config.window_size
        assert W >=1,config.window_size

        self.embed      = nn.Embedding(G, E).to(self.device)

        self.rnn      = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        self.word_enc = nn.Linear(W*E,E)
        self.word_dec = nn.Linear(E,W*E)
        # nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        # self.rnn_word_dec = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)



    _hidden_to_cats = None
    def _hidden_to_cats(self, yp, target, ret):
        # assert self._custom_hidden_to_cats == 1
        # CLS_NAME = self.__class__.__name__
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
            lp = (model_w + kp)
            return lp
        else:
            raise NotImplementedError( f'{ret!r} for {CLS_NAME!r}' )


    def _loss(self,item, ret):
        CLS_NAME = self.__class__.__name__

        source = item['source'] ### token sequence
        target = item['target'] ### token seq

        # source_embed = self.embed(source)
        # source_len = item['source_len']
        # source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]
        target_embed = self.embed(target)


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
        G = self.G
        E = self.embed_dim
        target_embed = self.embed(target)
        B = len(target)
        i = 0
        att = torch.ones((B,1,1),device=self.device)


        def s(v):
            if torch.isnan(v).any():
                import pdb; pdb.set_trace()

        if 1:

            target_isblank = item['target_isblank'] | ~target_notnull

            ## (B, T)
            '''
            Segmenting sequences into sub-sequences

            Restruct the input by cutting at blank position.

            bound maximum segment count
            '''



            idx_list = torch.zeros((B,T),dtype=torch.long,device=self.device)
            idx      = torch.zeros((B,1),dtype=torch.long,device=self.device)
            seg_list = torch.zeros((B,T),dtype=torch.long,device=self.device)
            seg      = torch.ones((B,1),dtype=torch.long,device=self.device)
            # seg_list = torch.zeros()
            for i in range(T):
                idx_list[:,i:i+1] = idx
                seg_list[:,i:i+1] = seg * target_notnull[:,i:i+1] + 0 * ~target_notnull[:,i:i+1]
                idx = target_isblank[:,i:i+1] * 0 + (idx+1) * ~target_isblank[:,i:i+1]
                new_segment = target_isblank[:,i:i+1] & target_notnull[:,i:i+1]
                seg = new_segment * (seg+1) + ~new_segment * seg
            # torch.scatter()

            S = self.S
            # W = self.W
            seg_list = seg_list.clip(0,S)
            pos_list = seg_list * W + idx_list
            segments = (G - 1)*torch.ones((B,S*W),dtype=torch.long,device=self.device)
            segments = segments.scatter_(src=target,index=pos_list,dim=1).reshape((B,S,W))
            segments[:,0]=0


            # print(idx_list[0])
            # print(seg_list[0])
            # import pdb; pdb.set_trace()
            ## (B, S, W)


            '''
            Encoding the segments
            with lower-level RNN
            with KV attention
            '''
            ## (B, S, W, E ) x (W, E, K) -> (B, S, K)
            segments_embed_seq = self.embed(segments)
            segments_embed_seq = segments_embed_seq.reshape((B*S,W,E))
            ## (B,S,W*E) -> (B,S,E)
            segment_embed = self.word_enc(segments_embed_seq.reshape((B,S,W*E)))
            segment_embed = self.std_norm(segment_embed,-1)



            '''
            Run RNN on encoded segments

            Make sure the encoding is calculated from left segments
            '''
            segment_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device), segment_embed],dim=1)[:,:-1]
            # segment_embed_parent = segment_embed_parent/segment_embed_parent.std(dim=-1,keepdims=True)
            h0 = torch.ones([1,B,E],device=self.device)
            c0 = torch.ones([1,B,E],device=self.device)

            prior = 0.
            prior_sum = 0.

            yp, h1 = self.rnn(segment_embed_parent, h0)
            # yp = yp /(1E-3 + yp.std(dim=-1, keepdims=True))
            yp = yp /(yp.std(dim=-1, keepdims=True))

            ## (B,S,E)
            segment_decoded = yp

            if ret=='encode': return yp

            '''
            Decoding the predicted segments into predicted words

            make sure the decoder does not see the required character
            '''

            ## (B,S,E) -> (B,S,W,E)
            segments_decoded_seq = self.word_dec(segment_decoded).reshape((B,S,W,E))
            # h0 = torch.ones([1,B*S,E],device=self.device,dtype=torch.float)
            # h0 = h0 /(h0.std(dim=-1, keepdims=True))



            ## (B,S,W,G)
            lpc = self.unembed(segments_decoded_seq).log_softmax(-1).reshape((B,S,W,G))
            lp = torch.gather(lpc,index=segments.unsqueeze(-1),dim=-1).squeeze(-1)


            '''
            Concat segments back into seqs
            '''

            lp = torch.gather(lp.reshape((B,S*W)),index=pos_list,dim=-1)
            loss_per_loc = lp
            # logp_sum = (lp * target_notnull_segment).sum(dim=(1,2))
            logp_sum = (lp * target_notnull).sum(dim=1)


        ### normalise by average token count, but sum by batchsize
        # lossVal = lossVal/ (source_notnull.sum() / B)
        # print('[mean]',target_notnull.sum(1).float().mean())

        logp_per_token = logp_sum/ target_notnull.sum(dim=1)
        # loss = -logp_per_token
        loss = -logp_sum

        if ret=='forward':
            return loss,att
        elif ret in 'loss grad_loss'.split():
            return loss
        # elif ret=='grad_loss':
        #     return loss
        elif ret=='loss_per_loc':
            return -loss_per_loc,target_notnull
        else:
            raise NotImplementedError(f'''{ret} for ._loss()''')

class DLM104(DLMPrototype):
    '''
    Conv-GRU
    '''

    _custom_hidden_to_cats = 1

    def __init__(self,device,config,_=None):
        super().__init__(device,config)

        G = config.graph_dim +1
        self.G = G
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        self.S = 31
        self.W = W = config.window_size
        assert W >=1,config.window_size

        self.embed      = nn.Embedding(G, E).to(self.device)

        self.rnn      = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        self.word_enc = nn.Linear(W*E,E)
        self.word_enc_2 = nn.Linear(E,E)
        # self.word_enc_3 = nn.Linear(E,E)

        self.word_dec = nn.Linear(E,W*E)
        self.word_dec_2 = nn.Linear(E,E)
        # self.word_dec_3 = nn.Linear(E,E)

        # nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        # self.rnn_word_dec = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)



    _hidden_to_cats = None
    def _hidden_to_cats(self, yp, target, ret):
        # assert self._custom_hidden_to_cats == 1
        # CLS_NAME = self.__class__.__name__
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
            lp = (model_w + kp)
            return lp
        else:
            raise NotImplementedError( f'{ret!r} for {CLS_NAME!r}' )


    def _loss(self,item, ret):
        CLS_NAME = self.__class__.__name__

        source = item['source'] ### token sequence
        target = item['target'] ### token seq

        # source_embed = self.embed(source)
        # source_len = item['source_len']
        # source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]
        target_embed = self.embed(target)


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
        G = self.G
        E = self.embed_dim
        target_embed = self.embed(target)
        B = len(target)
        i = 0
        att = torch.ones((B,1,1),device=self.device)


        def s(v):
            if torch.isnan(v).any():
                import pdb; pdb.set_trace()

        if 1:

            target_isblank = item['target_isblank'] | ~target_notnull

            ## (B, T)
            '''
            Segmenting sequences into sub-sequences

            Restruct the input by cutting at blank position.

            bound maximum segment count
            '''



            idx_list = torch.zeros((B,T),dtype=torch.long,device=self.device)
            idx      = torch.zeros((B,1),dtype=torch.long,device=self.device)
            seg_list = torch.zeros((B,T),dtype=torch.long,device=self.device)
            seg      = torch.ones((B,1),dtype=torch.long,device=self.device)
            # seg_list = torch.zeros()
            for i in range(T):
                idx_list[:,i:i+1] = idx
                seg_list[:,i:i+1] = seg * target_notnull[:,i:i+1] + 0 * ~target_notnull[:,i:i+1]
                idx = target_isblank[:,i:i+1] * 0 + (idx+1) * ~target_isblank[:,i:i+1]
                new_segment = target_isblank[:,i:i+1] & target_notnull[:,i:i+1]
                seg = new_segment * (seg+1) + ~new_segment * seg
            # torch.scatter()

            S = self.S
            # W = self.W
            seg_list = seg_list.clip(0,S)
            pos_list = seg_list * W + idx_list
            segments = (G - 1)*torch.ones((B,S*W),dtype=torch.long,device=self.device)
            segments = segments.scatter_(src=target,index=pos_list,dim=1).reshape((B,S,W))
            segments[:,0]=0


            # print(idx_list[0])
            # print(seg_list[0])
            # import pdb; pdb.set_trace()
            ## (B, S, W)


            '''
            Encoding the segments
            with lower-level RNN
            with KV attention
            '''
            ## (B, S, W, E ) x (W, E, K) -> (B, S, K)
            segments_embed_seq = self.embed(segments)
            segments_embed_seq = segments_embed_seq.reshape((B*S,W,E))
            ## (B,S,W*E) -> (B,S,E)
            segment_embed = self.word_enc(segments_embed_seq.reshape((B,S,W*E))).relu()
            # segment_embed = segment_embed + self.word_enc_2(segment_embed).relu()
            # segment_embed = segment_embed + self.word_enc_3(segment_embed)
            segment_embed = self.word_enc_2(segment_embed)
            segment_embed = self.std_norm(segment_embed,-1)



            '''
            Run RNN on encoded segments

            Make sure the encoding is calculated from left segments
            '''
            segment_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device), segment_embed],dim=1)[:,:-1]
            # segment_embed_parent = segment_embed_parent/segment_embed_parent.std(dim=-1,keepdims=True)
            h0 = torch.ones([1,B,E],device=self.device)
            c0 = torch.ones([1,B,E],device=self.device)

            prior = 0.
            prior_sum = 0.

            yp, h1 = self.rnn(segment_embed_parent, h0)
            # yp = yp /(1E-3 + yp.std(dim=-1, keepdims=True))
            yp = yp /(yp.std(dim=-1, keepdims=True))

            ## (B,S,E)
            segment_decoded = yp

            if ret=='encode': return yp

            '''
            Decoding the predicted segments into predicted words

            make sure the decoder does not see the required character
            '''

            ## (B,S,E) -> (B,S,W,E)
            # segment_decoded = segment_decoded+self.word_dec_3(segment_decoded).relu()
            # segment_decoded = segment_decoded+ self.word_dec_2(segment_decoded).relu()
            segment_decoded = self.word_dec_2(segment_decoded).relu()
            segments_decoded_seq = self.word_dec(segment_decoded).reshape((B,S,W,E))
            # h0 = torch.ones([1,B*S,E],device=self.device,dtype=torch.float)
            # h0 = h0 /(h0.std(dim=-1, keepdims=True))
            segments_decoded_seq = self.std_norm(segments_decoded_seq,-1)



            ## (B,S,W,G)
            lpc = self.unembed(segments_decoded_seq).log_softmax(-1).reshape((B,S,W,G))
            lp = torch.gather(lpc,index=segments.unsqueeze(-1),dim=-1).squeeze(-1)


            '''
            Concat segments back into seqs
            '''

            lp = torch.gather(lp.reshape((B,S*W)),index=pos_list,dim=-1)
            loss_per_loc = lp
            # logp_sum = (lp * target_notnull_segment).sum(dim=(1,2))
            logp_sum = (lp * target_notnull).sum(dim=1)


        ### normalise by average token count, but sum by batchsize
        # lossVal = lossVal/ (source_notnull.sum() / B)
        # print('[mean]',target_notnull.sum(1).float().mean())

        logp_per_token = logp_sum/ target_notnull.sum(dim=1)
        # loss = -logp_per_token
        loss = -logp_sum

        if ret=='forward':
            return loss,att
        elif ret in 'loss grad_loss'.split():
            return loss
        # elif ret=='grad_loss':
        #     return loss
        elif ret=='loss_per_loc':
            return -loss_per_loc,target_notnull
        else:
            raise NotImplementedError(f'''{ret} for ._loss()''')


class DLM106(DLMPrototype):
    '''
    Conv-GRU
    '''

    _custom_hidden_to_cats = 1

    def __init__(self,device,config,_=None):
        super().__init__(device,config)

        ### 1 extra token for stop token
        G = config.graph_dim +1
        self.G = G
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        self.S = 101
        self.W = W = config.window_size
        assert W >=1,config.window_size

        self.embed      = nn.Embedding(G, E).to(self.device)

        self.rnn      = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        self.word_enc = nn.Linear(W*E,E)
        self.word_enc_2 = nn.Linear(E,E)
        # self.word_enc_3 = nn.Linear(E,E)

        self.word_dec = nn.Linear(E,W*E)
        self.word_dec_2 = nn.Linear(E,E)
        # self.word_dec_3 = nn.Linear(E,E)
        self.parser_rnn      = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        self.project_isblank = nn.Linear(E,2)
        # self.parser_rnn = nn.GRU()
        # nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        # self.rnn_word_dec = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)



    _hidden_to_cats = None
    def _hidden_to_cats(self, yp, target, ret):
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
            lp = (model_w + kp)
            return lp
        else:
            raise NotImplementedError( f'{ret!r} for {CLS_NAME!r}' )


    def _loss(self,item, ret):
        CLS_NAME = self.__class__.__name__

        source = item['source'] ### token sequence
        target = item['target'] ### token seq

        # source_embed = self.embed(source)
        # source_len = item['source_len']
        # source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]
        target_embed = self.embed(target)


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
        G = self.G
        E = self.embed_dim
        target_embed = self.embed(target)
        B = len(target)
        i = 0
        att = torch.ones((B,1,1),device=self.device)


        def s(v):
            if torch.isnan(v).any():
                import pdb; pdb.set_trace()

        # if 1:
        K = self.config.kernel_size
        lp_prior_list = torch.zeros((B,K),device=self.device,dtype=torch.float)
        lp_encode_list = torch.zeros((B,K),device=self.device,dtype=torch.float)
        # lp_weight_list = torch.zeros((B,K),device=self.device,dtype=torch.float)
        for ik in range(K):


            ## (B, T)
            '''
            Segmenting sequences into sub-sequences

            Restruct the input by cutting at blank position.

            bound maximum segment count
            '''


            h0 = torch.ones([1,B,E],device=self.device)
            c0 = torch.ones([1,B,E],device=self.device)

            S = self.S
            # W = self.W
            STOP_TOKEN=G-1


            idx_list = torch.zeros((B,T),dtype=torch.long,device=self.device)
            idx      = torch.zeros((B,1),dtype=torch.long,device=self.device)
            seg_list = torch.zeros((B,T),dtype=torch.long,device=self.device)
            seg      = torch.ones ((B,1),dtype=torch.long,device=self.device)
            # seg_list = torch.zeros()
            # target_isblank =
            if CLS_NAME in 'DLM106'.split():
                target_isblank = item['target_isblank'] | ~target_notnull
            elif CLS_NAME in 'DLM107 DLM108'.split():
                target_isblank_logp = self.project_isblank(self.parser_rnn(target_embed,h0)[0]).log_softmax(-1)
                lp_isblank, idx_isblank = self.sample_logp(target_isblank_logp,return_logp=True,dim=-1)
                target_isblank = idx_isblank==1
            else:
                raise NotImplementedError

            for i in range(T):
                idx_list[:,i:i+1] = idx
                # seg_list[:,i:i+1] = seg * target_notnull[:,i:i+1] + 0 * ~target_notnull[:,i:i+1]
                seg_list[:,i:i+1] = seg
                idx = target_isblank[:,i:i+1] * 0 + (idx+1) * ~target_isblank[:,i:i+1]
                idx = idx.clip(None,W-2)
                new_segment = target_isblank[:,i:i+1] & target_notnull[:,i:i+1]
                seg = new_segment * (seg+1) + ~new_segment * seg


            '''
            Needs to repack the segments with <s> token
            '''
            # import pdb; pdb.set_trace()
            # torch.scatter()

            seg_list = seg_list.clip(0,S-1)
            pos_list = seg_list * W + idx_list
            segments = STOP_TOKEN*torch.ones((B,S*W),dtype=torch.long,device=self.device)
            segments = segments.scatter_(src=target,index=pos_list,dim=1).reshape((B,S,W))
            segments_notnull = False * torch.ones((B,S*W),dtype=torch.bool,device=self.device)
            segments_notnull = segments_notnull.scatter_(src=target_notnull,index=pos_list,dim=1).reshape((B,S,W))
            segments_notnull = segments_notnull | segments_notnull.roll((1,),dims=(2,))
            segments[:,0]=0


                # assert (seg<=S-1).all(),seg.max()
                # assert (idx<=W-1).all()

            idx_list_new = torch.zeros((B,T+S),dtype=torch.long,device=self.device)
            idx          = torch.zeros((B,1  ),dtype=torch.long,device=self.device)
            seg_list_new = torch.zeros((B,T+S),dtype=torch.long,device=self.device)
            seg          = torch.ones ((B,1  ),dtype=torch.long,device=self.device)
            offset       = torch.zeros((B,1  ),dtype=torch.long,device=self.device)

            bid = torch.arange(B,device=self.device)[:,None]
            for i in range(T+S):
                '''
                Keep a pointer to segments. with seg and idx
                '''
                assert (seg<=S-1).all(),seg.max()
                assert (idx<=W-1).all()
                val = segments[bid,seg,idx]
                seg_list_new[:,i:i+1]=seg
                idx_list_new[:,i:i+1]=idx

                is_stop =  val == STOP_TOKEN
                is_end = is_stop & (seg==S-1)
                is_inc = ~is_end & is_stop
                seg = is_inc * (seg+1) + ~is_inc *seg
                idx = (~is_end & is_stop) * 0 + (~is_end & ~is_stop) *(idx+1) + is_end * idx
                # is_end =
                '''
                if is_stop, increment segment number
                '''

            pos_list_new = seg_list_new * W + idx_list_new
            target_mapped = torch.gather(segments.reshape((B,S*W)),index=pos_list_new,dim=-1)
            target_mapped_notnull = torch.gather(segments_notnull.reshape((B,S*W)),index=pos_list_new,dim=-1)
            # import pdb; pdb.set_trace()

            if CLS_NAME in 'DLM106 DLM107'.split():
                ## (B, S, W)
                '''
                Encoding the segments
                with lower-level RNN
                with KV attention
                '''
                ## (B, S, W, E ) x (W, E, K) -> (B, S, K)
                segments_embed_seq = self.embed(segments)
                segments_embed_seq = segments_embed_seq.reshape((B*S,W,E))
                ## (B,S,W*E) -> (B,S,E)
                segment_embed = self.word_enc(segments_embed_seq.reshape((B,S,W*E))).relu()
                segment_embed = self.word_enc_2(segment_embed)
                segment_embed = self.std_norm(segment_embed,-1)


                '''
                Run RNN on encoded segments

                Make sure the encoding is calculated from left segments
                '''
                segment_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device), segment_embed],dim=1)[:,:-1]
                # segment_embed_parent = segment_embed_parent/segment_embed_parent.std(dim=-1,keepdims=True)

                prior = 0.
                prior_sum = 0.

                yp, h1 = self.rnn(segment_embed_parent, h0)
                # yp = yp /(1E-3 + yp.std(dim=-1, keepdims=True))
                yp = yp /(yp.std(dim=-1, keepdims=True))

                ## (B,S,E)
                segment_decoded = yp

                if ret=='encode': return yp

                '''
                Decoding the predicted segments into predicted words

                make sure the decoder does not see the required character
                '''

                ## (B,S,E) -> (B,S,W,E)
                segment_decoded      = self.word_dec_2(segment_decoded).relu()
                segments_decoded_seq = self.word_dec(segment_decoded).reshape((B,S,W,E))
                # h0 = torch.ones([1,B*S,E],device=self.device,dtype=torch.float)
                # h0 = h0 /(h0.std(dim=-1, keepdims=True))
                segments_decoded_seq = self.std_norm(segments_decoded_seq,-1)

                ## (B,S,W,G)
                lpc = self.unembed(segments_decoded_seq).log_softmax(-1).reshape((B,S,W,G))

            elif CLS_NAME in 'DLM108':

                ## (B, S, W)
                '''
                Encoding the segments
                with lower-level RNN
                with KV attention
                '''
                ## (B, S, W, E ) x (W, E, K) -> (B, S, K)
                segments_embed_seq = self.embed(segments)
                segments_embed_seq = segments_embed_seq.reshape((B*S,W,E))

                h0 = torch.ones([1,B*S,E],device=self.device)
                v1 , h1 = self.rnn_word_enc(segments_embed_seq,h0)
                segment_embed = h1.reshape((B,S,E))

                '''
                Run RNN on encoded segments

                Make sure the encoding is calculated from left segments
                '''
                segment_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device), segment_embed],dim=1)[:,:-1]
                # segment_embed_parent = segment_embed_parent/segment_embed_parent.std(dim=-1,keepdims=True)
                h0 = torch.ones([1,B,E],device=self.device)
                c0 = torch.ones([1,B,E],device=self.device)

                prior = 0.
                prior_sum = 0.

                yp, h1 = self.rnn(segment_embed_parent, h0)
                # yp = yp /(1E-3 + yp.std(dim=-1, keepdims=True))
                yp = yp /(yp.std(dim=-1, keepdims=True))

                ## (B,S,E)
                segment_decoded = yp

                if ret=='encode': return yp

                '''
                Decoding the predicted segments into predicted words

                make sure the decoder does not see the required character
                '''

                h0 = torch.ones([1,B*S,E],device=self.device,dtype=torch.float)
                # h0 = h0 /(h0.std(dim=-1, keepdims=True))


                if 0:
                    # h0 = h0 /(h0.std(dim=-1, keepdims=True))
                    s1,_,s3 =  segments_embed_seq.shape
                    segments_embed_seq_parent = torch.cat([segment_decoded.reshape((B*S,1,E)), segments_embed_seq],dim=1)[:,:-1]
                    # segments_embed_seq_parent = self.std_norm(segments_embed_seq_parent,-1)
                    # segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq_parent , h0 )
                    segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq_parent , segment_decoded.reshape((1,B*S,E)) )
                    # segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq    _parent , h0)


                if 0:
                    # h0 = h0 /(h0.std(dim=-1, keepdims=True))
                    s1,_,s3 =  segments_embed_seq.shape
                    segments_embed_seq_parent = torch.cat([torch.ones((s1,1,s3),device=self.device), segments_embed_seq],dim=1)[:,:-1]
                    segments_embed_seq_parent = (segments_embed_seq_parent + segment_decoded.reshape((B*S,1,E))) * 0.5
                    # segments_embed_seq_parent = torch.cat([ segments_embed_seq_parent[:,:,:E//2], segment_decoded.reshape((B*S,1,E)).repeat((1,W,1))[:,:,:E//2]],dim=-1)

                    # segments_embed_seq_parent = self.std_norm(segments_embed_seq_parent,-1)
                    segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq_parent , h0 )
                    # segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq    _parent , h0)

                if 0:
                    s1,_,s3 =  segments_embed_seq.shape
                    segments_embed_seq_parent = torch.cat([torch.ones((s1,1,s3),device=self.device), segments_embed_seq],dim=1)[:,:-1]
                    # segments_embed_seq_parent = self.std_norm(segments_embed_seq_parent,-1)
                    segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq_parent , segment_decoded.reshape((1,B*S,E)) )
                    # segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq    _parent , h0)

                if 1:
                    '''
                    Best so far
                    '''
                    s1,_,s3 =  segments_embed_seq.shape
                    # segments_embed_seq_parent = torch.cat([torch.ones((s1,1,s3),device=self.device), segments_embed_seq],dim=1)[:,:-1]
                    segments_embed_seq_parent = torch.cat([segment_decoded.reshape((B*S,1,E)), segments_embed_seq],dim=1)[:,:-1]
                    # segments_embed_seq_parent = self.std_norm(segments_embed_seq_parent,-1)
                    # segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq_parent , segment_decoded.reshape((1,B*S,E)) )
                    segments_embed_seq_decoded, h1 = self.rnn_word_dec( segments_embed_seq_parent , h0)
                    # segment_decoded.reshape((1,B*S,E)) )
                    # segments_embed_seq_decoded = self.std_norm(segments_embed_seq_decoded,-1)

                ## (B,S,W,G)
                lpc = self.unembed(segments_embed_seq_decoded).log_softmax(-1).reshape((B,S,W,G))

            else:
                raise NotImplementedError



            lpc
            lp_seg = torch.gather(lpc,index=segments.unsqueeze(-1),dim=-1).squeeze(-1)


            '''
            Concat segments back into seqs

            Needs to sample concatenation tokens at the end of segment.
            '''

            lp = torch.gather(lp_seg.reshape((B,S*W)),index=pos_list_new,dim=-1)
            loss_per_loc = lp
            # encode_lp =
            lp_prior = (lp * target_mapped_notnull).sum(dim=1)
            if CLS_NAME in 'DLM106'.split():
                lp_encode = 0
            elif CLS_NAME in 'DLM107 DLM108'.split():

                lp_encode = (lp_isblank * target_notnull).sum(dim=1)
            else:
                raise NotImplementedError
            # lp_prior_sum =
            # lp_prior_sum = lp_prior_sum + lp_prior
            lp_prior_list[:,ik] = lp_prior
            lp_encode_list[:,ik] = lp_encode
            # lp_weight[:,ik] = (lp_prior-lp_encode).detach()

            # logp_sum = logp_sum +  (lp_prior + lp_encode * (lp_prior - lp_encode).detach())
            # logp_sum
        lp_weight_list = (lp_prior_list - lp_encode_list).detach()
        if K>1:
            lp_weight_list = lp_weight_list - lp_weight_list.mean(dim=1,keepdims=True)

        logp_sum_grad = (lp_prior_list + lp_encode_list * lp_weight_list).mean(dim=1)
        # logp_sum = logp_sum / (1.+ik)
        logp_sum = (lp_prior_list - lp_encode_list).detach().mean(1)
        print(lp_prior_list.mean(),lp_encode_list.mean())

        ### normalise by average token count, but sum by batchsize
        # lossVal = lossVal/ (source_notnull.sum() / B)
        # print('[mean]',target_notnull.sum(1).float().mean())

        logp_per_token = logp_sum/ target_mapped_notnull.sum(dim=1)
        # loss = -logp_per_token
        # logp_sum_grad

        if ret=='forward':
            return loss,att
        elif ret in 'loss '.split():
            return -logp_sum
        elif ret=='grad_loss':
            return -logp_sum_grad
        elif ret=='loss_per_loc':
            return -loss_per_loc,target_mapped_notnull
        else:
            raise NotImplementedError(f'''{ret} for ._loss()''')



class DLM107(DLM106):
    pass

class DLM108(DLM106):
    pass
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        # self.G = G
        E = config.embed_dim
        self.rnn_word_enc = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        self.rnn_word_dec = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)


class DLM105(DLMPrototype):
    '''
    Conv-GRU with lower channelcount for chars
    '''
    _custom_hidden_to_cats = 1
    def __init__(self,device,config,_=None):
        super().__init__(device,config)


        G = config.graph_dim +1
        self.G = G
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        self.S = 31
        self.W = W = config.window_size
        assert W >=1,config.window_size

        self.embed      = nn.Embedding(G, E).to(self.device)

        self.rnn      = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        self.word_enc = nn.Linear(W*E//4,E)
        self.word_enc_2 = nn.Linear(E,E)
        # self.word_enc_3 = nn.Linear(E,E)

        self.word_dec = nn.Linear(E,W*E//4)
        self.word_dec_2 = nn.Linear(E,E)
        # self.word_dec_3 = nn.Linear(E,E)
        # nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        # self.rnn_word_dec = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)



    _hidden_to_cats = None
    def _hidden_to_cats(self, yp, target, ret):
        # assert self._custom_hidden_to_cats == 1
        # CLS_NAME = self.__class__.__name__
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
            lp = (model_w + kp)
            return lp
        else:
            raise NotImplementedError( f'{ret!r} for {CLS_NAME!r}' )


    def _loss(self,item, ret):
        CLS_NAME = self.__class__.__name__

        source = item['source'] ### token sequence
        target = item['target'] ### token seq

        # source_embed = self.embed(source)
        # source_len = item['source_len']
        # source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]
        target_embed = self.embed(target)


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
        G = self.G
        E = self.embed_dim
        target_embed = self.embed(target)
        B = len(target)
        i = 0
        att = torch.ones((B,1,1),device=self.device)


        def s(v):
            if torch.isnan(v).any():
                import pdb; pdb.set_trace()

        if 1:

            target_isblank = item['target_isblank'] | ~target_notnull

            ## (B, T)
            '''
            Segmenting sequences into sub-sequences

            Restruct the input by cutting at blank position.

            bound maximum segment count
            '''



            idx_list = torch.zeros((B,T),dtype=torch.long,device=self.device)
            idx      = torch.zeros((B,1),dtype=torch.long,device=self.device)
            seg_list = torch.zeros((B,T),dtype=torch.long,device=self.device)
            seg      = torch.ones((B,1),dtype=torch.long,device=self.device)
            # seg_list = torch.zeros()
            for i in range(T):
                idx_list[:,i:i+1] = idx
                seg_list[:,i:i+1] = seg * target_notnull[:,i:i+1] + 0 * ~target_notnull[:,i:i+1]
                idx = target_isblank[:,i:i+1] * 0 + (idx+1) * ~target_isblank[:,i:i+1]
                new_segment = target_isblank[:,i:i+1] & target_notnull[:,i:i+1]
                seg = new_segment * (seg+1) + ~new_segment * seg
            # torch.scatter()

            S = self.S
            # W = self.W
            seg_list = seg_list.clip(0,S)
            pos_list = seg_list * W + idx_list
            segments = (G - 1)*torch.ones((B,S*W),dtype=torch.long,device=self.device)
            segments = segments.scatter_(src=target,index=pos_list,dim=1).reshape((B,S,W))
            segments[:,0]=0


            # print(idx_list[0])
            # print(seg_list[0])
            # import pdb; pdb.set_trace()
            ## (B, S, W)


            '''
            Encoding the segments
            with lower-level RNN
            with KV attention
            '''
            ## (B, S, W, E ) x (W, E, K) -> (B, S, K)
            segments_embed_seq = self.embed(segments)
            segments_embed_seq = segments_embed_seq.reshape((B*S,W,E))
            ## (B,S,W*E) -> (B,S,E)
            segment_embed = self.word_enc(segments_embed_seq[:,:,:E//4].reshape((B,S,W*E//4))).relu()
            # segment_embed = segment_embed + self.word_enc_2(segment_embed).relu()
            # segment_embed = segment_embed + self.word_enc_3(segment_embed)
            segment_embed = self.word_enc_2(segment_embed)
            segment_embed = self.std_norm(segment_embed,-1)



            '''
            Run RNN on encoded segments

            Make sure the encoding is calculated from left segments
            '''
            segment_embed_parent = torch.cat([torch.ones((B,1,E),device=self.device), segment_embed],dim=1)[:,:-1]
            # segment_embed_parent = segment_embed_parent/segment_embed_parent.std(dim=-1,keepdims=True)
            h0 = torch.ones([1,B,E],device=self.device)
            c0 = torch.ones([1,B,E],device=self.device)

            prior = 0.
            prior_sum = 0.

            yp, h1 = self.rnn(segment_embed_parent, h0)
            # yp = yp /(1E-3 + yp.std(dim=-1, keepdims=True))
            yp = yp /(yp.std(dim=-1, keepdims=True))

            ## (B,S,E)
            segment_decoded = yp

            if ret=='encode': return yp

            '''
            Decoding the predicted segments into predicted words

            make sure the decoder does not see the required character
            '''

            ## (B,S,E) -> (B,S,W,E)
            # segment_decoded = segment_decoded+self.word_dec_3(segment_decoded).relu()
            # segment_decoded = segment_decoded+ self.word_dec_2(segment_decoded).relu()
            segment_decoded = self.word_dec_2(segment_decoded).relu()
            segments_decoded_seq = self.word_dec(segment_decoded).reshape((B,S,W,E//4)).repeat((1,1,1,4))
            # h0 = torch.ones([1,B*S,E],device=self.device,dtype=torch.float)
            # h0 = h0 /(h0.std(dim=-1, keepdims=True))
            segments_decoded_seq = self.std_norm(segments_decoded_seq,-1)



            ## (B,S,W,G)
            lpc = self.unembed(segments_decoded_seq).log_softmax(-1).reshape((B,S,W,G))
            lp = torch.gather(lpc,index=segments.unsqueeze(-1),dim=-1).squeeze(-1)


            '''
            Concat segments back into seqs
            '''

            lp = torch.gather(lp.reshape((B,S*W)),index=pos_list,dim=-1)
            loss_per_loc = lp
            # logp_sum = (lp * target_notnull_segment).sum(dim=(1,2))
            logp_sum = (lp * target_notnull).sum(dim=1)


        ### normalise by average token count, but sum by batchsize
        # lossVal = lossVal/ (source_notnull.sum() / B)
        # print('[mean]',target_notnull.sum(1).float().mean())

        logp_per_token = logp_sum/ target_notnull.sum(dim=1)
        # loss = -logp_per_token
        loss = -logp_sum

        if ret=='forward':
            return loss,att
        elif ret in 'loss grad_loss'.split():
            return loss
        # elif ret=='grad_loss':
        #     return loss
        elif ret=='loss_per_loc':
            return -loss_per_loc,target_notnull
        else:
            raise NotImplementedError(f'''{ret} for ._loss()''')

from markov_lm.util_html import write_png_tag
import math

class DLM112(DLMPrototype):
    '''
    Conv-GRU with lower channelcount for chars
    '''
    _custom_hidden_to_cats = 1
    debug = 0
    def __init__(self,device,config,_=None):
        super().__init__(device,config)


        G = config.graph_dim +1
        self.G = G
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        # self.S = 31
        self.W = W = config.window_size
        # assert W >=1,config.window_size

        self.embed      = nn.Embedding(G, E).to(self.device)

        x = nn.Linear(1, E ).to(self.device)
        self.prior_mu   =  nn.Parameter(x.weight.T[None])

        x = nn.Linear(1, E ).to(self.device)
        self.prior_beta   =  nn.Parameter(x.weight.T[None])

        x = nn.Linear(1, E ).to(self.device)
        self.encoder_beta   =  nn.Parameter(x.weight.T[None])

        # self.rnn      = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)


    def log_param(self,buf,plt):
        # return
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        key = 'prior_beta'
        mat = getattr(self,key).cpu().detach().ravel()[:50,None]
        # mat = self.prior_beta.cpu().detach().ravel()[:50,None]
        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat.T,vmin=None,vmax=None,origin='lower')
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        ax.set_title(f'[log_param]{key}\nModel_name={self.config.model_name}\nEpoch:{epoch}\nmeanSquared:{mat.square().mean()}')
        plt.suptitle('')
        buf.write(write_png_tag(fig))

        fig,ax = plt.subplots(1,1,figsize=[10,10])
        key = 'prior_mu'
        mat = getattr(self,key).cpu().detach().ravel()[:50,None]
        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat.T,vmin=None,vmax=None,origin='lower')
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        ax.set_title(f'[log_param]{key}\nModel_name={self.config.model_name}\nEpoch:{epoch}\nmeanSquared:{mat.square().mean()}')
        plt.suptitle('')
        buf.write(write_png_tag(fig))


        fig,ax = plt.subplots(1,1,figsize=[10,10])
        key = 'encoder_beta'
        mat = getattr(self,key).cpu().detach().ravel()[:50,None]
        # mat = getattr(self,key).cpu().detach().ravel()[:,None]
        # im = ax.imshow(mat,vmin=0.0,vmax=0.5)
        im = ax.imshow(mat.T,vmin=None,vmax=None,origin='lower')
        # im = ax.imshow(mat,)
        plt.sca(ax)
        plt.colorbar(im)
        epoch = self.meta['epoch']
        ax.set_title(f'[log_param]{key}\nModel_name={self.config.model_name}\nEpoch:{epoch}\nmeanSquared:{mat.square().mean()}')
        plt.suptitle('')
        buf.write(write_png_tag(fig))

        pass


    _hidden_to_cats = None
    def _hidden_to_cats(self, yp, target, ret):
        # assert self._custom_hidden_to_cats == 1
        # CLS_NAME = self.__class__.__name__
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
            lp = (model_w + kp)
            return lp
        else:
            raise NotImplementedError( f'{ret!r} for {CLS_NAME!r}' )
    @staticmethod
    def _diag_normal_logp(zs,mu_p,beta_p):
        return (( -0.5*(( zs - mu_p)*beta_p).square() + beta_p.log() - 0.5 * math.log(2*math.pi) ))

    @staticmethod
    def _diag_normal_wrapped_logp(zs,mu_p,beta_p):
        '''
        https://en.wikipedia.org/wiki/Error_function

        wrapped normal dist. reweight by modified parition function
        beta = \sqrt{c}

        property:
            this function, exponentiated, integrates to one on [-1, 1] for zs

        sampling:
            needs to use inverse cdf
        '''
        zs = (zs + 1.- mu_p) % 2. - 1.
        return - ((zs )* beta_p).square() - (torch.erf(1*beta_p) - torch.erf(-1*beta_p)).log() - math.log(0.5) - 0.5 *math.log(math.pi) + torch.log(beta_p)

    @staticmethod
    def _diag_normal_wrapped_sample(zs,mu_p,beta_p):
        '''
        zs must be between 0 and 1
        '''
        if isinstance(zs,(tuple,list)):
            zs = torch.rand(zs,device=mu_p.device)
        be = torch.erf(beta_p)
        xs = torch.erfinv( be * (2*zs - 1) ) / beta_p
        xs = ((xs + mu_p) + 1)%2-1.
        return xs

    def shift_pad_left(self,target_embed, n=1,c=1):
        B,_,E = target_embed.shape
        target_embed_parent = torch.cat([c*torch.ones((B,n,E),device=self.device),target_embed],dim=1)[:,:-n]
        return target_embed_parent

    def _loss(self,item, ret):
        CLS_NAME = self.__class__.__name__
        source = item['source'] ### token sequence
        target = item['target'] ### token seq
        # source_embed = self.embed(source)
        # source_len = item['source_len']
        # source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]
        target_embed = self.embed(target)


        T = target.size(1)
        B = target.size(0)
        E = self.config.embed_dim
        N = W = self.config.window_size
        K = self.config.kernel_size
        target_len = item['target_len']
        target_notnull = torch.arange(target.size(1),device=self.device)[None,:]<target_len[:,None]

        if (item['has_start_token'].__class__==int and item['has_start_token']==1) or  item['has_start_token'].ravel()[0]==1:
            target_notnull[:,0] = False

        lp_prior_list = torch.zeros((B,W),device=self.device,dtype=torch.float)
        lp_encode_list = torch.zeros((B,W),device=self.device,dtype=torch.float)
        lp_decode_list = torch.zeros((B,W),device=self.device,dtype=torch.float)
        lp_weight_list = torch.zeros((B,W),device=self.device,dtype=torch.float)
        for iw in range(W):
            # target_embed_parent = target_embed_parent/target_embed_parent.std(dim=1,keepdims=True)
            # h0 = torch.ones([1,B,E],device=self.device)
            # c0 = torch.ones([1,B,E],device=self.device)
            '''
            Resampling Encoder
            '''
            ### (B,T,E)
            ### (1,1,E)
            encoder_mu    = target_embed
            encoder_beta  = self.encoder_beta.exp()

            prior_mu   = self.prior_mu
            prior_beta = self.prior_beta.exp()

            if ret=='encode':
                return encoder_mu

            lpc = None
            ### (B*K,T,E)
            if CLS_NAME in 'DLM112 DLM113'.split():
                encoder_beta = encoder_beta.clip(0.4,None)

                zs = torch.normal( 0, 1, encoder_mu.shape, device=self.device) / encoder_beta + encoder_mu
                lp_prior  = self._diag_normal_logp(zs, prior_mu, prior_beta).sum(-1)
                lp_encode = self._diag_normal_logp(zs, encoder_mu,  encoder_beta).sum(-1)

            elif CLS_NAME in 'DLM114'.split():
                encoder_beta = encoder_beta.clip(0.4,None)
                zs = torch.normal( 0, 1, encoder_mu.shape, device=self.device) / encoder_beta + encoder_mu
                # (B,T,E,1) - (1,1,E,K)
                lp_prior  = self._diag_normal_logp(zs.unsqueeze(-1), prior_mu, prior_beta)
                lp_prior = ((lp_prior ).sum(2) - math.log(K) ).logsumexp(-1)

                lp_encode = self._diag_normal_logp( zs, encoder_mu,  encoder_beta).sum(-1)

            elif CLS_NAME in 'DLM115'.split():
                encoder_beta = encoder_beta.clip(0.4,None)
                zs = torch.normal( 0, 1, encoder_mu.shape, device=self.device) / encoder_beta + encoder_mu
                # (B,T,E,1) - (1,1,E,K)
                lp_prior  = torch.cat(
                [self._diag_normal_logp(zs.unsqueeze(-1), prior_mu,                                prior_beta)[:,:,:,:-1],
                 self._diag_normal_logp(zs.unsqueeze(-1), self.shift_pad_left(self.lin_layer(zs)).unsqueeze(-1), prior_beta),
                 ],dim=-1)

                lp_prior = ((lp_prior ).sum(2) - math.log(K) ).logsumexp(-1)

                lp_encode = self._diag_normal_logp( zs, encoder_mu,  encoder_beta).sum(-1)
            elif CLS_NAME in 'DLM116'.split():
                encoder_mu = self.project((target_embed*target_notnull.unsqueeze(-1)).reshape((B,1,T*E)))

                zs = torch.normal( 0, 1, encoder_mu.shape, device=self.device) / encoder_beta + encoder_mu
                # (B,T,E,1) - (1,1,E,K)
                lp_prior  = torch.cat(
                [self._diag_normal_logp(zs.unsqueeze(-1), prior_mu,                                prior_beta)[:,:,:,:-1],
                 self._diag_normal_logp(zs.unsqueeze(-1), self.shift_pad_left(self.lin_layer(zs)).unsqueeze(-1), prior_beta),
                 ],dim=-1)
                lp_prior = ((lp_prior ).sum(2) - math.log(K) ).logsumexp(-1)
                lp_encode = self._diag_normal_logp( zs, encoder_mu,  encoder_beta).sum(-1)
                zs  =self.unproject(zs).reshape((B,T,E))

            elif CLS_NAME in 'DLM117'.split():
                encoder_mu = self.conv_layer(encoder_mu.transpose(1,2)).transpose(1,2)
                encoder_beta = encoder_beta.clip(0.4,None)
                zs = torch.normal( 0, 1, encoder_mu.shape, device=self.device) / encoder_beta + encoder_mu
                # (B,T,E,1) - (1,1,E,K)
                lp_prior  = self._diag_normal_logp(zs.unsqueeze(-1), prior_mu, prior_beta)
                lp_prior = ((lp_prior ).sum(2) - math.log(K) ).logsumexp(-1)

                lp_encode = self._diag_normal_logp( zs, encoder_mu,  encoder_beta).sum(-1)
                zs = self.convt_layer(zs.transpose(1,2)).transpose(1,2)
            elif CLS_NAME in 'DLM118'.split():
                x0 = encoder_mu
                encoder_mu = self.conv_layer(x0.transpose(1,2)).transpose(1,2)
                encoder_mu = x0 + encoder_mu
                encoder_beta = encoder_beta.clip(0.4,None)

                zs = torch.normal( 0, 1, encoder_mu.shape, device=self.device) / encoder_beta + encoder_mu

                # (B,T,E,1) - (1,1,E,K)
                lp_prior  = self._diag_normal_logp(zs.unsqueeze(-1), prior_mu, prior_beta)
                lp_prior = ((lp_prior ).sum(2) - math.log(K) ).logsumexp(-1)

                lp_encode = self._diag_normal_logp( zs, encoder_mu,  encoder_beta).sum(-1)
                zs = self.convt_layer(zs.transpose(1,2)).transpose(1,2) + zs
            elif CLS_NAME in 'DLM119 DLM120'.split():
                x0 = encoder_mu
                encoder_mu = self.conv_layer(x0.transpose(1,2)).transpose(1,2)
                if CLS_NAME in 'DLM119'.split():
                    encoder_mu = x0 + encoder_mu
                elif CLS_NAME in 'DLM120'.split():
                    encoder_mu
                else:
                    raise NotImplementedError
                encoder_beta = encoder_beta.clip(0.4,None)

                zs = torch.normal( 0, 1, encoder_mu.shape, device=self.device) / encoder_beta + encoder_mu

                # (B,T,E,1) - (1,1,E,K)
                lp_prior  = self._diag_normal_logp(zs.unsqueeze(-1), prior_mu, prior_beta)
                lp_prior = ((lp_prior ).sum(2) - math.log(K) ).logsumexp(-1)

                lp_encode = self._diag_normal_logp( zs, encoder_mu,  encoder_beta).sum(-1)
                zs =self.convt_layer(zs.tanh().transpose(1,2)).transpose(1,2)
                # zs = self.convt_layer(zs.transpose(1,2)).transpose(1,2) + zs
            elif CLS_NAME in 'DLM121'.split():
                x0 = encoder_mu
                encoder_mu = encoder_mu + self.conv_layer(encoder_mu.transpose(1,2)).transpose(1,2)
                encoder_mu = encoder_mu + self.conv_layer(encoder_mu.transpose(1,2)).transpose(1,2)
                encoder_beta = encoder_beta.clip(0.4,None)

                zs = torch.normal( 0, 1, encoder_mu.shape, device=self.device) / encoder_beta + encoder_mu

                # (B,T,E,1) - (1,1,E,K)
                lp_prior  = self._diag_normal_logp(zs.unsqueeze(-1), prior_mu, prior_beta)
                lp_prior = ((lp_prior ).sum(2) - math.log(K) ).logsumexp(-1)

                lp_encode = self._diag_normal_logp( zs, encoder_mu,  encoder_beta).sum(-1)
                zs =self.convt_layer(zs.tanh().transpose(1,2)).transpose(1,2)
                zs =self.convt_layer(zs.tanh().transpose(1,2)).transpose(1,2)
                # zs = self.convt_layer(zs.transpose(1,2)).transpose(1,2) + zs
            elif CLS_NAME in 'DLM123 DLM125 DLM127'.split():
                x0 = encoder_mu
                D = self.config.depth
                for i in range(D):
                    encoder_mu = encoder_mu + self.conv_layer(encoder_mu.transpose(1,2)).transpose(1,2)
                # encoder_mu = encoder_mu + self.conv_layer(encoder_mu.transpose(1,2)).transpose(1,2)
                # encoder_mu = encoder_mu + self.conv_layer(encoder_mu.transpose(1,2)).transpose(1,2)
                encoder_beta = encoder_beta.clip(0.4,None)

                if CLS_NAME in 'DLM125'.split():
                    '''
                    noise-free model
                    '''
                    zs = encoder_mu
                    lp_prior  = self._diag_normal_logp(zs.unsqueeze(-1), prior_mu, prior_beta)
                    lp_prior = ((lp_prior ).sum(2) - math.log(K) ).logsumexp(-1)

                    lp_encode = self._diag_normal_logp( zs, encoder_mu,  encoder_beta).sum(-1)

                elif CLS_NAME in 'DLM123 DLM127'.split():
                    zs = torch.normal( 0, 1, encoder_mu.shape, device=self.device) / encoder_beta + encoder_mu

                    # (B,T,E,1) - (1,1,E,K)
                    lp_prior  = self._diag_normal_logp(zs.unsqueeze(-1), prior_mu, prior_beta)
                    lp_prior = ((lp_prior ).sum(2) - math.log(K) ).logsumexp(-1)

                    lp_encode = self._diag_normal_logp( zs, encoder_mu,  encoder_beta).sum(-1)
                else:
                    raise NotImplementedError
                for i in range(D):
                    zs =self.convt_layer(zs.tanh().transpose(1,2)).transpose(1,2)
                # zs =self.convt_layer(zs.tanh().transpose(1,2)).transpose(1,2)
                # zs =self.convt_layer(zs.tanh().transpose(1,2)).transpose(1,2)
                # zs = self.convt_layer(zs.transpose(1,2)).transpose(1,2) + zs

            elif CLS_NAME in 'DLM128 DLM129'.split():
                def set_requires_grad(module, val):
                    for p in module.parameters():
                        p.requires_grad = val
                '''
                Inferring the latent
                '''
                encoder_mu_0 = encoder_mu
                D = self.config.depth
                for i in range(D):
                    encoder_mu = encoder_mu + self.conv_layer_list[i](encoder_mu.transpose(1,2)).transpose(1,2)
                # encoder_mu = encoder_mu + self.conv_layer(encoder_mu.transpose(1,2)).transpose(1,2)
                # encoder_mu = encoder_mu + self.conv_layer(encoder_mu.transpose(1,2)).transpose(1,2)
                # encoder_beta = encoder_beta.clip(0.8,None)
                encoder_mu    = encoder_mu * target_notnull.unsqueeze(-1)
                encoder_beta  = encoder_beta.clip(self.config.p_null,None)
                # prior_beta = prior_beta.clip(0.8,None)

                '''
                Required by the importance sampling
                '''
                sampled_zs = zs = torch.normal( 0, 1, encoder_mu.shape, device=self.device) / encoder_beta + encoder_mu

                'zs is the sampled code'

                # (B,T,E,1) - (1,1,E,K)
                lp_prior  = self._diag_normal_logp(zs.unsqueeze(-1), prior_mu, prior_beta)
                lp_prior = ((lp_prior ).sum(2) - math.log(K) ).logsumexp(-1)
                lp_encode = self._diag_normal_logp( zs, encoder_mu,  encoder_beta).sum(-1)

                lpc,recovered  = self.decode(zs)
                ###
                '''
                zs is the recovered target
                lpc is the decod
                '''
                if ret=='debug':
                    return locals()



            elif CLS_NAME in 'DLM126'.split():
                x0 = encoder_mu
                D = self.config.depth
                for i in range(D):
                    encoder_mu = self.conv_layer(encoder_mu.transpose(1,2)).transpose(1,2)
                # encoder_mu = encoder_mu + self.conv_layer(encoder_mu.transpose(1,2)).transpose(1,2)
                # encoder_mu = encoder_mu + self.conv_layer(encoder_mu.transpose(1,2)).transpose(1,2)
                encoder_beta = encoder_beta.clip(self.config.beta,None).clip(0.4,None)

                if CLS_NAME in 'DLM126'.split():
                    zs = torch.normal( 0, 1, encoder_mu.shape, device=self.device) / encoder_beta + encoder_mu

                    # (B,T,E,1) - (1,1,E,K)
                    lp_prior  = self._diag_normal_logp(zs.unsqueeze(-1), prior_mu, prior_beta)
                    lp_prior = ((lp_prior ).sum(2) - math.log(K) ).logsumexp(-1)

                    lp_encode = self._diag_normal_logp( zs, encoder_mu,  encoder_beta).sum(-1)
                else:
                    raise NotImplementedError

                for i in range(D):
                    zs =self.convt_layer(zs.tanh().transpose(1,2)).transpose(1,2)
                # zs =self.convt_layer(zs.tanh().transpose(1,2)).transpose(1,2)
                # zs =self.convt_layer(zs.tanh().transpose(1,2)).transpose(1,2)
                # zs = self.convt_layer(zs.transpose(1,2)).transpose(1,2) + zs

            elif CLS_NAME in 'DLM127'.split():
                x0 = encoder_mu
                D = self.config.depth
                for i in range(D):
                    encoder_mu = self.conv_layer(encoder_mu.transpose(1,2)).transpose(1,2)

                # if CLS_NAME in 'DLM126'.split():
                lpk = self.lpk_layer(encoder_mu).log_softmax(-1)
                logp_cs,idx_cs = self.sample_logp(lpk,return_logp=True,dim=-1)

                #(1,1,E,K)
                encoder_mu = prior_mu
                # encdoer_mu =
                # encoder_beta = prior_beta
                encoder_beta = encoder_beta.unsqueeze(-1)
                # encoder_beta = encoder_beta.clip(self.config.beta,None).clip(0.4,None)
                encoder_beta = encoder_beta.clip(0.4,None)
                ### (1,1,)
                zs = torch.normal( 0, 1, (B,T,E), device=self.device) / encoder_beta.squeeze(-1)
                # lp_zs = self._diag_normal_logp(zs, 0., encoder_beta)
                zs = encoder_mu[0,0].T[idx_cs] + zs ## (B,T,E)

                ## (B,T,E,1) - (1,1,E,K)
                lp_prior  = self._diag_normal_logp(zs.unsqueeze(-1), prior_mu, prior_beta)
                lp_prior = ((lp_prior ).sum(2) + self.prior_k.log_softmax(-1) ).logsumexp(-1)

                ## (B,T,E,K)
                lp_encode = self._diag_normal_logp( zs.unsqueeze(-1), encoder_mu,  encoder_beta)
                lp_encode = ((lp_encode ).sum(2) + lpk  ).logsumexp(-1)
                # else:
                #     raise NotImplementedError

                for i in range(D):
                    zs =self.convt_layer(zs.tanh().transpose(1,2)).transpose(1,2)
                    # zs =self.convt_layer(zs.transpose(1,2)).transpose(1,2)
                # zs =self.convt_layer(zs.tanh().transpose(1,2)).transpose(1,2)
                # zs =self.convt_layer(zs.tanh().transpose(1,2)).transpose(1,2)
                # zs = self.convt_layer(zs.transpose(1,2)).transpose(1,2) + zs
                lpc = self.unembed(zs).log_softmax(-1)

            elif CLS_NAME in 'DLM124'.split():
                x0 = encoder_mu
                D = self.config.depth
                for i in range(D):
                    encoder_mu = encoder_mu + self.conv_layer(encoder_mu.tanh().transpose(1,2)).transpose(1,2)
                # encoder_mu = encoder_mu + self.conv_layer(encoder_mu.transpose(1,2)).transpose(1,2)
                # encoder_mu = encoder_mu + self.conv_layer(encoder_mu.transpose(1,2)).transpose(1,2)
                encoder_beta = encoder_beta.clip(0.6,None)

                zs = torch.normal( 0, 1, encoder_mu.shape, device=self.device) / encoder_beta + encoder_mu

                # (B,T,E,1) - (1,1,E,K)
                lp_prior  = self._diag_normal_logp(zs.unsqueeze(-1), prior_mu, prior_beta)
                lp_prior = ((lp_prior ).sum(2) - math.log(K) ).logsumexp(-1)

                lp_encode = self._diag_normal_logp( zs, encoder_mu,  encoder_beta).sum(-1)
                for i in range(D):
                    zs =self.convt_layer(zs.tanh().transpose(1,2)).transpose(1,2)
                # zs =self.convt_layer(zs.tanh().transpose(1,2)).transpose(1,2)
                # zs =self.convt_layer(zs.tanh().transpose(1,2)).transpose(1,2)
                # zs = self.convt_layer(zs.transpose(1,2)).transpose(1,2) + zs

            elif CLS_NAME in 'DLM122'.split():
                x0 = encoder_mu
                encoder_mu = encoder_mu + self.conv_layer(encoder_mu.transpose(1,2)).transpose(1,2)
                encoder_mu = encoder_mu + self.conv_layer_2(encoder_mu.transpose(1,2)).transpose(1,2)
                encoder_beta = encoder_beta.clip(0.4,None)

                zs = torch.normal( 0, 1, encoder_mu.shape, device=self.device) / encoder_beta + encoder_mu

                # (B,T,E,1) - (1,1,E,K)
                lp_prior  = self._diag_normal_logp(zs.unsqueeze(-1), prior_mu, prior_beta)
                lp_prior = ((lp_prior ).sum(2) - math.log(K) ).logsumexp(-1)

                lp_encode = self._diag_normal_logp( zs, encoder_mu,  encoder_beta).sum(-1)
                zs =self.convt_layer(zs.tanh().transpose(1,2)).transpose(1,2)
                zs =self.convt_layer_2(zs.tanh().transpose(1,2)).transpose(1,2)
                # zs = self.convt_layer(zs.transpose(1,2)).transpose(1,2) + zs

            else:
                raise NotImplementedError


            if CLS_NAME in 'DLM113'.split():
                zs = self.lin_layer(zs.relu())
            if lpc is None:
                lpc = self.unembed(zs).log_softmax(-1)
            # lpc = self.unembed(self.std_norm(zs,-1)).log_softmax(-1)
            lp_decode = torch.gather(lpc,index=target.unsqueeze(-1),dim=-1).squeeze(-1)

            lp_prior_list[:,iw] = (lp_prior*target_notnull).sum(-1)
            lp_decode_list[:,iw]= (lp_decode*target_notnull).sum(-1)
            lp_encode_list[:,iw]= (lp_encode*target_notnull).sum(-1)
            loss_per_loc = -(lp_prior + lp_decode - lp_encode)

        lp_weight_list = (lp_prior_list+lp_decode_list - lp_encode_list).detach()
        if W==1:
            pass
        else:
            pass
            # lp_weight_list = lp_weight_list-lp_weight_list.mean(-1,keepdims=True)

        logp_sum_grad = (lp_prior_list + lp_decode_list + lp_encode_list * lp_weight_list).mean(dim=1)
        logp_sum = (lp_prior_list + lp_decode_list - lp_encode_list).detach().mean(1)
        ### normalise by average token count, but sum by batchsize
        # lossVal = lossVal/ (source_notnull.sum() / B)
        # print('[mean]',target_notnull.sum(1).float().mean())
        self.debug = 0
        if self.debug:
            print(f'e_beta:{encoder_beta.mean():.4f} w_beta:{prior_beta.mean():.4f} '
            f'e_mu_mse:{encoder_mu.square().mean():.4f} w_mu_mse:{prior_mu.square().mean():.4f} ')
            print(f'lp_prior:{lp_prior_list.mean():.4f}  lp_decode:{lp_decode_list.mean():.4f} '
            f'lp_encode:{lp_encode_list.mean():.4f}')

        # import pdb; pdb.set_trace()
        logp_per_token = logp_sum/ target_notnull.sum(dim=1).clip(1,None)
        loss = -logp_sum
        att = torch.zeros((B,T,1),device = self.device)
        # import pdb; pdb.set_trace()
        if ret=='forward':
            return logp_sum.unsqueeze(-1),att
        elif ret in 'loss'.split():
            return -logp_sum
            # return loss_per_loc *
            # return (loss_per_loc * target_notnull).sum(dim=-1)
            # return -logp_per_token
            # sum / target_notnull.sum(dim=1)
        elif ret=='grad_loss':
            return -logp_sum_grad
        elif ret=='loss_per_loc':
            return loss_per_loc,target_notnull
        else:
            raise NotImplementedError(f'''{ret} for ._loss()''')




class DLM130(DLMPrototype):
    '''
    HMM copy
    '''
    _custom_hidden_to_cats = 1
    debug = 0
    def __init__(self,device,config,_=None):
        super().__init__(device,config)

        G = config.graph_dim +1
        self.G = G
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        x = nn.Linear(1, 1 ).to(self.device)
        self.jump_prob   =  nn.Parameter(x.weight[0,0])
        # .T[None])
        # self.emission = x
        # self.S = 31
        self.W = W = config.window_size
        # assert W >=1,config.window_size

        self.embed      = nn.Embedding(G, E).to(self.device)

        # x = nn.Linear(1, E ).to(self.device)
        # self.prior_mu   =  nn.Parameter(x.weight.T[None])
        #
        # x = nn.Linear(1, E ).to(self.device)
        # self.prior_beta   =  nn.Parameter(x.weight.T[None])
        #
        # x = nn.Linear(1, E ).to(self.device)
        # self.encoder_beta   =  nn.Parameter(x.weight.T[None])
        #
        # self.rnn      = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

    @staticmethod
    def _diag_normal_logp(zs,mu_p,beta_p):
        return (( -0.5*(( zs - mu_p)*beta_p).square() + beta_p.log() - 0.5 * math.log(2*math.pi) ))

    def shift_pad_left(self,target_embed, n=1,c=1):
        B,_,E = target_embed.shape
        target_embed_parent = torch.cat([c*torch.ones((B,n,E),device=self.device),target_embed],dim=1)[:,:-n]
        return target_embed_parent

    def _loss(self,item, ret,rng=None):
        CLS_NAME = self.__class__.__name__
        source = item['source'] ### token sequence
        target = item['target'] ### token seq
        # source_embed = self.embed(source)
        # source_len = item['source_len']
        # source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]
        target_embed = self.embed(target)


        T = target.size(1)
        B = target.size(0)
        E = self.config.embed_dim
        N = W = self.config.window_size
        K = self.config.kernel_size
        target_len = item['target_len']
        target_notnull = torch.arange(target.size(1),device=self.device)[None,:]<target_len[:,None]


        log_pzo_list = torch.ones((B,T,E),device=self.device,dtype=torch.float)
        log_pz = torch.ones((B,E,1),device=self.device,dtype=torch.float)
        log_pz = log_pz.log_softmax(1)
        log_pz_0 = log_pz
        # bp = self.
        pb = self.jump_prob.sigmoid()
        # beta_p = torch.tensor(self.config.beta).sigmoid()
        ## self.emission (E,G)
        _emit_p = self.embed.weight.T.log_softmax(-1)
        self.v = _emit_p.argmax(1)
        if 0:
            vvv = []
            for vv in conf.model.v: vvv.append(conf.dataset.vocab.wordize(min(vv,conf.model.G-1-1)))
            print(''.join(vvv))

        for it in range(T):
            # target
            ## self.emission.T[target[:,it:it+1]] (B,1,E)
            # log_e = self.emission.T[target[:,it:it+1]].transpsoe(2,1)
            log_e = _emit_p.T[target[:,it:it+1]].transpose(2,1)
            log_pzo = (log_pz +log_e)
            log_pzo_list[:,it:it+1]=log_pzo.transpose(2,1)

            log_pz = torch.cat([
                (pb.log() + log_pzo.roll(1,dims=(1))),
                (1-pb).log() + log_pz_0,
            ],dim=2).logsumexp(2).unsqueeze(-1)
            print(log_pzo.logsumexp(1).mean())
            pass

        ### directly calculate logp_sum
        # log_a
        # log_t
        logp_per_loc = log_pzo_list.logsumexp(2)
        logp_per_loc = logp_per_loc-logp_per_loc.roll(1,dims=(1,))
        loss_per_loc = -logp_per_loc

        logp_sum = (logp_per_loc*target_notnull).sum(1)
        # logp_sum = log_pzo.logsumexp(1).squeeze(-1)
        logp_sum_grad = -logp_sum

        ### normalise by average token count, but sum by batchsize
        # lossVal = lossVal/ (source_notnull.sum() / B)
        # print('[mean]',target_notnull.sum(1).float().mean())
        self.debug = 0
        if self.debug:
            print(f'e_beta:{encoder_beta.mean():.4f} w_beta:{prior_beta.mean():.4f} '
            f'e_mu_mse:{encoder_mu.square().mean():.4f} w_mu_mse:{prior_mu.square().mean():.4f} ')
            print(f'lp_prior:{lp_prior_list.mean():.4f}  lp_decode:{lp_decode_list.mean():.4f} '
            f'lp_encode:{lp_encode_list.mean():.4f}')

        # import pdb; pdb.set_trace()
        logp_per_token = logp_sum/ target_notnull.sum(dim=1).clip(1,None)
        loss = -logp_sum
        att = torch.zeros((B,T,1),device = self.device)
        # import pdb; pdb.set_trace()
        if ret=='encode':
            return att
        if ret=='forward':
            return logp_sum.unsqueeze(-1),att
        elif ret in 'loss'.split():
            return -logp_sum
            # return loss_per_loc *
            # return (loss_per_loc * target_notnull).sum(dim=-1)
            # return -logp_per_token
            # sum / target_notnull.sum(dim=1)
        elif ret=='grad_loss':
            return -logp_sum_grad
        elif ret=='loss_per_loc':
            return loss_per_loc,target_notnull
        else:
            raise NotImplementedError(f'''{ret} for ._loss()''')
# class DLM113(DLMPrototype):

class DLM113(DLM112):
    # Prototype):
    '''
    Conv-GRU with lower channelcount for chars
    '''
    _custom_hidden_to_cats = 1
    def __init__(self,device,config,_=None):
        super().__init__(device,config)


        G = config.graph_dim +1
        self.G = G
        E = config.embed_dim
        # self.S = 31
        self.W = W = config.window_size
        # assert W >=1,config.window_size
        self.lin_layer = nn.Linear(E, E ).to(self.device)

class DLM114(DLM112):
    # Prototype):
    '''
    Conv-GRU with lower channelcount for chars
    '''
    _custom_hidden_to_cats = 1
    def __init__(self,device,config,_=None):
        super().__init__(device,config)


        G = config.graph_dim +1
        self.G = G
        K = config.kernel_size
        assert K >=1,config
        E = config.embed_dim
        # self.S = 31
        self.W = W = config.window_size
        self.K = K = config.kernel_size

        x = nn.Linear(E,K).to(self.device)
        self.prior_mu   =  nn.Parameter(x.weight.T[None,None])

        x = nn.Linear(E,1 ).to(self.device)
        self.prior_beta   =  nn.Parameter(x.weight.T[None,None])

        # assert W >=1,config.window_size
        # self.lin_layer = nn.Linear(E, E ).to(self.device)

class DLM117(DLM112):
    # Prototype):
    '''
    Conv-GRU with lower channelcount for chars
    '''
    _custom_hidden_to_cats = 1
    def __init__(self,device,config,_=None):
        super().__init__(device,config)


        G = config.graph_dim +1
        self.G = G
        K = config.kernel_size
        assert K >=1,config
        E = config.embed_dim
        # self.S = 31
        self.W = W = config.window_size
        self.K = K = config.kernel_size

        x = nn.Linear(E,K).to(self.device)
        self.prior_mu   =  nn.Parameter(x.weight.T[None,None])

        x = nn.Linear(E,1 ).to(self.device)
        self.prior_beta   =  nn.Parameter(x.weight.T[None,None])

        self.conv_layer = nn.Conv1d(E,E,kernel_size = 5, padding='same')
        self.convt_layer = nn.Conv1d(E,E,kernel_size = 5, padding='same')
        # self.convt_layer = nn.ConvTranspose1d(E,E,kernel_size = 5, padding='same')

        # assert W >=1,config.window_size
        # self.lin_layer = nn.Linear(E, E ).to(self.device)


class DLM118(DLM117):
    pass
class DLM119(DLM117):
    pass
class DLM120(DLM117):
    pass
class DLM121(DLM117):
    pass
class DLM123(DLM117):
    pass
class DLM124(DLM117):
    pass
class DLM125(DLM117):
    pass
class DLM126(DLM117):
    pass


class DLM127(DLM117):
    pass
    def unembed(self,x):
        y = x.matmul(self.embed.weight.T.tanh()*self.config.beta)
        return y

import numpy as np
class U(object):
    '''
    Utility class
    '''
    @staticmethod
    def N(v):
        if isinstance(v,torch.Tensor):
            return v.detach().cpu().numpy()
        elif isinstance(v, list):
            return v
        elif isinstance(v,np.ndarray):
            return v
        else:
            raise NotImplementedError(f"Not impl for {v!r}")

class DLM128(DLM112):
    # Prototype):
    '''
    Conv-GRU with lower channelcount for chars
    '''
    _custom_hidden_to_cats = 1
    def __init__(self,device,config,_=None):
        super().__init__(device,config)


        G = config.graph_dim +1
        self.G = G
        K = config.kernel_size
        assert K >=1,config
        E = config.embed_dim
        # self.S = 31
        self.W = W = config.window_size
        self.K = K = config.kernel_size

        x = nn.Linear(E,K).to(self.device)
        self.prior_mu   =  nn.Parameter(x.weight.T[None,None])

        x = nn.Linear(E,1 ).to(self.device)
        self.prior_beta   =  nn.Parameter(x.weight.T[None,None])

        D = self.config.depth
        self.conv_layer_list  = nn.ModuleList( [ nn.Conv1d(E,E,kernel_size = 5, padding='same') for i in range(D) ])
        self.convt_layer_list = nn.ModuleList( [ nn.Conv1d(E,E,kernel_size = 5, padding='same') for i in range(D) ])
        # self.convt_layer = nn.ConvTranspose1d(E,E,kernel_size = 5, padding='same')

        # assert W >=1,config.window_size
        # self.lin_layer = nn.Linear(E, E ).to(self.device)
    def decode(self,zs):
        D = self.config.depth
        for i in range(D):
            zs = self.convt_layer_list[i](zs.tanh().transpose(1,2)).transpose(1,2)
        lpc = self.unembed(zs).log_softmax(-1)
        return lpc,zs


    def sample_token(self,B,T,prompt=None):
        # encoder_mu    = target_embed
        # encoder_beta  = self.encoder_beta.exp()

        prior_mu   = self.prior_mu
        prior_beta = self.prior_beta.exp()

        E = self.config.embed_dim
        T = self.config.n_step
        K = self.config.kernel_size
        # mu_p   = self.h_prior[0:1]
        # beta_p = self.h_prior[1:2].exp()
        h1r = torch.normal( 0, 1, (B,T,E,1), device=self.device) / prior_beta + prior_mu
        prior_k = torch.ones((1,1,1,K),device=self.device).log_softmax(-1)
        k = self.sample_logp(prior_k.repeat((B,T,1,1)),dim=-1,return_logp=False)
        h1r = torch.gather(h1r,index=k.repeat((1,1,E)).unsqueeze(-1),dim=-1).squeeze(-1)
        return self.sample_token_from_latent(h1r)

    def sample_token_from_latent(self, h1r,return_logp=False):
        lpc = self.decode(h1r)[0]
        return self.sample_logp(lpc,-1,return_logp)
        return None

        yp = self.latent_to_emittor(h1r)
        ## (B,T,C) tensor
        logp = self._hidden_to_cats(yp,ret='full',target=None)
        return self.sample_logp(logp,-1,return_logp)


    @staticmethod
    def callback_after_test_all(conf, model,  item):
        vis = conf.vis;  env = conf._session_name;
        if vis is None: return;
        # key = 'debug'
        xd = model._loss(item,ret='debug')

        key = 'encoder_mu_0'
        if xd.get(key,None) is None:
            pass
        else:
            yp = xd[key][:10]
            vis.heatmap(U.N(torch.cat(list(yp[:3]),dim=1)),env=env,opts=dict(title=key),win=key)


        key = 'encoder_mu'
        if xd.get(key,None) is None:
            pass
        else:
            yp = xd[key][:10]
            vis.heatmap(U.N(torch.cat(list(yp[:3]),dim=1)),env=env,opts=dict(title=key),win=key)

        key = 'recovered'
        if xd.get(key,None) is None:
            pass
        else:
            yp = xd[key][:3]
            xmin=None;xmax=None
            vis.heatmap(U.N(torch.cat(list(yp),dim=1)),env=env,opts=dict(title=key,xmin=xmin,xmax=xmax),win=key)

        key = 'sampled_zs'
        if xd.get(key,None) is None:
            pass
        else:
            yp = xd[key][:3]
            xmin=None;xmax=None
            vis.heatmap(U.N(torch.cat(list(yp),dim=1)),env=env,opts=dict(title=key,xmin=xmin,xmax=xmax),win=key)

        key = 'prior_beta'
        vis.heatmap(U.N(model.prior_beta[0,0].exp()),env=env,opts=dict(title=key,xmin=0,xmax=3),win=key)

        key = 'encoder_beta'
        vis.heatmap(U.N(model.encoder_beta.ravel().exp()[None]),env=env,opts=dict(title=key,xmin=0,xmax=3),win=key)

        meta = getattr(model,'meta',None)
        # import pdb; pdb.set_trace()
        meta = meta or {'test_losses':[],'epoch':-1}
        if len(meta['test_losses']):
            key = 'test_losses'
            vis.line( np.clip(U.N(meta['test_losses']),None,200),X=meta['epoch_list'],opts=dict(title=key,),win=key, env=env)
        # vis.line(U.)
            vis.text(f'''
            Epoch:      {meta['epoch']}<br/>
            test_loss: {meta['test_losses'][-1]}
            ''',
            env=env,win='title')

    @staticmethod
    def callback_checkpoint(conf, model,  item):
        vis = conf.vis;  env = conf._session_name;
        if vis is None: return;
        env = conf._session_name
        # meta = model.meta

        return



class DLM131(DLM128):

    def _get_loss(self,zs, encoder_mu,target,target_notnull):
        K = self.K
        encoder_beta  = self.encoder_beta.exp()
        encoder_beta  = encoder_beta.clip(self.config.p_null,None)
        prior_mu   = self.prior_mu
        prior_beta = self.prior_beta.exp()

        lpc,recovered  = self.decode(zs)
        # def get_
        lp_prior  = self._diag_normal_logp( zs.unsqueeze(-1), prior_mu, prior_beta)
        lp_prior  = ((lp_prior ).sum(2) - math.log(K) ).logsumexp(-1)
        lp_encode = self._diag_normal_logp( zs, encoder_mu,   encoder_beta).sum(-1)
        # lpc = self.unembed(self.std_norm(zs,-1)).log_softmax(-1)
        lp_decode = torch.gather(lpc,index=target.unsqueeze(-1),dim=-1).squeeze(-1) * target_notnull

        lp_weight      = (lp_prior + lp_decode - lp_encode).detach()
        logp_sum_grad  = (lp_prior + lp_decode + lp_encode * lp_weight)#.mean(dim=1)
        logp_sum       = (lp_prior + lp_decode - lp_encode).detach()#.mean(1)
        # logp_per_token = logp_sum
        logp_sum       = (logp_sum )
        loss_per_loc   = logp_sum
        logp_sum       = logp_sum.sum(1)
        logp_sum_grad  = (logp_sum_grad ).sum(1)
        return locals()

    def _loss(self,item, ret):
        CLS_NAME = self.__class__.__name__
        source = item['source'] ### token sequence
        target = item['target'] ### token seq
        # source_embed = self.embed(source)
        # source_len = item['source_len']
        # source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]
        target_embed = self.embed(target)

        T = target.size(1)
        B = target.size(0)
        E = self.config.embed_dim
        N = W = self.config.window_size
        K = self.config.kernel_size
        target_len = item['target_len']
        xT  =  torch.arange(target.size(1),device=self.device)[None,:]
        target_notnull = (xT<target_len[:,None]) & (xT>=2)
        # target_notnull =

        if (item['has_start_token'].__class__==int
            and item['has_start_token']==1) or  item['has_start_token'].ravel()[0]==1:
            target_notnull[:,0] = False

        if 1:
            '''
            Resampling Encoder
            '''
            ### (B,T,E)
            ### (1,1,E)
            if ret=='encode':
                return encoder_mu

            def set_requires_grad(module, val):
                for p in module.parameters():
                    p.requires_grad = val
            '''
            Inferring the latent
            '''
            D = self.config.depth


            encoder_mu    = target_embed
            encoder_mu_0  = encoder_mu
            if CLS_NAME in 'DLM131'.split():
                for i in range(D):
                    encoder_mu = encoder_mu + self.conv_layer_list[i](encoder_mu.tanh().transpose(1,2)).transpose(1,2)
                encoder_mu    = encoder_mu * target_notnull.unsqueeze(-1)

                nrd = torch.normal( 0, 1, xd['encoder_mu'].shape, device=self.device)
                sampled_zs = nrd / xd['encoder_beta'] + xd['encoder_mu']

            elif CLS_NAME in 'DLM132 DLM133 DLM134 DLM135 DLM137'.split():
                set_requires_grad(self,False)
                # encoder_mu =  torch.normal( 0, 1, encoder_mu.shape, device=self.device,requires_grad=True)
                encoder_mu = self.lat_init(**locals())
                # encoder_mu = torch.tensor(0*encoder_mu,device=self.device,requires_grad=True)
                # / encoder_beta + encoder_mu
                # encoder_mu.requires_grad = True
                for i in range(10):
                    self.train()
                    encoder_mu.requires_grad = True
                    xd = self._get_loss(encoder_mu, encoder_mu.detach(), target, target_notnull)
                    lp = (xd['lp_prior'].sum(1) + xd['lp_decode'].sum(1))
                    lp = (lp).sum(0)
                    # print(f'''{lp.item():.3f}''')
                    lp.backward(retain_graph=True)
                    with torch.no_grad():
                        encoder_mu = encoder_mu.add_( 0.1 * encoder_mu.grad)
                    # print(encoder_mu.grad.std())
                encoder_mu.requires_grad = False
                encoder_mu = encoder_mu.detach()
                set_requires_grad(self,True)

                sampled_zs = self.sample_zs(xd, self.device)
            elif CLS_NAME in 'DLM136'.split():
                h0 = torch.ones([self.D,B,E],device=self.device)
                encoder_mu = self.rnn_enc( target_embed, h0)[0]
                sampled_zs = self.sample_zs(dict(encoder_mu = encoder_mu),self.device)
            elif CLS_NAME in 'DLM138'.split():
                h0 = torch.ones([self.D,B,E],device=self.device)
                encoder_mu = self.rnn_enc( self.shift_pad_left(target_embed,1), h0)[0]
                sampled_zs = encoder_mu

            elif CLS_NAME in 'DLM140 DLM141'.split():
                encoder_mu= target_embed
                sampled_zs = encoder_mu
            else:
                raise NotImplementedError

            '''
            recalc encoder_mu according to pointwise noise-free loss
            Required by the importance sampling
            '''
            # (B,T,E,1) - (1,1,E,K)
            '''
            Given zs, calculate lprior + ldecode - lencode
            '''


        xd = self._get_loss(sampled_zs, encoder_mu, target,target_notnull)
        if ret=='debug':
            sampled_zs = xd['sampled_zs']
            locals().update(xd)
            return locals()


        att = torch.zeros((B,T,1),device = self.device)
        ### normalise by average token count, but sum by batchsize
        # lossVal = lossVal/ (source_notnull.sum() / B)
        self.debug = 0
        if self.debug:
            print(f'''{xd['lp_prior'].mean().item():.3f}, {xd['lp_decode'].mean().item():.3f}, {xd['lp_encode'].mean().item():.3f}''')
            # print(f'e_beta:{encoder_beta.mean():.4f} w_beta:{prior_beta.mean():.4f} '
            # f'e_mu_mse:{encoder_mu.square().mean():.4f} w_mu_mse:{prior_mu.square().mean():.4f} ')
            # print(f'lp_prior:{lp_prior_list.mean():.4f}  lp_decode:{lp_decode_list.mean():.4f} '
            # f'lp_encode:{lp_encode_list.mean():.4f}')

        if ret=='forward':
            return xd['logp_sum'].unsqueeze(-1),att
        elif ret in 'loss'.split():
            return -xd['logp_sum']
        elif ret=='grad_loss':
            return -xd['logp_sum_grad']
        elif ret=='loss_per_loc':
            return xd['loss_per_loc'],target_notnull
        else:
            raise NotImplementedError(f'''{ret} for ._loss()''')


class DLM132(DLM131):
    pass
    @staticmethod
    def lat_init(self,encoder_mu,**kw):
        return torch.tensor(0*encoder_mu,device=self.device,requires_grad=True)
    def decode(self,zs):
        D = self.config.depth
        for i in range(D):
            zs = self.convt_layer_list[i](zs.transpose(1,2)).transpose(1,2)
        ### zs (B,T,E)  (1,1,E,G)
        # lpc = zs.unsqueeze(-1) - self.embed.weight.T[None,None]
        # lpc = (-lpc.square().mean(2)).log_softmax(-1)
        lpc = self.unembed(zs).log_softmax(-1)
        return lpc,zs
    def sample_zs(self,xd,device):
        nrd = torch.normal( 0, 1, xd['encoder_mu'].shape, device=self.device)
        sampled_zs = nrd / xd['encoder_beta'] + xd['encoder_mu']
        return sampled_zs

class DLM133(DLM132):
    pass

    def _get_loss(self,zs, encoder_mu,target,target_notnull):
        K = self.K
        encoder_beta  = self.encoder_beta.exp()
        encoder_beta  = encoder_beta.clip(self.config.p_null,None)
        prior_mu   = self.prior_mu[:,:,:,0:1]
        prior_beta = self.prior_beta.exp()

        lpc,recovered  = self.decode(zs)
        # def get_
        # lp_prior  = self._diag_normal_logp( zs.unsqueeze(-1), prior_mu, prior_beta)
        # lp_prior  = ((lp_prior ).sum(2) - math.log(K) ).logsumexp(-1)

        # zsd  = torch.cat([zs[:,0:1], zs[:,:-1] - zs[:,1:]],dim=1)
        # lp_prior  = self._diag_normal_logp( zsd.unsqueeze(-1), prior_mu, prior_beta).squeeze(-1).sum(-1)
        lp_encode = self._diag_normal_logp( zs, encoder_mu,   encoder_beta).sum(-1)
        lp_prior = lp_encode * 0.
        # lpc = self.unembed(self.std_norm(zs,-1)).log_softmax(-1)
        lp_decode = torch.gather(lpc,index=target.unsqueeze(-1),dim=-1).squeeze(-1) * target_notnull

        lp_weight      = (lp_prior + lp_decode - lp_encode).detach()
        logp_sum_grad  = (lp_prior + lp_decode + lp_encode * lp_weight)#.mean(dim=1)
        logp_sum       = (lp_prior + lp_decode - lp_encode).detach()#.mean(1)
        # logp_per_token = logp_sum
        logp_sum       = (logp_sum )
        loss_per_loc   = logp_sum
        logp_sum       = logp_sum.mean(1)
        logp_sum_grad  = (logp_sum_grad ).mean(1)
        return locals()


if __name__=='__main__':
    # torch.random
    func = DLM112._diag_normal_wrapped_logp
    print(f'[testing]{func.__name__}')

    xs0 = xs = torch.linspace(-1.,1,200)
    # for mu_p in (0.5)
    dx = xs[1]-xs[0]
    xs = xs[:-1] + dx
    xs = xs[:,None,None]
    mu_p    = torch.linspace(-0.5,0.5, 5)[None,:,None]
    beta_p  = torch.linspace(0.5,2.5, 5)[None,None,:]
    xsample = DLM112._diag_normal_wrapped_sample((5000,1,1),mu_p,beta_p)
    lp = func(xs,mu_p,beta_p)
    p = lp.exp()
    spdf = lp.exp().sum(0)*dx
    assert torch.allclose(torch.tensor(1.),spdf),spdf
    import matplotlib;matplotlib.use('agg')
    import matplotlib.pyplot as plt
    print(f'spdf={spdf}')
    with open(__file__+'.test.html','w') as buf:
        ps = p.reshape((p.shape[0],-1)).T
        xsample_list = xsample.reshape((len(xsample),-1)).T
        for i,pss in enumerate(ps):
            xsr = xs.ravel()
            plt.plot(xs.ravel(), pss)
            xsss = xsample_list[i]
            # import pdb; pdb.set_trace()
            cts,edges = torch.histogram( xsample_list[i], xs0.ravel())
            # plt.savefig()
            plt.plot(xsr,cts/xsample_list.shape[1]/(xsr[1]-xsr[0]),'r--')
            # plt.bar(xsr, cts,alpha=0.15,wids=xsr[1]-xsr[0])
            plt.ylim(0,2)
            plt.xlim(-1,1)
            fig = plt.gcf()
            buf.write(write_png_tag(fig))
            plt.close(fig)


class DLM134(DLM132):
    def decode(self,zs):
        D = self.config.depth
        for i in range(D):
            # zs = (0.5*z   s + 0.5*self.convt_layer_list[i](zs.transpose(1,2)).transpose(1,2))
            zs = (zs + self.convt_layer_list[i](zs.transpose(1,2)).transpose(1,2))
        lpc = self.unembed(zs).log_softmax(-1)
        return lpc,zs

    def _get_loss(self,zs, encoder_mu,target,target_notnull):
        '''
        encoder_mu:      is the noise-free param
        zs:              is the sampled latent
        target:          the wanted decoding result
        target_notnull:  mask to ignore unspecified

        describ:
            zs is sampled in the free domain
            encoder_mu is in the free domain
              but this would be problematic, since the loss function cannot be calculated
              in the wrapped domain...
        we cannot expect to avoid the wrapped domain caculation
        '''
        K = self.K
        encoder_beta  = self.encoder_beta.exp()
        encoder_beta  = encoder_beta.clip(self.config.p_null,None)
        prior_mu   = self.prior_mu[:,:,:,0:1]
        prior_beta = self.prior_beta.exp()

        zs = zs * (torch.arange(zs.shape[1],device=self.device)[None,:,None] % self.config.depth ==0)
        lpc,recovered  = self.decode(zs.sin())
        # torch.autograd.set_detect_anomaly(True)
        # lp_prior  = self._diag_normal_logp( zs.unsqueeze(-1), prior_mu, prior_beta)
        # lp_prior  = ((lp_prior ).sum(2) - math.log(K) ).logsumexp(-1)

        # zsd  = torch.cat([zs[:,0:1], zs[:,:-1] - zs[:,1:]],dim=1)
        # lp_prior  = self._diag_normal_logp( zsd.unsqueeze(-1), prior_mu, prior_beta).squeeze(-1).sum(-1)
        # zs = zs.sin()
        # _encoder_mu = encoder_mu
        # encoder_mu = _encoder_mu.sin()
        lp_encode = self._diag_normal_wrapped_logp(zs.sin(), encoder_mu.sin(), encoder_beta).sum(-1)
        # lp_prior  = self._diag_normal_wrapped_logp(zs.sin(), prior_mu.squeeze(-1).sin(), prior_beta.squeeze(-1)).sum(-1)
        # lp_encode = self._diag_normal_logp( zs, encoder_mu,   encoder_beta).sum(-1)
        # lp_encode = lp_encode * 0.
        lp_prior = -math.log(2) * torch.ones_like(zs, device = self.device).sum(-1)

        '''
        Use a fixed prior log(1./2^D) = - D * log(2)
        '''
        # lp_prior = lp_encode * 0.
        # lpc = self.unembed(self.std_norm(zs,-1)).log_softmax(-1)
        lp_decode = torch.gather(lpc,index=target.unsqueeze(-1),dim=-1).squeeze(-1) * target_notnull
        # lp_decode = lp_decode(-1)
        # print(lp_prior.mean().item(), lp_decode.mean().item(),lp_encode.mean().item())
        # import pdb; pdb.set_trace()
        lp_weight      = (lp_prior.sum(1) + lp_decode.sum(1) - lp_encode.sum(1)).detach()
        logp_sum_grad  = (lp_prior.sum(1) + lp_decode.sum(1) + lp_encode.sum(1) * lp_weight)#.mean(dim=1)
        logp_sum       = (lp_prior.sum(1) + lp_decode.sum(1) - lp_encode.sum(1)).detach()#.mean(1)
        # logp_per_token = logp_sum
        logp_sum       = (logp_sum )
        loss_per_loc   = logp_sum
        # logp_sum       = logp_sum.sum(1)
        logp_sum_grad  = (logp_sum_grad )
        #.sum(1)
        return locals()

    def sample_zs(self,xd,device):

        sampled_zs = self._diag_normal_wrapped_sample(xd['encoder_mu'].shape, xd['encoder_mu'].sin(), xd['encoder_beta'])
        sampled_zs = sampled_zs.arcsin()
        # sampled_zs = xd['encoder_mu']
        return sampled_zs




class DLM136(DLM131):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = self.config.embed_dim
        D = self.config.depth
        self.D = D
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=D, )
        self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=D, )
        return

    def decode(self,zs):
        ### (B, H, E )  @ ( (T, E) @ (E, E) )
        recovered = self.rnn_dec(zs, torch.ones_like(zs[None,:,0,:]).repeat((self.D,1,1)))[0]
        lpc = self.unembed(recovered).log_softmax(-1)
        return lpc,recovered

    def _get_loss(self,zs, encoder_mu,target,target_notnull):
        '''
        encoder_mu:      is the noise-free param
        zs:              is the sampled latent
        target:          the wanted decoding result
        target_notnull:  mask to ignore unspecified

        describ:

        convolution seems to cause strong correlation between neighbouring cells
        and is unwanted.
        '''
        K = self.K
        encoder_beta  = self.encoder_beta.exp()
        encoder_beta  = encoder_beta.clip(self.config.p_null,None)
        prior_mu   = self.prior_mu[:,:,:,0:1]
        prior_beta = self.prior_beta.exp()
        # zs =

        # zs = zs * (torch.arange(zs.shape[1],device=self.device)[None,:,None] % self.config.depth ==0)
        lpc,recovered  = self.decode(zs.sin())

        # _encoder_mu = encoder_mu
        # encoder_mu = _encoder_mu.sin()
        lp_encode = self._diag_normal_wrapped_logp(zs.sin(), encoder_mu.sin(), encoder_beta).sum(-1)
        lp_prior  = self._diag_normal_wrapped_logp(zs.sin(), prior_mu.squeeze(-1).sin(), prior_beta.squeeze(-1)).sum(-1)
        # lp_encode = self._diag_normal_logp( zs, encoder_mu,   encoder_beta).sum(-1)
        # lp_encode = lp_encode * 0.
        # lp_prior = -math.log(2) * torch.ones_like(zs, device = self.device).sum(-1)

        '''
        Use a fixed prior log(1./2^D) = - D * log(2)
        '''
        # lpc = self.unembed(self.std_norm(zs,-1)).log_softmax(-1)
        lp_decode = torch.gather(lpc,index=target.unsqueeze(-1),dim=-1).squeeze(-1) * target_notnull
        # lp_decode = lp_decode*0.
        # lp_prior = lp_prior * 0.
        # lp_encode = lp_encode*0.
        # lp_decode = lp_decode(-1)
        # print(lp_prior.mean().item(), lp_decode.mean().item(),lp_encode.mean().item())
        # import pdb; pdb.set_trace()
        lp_weight      = (lp_prior.sum(1) + lp_decode.sum(1) - lp_encode.sum(1)).detach()
        logp_sum_grad  = (lp_prior.sum(1) + lp_decode.sum(1) + lp_encode.sum(1) * lp_weight)#.mean(dim=1)
        logp_sum       = (lp_prior.sum(1) + lp_decode.sum(1) - lp_encode.sum(1)).detach()#.mean(1)
        # logp_per_token = logp_sum
        # logp_sum       = (logp_sum )
        loss_per_loc   = logp_sum
        # logp_sum       = logp_sum.sum(1)
        logp_sum_grad  = (logp_sum_grad )

        recovered = recovered * target_notnull.unsqueeze(-1)
        return locals()


    def sample_zs(self,xd,device):
        encoder_beta = self.encoder_beta.exp()
        encoder_beta  = encoder_beta.clip(self.config.p_null,None)
        sampled_zs = self._diag_normal_wrapped_sample(xd['encoder_mu'].shape, xd['encoder_mu'].sin(), encoder_beta)
        # xd['encoder_beta'])
        sampled_zs = sampled_zs.clip(-0.999,0.9999).arcsin()
        return sampled_zs

class DLM137(DLM136):
    pass
    @staticmethod
    def lat_init(self,encoder_mu,**kw):
        return torch.tensor(0*encoder_mu,device=self.device,requires_grad=True)


class DLM138(DLM137):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = self.config.embed_dim
        D = self.config.depth
        self.D = D
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=D, )
        # self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=D, )
        return

    def decode(self,zs):
        ### (B, H, E )  @ ( (T, E) @ (E, E) )
        recovered = self.rnn_dec(zs, torch.ones_like(zs[None,:,0,:]).repeat((self.D,1,1)))[0]
        lpc = self.unembed(recovered).log_softmax(-1)
        return lpc,recovered

    def _get_loss(self, zs, encoder_mu,target,target_notnull):
        '''
        encoder_mu:      is the noise-free param
        zs:              is the sampled latent
        target:          the wanted decoding result
        target_notnull:  mask to ignore unspecified

        describ:

        convolution seems to cause strong correlation between neighbouring cells
        and is unwanted.
        '''
        K = self.K
        encoder_beta  = self.encoder_beta.exp()
        encoder_beta  = encoder_beta.clip(self.config.p_null,None)
        # prior_mu   = self.prior_mu[:,:,:,0:1]
        # prior_beta = self.prior_beta.exp()

        lpc,recovered  = self.decode(zs)
        lp_prior = 0. * torch.ones_like(zs, device = self.device).sum(-1)
        lp_encode = lp_prior

        '''
        Use a fixed prior log(1./2^D) = - D * log(2)
        '''
        # lpc = self.unembed(self.std_norm(zs,-1)).log_softmax(-1)
        lp_decode = torch.gather(lpc,index=target.unsqueeze(-1),dim=-1).squeeze(-1) * target_notnull
        # lp_decode = lp_decode*0.
        # lp_prior = lp_prior * 0.
        # lp_encode = lp_encode*0.
        # lp_decode = lp_decode(-1)
        # print(lp_prior.mean().item(), lp_decode.mean().item(),lp_encode.mean().item())
        # import pdb; pdb.set_trace()
        lp_weight      = (lp_prior.sum(1) + lp_decode.sum(1) - lp_encode.sum(1)).detach()
        logp_sum_grad  = (lp_prior.sum(1) + lp_decode.sum(1) + lp_encode.sum(1) * lp_weight)#.mean(dim=1)
        logp_sum       = (lp_prior.sum(1) + lp_decode.sum(1) - lp_encode.sum(1)).detach()#.mean(1)
        # logp_per_token = logp_sum
        # logp_sum       = (logp_sum )
        loss_per_loc   = logp_sum
        # logp_sum       = logp_sum.sum(1)
        logp_sum_grad  = (logp_sum_grad )
        logp_sum = logp_sum #/ target_notnull.sum(1).clip(1,None)

        recovered = recovered * target_notnull.unsqueeze(-1)
        return locals()


    def sample_zs(self,xd,device):
        encoder_beta = self.encoder_beta.exp()
        encoder_beta  = encoder_beta.clip(self.config.p_null,None)
        sampled_zs = self._diag_normal_wrapped_sample(xd['encoder_mu'].shape, xd['encoder_mu'].sin(), encoder_beta)
        # xd['encoder_beta'])
        sampled_zs = sampled_zs.clip(-0.999,0.9999).arcsin()
        return sampled_zs


class DLM135(DLM131):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)

        self.H = 15

        G = config.graph_dim +1
        self.G = G
        E = config.embed_dim
        x = nn.Linear(1, 1 ).to(self.device)
        self.W = W = config.window_size
        self.embed      = nn.Embedding(G, E).to(self.device)

        T = 100
        x = nn.Linear(T, E ).to(self.device)
        self.grid_embedding = nn.Parameter(1000*(x.weight.T))

        x = nn.Linear(E, E ).to(self.device)
        self.w_k = nn.Parameter(x.weight.T)

        x = nn.Linear(E, E ).to(self.device)
        self.w_v = nn.Parameter(x.weight.T)

        # self.rnn      = nn.GRU(input_size = E,hidden_size=E, batch_first= True, num_layers=1)
        K = config.kernel_size
        assert K >=1,config
        # self.unembed = nn.Linear(config.embed_dim, config.graph_dim).to(self.device)

    @staticmethod
    def lat_init(self,encoder_mu,**kw):
        H = self.H
        B,T,E = (encoder_mu.shape)
        # ret = torch.eye(H,E,device=self.device)[None].repeat((B,1,1))
        # return ret
        return torch.rand((B,H,E),device=self.device)

    def decode(self,zs):
        ### (B, H, E )  @ ( (T, E) @ (E, E) )
        for i in range(self.config.depth):
            att0 = zs @ ((self.grid_embedding @ self.w_k)).T
            att = att0.relu().softmax(1)
            # zs = zs + att.transpose(2,1) @ ((zs @ self.w_v))
            zs = zs + att0.relu().softmax(2) @ (att.transpose(2,1) @ ((zs @ self.w_v)))
            # zs = zs + att @ (att.transpose(2,1) @ ((zs @ self.w_v)))
            # zs = zs +  ((zs.relu() @ self.w_v))
        recovered = att.transpose(2,1) @ ((zs @ self.w_v))

        # recovered = zs
        lpc = self.unembed(recovered).log_softmax(-1)
        return lpc,recovered

    def _get_loss(self,zs, encoder_mu,target,target_notnull):
        '''
        encoder_mu:      is the noise-free param
        zs:              is the sampled latent
        target:          the wanted decoding result
        target_notnull:  mask to ignore unspecified

        describ:

        convolution seems to cause strong correlation between neighbouring cells
        and is unwanted.
        '''
        K = self.K
        encoder_beta  = self.encoder_beta.exp()
        encoder_beta  = encoder_beta.clip(self.config.p_null,None)
        prior_mu   = self.prior_mu[:,:,:,0:1]
        prior_beta = self.prior_beta.exp()
        # zs =

        # zs = zs * (torch.arange(zs.shape[1],device=self.device)[None,:,None] % self.config.depth ==0)
        lpc,recovered  = self.decode(zs.sin())

        # _encoder_mu = encoder_mu
        # encoder_mu = _encoder_mu.sin()
        lp_encode = self._diag_normal_wrapped_logp(zs.sin(), encoder_mu.sin(), encoder_beta).sum(-1)
        # lp_prior  = self._diag_normal_wrapped_logp(zs.sin(), prior_mu.squeeze(-1).sin(), prior_beta.squeeze(-1)).sum(-1)
        # lp_encode = self._diag_normal_logp( zs, encoder_mu,   encoder_beta).sum(-1)
        # lp_encode = lp_encode * 0.
        lp_prior = -math.log(2) * torch.ones_like(zs, device = self.device).sum(-1)

        '''
        Use a fixed prior log(1./2^D) = - D * log(2)
        '''
        lp_prior = lp_encode * 0.
        # lpc = self.unembed(self.std_norm(zs,-1)).log_softmax(-1)
        lp_decode = torch.gather(lpc,index=target.unsqueeze(-1),dim=-1).squeeze(-1) * target_notnull
        # lp_decode = lp_decode(-1)
        # print(lp_prior.mean().item(), lp_decode.mean().item(),lp_encode.mean().item())
        # import pdb; pdb.set_trace()
        lp_weight      = (lp_prior.sum(1) + lp_decode.sum(1) - lp_encode.sum(1)).detach()
        logp_sum_grad  = (lp_prior.sum(1) + lp_decode.sum(1) + lp_encode.sum(1) * lp_weight)#.mean(dim=1)
        logp_sum       = (lp_prior.sum(1) + lp_decode.sum(1) - lp_encode.sum(1)).detach()#.mean(1)
        # logp_per_token = logp_sum
        # logp_sum       = (logp_sum )
        loss_per_loc   = logp_sum
        # logp_sum       = logp_sum.sum(1)
        logp_sum_grad  = (logp_sum_grad )

        recovered = recovered * target_notnull.unsqueeze(-1)
        return locals()

    def sample_zs(self,xd,device):

        sampled_zs = self._diag_normal_wrapped_sample(xd['encoder_mu'].shape, xd['encoder_mu'].sin(), xd['encoder_beta'])
        sampled_zs = sampled_zs.arcsin()
        return sampled_zs


class DLM137(DLM136):
    @staticmethod
    def lat_init(self,encoder_mu,**kw):
        return torch.tensor(0*encoder_mu,device=self.device,requires_grad=True)



class DLM140(DLM128):
    '''
    recover states from noise
    '''
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = self.config.embed_dim
        D = self.config.depth
        self.D = D
        self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=D, )
        self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=D, )
        x = nn.Linear(E,2).to(self.device)
        self.is_kept   =  x
        self.is_predict_source = (self.config.beta>0.5)
        # nn.Parameter(x.weight.T[None,None])
        self.W = 20
        self.K = 5

        return

    def decode(self, t2_embed):
        D = self.config.depth
        (BW, T, E)= t2_embed.shape
        h0 = torch.ones((D,BW,E),device=self.device)
        zs,_ = self.rnn_enc(t2_embed,h0)
        zs,_ = self.rnn_dec(zs.flip([2,]),h0)
        zs = zs.flip([2,])

        '''
        Allow the model choose to keep the token or inserting the token
        '''
        is_kept_lp = self.is_kept(zs).log_softmax(-1)

        logp_repl = self.unembed(zs).log_softmax(-1)

        logp_kept = self.unembed(t2_embed).log_softmax(-1)
        x = torch.stack([logp_repl,logp_kept],dim=2)
        logp = (is_kept_lp.unsqueeze(-1) + x).logsumexp(2)

        # logp = logp_repl

        return zs,logp

    @staticmethod
    def callback_after_test_all(conf, model,  item):
        vis = conf.vis;  env = conf._session_name;
        if vis is None: return;
        DLM128.callback_after_test_all(conf,model,item)
        xd = model._loss(item,ret='debug')
        key = 'logp_sum_byt'

        if xd.get(key,None) is not None:
            val = xd[key].mean(0)
            _min = None
            _max = 200
            vis.line( np.clip(U.N(val),_min,_max),X=np.arange(len(val)),opts=dict(title=key,),win=key, env=env)

        key = 'random_sample'
        with torch.no_grad():
            val0 = model.sample(*item['target'].shape, item=item)
        if val0 is not None:
            val = val0.unsqueeze(-1)
            vis.heatmap(U.N(torch.cat(list(val[:3]),dim=1)),env=env,opts=dict(title=key),win=key)
            arr = U.N(val0[:5]).clip(0, conf.dataset.vocab.__len__()-1)
            sents = [ ''.join([conf.dataset.tgt_wordize(vv).replace('<','|').replace('>','|') for vv in arrr])  for arrr in arr]
            joined_sents =  '<br/>'.join(sents)
            vis.text(f'''
RandomSample:
<br/>
{joined_sents}
            ''',
            env=env,win=key + '_text')
        # key = 'debug'

    def _sample_step(self, xs, ):
        ys,logp,_ = self.decode(self.embed(xs))
        if logp.shape.__len__()==4:
            logp = logp.logsumexp(0)
        lp, xs = self.sample_logp(logp, dim=-1,return_logp=True)
        return xs

    def sample(self, B,T, item=None):
        ### randomly init the tokens
        xs = torch.randint(self.G,[B,T], device=self.device)
        for k in range(self.W*4):
            ### improving the token
            xs = self._sample_step(xs)
            # import pdb; pdb.set_trace()
        return xs


    def corrupt_target(self,target,target_notnull,generator=None):
        W = self.W
        K = self.K

        p_null = self.config.p_null
        B,T = target.shape
        D = self.config.depth

        t1_mask     = torch.rand( [B,W+1,T], device=self.device,generator=generator)
        t1_mask     = (  t1_mask < (p_null * target_notnull.unsqueeze(1)) )
        t2_mask     = t1_mask[:,1:]
        t1_mask     = t1_mask[:,:-1]
        t1_mask[:,0] = 0

        t1_mask_cum = t1_mask.cumsum(dim=1) > 0

        t1_rint     = torch.randint(self.G,[B,W,T], device=self.device,generator=generator)
        t2_rint     = torch.randint(self.G,[B,W,T], device=self.device,generator=generator)
        t1 = target.unsqueeze(1) * ~t1_mask_cum + t1_rint * t1_mask_cum
        t2 = t1 * ~t2_mask + t2_rint* t2_mask

        t2 = t2.reshape((B*W,T))
        '''
        Predict the original
        '''

        if self.is_predict_source:
            # t1 = target.unsqueeze(1).repeat((1,W,1)).reshape((B*W,T))
            t1 = target.unsqueeze(1).repeat((1,W,1))
        else:
            pass
        t1 = t1.reshape((B*W,T))
        recovered = t1
        corrupted = t2
        return recovered, corrupted

    def add_target_notnull(self,item):
        target_len = item['target_len']
        target  = item['target']
        xT  =  torch.arange(target.size(1),device=self.device)[None,:]
        target_notnull = (xT<target_len[:,None]) & (xT>=2)
        item['target_notnull'] = target_notnull
        return item

    def _get_loss(self, ret, encoder_mu,target,target_notnull,generator=None):
        '''
        encoder_mu:      is the noise-free param
        zs:              is the sampled latent
        target:          the wanted decoding result
        target_notnull:  mask to ignore unspecified

        describ:
            use a fixed noise to perturb target recursively.

        因为替换一次和替换两次是没有区别的，所以对于多次采样可以利用这种等价性。
        每次固定只perturb k个位置，这个其实不是很poisson，或者不是条件独立的。
        如果poisson一点，那其实每个位置被corrupt的概率均等，采样起来比较方便。
        '''
        cname = self.__class__.__name__

        W = self.W
        K = self.K
        E = self.embed_dim


        p_null = self.config.p_null
        B,T = target.shape
        D = self.config.depth


        t1, t2 = self.corrupt_target(target, target_notnull, generator)
        t2_embed = self.embed(t2)
        zs,logps,decode_dict  = self.decode(t2_embed)

        if logps.shape.__len__() == 3:
            logps = logps.unsqueeze(0)
        # att = logps.log_softmax(0)
        # logp = logps.logsumexp(0)

        if cname in 'DLM150'.split():
            logp = logps.logsumexp(0)
            n = 5
            _logMC, tm = self.sample_logp(logp, dim=-1,return_logp=True, n=n)
            # sshape = (n*B*W,T,-1)
            tm = tm.transpose(-1,0).reshape( (n*B*W,T,) )

            _, logp  = self.decode(self.embed(tm))
            logp = logp.reshape( (n,B*W,T,-1) )
            # assert 0, logp.shape
            lps   = torch.gather(logp,index=t1.unsqueeze(-1).unsqueeze(0).expand(*( (n,)+ (-1,)*(len(logp.shape)-1) ) ),dim=-1).squeeze(-1)
            lps   = lps.reshape((n,B*W,T))
            _logMC = _logMC.reshape( (n,B*W,T,) )
            ### lps is log(O|M), _log
            _logMCM = _logMC.mean(-1)#.log_softmax(0)

            if 1:
                'REINFORCE trick'
                reward = (lps.mean(-1).exp())
                lgrad = ( reward.detach() * _logMCM )  + reward
            else:
                lgrad =  (lps.mean(-1).exp() )

            # logp_sum      = (lps*target_notnull.unsqueeze(1)).sum(-1)
            # logp_sum      = logp_sum.mean(-1)
            logp_sum_grad = lgrad.reshape((n,B,W)).mean(0).mean(-1) * T
            logp_sum      = logp_sum_grad
            loss_per_loc  = -lps.reshape((n,B,W,T)).mean(0)

            # del logp
        else:
            # logp = logps.logsumexp(0)
            # lps = torch.gather(logp,index=t1.unsqueeze(-1),dim=-1).squeeze(-1)
            ### (D,BW,T,G)
            D2 = logps.shape[0]
            lpsa = torch.gather(logps,index=t1.unsqueeze(-1).unsqueeze(0).repeat((D2,1,1,1)),dim=-1).squeeze(-1)

            ###  (D,BW,T)
            ###  (BW,D,T) transposed
            #### (B,W,D,T,) reshaped
            lpsa = lpsa.transpose(0,1).reshape((B,W,D2,T))

            lps  = lpsa.logsumexp(2)
            lps           = lps.reshape((B,W,T))

            logp_sum_byt  = (lps*target_notnull.unsqueeze(1)).sum(-1)
            logp_sum      = logp_sum_byt.mean(-1)
            logp_sum_grad = logp_sum

            pass

        if 1:
            #

            ### lps for logp_s

            encoder_mu_0  = self.embed(target)
            encoder_mu    = t2_embed.reshape((B,W,T,E))[:,0]

            recovered     = zs.reshape((B,W,T,E))[:,0]
            sampled_zs    =  t2_embed.reshape((B,W,T,E))[:,0] -  self.embed(t1).reshape((B,W,T,E))[:,0]
            loss_per_loc  = -lps



        # ### normalise by average token count, but sum by batchsize
        # # lossVal = lossVal/ (source_notnull.sum() / B)
        # self.debug = 0
        att = torch.zeros((B,T,1),device = self.device)
        masked_loss_per_loc = loss_per_loc *target_notnull[:,None,:]

        xd = locals()

        # if self.debug:
        #     print(f'''{xd['lp_prior'].mean().item():.3f}, {xd['lp_decode'].mean().item():.3f}, {xd['lp_encode'].mean().item():.3f}''')
        if ret=='debug':
            return xd
        elif ret=='forward':
            return xd['logp_sum'].unsqueeze(-1),att
        elif ret in 'loss'.split():
            return -xd['logp_sum']
        elif ret=='grad_loss':
            return -xd['logp_sum_grad']
        elif ret=='loss_per_loc':
            return xd['loss_per_loc']
        elif ret=='masked_loss_per_loc':
            return xd[ret]
        else:
            raise NotImplementedError(f'''{ret} for ._loss()''')



    def _loss(self,item, ret, generator= None):
        # if rng is not None:
        #     # assert 0
        #     print(rng)
        #     torch.set_rng_state(rng)
        item = self.add_target_notnull(item)

        CLS_NAME = self.__class__.__name__
        source = item['source'] ### token sequence
        target = item['target'] ### token seq
        # source_embed = self.embed(source)
        # source_len = item['source_len']
        # source_notnull = torch.arange(source.size(1),device=self.device)[None,:]<source_len[:,None]
        target_embed = self.embed(target)

        T = target.size(1)
        B = target.size(0)
        E = self.config.embed_dim
        N = W = self.config.window_size
        K = self.config.kernel_size

        # if (item['has_start_token'].__class__==int
        #     and item['has_start_token']==1) or  item['has_start_token'].ravel()[0]==1:
        #     target_notnull[:,0] = False

        if 1:
            '''
            Legacy variables for plotting in debug locals()
            '''
            D             = self.config.depth
            encoder_mu    = target_embed
            encoder_mu_0  = encoder_mu
            sampled_zs    = encoder_mu


        xd = self._get_loss(ret, None, target, item['target_notnull'],generator=generator)
        return xd

    @staticmethod
    def get_debug_data(item,confs):

        conf1 = confs[0]
        # dataset = conf1.dataset
        generator = torch.Generator(device=conf1.device)
        # rng = generator.get_state()
        # with torch.no_grad():
        seed = generator.seed()
        loss_per_loc = []
        debug_dicts  = []
        for i, conf in enumerate(confs):
            model = conf.model
            generator = generator.manual_seed(seed)
            item = model.add_target_notnull(item)
            T = item['target'].shape[-1]
            t1, t2 = model.corrupt_target(item['target'], item['target_notnull'],generator)
            generator = generator.manual_seed(seed)
            xd = model._loss(item,'debug', generator = generator )
            loss_per_loc += [xd['masked_loss_per_loc'].reshape((-1,T))]
            debug_dicts+= [xd]

        # loss1,loss2 = map((lambda x:) ,[conf1,conf2])
        # loss1,loss2 = map((lambda x:x.model._loss(item,'loss', generator )) ,[conf1,conf2])
        # mat = loss_sum = U.N(torch.stack(loss,dim=-1).sum(2).mean(1))
        mat = loss_sum = U.N(torch.stack(loss_per_loc,dim=-1).sum(1))
        wordize = np.vectorize(conf1.dataset.tgt_wordize)
        G = conf1.dataset.graph_dim

        # T = t1.shape[-1]
        t1 = t1.reshape((-1,T))
        t2 = t2.reshape((-1,T))

        xw1 = wordize(U.N(t1).clip(0,G-1))
        xw2 = wordize(U.N(t2).clip(0,G-1))


        import collections
        cls = collections.namedtuple('_fret','loss_sum loss_per_loc t1 t2 xw1 xw2 debug_dicts')

        return cls(
            loss_sum = U.N(loss_sum),
            loss_per_loc = list(map(U.N,loss_per_loc)), t1 = U.N(t1), t2=U.N( t2 ),
            xw1=xw1,xw2=xw2,
            debug_dicts = debug_dicts,
            )

class DLM141(DLM140):
    '''
    recover states from noise
    '''

    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = self.config.embed_dim
        D = self.config.depth
        self.D = D
        self.conv_layer_list = nn.ModuleList([nn.Conv1d(E,E,kernel_size = 5, padding='same') for _ in range(D)])

        self.conv_layer = nn.Conv1d(E,E,kernel_size = 5, padding='same')
        self.conv_layer_2 = nn.Conv1d(E,E,kernel_size = 5, padding='same')
        self.convt_layer = nn.Conv1d(E,E,kernel_size = 5, padding='same')
        self.convt_layer_2 = nn.Conv1d(E,E,kernel_size = 5, padding='same')

        # self.rnn_enc = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=D, )
        # self.rnn_dec = nn.GRU(input_size = E, hidden_size=E, batch_first= True, num_layers=D, )
        x = nn.Linear(E,2).to(self.device)
        self.is_kept   =  x
        # nn.Parameter(x.weight.T[None,None])
        return

    def decode(self, t2_embed):
        D = self.config.depth
        (BW, T, E)= t2_embed.shape
        h0 = torch.ones((D,BW,E),device=self.device)
        zs = t2_embed
        zs = zs.transpose(1,2)

        for layer in [self.conv_layer, self.conv_layer_2, self.convt_layer, self.convt_layer_2]:
            # for k in range(3):
                # zs = zs + layer(zs. relu())
            zs = zs + layer(zs).tanh()

        zs = zs.transpose(1,2)

        is_kept_lp = self.is_kept(zs).log_softmax(-1)

        logp_repl = self.unembed(zs).log_softmax(-1)

        logp_kept = self.unembed(t2_embed).log_softmax(-1)
        x = torch.stack([logp_repl,logp_kept],dim=2)
        logp = (is_kept_lp.unsqueeze(-1) + x).logsumexp(2)

        return zs,logp,locals()




class DLM142(DLM140):
    '''
    recover states from noise
    '''

    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = self.config.embed_dim
        D = self.config.depth
        self.D = D
        self.conv_layer_list = nn.ModuleList([nn.Conv1d(E,E,kernel_size = 5, padding='same') for _ in range(D)])
        if hasattr(self,'rnn_dec'): del self.rnn_dec
        if hasattr(self,'rnn_enc'): del self.rnn_enc

        x = nn.Linear(E,2).to(self.device)
        self.is_kept   =  x
        # nn.Parameter(x.weight.T[None,None])
        return

    def decode(self, t2_embed):
        D = self.config.depth
        assert len(t2_embed.shape)==3,t2_embed.shape
        (BW, T, E)= t2_embed.shape
        h0 = torch.ones((D,BW,E),device=self.device)
        zs = t2_embed
        zs = zs.transpose(1,2)

        # for k in range(2):
        for layer in self.conv_layer_list:
            # zs = zs + layer(zs. relu())
            '''
            Important to stabilise the numbers over deep layers
            '''
            # zs = 0.5*zs + 0.5* layer(zs).tanh()
            zs = (zs + layer(zs)).tanh()
            # zs = (zs + layer(zs)).clip(-1, 1)
        ### (BW,T,E)
        zs = zs.transpose(1,2)
        ### (BW,T,2)
        is_kept_lp = self.is_kept(zs).log_softmax(-1)
        logp_repl  = self.unembed(zs).log_softmax(-1)

        self.USE_RESIDUAL=1
        if self.USE_RESIDUAL:
            logp_kept = self.unembed(t2_embed).log_softmax(-1)
            x = torch.stack([logp_repl,logp_kept],dim=2)
            assert 0,x.shape
            logp = (is_kept_lp.unsqueeze(2) + x).logsumexp(2)
        else:
            logp = logp_repl
        return zs,logp,locals()

import numpy as np
_U = U
class DLM152(DLM140):
    '''
    Systematically test the usefulness of a layer-alignment variable
    '''
    U = _U

    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = self.config.embed_dim
        D = self.config.depth
        self.D = D
        self.conv_layer_list = nn.ModuleList([nn.Conv1d(E,E,kernel_size = 5, padding='same') for _ in range(D)])
        if hasattr(self,'rnn_dec'): del self.rnn_dec
        if hasattr(self,'rnn_enc'): del self.rnn_enc

        x = nn.Linear(E,2).to(self.device)
        self.is_kept   =  x
        # self.dbd = {}
        # nn.Parameter(x.weight.T[None,None])
        return

    @staticmethod
    def callback_after_test_all(conf, model,  item):


        from markov_lm.util_plotly import plotly_heatmap_tracks
        # super()
        vis = conf.vis;  env = conf._session_name;
        if vis is None: return;
        DLM140.callback_after_test_all(conf,model,item)

        key = 'test'
        nelem = 0
        # ZMIN,ZMAX = (0.,10.)
        ZMIN,ZMAX = (None,None)
        ZMIN = -10
        YMAX = 60

        xd = model.get_debug_data(item,[conf])

        '''
        Need to visualise the attention matrix.
        for each corrupted sentences
        '''


        # wordize = np.vectorize( conf.dataset.tgt_wordize )
        # G = conf.dataset.graph_dim
        #
        # target = model.U.N(item['target'][:20])
        # wtarget = wordize( target.clip(0,G-1))

        ### (B,W,D,T)
        T = item['target'].shape[-1]
        D = model.config.depth
        lpsa = xd.debug_dicts[0]['lpsa']
        B,W,D,T = lpsa.shape
        xa = model.U.N(lpsa.reshape((-1,D,T)))

        pxa = model.U.N(torch.softmax(torch.tensor(xa),dim=1))

        for (key,ZMIN,xaa) in [('log_prob',-10,xa), ('att',0,pxa)]:
            # .softmax(1)
            # ZMIN = 0

            sep = xd.xw1.copy()
            sep[:] = '-'
            tz = []
            tz += [[sep, xaa[:,0]*0 + ZMIN], ]
            tz += [[xd.xw2, xaa[:,0]*0 + ZMIN], ]
            for idd in range(xaa.shape[1]):
                tz += [[xd.xw1, xaa[:,idd]], ]
            # tz += [[xd.xw1, xa[:,1]], ]
            # tz += [[xd.xw1, xa[:,2]], ]
            # tz += [[xd.xw1, xa[:,3]], ]
            # # tz = [[xd.xw1[:YMAX], xd.loss_per_loc[0][:YMAX]]]


            # tz = [[wtarget, target*0.,  ]]
            # self._loss()

            title = f'{key} {nelem}'
            # assert 0,(tz[0][0][:5][:5])
            # assert 0,(target.shape,target.dtype)
            fig = plotly_heatmap_tracks(tz, ZMIN=ZMIN,ZMAX=ZMAX,YMAX=YMAX,title = title)
            vis.plotlyplot(fig, env=env,win=key)

    def decode(self, t2_embed):
        D = self.config.depth
        assert len(t2_embed.shape)==3,t2_embed.shape
        (BW, T, E)= t2_embed.shape
        zs = t2_embed
        zs = zs.transpose(1,2)

        # for k in range(2):
        hh = torch.zeros((D+2,BW,E, T),device=self.device)
        # output = torch.
        hh[0] = zs
        hh[1] = (zs*0).detach()+1
        for xd, layer in enumerate(self.conv_layer_list):
            # zs = zs + layer(zs. relu())
            '''
            Important to stabilise the numbers over deep layers
            '''
            # zs = 0.5*zs + 0.5* layer(zs).tanh()
            zs = (zs + layer(zs)).tanh()
            hh[xd+2] = zs
            # zs = (zs + layer(zs)).clip(-1, 1)

        zs = zs.transpose(1,2)
        hh = hh.transpose(-2,-1)


        self.USE_RESIDUAL=2
        if self.USE_RESIDUAL==1:
            is_kept_lp = self.is_kept(zs).log_softmax(-1)
            logp_repl  = self.unembed(zs).log_softmax(-1)
            logp_kept = self.unembed(t2_embed).log_softmax(-1)
            x = torch.stack([logp_repl,logp_kept],dim=2)

            logps = (is_kept_lp.unsqueeze(-1) + x).logsumexp(2)
        elif self.USE_RESIDUAL==2:
            loghh = self.unembed(hh).log_softmax(-1)
            prior = torch.ones([hh.shape[0],1,1,1],device=self.device).log_softmax(0)
            logps = (prior + loghh)
            #.logsumexp(0)
        else:
            logp_repl  = self.unembed(zs).log_softmax(-1)

            logps = logp_repl

        return zs,logps,locals()


class DLM153(DLM152):
    '''
    Systematically test the usefulness of a layer-alignment variable
    '''
    U = _U

    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = self.config.embed_dim
        D = self.config.depth
        self.D = D
        self.conv_layer_list = nn.ModuleList([nn.Conv1d(E,E,kernel_size = 5, padding='same') for _ in range(D)])
        if hasattr(self,'rnn_dec'): del self.rnn_dec
        if hasattr(self,'rnn_enc'): del self.rnn_enc
        T = self.config.window_size
        self.prior = nn.Parameter( torch.ones([D+2,1,T,1],device=self.device) )

        x = nn.Linear(E,2).to(self.device)
        self.is_kept   =  x
        # self.dbd = {}
        # nn.Parameter(x.weight.T[None,None])
        return

    @staticmethod
    def callback_after_test_all(conf, model,  item):


        from markov_lm.util_plotly import plotly_heatmap_tracks
        self = model
        # super()
        vis = conf.vis;  env = conf._session_name;
        if vis is None: return;
        DLM152.callback_after_test_all(conf,model,item)

        key = 'prior'
        title = key
        # prior
        nelem = 0
        # ZMIN,ZMAX = (0.,10.)
        ZMIN,ZMAX = (None,None)
        # ZMIN = -10
        YMAX = 60

        # xd = model.get_debug_data(item,[conf])

        '''
        Need to visualise the attention matrix.
        for each corrupted sentences
        '''
        tz = [(None, x.squeeze(-1)) for x in model.U.N(model.prior.softmax(0))]
        fig = plotly_heatmap_tracks(tz, ZMIN=ZMIN,ZMAX=ZMAX,YMAX=YMAX,title = title)
        vis.plotlyplot(fig, env=env,win=key)




    def decode(self, t2_embed):
        D = self.config.depth
        assert len(t2_embed.shape)==3,t2_embed.shape
        (BW, T, E)= t2_embed.shape
        zs = t2_embed
        zs = zs.transpose(1,2)

        # for k in range(2):
        hh = torch.zeros((D+2,BW,E, T),device=self.device)
        # output = torch.
        self.config.hyp = 2
        if self.config.hyp == 0:
            self.config.offset =offset = 0

        if self.config.hyp == 1:
            hh[0] = zs
            self.config.offset =offset = 1
        elif  self.config.hyp == 2:
            # hh[0] = zs
            hh[1] = (zs*0).detach()+1
            self.config.offset =offset = 2
        else:
            raise NotImplementedError(f'self.config.hyp={self.config.hyp}')
        for xd, layer in enumerate(self.conv_layer_list):
            # zs = zs + layer(zs. relu())
            '''
            Important to stabilise the numbers over deep layers
            '''
            # zs = 0.5*zs + 0.5* layer(zs).tanh()
            zs = (zs + layer(zs)).tanh()
            hh[xd+offset] = zs
            # zs = (zs + layer(zs)).clip(-1, 1)

        zs = zs.transpose(1,2)
        hh = hh.transpose(-2,-1)


        self.USE_RESIDUAL=2
        if self.USE_RESIDUAL==1:
            is_kept_lp = self.is_kept(zs).log_softmax(-1)
            logp_repl  = self.unembed(zs).log_softmax(-1)
            logp_kept = self.unembed(t2_embed).log_softmax(-1)
            x = torch.stack([logp_repl,logp_kept],dim=2)

            logps = (is_kept_lp.unsqueeze(-1) + x).logsumexp(2)
        elif self.USE_RESIDUAL==2:
            loghh = self.unembed(hh).log_softmax(-1)
            prior = self.prior.log_softmax(0)[:hh.shape[0]]
            logps = (prior + loghh)
            #.logsumexp(0)
        else:
            logp_repl  = self.unembed(zs).log_softmax(-1)

            logps = logp_repl

        return zs,logps,locals()

class DLM155(DLM152):
    '''
    Systematically test the usefulness of a layer-alignment variable
    '''
    U = _U

    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = self.config.embed_dim
        D = self.config.depth
        T = self.config.n_step
        K = self.config.kernel_size
        self.D = D
        self.conv_layer_list = nn.ModuleList([nn.Conv1d(E,E,kernel_size = 5, padding='same') for _ in range(D)])
        if hasattr(self,'rnn_dec'): del self.rnn_dec
        if hasattr(self,'rnn_enc'): del self.rnn_enc
        self.dense_layer_list   = nn.ModuleList([nn.Linear(E,E) for _ in range(K)])
        # self.dense_layer_list_2 = nn.ModuleList([nn.Linear(E,E) for _ in range(K)])
        self.prior = nn.Parameter( torch.ones([K+2,1,T,1],device=self.device) )

        x = nn.Linear(E,2).to(self.device)
        self.is_kept   =  x
        return

    @staticmethod
    def callback_after_test_all(conf, model,  item):


        from markov_lm.util_plotly import plotly_heatmap_tracks
        self = model
        # super()
        vis = conf.vis;  env = conf._session_name;
        if vis is None: return;
        DLM152.callback_after_test_all(conf,model,item)

        key = 'prior'
        title = key
        # prior
        nelem = 0
        # ZMIN,ZMAX = (0.,10.)
        ZMIN,ZMAX = (0,1)
        # ZMIN = -10
        YMAX = 60

        # xd = model.get_debug_data(item,[conf])

        '''
        Need to visualise the attention matrix.
        for each corrupted sentences
        '''
        tz = [(None, x.squeeze(-1)) for x in model.U.N(model.prior.softmax(0))]
        fig = plotly_heatmap_tracks(tz, ZMIN=ZMIN,ZMAX=ZMAX,YMAX=YMAX,title = title)
        vis.plotlyplot(fig, env=env,win=key)




    def decode(self, t2_embed):
        D = self.config.depth
        assert len(t2_embed.shape)==3,t2_embed.shape
        (BW, T, E)= t2_embed.shape
        zs = t2_embed
        zs = zs.transpose(1,2)

        # for k in range(2):
        K = self.config.kernel_size
        hh = torch.zeros((K+2,BW,E, T),device=self.device)
        for i in range(1):
            # for xd, layer in enumerate(self.conv_layer_list):
            for xd, layer in enumerate(self.conv_layer_list[:]):
                # zs = zs + layer(zs. relu())
                '''
                Important to stabilise the numbers over deep layers
                '''
                # zs = 0.5*zs + 0.5* layer(zs).tanh()
                zs = (zs + layer(zs)).tanh()
        # prior_mask

        self.config.offset =offset = 0
        # offset = 0
        zs = zs.transpose(1,2)
        hh = hh.transpose(-2,-1)
        prior = self.prior.clone()
        for xd, layer in enumerate(self.dense_layer_list):
            zs1 = (zs + layer(zs)).tanh()
            hh[xd+offset] = zs1
            prior[xd+offset] += 100.
            # layer
        '''
        DLM155D7E50, Ep67L49, Ep10L89
        '''
            # zs = (zs + layer(zs)).clip(-1, 1)


        CLS_NAME = self.__class__.__name__
        # self.USE_RESIDUAL=2
        # if self.USE_RESIDUAL==2:

        loghh = self.unembed(hh).log_softmax(-1)
        if CLS_NAME in 'DLM155'.split():
            # prior =
            prior = prior.log_softmax(0)[:hh.shape[0]]
        elif CLS_NAME in 'DLM156'.split():
            ### (K+2,BW,T ,G)
            prior = prior + self.dense_layer_list_switch(zs).permute((2,0,1)).unsqueeze(-1)
            prior = prior.log_softmax(0)[:hh.shape[0]]

        else:
            raise NotImplementedError(CLS_NAME)
        logps = (prior + loghh)
        return zs,logps,locals()


class DLM158(DLM152):
    '''
    Systematically test the usefulness of a layer-alignment variable
    '''
    U = _U

    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = self.config.embed_dim
        D = self.config.depth
        T = self.config.n_step
        K = self.config.kernel_size
        self.D = D
        self.conv_layer_list = nn.ModuleList([nn.Conv1d(E,E,kernel_size = 5, padding='same') for _ in range(D)])
        if hasattr(self,'rnn_dec'): del self.rnn_dec
        if hasattr(self,'rnn_enc'): del self.rnn_enc
        # self.dense_layer_list   = nn.ModuleList([nn.Linear(E,E) for _ in range(K)])
        # self.dense_layer_list_2 = nn.ModuleList([nn.Linear(E,E) for _ in range(K)])
        self.prior = nn.Parameter( torch.ones([K+2,1,T,1],device=self.device) )

        x = nn.Linear(E,2).to(self.device)
        self.is_kept   =  x
        return

    @staticmethod
    def callback_after_test_all(conf, model,  item):


        from markov_lm.util_plotly import plotly_heatmap_tracks
        self = model
        # super()
        vis = conf.vis;  env = conf._session_name;
        if vis is None: return;
        DLM152.callback_after_test_all(conf,model,item)

        key = 'prior'
        title = key
        # prior
        nelem = 0
        # ZMIN,ZMAX = (0.,10.)
        ZMIN,ZMAX = (0,1)
        # ZMIN = -10
        YMAX = 60

        # xd = model.get_debug_data(item,[conf])

        '''
        Need to visualise the attention matrix.
        for each corrupted sentences
        '''
        tz = [(None, x.squeeze(-1)) for x in model.U.N(model.prior.softmax(0))]
        fig = plotly_heatmap_tracks(tz, ZMIN=ZMIN,ZMAX=ZMAX,YMAX=YMAX,title = title)
        vis.plotlyplot(fig, env=env,win=key)




    def decode(self, t2_embed):
        D = self.config.depth
        assert len(t2_embed.shape)==3,t2_embed.shape
        (BW, T, E)= t2_embed.shape
        zs = t2_embed
        zs = zs.transpose(1,2)

        # for k in range(2):
        K = self.config.kernel_size
        hh = torch.zeros((K+2,BW,E, T),device=self.device)
        # for xd, layer in enumerate(self.conv_layer_list):
        self.config.offset = offset =0
        for xd, layer in enumerate(self.conv_layer_list[:]):
            '''
            Important to stabilise the numbers over deep layers
            '''
            zs = (zs + layer(zs)).tanh()

        zs = zs.transpose(1,2)
        hh = hh.transpose(-2,-1)
        prior = self.prior.clone()

        hh = torch.stack(
            [torch.roll(zs,-ik-K//2,1) for ik in range(K//2*2-1)], dim=0)

        # for xd, layer in enumerate(self.dense_layer_list):
        #     zs1 = (zs + layer(zs)).tanh()
        #     hh[xd+offset] = zs1
        #     prior[xd+offset] += 100.
            # layer
        '''
        DLM155D7E50, Ep67L49, Ep10L89
        '''

        CLS_NAME = self.__class__.__name__
        # self.USE_RESIDUAL=2
        # if self.USE_RESIDUAL==2:

        loghh = self.unembed(hh).log_softmax(-1)
        prior = prior.log_softmax(0)[:hh.shape[0]]
        logps = (prior + loghh)
        return zs,logps,locals()

class DLM156(DLM155):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = self.config.embed_dim
        D = self.config.depth
        T = self.config.n_step
        K = self.config.kernel_size
        self.D = D
        # self.dense_layer_list_2   = nn.ModuleList([nn.Linear(E,E) for _ in range(K)])
        self.dense_layer_list_switch   = nn.Linear(E,K+2)
        # nn.ModuleList([nn.Linear(E,) for _ in range(K)])
class DLM157(DLM155):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = self.config.embed_dim
        D = self.config.depth
        T = self.config.n_step
        K = self.config.kernel_size
        self.D = D
        # self.dense_layer_list_2   = nn.ModuleList([nn.Linear(E,E) for _ in range(K)])
        self.conv_layer_list_list = nn.ModuleList( [nn.ModuleList([nn.Conv1d(E,E,kernel_size = 5, padding='same') for _ in range(D)]) for _ in range(K)])
        self.dense_layer_list_switch   = nn.Linear(E,K+2)
        # self.dense_layer_list_switch   = nn.Linear(E,K+2)
        # nn.ModuleList([nn.Linear(E,) for _ in range(K)])
    def conv_func(self, zs, layer_list):
        zs = zs.transpose(1,2)
        for xd, layer in enumerate(layer_list):
            # zs = zs + layer(zs. relu())
            '''
            Important to stabilise the numbers over deep layers
            '''
            # zs = 0.5*zs + 0.5* layer(zs).tanh()
            zs = (zs + layer(zs)).tanh()
        zs = zs.transpose(1,2)
        return zs


    def decode(self, t2_embed):
        D = self.config.depth
        assert len(t2_embed.shape)==3,t2_embed.shape
        (BW, T, E)= t2_embed.shape
        zs = t2_embed

        # for k in range(2):
        K = self.config.kernel_size
        # prior_mask
        hh = torch.ones((K+2,BW,T,E),device=self.device)
        # self.offset = offset = 1
        self.config.offset = offset = 1
        for ik,conv_layer_list in enumerate(self.conv_layer_list_list):
            zs1 = self.conv_func(zs,conv_layer_list)
            if ik==0:
                prior = self.dense_layer_list_switch(zs).permute((2,0,1)).unsqueeze(-1)
                continue
            else:
                zs1 = self.conv_func(zs,conv_layer_list)
                hh[offset+ik]= zs1


        # hh = hh.transpose(-2,-1)
        '''
        DLM155D7E50,  Ep10L50.3 Ep20L47.1
        '''
            # zs = (zs + layer(zs)).clip(-1, 1)


        CLS_NAME = self.__class__.__name__

        loghh = self.unembed(hh).log_softmax(-1)
        prior = prior.log_softmax(0)[:hh.shape[0]]
        logps = (prior + loghh)
        return zs,logps,locals()

class DLM154(DLM152):
    '''
    Systematically test the usefulness of a layer-alignment variable
    '''
    U = _U

    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = self.config.embed_dim
        D = self.config.depth
        self.D = D
        self.conv_layer_list = nn.ModuleList([nn.Conv1d(E,E,kernel_size = 5, padding='same') for _ in range(D)])
        if hasattr(self,'rnn_dec'): del self.rnn_dec
        if hasattr(self,'rnn_enc'): del self.rnn_enc
        T = self.config.window_size
        self.prior = nn.Parameter( torch.ones([D+2,1,T,1],device=self.device) )

        x = nn.Linear(E,2).to(self.device)
        self.is_kept   =  x
        # self.dbd = {}
        # nn.Parameter(x.weight.T[None,None])
        return

    @staticmethod
    def callback_after_test_all(conf, model,  item):


        from markov_lm.util_plotly import plotly_heatmap_tracks
        self = model
        # super()
        vis = conf.vis;  env = conf._session_name;
        if vis is None: return;
        DLM152.callback_after_test_all(conf,model,item)

        key = 'prior'
        title = key
        # prior
        nelem = 0
        # ZMIN,ZMAX = (0.,10.)
        ZMIN,ZMAX = (None,None)
        # ZMIN = -10
        YMAX = 20

        # xd = model.get_debug_data(item,[conf])

        '''
        Need to visualise the attention matrix.
        for each corrupted sentences
        '''
        tz = [(None, x.squeeze(-1)) for x in model.U.N(model.prior.softmax(0))]
        fig = plotly_heatmap_tracks(tz, ZMIN=ZMIN,ZMAX=ZMAX,YMAX=YMAX,title = title)
        vis.plotlyplot(fig, env=env,win=key)




    def decode(self, t2_embed):
        D = self.config.depth
        assert len(t2_embed.shape)==3,t2_embed.shape
        (BW, T, E)= t2_embed.shape
        zs = t2_embed
        zs = zs.transpose(1,2)

        # for k in range(2):
        hh = torch.zeros((D+2,BW,E, T),device=self.device)
        # output = torch.
        self.config.hyp = 2
        if self.config.hyp == 0:
            self.config.offset =offset = 0

        if self.config.hyp == 1:
            hh[0] = zs
            self.config.offset =offset = 1
        elif  self.config.hyp == 2:
            # hh[0] = zs
            hh[1] = (zs*0).detach()+1
            self.config.offset =offset = 2
        else:
            raise NotImplementedError(f'self.config.hyp={self.config.hyp}')

        zs0 = zs
        for xd, layer in enumerate(self.conv_layer_list):
            # zs = zs + layer(zs. relu())
            '''
            Important to stabilise the numbers over deep layers
            '''
            # zs = 0.5*zs + 0.5* layer(zs).tanh()
            # zs0 = (zs + layer(zs0)).tanh()
            zs = (layer(zs0)).tanh()
            # zs = zs0 + (layer(zs0)).tanh()
            hh[xd+offset] = zs
            # zs = (zs + layer(zs)).clip(-1, 1)

        zs = zs.transpose(1,2)
        hh = hh.transpose(-2,-1)


        loghh = self.unembed(hh).log_softmax(-1)
        prior = self.prior.log_softmax(0)[:hh.shape[0]]
        logps = (prior + loghh)

        return zs,logps,locals()




class DLM150(DLM142):
    '''
    DLM142 provide the .__init__ and the .decode

    This is an experimental class to try reinforce on diffusion
    '''
    pass




import  markov_lm.external.clm_transformer_model
class DLM147(DLM140):
    def __init__(self,device,config,_=None):
        super().__init__(device,config)

        self.submodel = markov_lm.external.clm_transformer_model.NextCharTransformer(
            vocab_size = self.config.graph_dim + 1,
            n_layers = self.config.depth,
            hidden_size = self.config.embed_dim,
            inner_linear = self.config.embed_dim,
            n_heads = 4,
            dropout = 0.55,
            tied = False,
            max_sequence_len = 100,
            intermediate_layer_predictions = False,
            # intermediate_layer_predictions = True,
        )

        E = self.config.embed_dim
        D = self.config.depth
        self.D = D
        # self.conv_layer_list = nn.ModuleList([nn.Conv1d(E,E,kernel_size = 5, padding='same') for _ in range(D)])
        # if hasattr(self,'rnn_dec'): del self.rnn_dec
        # if hasattr(self,'rnn_enc'): del self.rnn_enc

        x = nn.Linear(E,2).to(self.device)
        self.is_kept   =  x
        # nn.Parameter(x.weight.T[None,None])
        return

    def decode(self, t2_embed):
        D = self.config.depth
        (BW, T, E)= t2_embed.shape
        h0 = torch.ones((D,BW,E),device=self.device)
        zs = t2_embed
        mask = torch.ones((BW,T,T),device=self.device)
        # import pdb; pdb.set_trace()
        # mask =
        zs, intermediate_predictions = self.submodel.encoder(t2_embed, mask)



        is_kept_lp = self.is_kept(zs).log_softmax(-1)

        logp_repl = self.unembed(zs).log_softmax(-1)

        self.USE_RESIDUAL=1
        if self.USE_RESIDUAL:
            logp_kept = self.unembed(t2_embed).log_softmax(-1)
            x = torch.stack([logp_repl,logp_kept],dim=2)
            logp = (is_kept_lp.unsqueeze(-1) + x).logsumexp(2)
        else:
            logp = logp_repl
        return zs,logp,locals()


    pass

class DLM143(DLM140):
    '''
    recover states from noise
    '''

    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = self.config.embed_dim
        D = self.config.depth
        self.D = D
        self.conv_layer_list = nn.ModuleList([nn.Conv1d(E,E,kernel_size = 5, padding='same') for _ in range(D)])
        if hasattr(self,'rnn_dec'): del self.rnn_dec
        if hasattr(self,'rnn_enc'): del self.rnn_enc

        x = nn.Linear(E,10).to(self.device)
        self.is_kept   =  x
        # nn.Parameter(x.weight.T[None,None])
        return

    def decode(self, t2_embed):
        D = self.config.depth
        (BW, T, E)= t2_embed.shape
        h0 = torch.ones((D,BW,E),device=self.device)
        zs = t2_embed
        zs = zs.transpose(1,2)

        # for k in range(2):
        for layer in self.conv_layer_list:
            # zs = zs + layer(zs. relu())
            '''
            Important to stabilise the numbers over deep layers
            '''
            # zs = 0.5*zs + 0.5* layer(zs).tanh()
            # zs = (zs + layer(zs)).tanh()
            zs = (layer(zs)).tanh()
            # zs = (zs + layer(zs)).clip(-1, 1)

        zs = zs.transpose(1,2)
        # zs = zs.flip([2,])


        is_kept_lp = self.is_kept(zs).log_softmax(-1).reshape((BW,-1,5,2)).logsumexp(2)

        logp_repl = self.unembed(zs).log_softmax(-1)

        self.USE_RESIDUAL=1
        if self.USE_RESIDUAL:
            logp_kept = self.unembed(t2_embed).log_softmax(-1)
            x = torch.stack([logp_repl,logp_kept],dim=2)
            logp = (is_kept_lp.unsqueeze(-1) + x).logsumexp(2)
        else:
            logp = logp_repl

        return zs,logp,locals()


class CustomTransformer(nn.Module):
    def __init__(self, E, K):
        # E = config.embed_dim
        super().__init__()

        self.w_k = nn.ModuleList([nn.Linear( E, E) for _ in range(K)])
        self.w_v = nn.ModuleList([nn.Linear( E, E) for _ in range(K)])
        self.K = K

    def forward(self, x):
        B,T,E = x.shape
        xy = 0
        for k,w_k in enumerate(self.w_k):
        # K = self.K
        # xx  = (self.w_k(x)).reshape((B,T,K,E)) @ x.transpose(-1,-2)
            xx  = (self.w_k[k](x)) @ x.transpose(-1,-2)
            att = xx.softmax(-1)
            xy  = xy + att @ (self.w_v[k](x) )
        return xy




class DLM144(DLM140):
    '''
    recover states from noise
    '''

    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = self.config.embed_dim
        D = self.config.depth
        self.D = D
        # self.
        KT = 2
        # self.conv_layer_list = nn.ModuleList([nn.Conv1d(E,E,kernel_size = 5, padding='same') for _ in range(D)])
        self.conv_layer_list = nn.ModuleList([CustomTransformer(E,KT) for _ in range(D)])
        if hasattr(self,'rnn_dec'): del self.rnn_dec
        if hasattr(self,'rnn_enc'): del self.rnn_enc

        x = nn.Linear(E,10).to(self.device)
        self.is_kept   =  x
        # nn.Parameter(x.weight.T[None,None])
        return

    def decode(self, t2_embed):
        D = self.config.depth
        (BW, T, E)= t2_embed.shape
        h0 = torch.ones((D,BW,E),device=self.device)
        zs = t2_embed
        # zs = zs.transpose(1,2)

        # for k in range(2):
        for layer in self.conv_layer_list:
            # zs = zs + layer(zs. relu())
            '''
            Important to stabilise the numbers over deep layers
            '''
            # zs = 0.5*zs + 0.5* layer(zs).tanh()
            zs = (zs + layer(zs)).tanh()
            # zs = (layer(zs)).tanh()
            # zs = (zs + layer(zs)).clip(-1, 1)

        # zs = zs.transpose(1,2)
        # zs = zs.flip([2,])

        is_kept_lp = self.is_kept(zs).log_softmax(-1).reshape((BW,-1,5,2)).logsumexp(2)

        logp_repl = self.unembed(zs).log_softmax(-1)

        self.USE_RESIDUAL=1
        if self.USE_RESIDUAL:
            logp_kept = self.unembed(t2_embed).log_softmax(-1)
            x = torch.stack([logp_repl,logp_kept],dim=2)
            logp = (is_kept_lp.unsqueeze(-1) + x).logsumexp(2)
        else:
            logp = logp_repl

        return zs,logp,locals()



class DLM145(DLM140):
    '''
    recover states from noise
    '''

    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = self.config.embed_dim
        D = self.config.depth
        self.D = D
        self.conv_layer_list = nn.ModuleList([nn.Conv1d(E,E,kernel_size = 5, padding='same') for _ in range(D)])
        if hasattr(self,'rnn_dec'): del self.rnn_dec
        if hasattr(self,'rnn_enc'): del self.rnn_enc

        x = nn.Linear(E,2).to(self.device)
        self.is_kept   =  x
        # nn.Parameter(x.weight.T[None,None])
        return

    def decode(self, t2_embed):
        D = self.config.depth
        (BW, T, E)= t2_embed.shape
        h0 = torch.ones((D,BW,E),device=self.device)
        zs = t2_embed
        zs = zs.transpose(1,2)

        # for k in range(2):
        for layer in self.conv_layer_list:
            # zs = zs + layer(zs. relu())
            '''
            Important to stabilise the numbers over deep layers
            '''
            # zs = 0.5*zs + 0.5* layer(zs).tanh()
            zs = (zs + layer(zs)).tanh()
            # zs = (zs + layer(zs)).clip(-1, 1)

        zs = zs.transpose(1,2)
        # zs = zs.flip([2,])


        is_kept_lp = self.is_kept(zs).log_softmax(-1)

        logp_repl = self.unembed(zs).log_softmax(-1)

        self.USE_RESIDUAL=1
        if self.USE_RESIDUAL:
            logp_kept = self.unembed(t2_embed).log_softmax(-1)
            x = torch.stack([logp_repl,logp_kept],dim=2)
            logp = (is_kept_lp.unsqueeze(-1) + x).logsumexp(2)
        else:
            logp = logp_repl
        return zs,logp


    def _get_loss(self, zs, encoder_mu,target,target_notnull):
        '''
        encoder_mu:      is the noise-free param
        zs:              is the sampled latent
        target:          the wanted decoding result
        target_notnull:  mask to ignore unspecified

        describ:
            use a fixed noise to perturb target recursively.

        因为替换一次和替换两次是没有区别的，所以对于多次采样可以利用这种等价性。
        每次固定只perturb k个位置，这个其实不是很poisson，或者不是条件独立的。
        如果poisson一点，那其实每个位置被corrupt的概率均等，采样起来比较方便。
        '''

        W = self.W
        K = self.K

        p_null = self.config.p_null
        B,T = target.shape
        D = self.config.depth

        t1_mask     = torch.rand( [B,W+1,T], device=self.device)
        t1_mask     = (  t1_mask < (p_null * target_notnull.unsqueeze(1)) )
        t2_mask     = t1_mask[:,1:]
        t1_mask     = t1_mask[:,:-1]
        t1_mask[:,0] = 0

        t1_mask_cum = t1_mask.cumsum(dim=1) > 0

        t1_rint     = torch.randint(self.G,[B,W,T], device=self.device)
        t2_rint     = torch.randint(self.G,[B,W,T], device=self.device)
        t1 = target.unsqueeze(1) * ~t1_mask_cum + t1_rint * t1_mask_cum
        t2 = t1 * ~t2_mask + t2_rint* t2_mask
        # print(t2_mask.sum(2).float().mean().item())
        E = self.embed_dim

        t2 = t2.reshape((B*W,T))
        t1 = t1.reshape((B*W,T))

        t2_embed = self.embed(t2)

        zs,logps,_  = self.decode(t2_embed)

        self.PREDICT_SOURCE=0
        if self.PREDICT_SOURCE:
            t1 = target.unsqueeze(1).repeat((1,W,1)).reshape((B*W,T))
        else:
            pass

        lps = torch.gather(logp,index=t1.unsqueeze(-1),dim=-1).squeeze(-1)
        lps = lps.reshape((B,W,T))
        logp_sum_byt = (lps*target_notnull.unsqueeze(1)).sum(-1)
        wt = torch.linspace(5,0,W,device=self.device)[None,:,]
        logp_sum     = logp_sum_byt.mean(-1)
        logp_sum_grad = (logp_sum_byt * wt.softmax(-1)).sum(-1)

        encoder_mu_0 = self.embed(target)
        encoder_mu   = t2_embed.reshape((B,W,T,E))[:,0]

        recovered = zs.reshape((B,W,T,E))[:,0]
        sampled_zs =  t2_embed.reshape((B,W,T,E))[:,0] -  self.embed(t1).reshape((B,W,T,E))[:,0]
        return locals()



class DLM146(DLM140):
    '''
    recover states from noise
    '''

    def __init__(self,device,config,_=None):
        super().__init__(device,config)
        E = self.config.embed_dim
        D = self.config.depth
        self.D = D
        self.conv_layer_list_1 = nn.ModuleList([nn.Conv1d(E,E,kernel_size = 5, padding='same') for _ in range(D//2)])
        self.conv_layer_list_2 = nn.ModuleList([nn.Conv1d(E,E,kernel_size = 5, padding='same') for _ in range(D//2)])
        self.conv_layer_list_3 = nn.ModuleList([nn.Conv1d(E,E,kernel_size = 5, padding='same') for _ in range(D//2)])
        if hasattr(self,'rnn_dec'): del self.rnn_dec
        if hasattr(self,'rnn_enc'): del self.rnn_enc

        x = nn.Linear(E,2).to(self.device)
        self.is_kept   =  x
        # nn.Parameter(x.weight.T[None,None])
        return

    def decode(self, t2_embed):
        D = self.config.depth
        (BW, T, E)= t2_embed.shape
        h0 = torch.ones((D,BW,E),device=self.device)
        zs = t2_embed
        zs = zs.transpose(1,2)

        # for k in range(2):
        for layer in self.conv_layer_list_1:
            # zs = zs + layer(zs. relu())
            '''
            Important to stabilise the numbers over deep layers
            '''
            # zs = 0.5*zs + 0.5* layer(zs).tanh()
            zs = (zs + layer(zs)).tanh()
        zs_inter = zs

        zs = zs_inter
        for layer in self.conv_layer_list_2:
            zs = (zs + layer(zs)).tanh()
        zs = zs.transpose(1,2)
        is_kept_lp = self.is_kept(zs).log_softmax(-1)

        # zs = zs.flip([2,])


        zs = zs_inter
        for layer in self.conv_layer_list_3:
            zs = (zs + layer(zs)).tanh()
        zs = zs.transpose(1,2)
        logp_repl = self.unembed(zs).log_softmax(-1)

        self.USE_RESIDUAL=1
        if self.USE_RESIDUAL:
            logp_kept = self.unembed(t2_embed).log_softmax(-1)
            x = torch.stack([logp_repl,logp_kept],dim=2)
            logp = (is_kept_lp.unsqueeze(-1) + x).logsumexp(2)
        else:
            logp = logp_repl
        return zs,logp,locals()



class DLM129(DLM128):
    # Prototype):
    '''
    Conv-GRU with lower channelcount for chars
    '''
    _custom_hidden_to_cats = 1

    def decode(self,zs):
        D = self.config.depth
        for i in range(D):
            zs = self.convt_layer_list[i](zs.sin().transpose(1,2)).transpose(1,2)
        lpc = self.unembed(zs).log_softmax(-1)
        return lpc,zs



import numpy as np


class DLM122(DLM117):
    _custom_hidden_to_cats = 1
    def __init__(self,device,config,_=None):
        super().__init__(device,config)


        G = config.graph_dim +1
        self.G = G
        K = config.kernel_size
        assert K >=1,config
        E = config.embed_dim
        # self.S = 31
        self.W = W = config.window_size
        self.K = K = config.kernel_size

        x = nn.Linear(E,K).to(self.device)
        self.prior_mu   =  nn.Parameter(x.weight.T[None,None])

        x = nn.Linear(E,1 ).to(self.device)
        self.prior_beta   =  nn.Parameter(x.weight.T[None,None])

        self.conv_layer = nn.Conv1d(E,E,kernel_size = 5, padding='same')
        self.conv_layer_2 = nn.Conv1d(E,E,kernel_size = 5, padding='same')
        self.convt_layer = nn.Conv1d(E,E,kernel_size = 5, padding='same')
        self.convt_layer_2 = nn.Conv1d(E,E,kernel_size = 5, padding='same')
        # self.convt_layer = nn.ConvTranspose1d(E,E,kernel_size = 5, padding='same')

        # assert W >=1,config.window_size
        # self.lin_layer = nn.Linear(E, E ).to(self.device)
class DLM115(DLM112):
    # Prototype):
    '''
    Conv-GRU with lower channelcount for chars
    '''
    _custom_hidden_to_cats = 1
    def __init__(self,device,config,_=None):
        super().__init__(device,config)


        G = config.graph_dim +1
        self.G = G
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        # self.S = 31
        self.W = W = config.window_size
        self.K = K = config.kernel_size

        x = nn.Linear(E,K).to(self.device)
        self.prior_mu   =  nn.Parameter(x.weight.T[None,None])

        x = nn.Linear(E,1 ).to(self.device)
        self.prior_beta   =  nn.Parameter(x.weight.T[None,None])
        self.lin_layer = nn.Linear(E, E ).to(self.device)

        # assert W >=1,config.window_size


class DLM116(DLM112):
    # Prototype):
    '''
    Conv-GRU with lower channelcount for chars
    '''
    _custom_hidden_to_cats = 1
    def __init__(self,device,config,_=None):
        super().__init__(device,config)


        G = config.graph_dim +1
        self.G = G
        # K = config.kernel_size
        # assert K >=1,config
        E = config.embed_dim
        # self.S = 31
        self.W = W = config.window_size
        self.K = K = config.kernel_size

        x = nn.Linear(E,K).to(self.device)
        self.prior_mu   =  nn.Parameter(x.weight.T[None,None])

        x = nn.Linear(E,1 ).to(self.device)
        self.prior_beta   =  nn.Parameter(x.weight.T[None,None])
        self.lin_layer = nn.Linear(E, E ).to(self.device)
        # T = self
        T = 100
        self.project = nn.Linear(E*T,E)
        self.unproject = nn.Linear(E,E*T)
        # assert W >=1,config.window_size

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

    def loss(self,item,):
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
