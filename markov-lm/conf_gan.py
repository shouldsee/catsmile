import torch
from dataclasses import dataclass

from markov_lm.c9007_util import tbuf_cls,fws,recur_detach
import shutil

import collections

class ConfigPrototype(object):
    def __init__(self,fn,is_sorted=False,
    field_width = [20,20] + [5]*30,
    section_ender = None):
        self.is_sorted = is_sorted
        self._session_name = None
        self.is_model_set = False
        if fn.endswith('.py'):
            fn = fn+'.html'
        self.f = open(fn,'w')
        self.tbuf = tbuf_cls()
        self.s = collections.OrderedDict()
        self.field_width = field_width[:]
        if section_ender is None:
            section_ender = '-'*15*len(field_width)
        self.section_ender = section_ender
            # sele.ct
        return

    def data_input_transform(self,item):
        raise NotImplementedError
    def get_ckpt_name(self):
        return self.__class__.__name__

    def _print(self,*x):
        f=  self.f
    # _print = lambda *x,f=f:
        f.write(f'<pre>{"".join(x)}</pre>\n')

    def init_s(self,k,v):
        s = self.s
        if k not in s:
            s[k]=[k,list(v.shape)]
        return
    def grad_loss(self,*a,**kw):
        return self.loss(*a,**kw)
    def loss(self,*a,**kw):
        raise NotImplementedError

    def callback_epoch_start(self,epoch):
        pass


    def callback_step(self,epoch,indexedItem,loss):
        pass
        tri,item = indexedItem
        f = self.f
        t = self.handler
        s = self.s
        _print = self._print
        self.append_activation = 0
        self.append_optimizer_avg=0
        if self.append_activation:
            for k,v in t.xdict.items():
                mae = v.abs().mean().item()
                # s.setdefault(k,[])
                if k not in s:
                    s[k]=[k,list(v.shape)]
                ss = s[k]
                ss.append(mae*1000)
                # ss.append(fws('%.3f'%mae,10))
                # print(v.id)
                # _print(fws(k,15)+fws(list(v.shape),10)+fws('%.3f'%mae,10))
        if self.append_optimizer_avg:

            opt = self.optimizer
            for p,v in opt.state.items():
                k = id(p)
                if k not in s:
                    s[k] = [id(p),list(getattr(p,'shape',[1]))]
                s[k].append( (1000*v.get('square_avg',torch.tensor(0.)).mean().item()))

    def callback_start(self,epoch,a,b):
        pass
    def callback_end(self,epoch,indexedItem,loss):
        s = self.s
        f = self.f
        is_sorted = self.is_sorted
        _print = self._print
        _print(f'Epoch{epoch}.Model-{self._session_name}.{loss}')
        CONFIG_HIDE = -100000000
        def _str(x,s):
            if isinstance(x,float):
                if x >CONFIG_HIDE:
                    x ='%.3f'%x
                else:
                    x = ''
            else:
                pass
            return fws(x,s)

        # wsl = [20,20] + [5]*30
        # field_width
        ks = list(s)
        if self.is_sorted:
            ks = sorted(ks)
        for k in ks:
            v = s.pop(k)
            _print(*[_str(vv,ws)+'|'+' ' for  vv, ws in zip(v,self.field_width)])

            _print(self.section_ender) if len(self.section_ender) else None

        # x = ['bias.1000']+list((1000*self.model.att_dense.bias).int().cpu().detach().numpy()[:10])
        # _print(*[_str(vv,10)+'|'+' '*3 for vv in x])
        _print('='*35)
        f.flush()

@dataclass
class GANConfigPrototype(ConfigPrototype):
    '''
    ### dataset yields real data
    ### model_g transforms N(0,I) into embedded sentences
    ### model_d takes a sentence and outputs a probability
    ### during training g, maximise model_d(g.sample())
    ### during training d, maximise model_d(real_sample) - model_d(g.sample())

    '''
    dataset: object = None
    model_g: object = None
    model_d: object = None
    _session_name: str = ''
    instance: int=0
    optimizer: object = None
    num_epoch: int =1000
    device: torch.device = None
    dataloader: torch.utils.data.DataLoader = None
    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        super().__init__(obj, *args, **kwargs)

        return obj
    # def __init__(self):
    #     pass
    def callback_epoch_start(self,epoch):
        print(f'[last_d_loss]{self.model.last_d_loss}')
        return

    def loss(self,item):
        return self.model.loss(item)


from markov_lm.Model_add import AddModelWithAttentionStacked,SASConfig
class AGANConfig(SASConfig):
    def to_model(self,device,charset):
        return AttentionGAN(device,self,charset)

## use LayerConfig
class AttentionGANUnit(AddModelWithAttentionStacked):
    def __init__(self,device,config,charset):
        config.graph_dim+=1
        # graph_dim+1
        super().__init__(device,config)
        self.device = device
        self.config = config
        self.is_random = config.is_random
        self.linear = nn.Linear(config.embed_dim,1)
    def sample(self,item):
        B,L = real_sample.shape[:2]
        [i,z,gradsq,xsa]
        # L = item['unmasked'].shape[1]
        # torch.random.
    def _batch_init(self,zi,masked):
        #### batch_init part
        ### state init
        z = masked
        if not self.is_random:
            z = torch.cat([z*0+self.config.graph_dim-1, z],dim=1)
            # z = torch.cat([self.config.graph_dim-1, z],dim=1)
            z = z
        else:
            z = torch.normal(0,1,z.shape+(self.config.embed_dim,),device=self.device)
        if not self.is_embed:
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

    def sample(self,item):
        self.is_random = 1
        self.is_embed  = 1
        outer,inner = self.forward(item)
        xsa = inner[-1]
        return xsa

    def log_prob(self,item,is_embed=1,detach=False):
        self.is_random = 0.
        self.is_embed = is_embed

        outer,inner = self.forward(item,detach=detach)
        xsa = inner[-1]
        # p = self.linear(xsa[:,:,]).mean(1)[:,0]
        # z = item['masked']
        # if not self.is_embed:
        #     z = self.embed(z)
        # z = self.norm(z)
        # p = xsa * z
        # print(p[0].mean(-1))
        # p = p.mean(-1).mean(-1)
        xsa = self.norm(xsa)
        print(xsa.std(-1)[0,:10].detach().cpu().numpy())
        # sum)
        p =( xsa[:,0,] @ self.norm(self.linear.weight).T)/self.embed_dim
        p = p[:,0]
        print(self.alias,p[:10].int().detach().cpu().numpy())
        # .mean(1)[:,0]
        # p = self.linear(xsa[:,:]).max(1)[0][:,0]
        # print(p)
        # [:,0]
        return p

import copy
import torch.nn as nn
class AttentionGAN(torch.nn.Module):
    def __init__(self,device,config,charset):
        super().__init__()
        self.device  = device
        self.config  = config
        config.is_random=0
        self.model_d = AttentionGANUnit(device,copy.copy(config),charset)
        config.is_random=1
        self.model_g = AttentionGANUnit(device,copy.copy(config),charset)
        self.model_d.alias = 'dis'
        self.model_g.alias = 'gen'
        self.last_d_loss = 0.
        self.last_g_loss = 0.
        self.last_unembed_nll=0.
    def loss(self,item):
        # beta = 0.001
        # alpha = 0.001
        alpha = beta = 1.
        beta = 0.001
        decay = 0.8
        real_sample = item['unmasked']
        B = real_sample.shape[0]
        mg = self.model_g
        md = self.model_d

        def get_d_loss():
            xl = 0.
            print('[l1]')
            xl = xl -  1 *  (- alpha * md.log_prob( dict(masked=mg.sample(item).detach() ) ) ).sigmoid()
            ### Negative probability of real sample being real
            print('[l2]')
            xl = xl -  1 * (alpha * md.log_prob( item, is_embed = 0)).sigmoid()
            xl = 0.5*xl
            d_loss = xl
            return d_loss

        ### Negative probability of fake sample being real
        get_g_loss = lambda: - 1 * (beta * md.log_prob( dict(masked=mg.sample(item)) ,detach=True) ).sigmoid()
        if 1:
            # self.last_d_loss = d_loss.mean().item()
            # self.last_g_loss = g_loss.mean().item()

            g_strong = -self.last_g_loss>0.8
            d_weak = -self.last_d_loss < 0.5


            # if (item['epoch'] == 0) or d_weak  or g_strong:
            g_loss = get_g_loss()
            self.last_g_loss *= decay
            self.last_g_loss += (1-decay)*g_loss.mean().item()

            d_loss = get_d_loss()
            self.last_d_loss *= decay
            self.last_d_loss += (1-decay)* d_loss.mean().item()
            if (item['epoch'] == 0) or -self.last_d_loss<-self.last_g_loss:
                # if g_strong and not d_weak:
                #     g_loss = get_g_loss()
                #     self.last_g_loss *= decay
                #     self.last_g_loss += (1-decay)*g_loss.mean().item()
                # d_loss = get_d_loss()
                # self.last_d_loss *= decay
                # self.last_d_loss += (1-decay)* d_loss.mean().item()
                return d_loss
            else:
                # # print('[epoch] generator')
                # g_loss = get_g_loss()
                # self.last_g_loss *= decay
                # self.last_g_loss += (1-decay)* g_loss.mean().item()
                return g_loss
        else:
            g_loss = get_g_loss()
            d_loss = get_d_loss()

            self.last_d_loss *= decay
            self.last_d_loss += (1-decay)* d_loss.mean().item()

            self.last_g_loss *= decay
            self.last_g_loss += (1-decay)* g_loss.mean().item()
            unembed_nll = - md.target_energy( md.vocab(md.embed(item['masked'])).log_softmax(-1), item['masked'] )
            self.last_unembed_nll *= decay
            self.last_unembed_nll += (1-decay)* unembed_nll.mean().item()
            xl = torch.maximum(g_loss,d_loss)  + unembed_nll.mean(-1)

                # xl = xl +  1 * (model_d.log_prob( model.d.embed(item['unmasked'] )) ).sigmoid()
            return xl

        # xl = xl
        # return 0.
    def data_input_transform(self,item):
        return item
