'''
- Author: Feng Geng
- Changelog:
  - 20220507-20220514: for (dataset=ArithmeticTest, model=AddModelWithAttention),
  tested different parameters. use_dense_relu=1/11 shows interesting capacity.
  per-position dropout greatly prevents overfitting for unknown reason.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import os,sys

import numpy as np
from torch import autograd

from markov_lm.util_html import write_png_tag

import sys
import glob
from pprint import pprint
import shutil
from tqdm import tqdm

from markov_lm.Model import cross_entropy
from markov_lm.c9007_util import tbuf_cls,fws,recur_detach

from markov_lm.Dataset.translation_dataset import ArithmeticTest
from markov_lm.Model_add import AddModelWithBert
from markov_lm.Model_add import AddModelWithAttention


import collections
import math

class ConfigPrototype(object):
    def __init__(self):
        self.f = open(__file__+'.log.html','w')
        self.tbuf = tbuf_cls()
        self.s = collections.OrderedDict()

        return
    def _print(self,*x):
        f=  self.f
    # _print = lambda *x,f=f:
        f.write(f'<pre>{"".join(x)}</pre>\n')

    def init_s(self,k,v):
        s = self.s
        if k not in s:
            s[k]=[k,list(v.shape)]
        return

    def callback_step(self,epoch,indexedItem,loss):
        pass
        tri,item = indexedItem
        f = self.f
        t = self.handler
        s = self.s
        _print = self._print
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
        _print = self._print
        _print(f'Epoch{epoch}.Model-{self.model.__class__.__name__}.{loss}')
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

        for k in list(s):
            v = s.pop(k)
            _print(*[_str(vv,10)+'|'+' '*3 for vv in v])
            _print('-'*15*len(v))

        # x = ['bias.1000']+list((1000*self.model.att_dense.bias).int().cpu().detach().numpy()[:10])
        # _print(*[_str(vv,10)+'|'+' '*3 for vv in x])
        _print('='*35)
        f.flush()


def parse_checkpoint(sys,):
    if '--auto' in sys.argv:
        LOAD_GET_AUTO = 1
    else:
        LOAD_GET_AUTO = 0

    os.makedirs('Checkpoints') if not os.path.exists('Checkpoints') else None

    if LOAD_GET_AUTO:
        res = glob.glob('Checkpoints/*pkl')
        getLoad = lambda x:int(x.replace('Checkpoints/Checkpoint','').split('.')[0])
        res = sorted(res,key=getLoad)[-1]
        LOAD= getLoad(res)
    else:
        LOAD = '-1'

    if '--LOAD' in sys.argv:
        LOAD = sys.argv[sys.argv.index('--LOAD')+1]
        # LOAD = int(LOAD)
    return LOAD


def init_conf(CUDA,shuffle, AddModelWithAttention=AddModelWithAttention):
    '''
    Runtime subclassing AddModelWithAttention()

    returns a config object that controls training process.
    binds (dataset,model,device)
    '''

    def add_hook_for_output_tensor(model,CONFIG_EXPAND_LEVEL,CONFIG_DETACH, CONFIG_PRINT_CLASS):
        '''

        returns a temp object with a dict of tensors outputted at each nn.Module with maximum depth of CONFIG_EXPAND_LEVEL.
        '''
        class Temp(object):
            xdict = collections.OrderedDict()
            handles = []
            def __init__(self,):
                pass
                # self.xdict= xdict
            def remove_hooks(self):
                # ,handles= handles):
                 [h.remove() for h in self.handles]
        t = Temp()
        handles = t.handles
        xdict = t.xdict


        def get_backward_hook(name,xdict=xdict):
            # def hook(model, input_grad, output_grad):
            def hook(output_grad):
                recur_detach(xdict,name,output_grad,CONFIG_DETACH,'/')
                # xdict[name] = output
            return hook

        '注入灵魂'
        for k,v in model.named_parameters():
            if k.count('.')<=CONFIG_EXPAND_LEVEL:
                if CONFIG_PRINT_CLASS:
                    print(fws(k,60) ,v.__class__);
                h = v.register_hook(get_backward_hook(k))
                # h = v.register_full_backward_hook(get_backward_hook(k))
                handles.append(h)

        return t


    conf = ConfigPrototype()

    '''
    Abusing attributes here
    [TBC]
    '''
    conf.instance= 29

    torch.manual_seed(conf.instance)
    conf.criterion = cross_entropy
    conf.embed_dim = 50
    conf.mixture_count = 8
    conf.state_count = 15
    conf.device =  torch.device('cuda:0' if CUDA else 'cpu')
    conf.num_epoch = 5000

    conf.learning_rate = 0.001
    conf.SAVE_INTERVAL = 10
    conf.batch_size = 60
    conf.tsi_max = 10

    ### This is a random dataset !!!! init after random seed is set
    conf.dataset = dataset = ArithmeticTest(CUDA=CUDA)
    ### test dataset works
    (conf.dataset[range(5)])
    conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)
    # dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)
    add_optimizer      =  lambda conf,params:torch.optim.RMSprop( params, lr=conf.learning_rate,)
    # add_optimizer      =  lambda conf,params:torch.optim.RMSprop( params, lr=conf.learning_rate,eps=0.01)

    '''
    now specify model
    '''
    conf.mixture_count=3
    conf.depth = 12
    conf.iter_per_layer=12
    # conf.learning_rate = 0.0001
    # conf.instance = 4;torch.manual_seed(conf.instance)
    conf.model = model = AddModelWithBert(graph_dim = dataset.graph_dim,
        depth=conf.depth,
        # state_count=conf.state_count,
        embed_dim=conf.embed_dim,device=conf.device,
        iter_per_layer = conf.iter_per_layer,
        mask_token_idx=dataset.mask_token_idx)

    conf.model.__class__.__name__ = f'AddModelWithBert-D{conf.depth}-1I{conf.instance}-Dropout{conf.model.bconf.hidden_dropout_prob:.2f}'

    class AddModelWithAttention(AddModelWithAttention):
        '''
        Adds callback to record energy
        '''
        bi = 0
        def callback_step(self,inner,outer,s=conf.s):
            energy= outer[-2].mean()
            k = f'energy_{self.bi}'
            conf.init_s(k,energy)
            s[k].append(energy.item()*1000)

        def callback_end(self,outer,inner,s=conf.s):
            self.bi += 1
        def train(self,*a,**kw):
            super().train(*a,**kw)
            self.bi = 0


    # AddModelWithAttention.callback_step = callback_step
    conf.kernel_size = 5
    conf.depth = 12
    conf.embed_dim = 40
    conf.use_dropout= 0.5
    conf.use_dense_relu = 11
    conf.use_layernorm = 1
    conf.use_gradnorm = 1
    conf.step_size = 0.05
    conf.model = model = AddModelWithAttention(graph_dim = dataset.graph_dim,
        depth=conf.depth,
        use_dropout = conf.use_dropout,
        kernel_size = conf.kernel_size,
        use_dense_relu = conf.use_dense_relu,
        use_layernorm = conf.use_layernorm,
        use_gradnorm = conf.use_gradnorm,
        embed_dim=conf.embed_dim,device=conf.device,
        iter_per_layer = conf.iter_per_layer,
        mask_token_idx=dataset.mask_token_idx,
        step_size = conf.step_size)
    conf.learning_rate = 0.001
    '''
    Class name is used to construct checkpoint filenames. manually record important params here, needs a better system.
    [TBC]
    '''
    conf.model.__class__.__name__ += f'-D{conf.depth}-1I{conf.instance}-DenseRelu{conf.use_dense_relu}-Layernorm{conf.use_layernorm}-Dropout{conf.use_dropout}-Gradnorm{conf.use_gradnorm}-loglr{math.log10(conf.learning_rate):.1f}'
    conf.model.__class__.__name__ += f'-V7'


    conf.model = model = model.to(conf.device)
    params = list(model.parameters())
    print(dict(model.named_parameters()).keys())

    #### using Adam with high learning_rate is catastrophic
    conf.optimizer = add_optimizer(conf,params)
    conf.handler = add_hook_for_output_tensor(model,CONFIG_DETACH=1,CONFIG_EXPAND_LEVEL=4,CONFIG_PRINT_CLASS=0)
    return conf

def main():
    CUDA = 1
    conf = init_conf(CUDA,shuffle=True)
    model = conf.model
    dataset = conf.dataset
    dataset
    CKPT = parse_checkpoint(sys,)
    if(CKPT!='-1'):

        epoch = CKPT
        # res = glob.glob(os.path.join("Checkpoints",f"{conf.model.__class__.__name__}_{CKPT}_*.pkl"))
        res = glob.glob(os.path.join("Checkpoints",f"{conf.model.__class__.__name__}_{epoch}*.pkl"))
        assert len(res)==1,['Which one to choose?',res]
        print(f'[LOADING]{res[0]}')

        checkpoint   = torch.load(res[0])
        test_losses  = checkpoint["test_losses"]
        train_losses = checkpoint["train_losses"]
        epoch        = checkpoint["epoch"]
        x            = checkpoint["model"]
        xx = {}
        for k,v in x.items():
            if k in dict(model.named_parameters()):
                xx[k] = v
            else:
                pass
                # print(k)
        x = xx
        STRICT_LOAD = '--nostrict' not in sys.argv
        model.load_state_dict(x,strict=STRICT_LOAD)
        if STRICT_LOAD:
            conf.optimizer.load_state_dict(checkpoint['optimizer'])

    else:
        test_losses = []
        train_losses = []
        epoch = -1


    loss_test_mean = 0
    n_mask = 4
    for _epoch in range(conf.num_epoch):
        # conf.dataset.op_extract_and_mask(n_mask)
        epoch += 1
        loss_train_sum = 0
        loss_test_sum = 0

        ### needs to random extract tokens into sets and sequences
        if(epoch % conf.SAVE_INTERVAL ==0):
            target_filename = os.path.join("Checkpoints",f"{conf.model.__class__.__name__}_{epoch}_{loss_test_mean:.5f}.pkl")
            torch.save({
                "model"       :conf.model.state_dict(),
                "optimizer"   :conf.optimizer.state_dict(),
                "epoch"       :epoch,
                "train_losses":train_losses,
                "test_losses" :test_losses
            },target_filename)
            linkFile = __file__+'.curr.ckpt.pkl'
            # os.unlink(linkFile) if os.path.exists(linkFile) else None
            # os.link(target_filename,linkFile)
            shutil.copy2(target_filename,linkFile+'.temp')
            shutil.move(linkFile+'.temp',linkFile)
            # loss = cross_entropy(x,y)

        model.eval()
        dataset.test()
        for tsi,item in enumerate(conf.dataloader):
            # print(zi.min())
            loss = model.loss(item).mean()
            # loss = model.loss(zi,x,y,z).mean()
            loss_test_sum +=  float(loss.item())
            if tsi==conf.tsi_max:
                break
                # print(tsi)

        model.train()
        dataset.train()
        model.zero_grad()
        conf.callback_start(epoch,None,None)
        for tri,item in enumerate(tqdm(conf.dataloader)):
            xx = model.grad_loss(item)
            grad_loss = xx.mean(0)
            loss = model.loss(item).mean()
            # loss.mean()
            loss_train_sum += float(loss.item())
            grad_loss.backward()
            conf.optimizer.step()
            conf.callback_step(epoch,(tri,item),loss)
        conf.callback_end(epoch,(tri,item),loss)

        if hasattr(model,'get_hidden'):
            fs_list = []
            zs = []
            idxs = []
            sents = []
            xsa = []
            for tri,item in enumerate(tqdm(conf.dataloader)):
                x    = item['english']
                zi   = item['index']
                y    = item['extracted']
                z    = item['masked']
                inner,outer = model.get_hidden(zi,x,y,z)
                fs= outer[-1]
                fs_list.append(fs)
                idxs.append(zi)
                sents.append(x)
                xsa.append(inner[-1])
                zs.append(outer[-2])
            sents = torch.cat(sents,dim=0)
            fs = torch.cat(fs_list,dim=0)
            idxs = torch.cat(zs,dim=0)
            xsa = torch.cat(xsa,dim=0)
            torch.save(dict(zs=zs,idxs=idxs,fs=fs,xsa=xsa, sents=sents),__file__+'.'+model.__class__.__name__+'.pkl')


        loss_train_mean = loss_train_sum/(1+tri)
        loss_test_mean = loss_test_sum/(1+tsi)
        print(f'Epoch: {epoch}')
        print(f'ModelClassName: {conf.model.__class__.__name__}')
        print(f'Training Loss: {loss_train_mean}')
        print(f'Testing Loss: {loss_test_mean}')

        train_losses.append(loss_train_mean)
        test_losses.append(loss_test_mean)

if __name__=='__main__':
    main()
