'''
- Author: Feng Geng
- Changelog:
  - 20220507-20220514: for (dataset=ArithmeticTest, model=AddModelWithAttention),
  tested different parameters. use_dense_relu=1/11 shows interesting capacity.
  per-position dropout greatly prevents overfitting for unknown reason.
  - 思考: 模型的内部计算经常是梯度的,但是模型参数的更新一般是基于梯度的.这不知道是否矛盾.
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

from markov_lm.c9007_util import tbuf_cls,fws,recur_detach

from markov_lm.Dataset.translation_dataset import ArithmeticTest
from markov_lm.Model_add import AddModelWithBert
from markov_lm.Model_add import AddModelWithAttention


import collections
import math
from markov_lm.conf_gan import ConfigPrototype
#
# class ConfigPrototype(object):
#     def __init__(self,fn,is_sorted=False):
#         self.is_sorted = is_sorted
#         self._session_name = None
#         self.is_model_set = False
#         self.f = open(fn+'.log.html','w')
#         self.tbuf = tbuf_cls()
#         self.s = collections.OrderedDict()
#         return
#
#     def data_input_transform(self,item):
#         raise NotImplementedError
#     def get_ckpt_name(self):
#         return self.__class__.__name__
#
#     def _print(self,*x):
#         f=  self.f
#     # _print = lambda *x,f=f:
#         f.write(f'<pre>{"".join(x)}</pre>\n')
#
#     def init_s(self,k,v):
#         s = self.s
#         if k not in s:
#             s[k]=[k,list(v.shape)]
#         return
#     def grad_loss(self,*a,**kw):
#         return self.loss(*a,**kw)
#     def loss(self,*a,**kw):
#         raise NotImplementedError
#
#     def callback_epoch_start(self,epoch):
#         pass
#
#
#     def callback_step(self,epoch,indexedItem,loss):
#         pass
#         tri,item = indexedItem
#         f = self.f
#         t = self.handler
#         s = self.s
#         _print = self._print
#         for k,v in t.xdict.items():
#             mae = v.abs().mean().item()
#             # s.setdefault(k,[])
#             if k not in s:
#                 s[k]=[k,list(v.shape)]
#             ss = s[k]
#             ss.append(mae*1000)
#             # ss.append(fws('%.3f'%mae,10))
#             # print(v.id)
#             # _print(fws(k,15)+fws(list(v.shape),10)+fws('%.3f'%mae,10))
#
#         opt = self.optimizer
#         for p,v in opt.state.items():
#             k = id(p)
#             if k not in s:
#                 s[k] = [id(p),list(getattr(p,'shape',[1]))]
#             s[k].append( (1000*v.get('square_avg',torch.tensor(0.)).mean().item()))
#
#     def callback_start(self,epoch,a,b):
#         pass
#     def callback_end(self,epoch,indexedItem,loss):
#         s = self.s
#         f = self.f
#         is_sorted = self.is_sorted
#         _print = self._print
#         _print(f'Epoch{epoch}.Model-{self._session_name}.{loss}')
#         CONFIG_HIDE = -100000000
#         def _str(x,s):
#             if isinstance(x,float):
#                 if x >CONFIG_HIDE:
#                     x ='%.3f'%x
#                 else:
#                     x = ''
#             else:
#                 pass
#             return fws(x,s)
#
#         wsl = [20,20] + [5]*30
#         ks = list(s)
#         if self.is_sorted:
#             ks = sorted(ks)
#         for k in ks:
#             v = s.pop(k)
#             _print(*[_str(vv,ws)+'|'+' ' for  vv, ws in zip(v,wsl)])
#             _print('-'*15*len(v))
#
#         # x = ['bias.1000']+list((1000*self.model.att_dense.bias).int().cpu().detach().numpy()[:10])
#         # _print(*[_str(vv,10)+'|'+' '*3 for vv in x])
#         _print('='*35)
#         f.flush()


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

def init_conf(CUDA,shuffle, AddModelWithAttention=AddModelWithAttention,ADD_MONITOR_HOOK=1):
    '''
    Runtime subclassing AddModelWithAttention()

    returns a config object that controls training process.
    binds (dataset,model,device)
    '''


    conf = ConfigPrototype(__file__)
    '''
    Abusing attributes here
    [TBC]
    '''
    # conf.task = 'add'
    # conf.task = 'refill'
    # conf.task = 'ner1'
    # conf.task = 'duie-mlm'
    conf.task = 'duie-ce'

    conf.instance= 29

    torch.manual_seed(conf.instance)
    conf.embed_dim     = 50
    conf.device        =  torch.device('cuda:0' if CUDA else 'cpu')
    conf.num_epoch     = 5000
    conf.learning_rate = 0.001
    conf.batch_size    = 15
    add_optimizer      = lambda conf,params:torch.optim.RMSprop( params, lr=conf.learning_rate,)
    # add_optimizer      =  lambda conf,params:torch.optim.RMSprop( params, lr=conf.learning_rate,eps=0.01)

    conf.SAVE_INTERVAL = 5
    conf.tsi_max = -1
    # print(conf.task)
    # conf.task='UNK'

    torch.manual_seed(conf.instance)
    if conf.task=='refill':
        from markov_lm.Dataset.translation_dataset import RefillDataset
        conf.dataset = dataset =RefillDataset(CUDA=CUDA)
        conf.data_input_transform = lambda item:dict(unmasked = item['english'],masked=item['masked'], mask=item['mask'])

        conf.loss = lambda item:conf.model.loss(item)
        conf.grad_loss = lambda item:conf.model.loss(item)
        conf.callback_epoch_start = lambda epoch: conf.dataset.op_extract_and_mask(n_mask=4)
        conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)


    elif conf.task=='add':
        ### This is a random dataset !!!! init after random seed is set
        conf.dataset = dataset = ArithmeticTest(CUDA=CUDA)
        # import pdb;
        conf.data_input_transform = lambda item: dict(unmasked = item['unmasked'],masked=item['masked'], mask=item['mask']);

        conf.loss = lambda item:conf.model.loss(item)
        conf.grad_loss = lambda item:conf.model.loss(item)

        ### test dataset works
        conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)
        # dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)
    elif conf.task in 'ner1 duie-mlm duie-ce'.split():
        from markov_lm.Dataset.DUIE_NER import DUIE_NER
        ### This is a random dataset !!!! init after random seed is set
        conf.dataset = dataset = DUIE_NER(CUDA=CUDA,task_mode=conf.task)
        conf.data_input_transform = lambda item: item

        if conf.task == 'duie-ce':
            def loss(item,conf=conf):
                model = conf.model
                seq1 = item['unmasked']
                outer,inner = conf.model.forward(dict(masked = seq1 ))
                out1 = inner[-1]

                def get_energy(model,seq1):
                    seq1_embed = model.norm(model.embed(seq1))
                    # seq2 = model.norm(seq2)
                    out1 = model.forward(dict(masked = seq1))[-1][-1]
                    out1 = model.norm( model.project(out1))
                    v1 = ((seq1_embed) * (out1)).mean(-1).mean(1)
                    return v1
                #
                # seq2 = model.norm(model.embed(seq2))
                # out2 = model.norm( model.project(out2))
                #
                # # seq3 = torch.randint(conf.model.config.graph_dim,size=item['masked'].shape,device=model.device)
                # # out3 = conf.model.forward(dict(masked = seq3 ))[-1][-1]
                # # seq3 = model.norm(model.embed(seq3))
                # # out3 = model.norm( model.project(out3))
                #
                # # out2 = model.norm(out2)
                #
                # ### minimum variation
                #
                # v1 = ((seq1) * (out1)).mean(-1).mean(1)
                #
                # v2 = 1.0*((seq2)* (out2) ).mean(-1).mean(1)


                repl = torch.randint(conf.model.config.graph_dim,size=item['mask'].shape,device=model.device)
                # repl = 0 * item['mask'] + model.config.mask_token_idx
                seq2 = torch.scatter(item['masked'],src=repl,index=item['mask'],dim=1)
                # out2 = conf.model.forward(dict(masked = seq2 ))[-1][-1]

                repl = torch.randint(conf.model.config.graph_dim,size=item['mask'].shape,device=model.device)
                seq3 = torch.scatter(item['masked'],src=repl,index=item['mask'],dim=1)

                repl = torch.randint(conf.model.config.graph_dim,size=item['mask'].shape,device=model.device)
                seq4 = torch.scatter(item['masked'],src=repl,index=item['mask'],dim=1)


                beta = 1.0
                v1 = get_energy(model,seq1)
                v2 = get_energy(model,seq2)
                vs = torch.stack([v1,v2,
                get_energy(model,seq3),
                get_energy(model,seq4),
                ],-1)
                lossVal = - vs.log_softmax(-1)[:,0]
                ## hard margin hinge loss
                # margin = 10.
                # lossVal = torch.relu(-v1 + margin + v2)

                ### log-loss soft hinge
                # lossVal = torch.log(1+torch.exp(-v1  + v2))
                # beta = 0.001
                # lossVal = torch.log(1+torch.exp(beta*(-v1  + v2)))
                '''
                #### log-loss is just log_softmax of the positive samples
                #### from-scratch model seems to be overfitting this means the
                the problems proposed by the generator is too easy
                to solve and does not require semantic knowledge to solve.
                Hence it's essential to build harder problems to
                '''

                # lossVal = -v1 + (conf.last_v1_mean - v1).square() +  v2 #+ 0.5*(seq3*out3).mean(-1).mean(1)
                # conf.last_v1_mean *= 0.8
                # conf.last_v1_mean += 0.2*v1.mean().detach()


                # lossVal = -(((seq1) * (out1)).mean(-1).mean(1)).sigmoid()  +  (1.0*((seq2)* (out2) ).mean(-1).mean(1)).sigmoid() #+ 0.5*(seq3*out3).mean(-1).mean(1)
                # lossVal = -(model.embed(seq1) * model.project(out1)).mean(-1).mean(1) + (model.embed(seq2)* model.project(out2) ).mean(-1).mean(1)
                # import pdb; pdb.set_trace()\
                return lossVal

            conf.loss = loss
            conf.grad_loss = loss
            conf.last_v1_mean = 0.
        else:
            conf.loss = lambda item:conf.model.loss(item)
            conf.grad_loss = lambda item:conf.model.loss(item)

        ### test dataset works
        conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)
        if conf.task in 'duie-mlm duie-ce':
            conf.callback_epoch_start = lambda epoch: conf.dataset.op_sample_mask(n_mask=10)
            # conf.
        # dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)
    else:
        raise NotImplementedError(conf.task)
        # op_extract_and_mask
    (conf.dataset[range(5)])

    # '''
    # now specify model
    # '''

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



    CLS = [AddModelWithAttention]
    def _add_model(conf,cls = CLS):
        conf.model = model = CLS[0](
            graph_dim      = dataset.graph_dim,
            depth          = conf.depth,
            use_dropout    = conf.use_dropout,
            kernel_size    = conf.kernel_size,
            use_dense_relu = conf.use_dense_relu,
            use_layernorm  = conf.use_layernorm,
            use_gradnorm   = conf.use_gradnorm,
            use_input_image= conf.use_input_image,
            embed_dim      = conf.embed_dim,device=conf.device,
            kernel_dim     = conf.kernel_dim,
            iter_per_layer = conf.iter_per_layer,
            mask_token_idx = dataset.mask_token_idx,
            n_choice       = conf.n_choice,# RefillLoss()

            step_size      = conf.step_size)
        return model
    from markov_lm.Model_add import LayerConfig#,SimpleConfig
    conf.kernel_dim = None
        # AddModelWithAttention.callback_step = callback_step
    conf.n_choice = 0
    conf.kernel_dim = 10
    if 1:
        def _add_model(conf):
            conf.model = model = CLS[0](device=conf.device, config=conf.lconf,charset = dataset.charset)
            return model

        conf.learning_rate = 0.001
        # conf.kernel_size = 15
        # conf.depth = 20
        # conf.embed_dim = 50
        conf.embed_dim = 100
        conf.lconf = LayerConfig(
            kernel_size = 10,
            depth = 4,
            embed_dim = conf.embed_dim,
            kernel_dim = 11,
            use_dropout= 0.48,
            use_dense_relu = 11,
            use_layernorm = 1,
            use_gradnorm = 1,
            use_input_image =0,
            step_size = 0.05,
            iter_per_layer= 100,
            mask_token_idx = dataset.mask_token_idx,
            graph_dim=dataset.graph_dim
        )
        conf.__dict__.update(conf.lconf.__dict__)
        assert conf.is_model_set is False
        conf.is_model_set =True
        # conf.learning_rate = 0.00001
        conf.batch_size = 60

        conf._session_name = ''

        from markov_lm.Model_add import AddModelWithAttentionStacked,AddModelBertInterface
        CLS[0] = AddModelWithAttentionStacked

        # add_optimizer      = lambda conf,params:torch.optim.SGD( params, lr=conf.learning_rate,)

        #
        # from markov_lm.Model_add import AddModelBertInterface, AddModelBertInterfaceConfig
        # CLS[0] = AddModelBertInterface
        # conf.lconf = AddModelBertInterfaceConfig(
        #     embed_dim=conf.embed_dim,
        #     # embed_dim=200,
        #     graph_dim=dataset.graph_dim,
        #     pretrain_model_name='bert-base-chinese',
        #     mask_token_idx = dataset.mask_token_idx,
        #     use_original_embedding=0,
        #     attach = 0,
        #     )
        # conf.depth = 5


        model = _add_model(conf)

        # print([getattr(x,'reset_parameters',lambda :'noop')() for x in model.children()])
        # conf._session_name += '-RESET'
        # model.reset_parameters()


    if 0:
        '''
        Adapter for RefillModelRNNConvolve
        '''
        conf.use_dropout= 0.0
        conf.use_dense_relu = 13
        conf.use_layernorm = 0
        conf.use_gradnorm = 0
        conf.use_input_image =0

        conf.kernel_size = 5
        conf.depth = 12
        conf.embed_dim = 40
        conf.use_mixture=0
        assert conf.is_model_set is False
        conf.is_model_set =True

        from markov_lm.Model_Refill import RefillModelRNNConvolve
        def _add_model(conf):
            conf.state_count   = conf.kernel_size
            conf.mixture_count = conf.kernel_size
            conf.model = model = RefillModelRNNConvolve(
                # total_length=conf.dataset.total_length(),
                total_length=1,
                # conf.dataset.total_length(),
                # min_len=dataset.min_len,
                min_len=-1,
                    # dataset.min_len,

                graph_dim     = dataset.graph_dim,
                mixture_count = conf.mixture_count,
                use_mixture   = conf.use_mixture,
                state_count   = conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.mask_token_idx)

            conf.model.use_mask = True
            # english_vocab['<mask>'])
            return model
        # conf.data_input_transform = lambda item:dict(unmasked = item['english'],masked=item['masked'], mask=item['mask'])
        conf.data_input_transform = lambda item:item
        if conf.task=='refill':
            conf.loss = lambda item:conf.model.loss(
                x    = item['english'],
                zi   = item['index'],
                y    = item['extracted'],
                z    = item['masked'],
                mask = item['mask'])
            conf.grad_loss = conf.loss
            conf.callback_epoch_start = lambda epoch: conf.dataset.op_extract_and_mask(n_mask=4)
        elif conf.task in 'add ner1 duie-mlm'.split():
            conf.loss = lambda item:conf.model.loss(
                x    = item['unmasked'],
                # zi   = item['index'],
                zi = None,
                y = None,
                # y    = item['extracted'],
                z    = item['masked'],
                mask = item['mask'])
            conf.grad_loss = conf.loss
        #lambda item:conf.loss(item)
        conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)

        conf.is_model_set =True


    # _add_model(conf)
    if isinstance(conf.model,AddModelWithAttention):
        def callback(outer,inner):
            (item,lptok,xexp) = inner
            state = conf.dataset.mode
            # state = 'train' if conf.dataset.state=='train'
            _tok = lambda x: '' if x ==0 else x
            for i in range(min(5,len(lptok))):
                k = f'token_{state}_{i}_word'
                # print(item['masked'].shape)
                vlist = item['masked'][i].detach().cpu().numpy().tolist()
                conf.s[k] = [k,list(item['masked'].shape)] +   vlist

                k = f'token_{state}_{i}_wordv'
                vlist = list(map(conf.dataset.charset.__getitem__,vlist))
                conf.s[k] = [k,list(item['masked'].shape)] +   vlist
                # [[v] for v in item['masked'][i].detach().cpu().numpy().tolist()]
                # list(map(_tok, (10*lptok[i].argmax(-1)).detach().cpu().numpy().tolist()))
                k = f'token_{state}_{i}_unmask'
                conf.s[k]  = [k,''] + list(map(_tok,(item['unmasked'][i]).detach().cpu().numpy()))
                k = f'token_{state}_{i}_pred'
                conf.s[k] = [k,''] +  list(map(_tok, (1*lptok[i].argmax(-1)).detach().cpu().numpy().tolist()))
                k = f'token_{state}_{i}_exp'
                conf.s[k]  = [k,''] + list(map(_tok,(xexp[i]).detach().cpu().numpy()))
                k = f'token_{state}_{i}_ws'
                conf.s[k]  = [k]
                # for k,v in conf.s.items():
                #     print(k,len(v))

                # conf._print(*[fws(xx,10) + '  ' for xx in  lptok[i].argmax(-1).detach().cpu().numpy()])
        conf.model.callback_end = callback

    '''
    Class name is used to construct checkpoint filenames. manually record important params here, needs a better system.
    [TBC]
    '''

    conf._session_name +=  f'-S{conf.instance}'
    conf._session_name += f'-task{conf.task}-shuffle{int(shuffle)}'
    conf._session_name += f'-{conf.model.__class__.__name__}-D{conf.depth}-E{conf.embed_dim}-K{conf.kernel_size}-KE{conf.kernel_dim}-IPL{conf.iter_per_layer}-'
    # conf._session_name += f'DenseRelu{conf.use_dense_relu}-Layernorm{conf.use_layernorm}-Dropout{conf.use_dropout}-Gradnorm{conf.use_gradnorm}-loglr{math.log10(conf.learning_rate):.1f}'
    conf._session_name += f'DenseRelu{conf.use_dense_relu}-Layernorm{conf.use_layernorm}-Dropout{conf.use_dropout}-Gradnorm{conf.use_gradnorm}-loglr{-4:.1f}'
    conf._session_name += f'-nchoice{conf.model.n_choice}' if conf.n_choice else ''
    conf._session_name += f'-UseInputImage{conf.use_input_image}-1i{conf.instance}'
    # try:
    #     conf._session_name+= '-'+conf.lconf.to_alias()
    # except:
    #     raise

    conf.model = model = conf.model.to(conf.device)
    params = list(model.parameters())
    print(dict(model.named_parameters()).keys())

    #### using Adam with high learning_rate is catastrophic
    conf.optimizer = add_optimizer(conf,params)


    if ADD_MONITOR_HOOK:
        conf.handler = add_hook_for_output_tensor(model,CONFIG_DETACH=1,CONFIG_EXPAND_LEVEL=4,CONFIG_PRINT_CLASS=0)
    else:
        class NullHanlder(object):
            xdict = {}
        conf.handler = NullHanlder()
    assert conf._session_name is not None,'Please set conf._session_name before finish init!'
    return conf

def get_model_test_loss(conf):
    '''
    For each test entry, obtain model's prediction loss, as a dict{index:loss}
    This is for evaluation of models performance
    '''
    model                = conf.model
    dataset              = conf.dataset
    loss                 = conf.loss
    data_input_transform = conf.data_input_transform
    dataloader           = conf.dataloader

    model.eval()
    dataset.test()
    index = []
    lsl = []
    for tsi,item in enumerate(dataloader):
        item = data_input_transform(item)
        _loss = loss(item).detach()
        if _loss.shape.__len__()>=2:
            _loss = _loss.mean(item.shape[1:])
        index.append(item['index'].to(_loss.device))
        lsl.append(_loss)
        # item['losses'])
    index = torch.cat(index,dim=0)
    lsl   = torch.cat(lsl,dim=0)
    st = index.argsort()
    v = torch.stack([index,lsl],dim=1)[st,:]

    return v
        # loss_test_sum +=  float(loss.item())

def main():
    CUDA    = ('--cpu' not in sys.argv)
    conf    = init_conf(CUDA,shuffle=True)
    model   = conf.model
    dataset = conf.dataset

    CKPT = parse_checkpoint(sys,)
    if(CKPT!='-1'):

        epoch = CKPT
        # res = glob.glob(os.path.join("Checkpoints",f"{conf._session_name}_{CKPT}_*.pkl"))
        res = glob.glob(os.path.join("Checkpoints",f"{conf._session_name}_{epoch}*.pkl"))
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
        # print('[strict]',STRICT_LOAD)
        v = ''
        k = '--load_blacklist'
        if k in sys.argv:
            v = sys.argv[sys.argv.index(k)+1]
        BLACKLIST = v.split(',')
        for k in BLACKLIST:
            if k:
                del x[k]

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
        curr_seed = torch.seed()
        conf.callback_epoch_start(epoch)


        loss_mat = get_model_test_loss(conf)
        loss_test_mean = loss_mat[:,1].mean()

        ### needs to random extract tokens into sets and sequences
        if(epoch % conf.SAVE_INTERVAL ==0):
            # target_filename = conf.get_ckpt_name(os.path.join("Checkpoints",f"{conf._session_name}_{epoch}_{loss_test_mean:.5f}.pkl"))
            target_filename = os.path.join("Checkpoints",f"{conf._session_name}_{epoch}_{loss_test_mean:.5f}.pkl")
            torch.save({
                "model"       :conf.model.state_dict(),
                "optimizer"   :conf.optimizer.state_dict(),
                "epoch"       :epoch,
                "train_losses":train_losses,
                "test_losses" :test_losses,
                "loss_mat"    :loss_mat,
                "curr_seed"   :[curr_seed, torch.seed()],
                "model_config":conf.model.__dict__.get('config',{})
            },target_filename)
            linkFile = __file__+'.curr.ckpt.pkl'
            # os.unlink(linkFile) if os.path.exists(linkFile) else None
            # os.link(target_filename,linkFile)
            shutil.copy2(target_filename,linkFile+'.temp')
            shutil.move(linkFile+'.temp',linkFile)
            # loss = cross_entropy(x,y)


        model.train()
        dataset.train()
        model.zero_grad()
        conf.callback_start(epoch,None,None)
        for tri,item in enumerate(tqdm(conf.dataloader)):
            item            = conf.data_input_transform(item)
            grad_loss       = conf.grad_loss(item).mean()
            loss            = conf.loss(item).mean()
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
                item = conf.data_input_transform(item)
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
            torch.save(dict(zs=zs,idxs=idxs,fs=fs,xsa=xsa, sents=sents), __file__+'.'+conf._session_name+'.pkl')


        loss_train_mean = loss_train_sum/(1+tri)
        # loss_test_mean = loss_test_sum/(1+tsi)
        print(f'Epoch: {epoch}')
        print(f'ModelClassName: {conf._session_name}')
        print(f'Training Loss: {loss_train_mean}')
        print(f'Testing Loss: {loss_test_mean}')

        train_losses.append(loss_train_mean)
        test_losses.append(loss_test_mean)

if __name__=='__main__':
    main()
