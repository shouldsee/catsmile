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
def attach_dataset_fashion_mnist_compress(conf):
    '''
    Image recover for fashion mnist
    '''

    from markov_lm.Dataset.fashion_mnist import fashion_mnist
    conf.dataset = fashion_mnist(CUDA=conf.CUDA)
    conf.dataloader = torch.utils.data.DataLoader(conf.dataset, batch_size=conf.batch_size, shuffle=conf.shuffle)
    conf.data_input_transform = lambda item:[item.__setitem__('epoch',conf.epoch),item][-1]
    conf.loss = lambda *a,**kw: conf.model.loss(*a,**kw)
    # conf.loss = lambda *a,**kw: conf.model.grad_loss(*a,**kw)
    conf.grad_loss = lambda *a,**kw: conf.model.grad_loss(*a,**kw)
    # conf.grad_loss = lambda *a,**kw: conf.model.loss(*a,**kw)
    # conf.grad_loss = lambda : conf.model.loss
    #dict(unmasked = item['english'],masked=item['masked'], mask=item['mask'])
    # import pdb; pdb.set_trace()

def init_conf(CUDA,shuffle, AddModelWithAttention=AddModelWithAttention,ADD_MONITOR_HOOK=1):
    '''
    Runtime subclassing AddModelWithAttention()

    returns a config object that controls training process.
    binds (dataset,model,device)
    '''


    conf = ConfigPrototype(__file__)
    conf.CUDA= CUDA
    conf.shuffle=  shuffle
    '''
    Abusing attributes here
    [TBC]
    '''
    # conf.task = 'add'
    # conf.task = 'refill'
    # conf.task = 'ner1'
    # conf.task = 'duie-mlm'
    conf.task = 'fashion-mnist-compress'
    # conf.task = 'duie-ce'

    conf.instance= 28

    torch.manual_seed(conf.instance)
    conf.embed_dim     = 50
    conf.device        =  torch.device('cuda:0' if CUDA else 'cpu')
    conf.num_epoch     = 600
    # 5000
    conf.learning_rate = 0.001
    conf.batch_size    = 360
    add_optimizer      = lambda conf,params:torch.optim.RMSprop( params, lr=conf.learning_rate,)
    # add_optimizer      =  lambda conf,params:torch.optim.RMSprop( params, lr=conf.learning_rate,eps=0.01)

    if '--save' in sys.argv:
        v= sys.argv[sys.argv.index('--save')+1]
    else:
        v = 30
    v = int(v)
    conf.SAVE_INTERVAL = v
    conf.tsi_max = -1
    # print(conf.task)
    # conf.task='UNK'
    conf.n_mask = 1
    conf.thin_sep = 5
    conf.max_len = 50

    torch.manual_seed(conf.instance)
    if 0:
        pass
    elif conf.task == 'fashion-mnist-compress':
        attach_dataset_fashion_mnist_compress(conf)

    else:
        raise NotImplementedError(conf.task)
        # op_extract_and_mask
    # (conf.dataset[range(5)])


    from markov_lm.Model_gmm import GMMLayerConfig
    # from markov_lm.Model_gmm import GlobalPCA
    CLS = []
#    CLS = [AddModelWithAttention]
    def _add_model(conf,cls = CLS):
        conf.lconf  =lconf= GMMLayerConfig(
            depth = 1,
            graph_dim = conf.dataset.graph_dim,
            # embed_dim = 20,
            iter_per_layer=-1,
            kernel_size = 50,
            n_step = 5,
            beta = 0.01,
            # beta = 1,
            embed_dim = 20,

            # beta = 1.,
            # beta = 1.,

            # model_name = 'GlobalMixtureEncoder',
            # model_name = 'GlobalMixtureEncoderEOL',
            # model_name = 'AutoEncoder',
            # model_name = 'MixedDiffEncoder',
            # model_name = 'GlobalMixtureEncoderDiffEOL',

            # 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size40-model-nameGlobalMixtureEncoderDiffEOLGrad-beta0.01-n-step5-loglr-1.0_30_936.48840.pkl'
            model_name = 'GlobalMixtureEncoderDiffEOLGrad',
            # model_name = 'NLAutoEncoder',
            # model_name = 'SKPCA',
            # kernel_size = 10,

            # embed_dim = 20,
            # model_name = 'GlobalMixtureEncoderEOL',

            # embed_dim = 20,
            # model_name = 'GlobalMixtureEncoderStratifiedEOL',

           # model_name = 'GlobalMixtureEncoderEOLGrad',
           # model_name = 'GlobalMixtureEncoderLEOP',
           # model_name = 'GlobalMixtureEncoderLEOPMLP',
           # model_name = 'GlobalGradMixtureEncoder',
           # model_name = 'GlobalGradMixtureDiffEncoder',
           # model_name = 'Conv2DMixtureEncoder',
           # model_name = 'Conv2DMixtureV1',
        )
        '''
        AutoEncoder          E20 1225.872217235451
        SKPCA                E20 1207
        GME                  E20 K80  120_1009.56818
        GME                  E20 K160 80_900
        ConvGME              E20 K160 80_830 380_809 420_800
        '''

        # conf.learning_rate = 0.0001
        # conf.learning_rate = 0.001
        # conf.learning_rate = 1.0
        conf.learning_rate = 0.1
        # conf.learning_rate = 0.0001
        # conf.learning_rate = 0.001
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

        conf.callback_epoch_start  = _callback

        conf.model = model = lconf.to_model(conf.device).to(conf.device)
        return model


    conf._session_name = ''
    # add_optimizer      = lambda conf,params:torch.optim.Adam( params, lr=conf.learning_rate,)
    # conf._session_name += 'opt-adam-'


    _add_model(conf)

    '''
    Class name is used to construct checkpoint filenames. manually record important params here, needs a better system.
    [TBC]
    '''

    conf._session_name +=  f'-S{conf.instance}'
    conf._session_name += f'-task{conf.task}-shuffle{int(shuffle)}'

    for k,v in conf.lconf.__dict__.items():
        k = k.replace('_','-')
        #k = k.replace('_','-')
        conf._session_name+= f'-{k}{v}'
    conf._session_name += f'-loglr{math.log10(conf.learning_rate):.1f}'
    # try:
    #     conf._session_name+= '-'+conf.lconf.to_alias()
    # except:
    #     raise

    conf.model = model = conf.model.to(conf.device)
    conf.params = params = list(model.parameters())
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
        conf.epoch = epoch
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
                "model_config":conf.model.__dict__.get('config',{}),
                "model_cls"   :conf.model.__class__,
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
#                item['epoch'] = epoch
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
