'''
- Author: Feng Geng
- Changelog:
  - 20220718: 解耦 dataset-model-runner
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
from markov_lm.conf_data import ConfigDatasetInstance as dconf
from markov_lm.conf_runner import conf_main_loop,conf_parse_all


from markov_lm.Model_gmm import GMMLayerConfig
def _add_model(conf,):
    # conf.lconf  =lconf= GMMLayerConfig(
    #     depth = 1,
    #     graph_dim = conf.dataset.graph_dim,
    #     # embed_dim = 20,
    #     iter_per_layer=-1,
    #     kernel_size = 11,
    #     n_step = 5,
    #     beta = 0.01,
    #     # beta = 10.,
    #     # beta = 1,
    #     embed_dim = 20,
    #     model_name = 'LLGAE',
    #     # model_name = 'GLGAE',
    #     # model_name = 'GLGAEGrad',
    #     p_null = 0.,
    #     # model_name = 'GLGAEGradEOL',
    #     # beta = 1.,
    #     # beta = 1.,
    # )

    # conf.lconf  =lconf= GMMLayerConfig(
    #     depth = 1,
    #    # graph_dim = conf.dataset.graph_dim,
    #
    #     # embed_dim = (14,14),
    #     # model_name = 'CGAE',
    #
    #     iter_per_layer=-1,
    #     # kernel_size = 100,
    #     n_step = 1,
    #     beta = 1.001,
    #
    #     kernel_size = (21,5),
    #     graph_dim = (28,28),
    #     embed_dim = (28,28),
    #     model_name = 'GibbsGGAE',
    #     # embed_dim =
    #     # model_name = 'KMEANS',
    #     # model_name = 'RandomKMEANS',
    #     p_null = 0.,
    # )
    # conf.learning_rate = 0.01

    #
    # conf.lconf  =lconf= GMMLayerConfig(
    #     depth = 1,
    #     iter_per_layer=-1,
    #     n_step = 5,
    #     # beta = 1.001,
    #     beta = 0.01001,
    #     graph_dim = conf.dataset.graph_dim,
    #     embed_dim = 20,
    #     kernel_size = 15,
    #     model_name = 'LLGAEWithResidual',
    #     p_null = 0.,
    # )
    # conf.learning_rate = 0.0001

    # conf.lconf  =lconf= GMMLayerConfig(
    #     depth = 1,
    #     iter_per_layer=-1,
    #     n_step = 15,
    #     # beta = 1.001,
    #     beta = 0.01001,
    #     graph_dim = conf.dataset.graph_dim,
    #     embed_dim = 20,
    #     kernel_size = 15,
    #     model_name = 'LLGAE',
    #     p_null = 0.,
    # )
    # conf.learning_rate = 0.0001


    conf.lconf  =lconf= GMMLayerConfig(
        depth = 1,
        iter_per_layer=-1,
        n_step = 1,
        # beta = 1.001,
        beta = 1.,
        graph_dim = conf.dataset.graph_dim,
        embed_dim = 20,
        kernel_size = 15,
        model_name = 'LLGAE',
        p_null = 0.,
    )
    conf.learning_rate = 0.0001
    #
    # conf.lconf  =lconf= GMMLayerConfig(
    #     depth = 1,
    #     iter_per_layer=-1,
    #     n_step = 1,
    #     # beta = 1.001,
    #     beta = 1.,
    #     graph_dim = conf.dataset.graph_dim,
    #     embed_dim = 20,
    #     kernel_size = 30,
    #     model_name = 'LLGAE',
    #     p_null = 0.,
    # )
    # conf.learning_rate = 0.0001
    #

    # conf.lconf  =lconf= GMMLayerConfig(
    #     depth = 1,
    #     iter_per_layer=-1,
    #     n_step = 5,
    #     # beta = 1.001,
    #     beta = 0.01001,
    #     graph_dim = conf.dataset.graph_dim,
    #     embed_dim = 20,
    #     kernel_size = 30,
    #     model_name = 'LLGAE',
    #     p_null = 0.,
    # )
    # conf.learning_rate = 0.0001
    #


    conf.lconf  =lconf= GMMLayerConfig(
        depth = 1,
        iter_per_layer=-1,
        n_step = 5,
        # beta = 1.001,
        beta = 0.01001,
        graph_dim = (1,28,28),
        # conf.dataset.graph_dim,
        # graph_dim = conf.dataset.graph_dim,
        embed_dim = 20,
        kernel_size = 30,
        model_name = 'BetaVAEConvLocalNoNoise',
        p_null = 0.,
    )
    conf.learning_rate = 0.0001
    # #
    # #
    # conf.lconf  =lconf= GMMLayerConfig(
    #     depth = 1,
    #     iter_per_layer=-1,
    #     n_step = 5,
    #     # beta = 1.001,
    #     beta = 0.01001,
    #     graph_dim = (1,28,28),
    #     # conf.dataset.graph_dim,
    #     # graph_dim = conf.dataset.graph_dim,
    #     embed_dim = 20,
    #     kernel_size = 0,
    #     # model_name = 'BetaVAENoNoise',
    #     model_name = 'AutoEncoderBakeOff',
    #     p_null = 0.,
    # )
    # conf.learning_rate = 0.0001

    #
    #
    #
    # conf.lconf  =lconf= GMMLayerConfig(
    #     depth = 1,
    #     iter_per_layer=-1,
    #     n_step = 5,
    #     beta = 0.010,
    #     # beta = 0.001001,
    #     # graph_dim = (1,28,28),
    #     graph_dim = conf.dataset.graph_dim,
    #     # graph_dim = conf.dataset.graph_dim,
    #     embed_dim = 20,
    #     kernel_size = 0,
    #     # model_name = 'BetaVAENoNoise',
    #     model_name = 'SOGAE2',
    #     p_null = 0.,
    # )
    # conf.learning_rate = 0.0001


    # conf.lconf  =lconf= GMMLayerConfig(
    #     depth = 3,
    #     iter_per_layer=-1,
    #     n_step = 1,
    #     beta = 1.001,
    #     # embed_dim =
    #     # graph_dim = conf.dataset.graph_dim,
    #     graph_dim = (28,28),
    #     embed_dim = 7,
    #     kernel_size = (7,7),
    #     model_name = 'ConvLocalAutoEncoder',
    #     p_null = 0.,
    # )
    # conf.learning_rate = 0.0001




    '''
    AutoEncoder          E20 1225.872217235451
    SKPCA                E20 1207
    GME                  E20 K80  120_1009.56818
    GME                  E20 K160 80_900
    ConvGME              E20 K160 80_830 380_809 420_800
    '''

    # conf.learning_rate = 0.0001
    # conf.learning_rate = 0.01
    # conf.learning_rate = 1.0
    # conf.learning_rate = 0.01
    # conf.learning_rate = 0.0001
    # conf.learning_rate = 0.01
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
    # conf.callback_epoch_start  = _callback
    conf.model = model = lconf.to_model(conf.device).to(conf.device)
    return model


def conf_init(CUDA,random_seed,shuffle,ADD_MONITOR_HOOK=1):
    '''
    returns a config object that controls training process.
    binds (dataset,model,device)
    '''


    '''
    Abusing attributes here
    [TBC]
    '''
    conf = ConfigPrototype(__file__)
    conf.CUDA          = CUDA
    conf.shuffle       = shuffle
    conf.device        = torch.device('cuda:0' if CUDA else 'cpu')
    conf.num_epoch     = 600
    conf.batch_size    = 280
    add_optimizer      = lambda conf,params:torch.optim.RMSprop( params, lr=conf.learning_rate,)
    # add_optimizer      =  lambda conf,params:torch.optim.RMSprop( params, lr=conf.learning_rate,eps=0.01)
    # add_optimizer      = lambda conf,params:torch.optim.Adam( params, lr=conf.learning_rate,)
    #### using Adam with high learning_rate is catastrophic
    # conf._session_name += 'opt-adam-'


    conf.instance= random_seed
    torch.manual_seed(conf.instance)

    #############################
    'Setting Dataset'
    # conf.task = 'add'
    # conf.task = 'refill'
    # conf.task = 'ner1'
    # conf.task = 'duie-mlm'
    conf.task = 'fashion-mnist-compress'
    # conf.task = 'duie-ce'
    dconf.attach_task_to_conf(conf,'fashion-mnist-compress')

    #############################
    'Setting Model'
    _add_model(conf)

    #############################
    'Setting Name'
    'Optimizer is not included in the name'
    conf.get_cache_name()
    conf.model = model = conf.model.to(conf.device)

    #############################
    'Check Parameters'
    'Set Parameter and optimizer'
    conf.params = params = list(model.parameters())
    print(dict(model.named_parameters()).keys())
    conf.optimizer = add_optimizer(conf,params)


    if ADD_MONITOR_HOOK:
        conf.handler = conf.add_hook_for_output_tensor(model,CONFIG_DETACH=1,CONFIG_EXPAND_LEVEL=4,CONFIG_PRINT_CLASS=0)
    else:
        class NullHanlder(object):
            xdict = {}
        conf.handler = NullHanlder()
    assert conf._session_name is not None,'Please set conf._session_name before finish init!'

    conf.callback_end = get_conf_callback_end(conf)
    return conf

def get_conf_callback_end(conf):
    def callback_end(epoch,tri,item,loss,conf=conf):
        '''
        This is a callback
        '''
        model = conf.model
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
    return callback_end

def main():
    CUDA, CKPT,STRICT_LOAD,BLACKLIST,SAVE_INTERVAL,RANDOM_SEED = conf_parse_all(sys.argv)
    conf    = conf_init(CUDA,RANDOM_SEED,shuffle=True)
    model   = conf.model
    dataset = conf.dataset
    conf_main_loop(conf,CKPT,STRICT_LOAD,BLACKLIST,SAVE_INTERVAL)


if __name__=='__main__':
    main()
