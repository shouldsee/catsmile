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


def main():
    CUDA, CKPT,STRICT_LOAD,BLACKLIST,SAVE_INTERVAL,RANDOM_SEED = conf_parse_all(sys.argv)
    # sess    = Session()
    initor  = 'conf_init_translate'
    conf    = ConfigPrototype(__file__)
    conf    = eval(initor)(conf, CUDA,RANDOM_SEED,shuffle=True)
    model   = conf.model
    dataset = conf.dataset
    conf_main_loop(conf,CKPT,STRICT_LOAD,BLACKLIST,SAVE_INTERVAL)


def conf_init_translate(conf, CUDA,random_seed,shuffle,ADD_MONITOR_HOOK=1):
    '''
    autoregression binding

    returns a config object that controls training process.
    binds (dataset,model,device)

    '''


    '''
    Abusing attributes here
    [TBC]
    '''
    conf.CUDA          = CUDA
    conf.shuffle       = shuffle
    conf.device        = torch.device('cuda:0' if CUDA else 'cpu')
    conf.num_epoch     = 6000
    conf.batch_size    = 100
    add_optimizer      = lambda conf,params:torch.optim.RMSprop( params, lr=conf.learning_rate,)
    # add_optimizer      =  lambda conf,params:torch.optim.RMSprop( params, lr=conf.learning_rate,eps=0.01)
    # add_optimizer      = lambda conf,params:torch.optim.Adam( params, lr=conf.learning_rate,)
    #### using Adam with high learning_rate is catastrophic
    # conf._session_name += 'opt-adam-'


    conf.instance = conf.rnd = random_seed
    torch.manual_seed(conf.instance)

    #############################
    'Setting Dataset'
    # conf.task = 'add'
    # conf.task = 'translate-german-english'
    # conf.task = 'translate-wmt14-de2en-5k'
    conf.task = 'translate-mutli30k-de2en-l50'
    # conf.task = 'translate-wmt14-de2en-50k'
    # conf.task = 'translate-wmt14-de2en-20k'
    dconf.attach_task_to_conf(conf,conf.task)

    #############################
    'Setting Model'
    # @staticmethod
    def _add_model( conf,):
        from markov_lm.Model_NLP import NLPLayerConfig
        conf.lconf = NLPLayerConfig(
            graph_dim = conf.dataset.graph_dim,
            embed_dim=128,
            # window_size=10,
            depth =1,
            model_name = 'Seq2SeqWithAttention',
            # model_name = 'Seq2SeqWithNoAttention',

            # depth =10,
            # model_name = 'Seq2SeqWithTransformer',

            # model_name = 'AlignmentModel',
            # model_name = 'SoftAlignmentModel',
            n_step = conf.dataset.data_dim,
            # model_name = 'Seq2SeqWithNoAttention',
        )
        conf.model = model = conf.lconf.to_model(conf.device).to(conf.device)
        conf.learning_rate = 0.00001
        # conf.learning_rate = 0.0001
        # conf.learning_rate = 0.001
        # conf.learning_rate = 0.01
        return model
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
    return conf


def conf_init_ar(conf, CUDA,random_seed,shuffle,ADD_MONITOR_HOOK=1):
    '''
    autoregression binding

    returns a config object that controls training process.
    binds (dataset,model,device)

    '''


    '''
    Abusing attributes here
    [TBC]
    '''
    conf.CUDA          = CUDA
    conf.shuffle       = shuffle
    conf.device        = torch.device('cuda:0' if CUDA else 'cpu')
    conf.num_epoch     = 6000
    conf.batch_size    = 280
    add_optimizer      = lambda conf,params:torch.optim.RMSprop( params, lr=conf.learning_rate,)
    # add_optimizer      =  lambda conf,params:torch.optim.RMSprop( params, lr=conf.learning_rate,eps=0.01)
    # add_optimizer      = lambda conf,params:torch.optim.Adam( params, lr=conf.learning_rate,)
    #### using Adam with high learning_rate is catastrophic
    # conf._session_name += 'opt-adam-'


    conf.instance = conf.rnd = random_seed
    torch.manual_seed(conf.instance)

    #############################
    'Setting Dataset'
    # conf.task = 'add'
    conf.task = 'refill'
    # conf.task = 'ner1'
    # conf.task = 'duie-mlm'
    # conf.task = 'fashion-mnist-compress'
    # conf.task = 'duie-ce'
    dconf.attach_task_to_conf(conf,conf.task)

    #############################
    'Setting Model'
    # @staticmethod
    def _add_model( conf,):
        from markov_lm.Model_NLP import NLPLayerConfig
        conf.lconf = NLPLayerConfig(
            graph_dim = conf.dataset.graph_dim,
            depth =3,
            embed_dim=11,
            window_size=10,
            model_name = 'SimpleDenseNet',
            # model_name = 'SimpleDenseNetTransformer',
            # model_name = 'SimpleDenseNetGaussDist',
            # model_name = 'SimpleDenseNetSquareSum',
            # loss_name = 'HSQ',
            # grad_loss_name = 'HSQ',
            # loss_name = 'SELERR',
            loss_name = 'KLD',
            grad_loss_name = 'KLD',
            # grad_loss_name = 'HSQ',
        )
        conf.model = model = conf.lconf.to_model(conf.device).to(conf.device)
        conf.learning_rate = 0.0001
        # conf.learning_rate = 0.001
        # conf.learning_rate = 0.01
        return model
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
    return conf


if __name__=='__main__':
    main()
