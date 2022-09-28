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
import toml
from datetime import datetime


def argv_to_conf(cli_argv):
    (CUDA, CKPT,STRICT_LOAD,BLACKLIST,SAVE_INTERVAL,RANDOM_SEED,
        model_dict,meta_dict) = conf_parse_all(cli_argv)
    # sess    = Session()
    # initor  = 'conf_init_translate'
    initor = 'conf_init_nlm'
    conf    = ConfigPrototype(__file__,**meta_dict)
    conf    = eval(initor)(conf, CUDA,RANDOM_SEED,model_dict = model_dict, shuffle=True)
    model   = conf.model
    dataset = conf.dataset
    for k in 'CKPT STRICT_LOAD BLACKLIST SAVE_INTERVAL'.split():
        setattr(conf,k,eval(k))

    return conf
    # ,  CKPT, STRICT_LOAD, BLACKLIST,SAVE_INTERVAL

import json

def main():
    if '--compare' in sys.argv:
        '''
        this function needs to specify two models.
        which means we need hierarchical structures,
        with each structure corresponds to a model.

        specifying hierarhical structures on CLI is a bad idea.
        here we uses toml to better represent these structs
        '''

        '''
        js console hacker

        app._reactRootContainer._internalRoot.current.child.stateNode.sendSocketMessage({event_type:'test'})
        thiss = app._reactRootContainer._internalRoot.current.child.stateNode;
        '''


        import visdom
        vis = visdom.Visdom(port=6006)
        '''
        在button的onclick方法里注入socket通信方法

        {"cmd":"forward_to_vis","data":{"target":"box1","eid":"testdev","event_type":"Click"}}
        '''


        win = 'box1'
        env = 'testdev'

        callback_target= f'{env}/{win}'

        injected_js = '''
        const data = new FormData(this.parentElement);
        const dataValue = Object.fromEntries(data.entries());
        const msg = {"cmd":"forward_to_vis","data":{"target":"%s","eid":"testdev","event_type":"SubmitForm", event_data: dataValue}}
        console.log(msg);

        app._reactRootContainer._internalRoot.current.child.stateNode.sendSocketMessage(msg);
        console.log('injected')
        '''%callback_target


        injected_js = injected_js.replace('\"','\'')
        injected_js = ';'.join(injected_js.splitlines())

        default_model1 = '''
        loglr = -4
        LOAD = "100_"

        [model]
        embed_dim=60
        model_name = "DLM142"
        kernel_size = 3
        window_size = 1
        depth = 12
        p_null = 0.05
        '''

        from markov_lm.util_base import toml_to_argv, dict_to_argv

        # x1 = dict_to_argv(toml.loads(default_model1))
        x1 = toml_to_argv(default_model1)
        style_of_input_box = "border:2px solid black; width:250px;height:150px;"
        def add_textarea(k,default, style = style_of_input_box):
            return f'''
            <label>{k}</label>
            <br/>
            <textarea name="{k}" style="{style}">{default}</textarea>
            '''

        k  = 'target_env'
        vd = 'compare0003'
        style = 'border:2px solid black;'

        v = f'<input type="text" name="{k}" value="{vd}" style="{style}"></input>'
        v += add_textarea('model_session_config_1', default_model1)
        v += '<br/>'
        v += add_textarea('model_session_config_2', default_model1)
        v += '<br/>'

        xx1 = toml.loads(default_model1)

        k = 'EventDataType'
        vd = 'MarkovComapreEventData'

        vis.text(
        f'''
        <form action="javascript:void(0);">
        <input type='hidden' name="{k}" value="{vd}" />
        <button onclick="javascript:{injected_js}">submit</button>
        <br/>
        {v}
        </form>
        '''
        ,env=env,win=win)


        from markov_lm.util_base import dset
        from datetime import datetime
        from typing import List, Optional
        from pydantic import BaseModel,validator
        class PaneData(BaseModel):
            command: str
            content: str
            title: str
            id:str
            type:str
            i: int
            contentID: str
            #height: Optional[int]=None

        import logging
        from typing import Literal, Union
        class EventData(BaseModel):
            EventDataType : Literal['EventData']
        class KeyPress(BaseModel):
            pass


        class MarkovComapreEventData(BaseModel):
            EventDataType: Literal['MarkovComapreEventData']
            model_session_config_1: dict
            model_session_config_2: dict
            target_env: str

            @validator("model_session_config_1","model_session_config_2",pre=True)
            def parse_toml(cls,v):
                return toml.loads(v)

            _example = r'''
            {"target": "testdev/box1",
             "eid": "testdev",
             "event_type": "SubmitForm",
             "event_data":
             {"EventDataType": "MarkovComapreEventData",
             "target_env": "compare0003",
             "model_session_config_1": "        loglr = -4\n        LOAD = \"100_\"\n\n        [model]\n        embed_dim=60\n        model_name = \"DLM142\"\n        kernel_size = 3\n        window_size = 1\n        depth = 12\n        p_null = 0.05\n        ",
             "model_session_config_2": "        loglr = -4\n        LOAD = \"100_\"\n\n        [model]\n        embed_dim=60\n        model_name = \"DLM142\"\n        kernel_size = 3\n        window_size = 1\n        depth = 12\n        p_null = 0.05\n        "}
             }
            '''
            def on_recv(self):
                logging.debug(f'[before_on_recv]{self}')
                self._on_recv()
                logging.debug(f'[after_on_recv]{self}')
            def _on_recv(self, ):
                # pcli_argv_1 =
                env = self.target_env
                from markov_lm.Model_NLM import U
                from markov_lm.util_base import toml_to_argv, dict_to_argv
                import pandas as pd

                conf1 = conf_main_loop(argv_to_conf(dict_to_argv(self.model_session_config_1)),'load')
                conf2 = conf_main_loop(argv_to_conf(dict_to_argv(self.model_session_config_2)),'load')

                dataset = conf1.dataset
                dataloader = conf1.dataloader
                vis = conf1.vis
                dataset.test()
                item = next(iter(dataloader))
                loss1,loss2 = map((lambda x:x.model.loss(item)),[conf1,conf2])

                key = 'test_loss_scatter'
                mat = U.N(torch.stack([loss1,loss2],dim=-1))
                x,y = mat.T
                MAX = int(mat.max())
                MIN = 0
                vis.scatter( mat, env=env, win = key,opts=dict(title=key,xtickmin=MIN,xtickmax=MAX, ytickmin=MIN,ytickmax=MAX,markersize=5,textlabels= list(range(len(mat)))))
                # vis.scatter( mat, env=env, win = key,opts=dict(title=key))
                # key = hist1

                key ='test_loss_boxplot'
                vis.boxplot( mat, env=env, win = key,opts=dict(title=key))
                # ,xtickmin=MIN,xtickmax=MAX, ytickmin=MIN,ytickmax=MAX))

                key ='test_loss_diff_histogram'
                vis.histogram( mat.T[1] - mat.T[0], env=env, win = key,opts=dict(title=key))


                target = item['target']
                xdiff = loss2 - loss1
                xsel = target[ xdiff < -7]

                x = np.vectorize(dataset.tgt_wordize)(U.N(xsel))
                # x = [''.join(xx) for xx in x]
                df = pd.DataFrame( x )
                key = 'very Negative xdiff'
                vis.text(df.to_html(), win=key,env=env)
                # loss2 + margin < loss1
                key ='Evaluation Dialogue'
                html_buffer = ''''<input></input>'''
                vis.text(html_buffer,win=key,env=env)



        cls = MarkovComapreEventData
        cls._example_value =  cls(**json.loads(cls._example)['event_data'])

        from typing import Union
        from pydantic import Field

        class VisdomCallbackMessage(BaseModel):
            eid: str
            event_type: str
            target: str
            pane_data: Optional[PaneData]  ### intermediaste
            event_data: Union[EventData, MarkovComapreEventData] = Field(default={},discriminator='EventDataType')

        def _callback(event,env = env, win='reply_win', vis=vis):
            '''
            event = {'eid': 'text_callbacks',
             'event_type': 'KeyPress',
             'key': 'd',
             'key_code': 68,
             'pane_data': {'command': 'window',
                           'content': 'This is a write demo notepad. Type below. Delete '
                                      'clears text:<br>a',
                           'contentID': '43af9d32-5185-4167-8ef2-52b34b3c84e2',
                           'height': None,
                           'i': 1,
                           'id': 'window_3b1fede0134b74',
                           'inflate': True,
                           'title': '',
                           'type': 'text',
                           'width': None},
             'target': 'window_3b1fede0134b74'}

            '''
            # if event['event_type'] =='SubmitForm':
            msg = VisdomCallbackMessage(**event)
            if isinstance( msg.event_data, MarkovComapreEventData):
                # import pdb; pdb.set_trace()
                msg.event_data.on_recv()
                vis.text(f'[{datetime.now().isoformat()}] Callback received <br/><textarea>{(msg.event_data.json())}</textarea>', win=win, env=env)
            else:
                print(msg)

            # BaseMessage()
        vis.register_event_handler(_callback, callback_target)


        ExampleBaseMessage = {'eid': 'text_callbacks',
         'event_type': 'KeyPress',
         'key': 'd',
         'key_code': 68,
         'pane_data': {'command': 'window',
                       'content': 'This is a write demo notepad. Type below. Delete '
                                  'clears text:<br>a',
                       'contentID': '43af9d32-5185-4167-8ef2-52b34b3c84e2',
                       'height': None,
                       'i': 1,
                       'id': 'window_3b1fede0134b74',
                       'inflate': True,
                       'title': '',
                       'type': 'text',
                       'width': None},
         'target': 'window_3b1fede0134b74'}

        xx = VisdomCallbackMessage(**ExampleBaseMessage)
        from pprint import pprint
        pprint(xx)
        import pdb; pdb.set_trace()

        input('Wating for callback')
        sys.exit(0)
        assert 0

        pcli_argv_1 = [
            '--model.embed_dim', '60',
            '--model.model_name', 'DLM142',
            '--model.kernel_size', '3',
            '--model.window_size', '1',
            '--model.depth', '12',
            '--loglr', '-4',
            '--model.p_null', '0.05',
            # '--batch_size','150',
            '--LOAD','100_'
        ]
        pcli_argv_2 = [
            '--model.embed_dim', '60',
            '--model.model_name', 'DLM142',
            '--model.kernel_size', '3',
            '--model.window_size', '1',
            '--model.depth', '12',
            '--loglr', '-4',
            '--model.p_null', '0.05',
            # '--batch_size','150',
            '--LOAD','20_'
        ]
        env = 'compare/0002'
        # confargv_to_conf(pcli_argv)
        # conf1,  CKPT, STRICT_LOAD, BLACKLIST,SAVE_INTERVAL = argv_to_conf(pcli_argv_1)
        # conf2,  CKPT, STRICT_LOAD, BLACKLIST,SAVE_INTERVAL =

        pass
    else:
        # conf,  CKPT, STRICT_LOAD, BLACKLIST,SAVE_INTERVAL =
        conf = argv_to_conf(sys.argv)
        conf_main_loop(conf)
        # , CKPT,STRICT_LOAD,BLACKLIST,SAVE_INTERVAL)


def conf_init_translate(conf, CUDA,random_seed,shuffle,model_dict,ADD_MONITOR_HOOK=1):
    '''
    autoregression binding

    returns a config object that controls training process.
    binds (dataset,model,device)

    '''


    '''
    Abusing attributes here
    [TBC]
    '''
    if conf.log10_learning_rate is not None:
        conf.learning_rate = 10**conf.log10_learning_rate
    conf.CUDA          = CUDA
    conf.shuffle       = shuffle
    conf.device        = torch.device('cuda:0' if CUDA else 'cpu')
    conf.num_epoch     = 6000
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

    conf.batch_size    = 280
    # conf.task = 'translate-mutli30k-de2en-l50'


    # conf.batch_size    = 20
    conf.batch_size    = 280
    conf.task = 'translate-multi30k-de2en-l20'
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
            # window_size=10,
            depth = 1,

            embed_dim=128,
            # model_name = 'Seq2SeqWithAttention',
            # model_name = 'Seq2SeqWithAttentionMixture',
            # model_name = 'Seq2SeqWithNoAttention',
#
            # depth =10,
            # model_name = 'Seq2SeqWithTransformer',

            # model_name = 'AlignmentModel',
            # model_name = 'SoftAlignmentModel',

            # beta = 0.5,
            # model_name = 'GaussianSoftAlignmentModel',

            # model_name = 'SharedSoftAlignmentModel',
            # model_name = 'PerTokenSoftAlignmentModel',
            # model_name = 'HardAlignmentModel',
            # model_name = 'SoftAlignmentModelAllowSourcePad',
            # model_name = 'SoftAlignmentModelSimpleMean',
            # model_name = 'SAM7',
            # beta = 0.5,
            # model_name = 'SAM10',

            # beta = 0.5,
            # kernel_size = 1,
            # model_name = 'SAM11',

            # embed_dim=256,

            # embed_dim=128,
            # beta = 1.0,
            # beta = 2.0,
            # beta = 0.5,
            # model_name = 'SAM13',

            # model_name = 'SAM14',
            model_name = 'SAM5',

            # model_name = 'SAM3',
            n_step = conf.dataset.data_dim,
            # model_name = 'Seq2SeqWithNoAttention',
        )
        conf.model = model = conf.lconf.to_model(conf.device).to(conf.device)
        if getattr(conf.learning_rate,None) is None:
            # conf.learning_rate = 0.00001
            # conf.learning_rate = 0.0001
            # conf.learning_rate = 0.001
            # conf.learning_rate = 0.01
            pass
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

def conf_init_nlm(conf, CUDA,random_seed,shuffle,model_dict,ADD_MONITOR_HOOK=1):
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
    # conf.num_epoch     = conf.target_epoch
    # 6000
    # add_optimizer      = lambda conf,params:torch.optim.RMSprop( params, lr=conf.learning_rate,)
    add_optimizer      =  lambda conf,params:torch.optim.RMSprop( params, lr=conf.learning_rate,eps=0.01)
    # add_optimizer      = lambda conf,params:torch.optim.Adam( params, lr=conf.learning_rate,)
    # add_optimizer      = lambda conf,params:torch.optim.SGD( params, lr=conf.learning_rate,)
    #### using Adam with high learning_rate is catastrophic
    # conf._session_name += 'opt-adam-'
    # conf.learning_rate = 0.0001
    # conf.learning_rate = 0.00001
    # conf.learning_rate = 0.00005
    # conf.learning_rate = 0.01
    # conf.learning_rate = 0.001
    # conf.learning_rate = 0.005
    # conf.learning_rate = 0.01
    # conf.learning_rate = 0.1

    conf.instance = conf.rnd = random_seed
    torch.manual_seed(conf.instance)

    #############################
    'Setting Dataset'
    # conf.task = 'add'
    # conf.task = 'translate-german-english'
    # conf.task = 'translate-wmt14-de2en-5k'

    # conf.batch_size    = 280
    # conf.task = 'translate-mutli30k-de2en-l50'


    # conf.batch_size    = 120
    # conf.batch_size    = 150
    # conf.batch_size    = 25
    # conf.batch_size    = 180
    # conf.batch_size    = 100
    # conf.batch_size    = 5
    conf.task = 'translate-multi30k-de2en-l20'
    conf.task = 'translate-multi30k-de2en-chardata-l100'
    # conf.task = 'translate-multi30k-de2en-chardata-l100-split'
    # conf.task = 'protein-cath-s35-l100'
    # conf.task = 'protein-cath-s35-l20'
    # conf.task = 'translate-ptb-l20'
    # conf.task = 'translate-ptb-l100'
    # conf.task = 'translate-wmt14-de2en-50k'
    # conf.task = 'translate-wmt14-de2en-20k'
    dconf.attach_task_to_conf(conf,conf.task)

    #############################
    'Setting Model'
    # @staticmethod
    def _add_model( conf,):
        # from markov_lm.Model_NLP import NLPLayerConfig
        from markov_lm.Model_NLM import NLMLayerConfig
        conf.lconf = config = NLMLayerConfig(
            graph_dim = conf.dataset.graph_dim,
            # window_size=4,
            depth =1,

            embed_dim=128,
            # # model_name = 'DLM1',
            # # model_name = 'DLM2',
            # model_name = 'DLM5',

            # kernel_size = 1,
            # model_name = 'DLM7',

            # model_name = 'DLM8',
            # kernel_size = 3,


            # model_name = 'DLM9',
            # kernel_size = 3,
            #

            # embed_dim = 300,
            # model_name = 'DLM10',
            # model_name = 'DLM20',

            # model_name = 'DLM25',
            # model_name = 'DLM24',
            # model_name = 'DLM22',
            # model_name = 'DLM26',
            model_name = 'DLM46',
            # model_name = 'DLM47',
            # model_name = 'DLM21',
            # model_name = 'DLM27',
            # kernel_size = 200,
            kernel_size = 400,
            window_size=-1,
            loss_name='0,4,5,7',

            #
            # window_size= 4,
            # model_name = 'DLM30',
            # kernel_size = 400,



            n_step = conf.dataset.data_dim,
            # model_name = 'Seq2SeqWithNoAttention',
        )
        for k,v in model_dict.items():
            type_dict = config.__dataclass_fields__

            assert k in type_dict,f'{config.__class__!r} does not has field {k!r}'
            # import pdb; pdb.set_trace()
            setattr(config, k, type_dict[k].type(v))
            # '10'))
        # config.__dict__.update(model_dict)
        conf.model = model = conf.lconf.to_model(conf.device).to(conf.device)

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



def conf_init_ar(conf, CUDA,random_seed,shuffle,model_dict,ADD_MONITOR_HOOK=1):
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
    # conf.num_epoch     = 6000
    conf.batch_size    = 280
    # conf.batch_size    = 180
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
