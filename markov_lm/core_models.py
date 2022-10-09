'''
Custom pydantic DataModels
'''
import logging
import json,toml

from datetime import datetime
from typing import List, Optional, Union,Literal, ClassVar
from pydantic import BaseModel,validator
from pydantic import Field

from markov_lm.util_base import dset
from markov_lm.util_base import toml_to_argv, dict_to_argv
from markov_lm.util_base import js_inject_prep, vis_html_jump_button, add_textarea, rapp


import plotly.graph_objects as go
class InterfaceDev(object):
    @classmethod
    def bind_visdom(cls, vis):
        z = [[1, 20, 30],
          [20, 1, 60],
          [30, 60, 1]]
        text = [['one', 'twenty', 'thirty'],
              ['twenty', 'one', 'sixty'],
              ['thirty', 'sixty', 'one']]
        fig = go.Figure(data=go.Heatmap(
                            z=z,
                            text=text,
                            texttemplate="%{text}",
                            textfont={"size":20}))

        # fig.show()
        # print(fig)
        vis.plotlyplot(fig, env='testdev',win='array')

class PaneData(BaseModel):
    command: str
    content: str
    title: str
    id:str
    type:str
    i: int
    contentID: str


class EventData(BaseModel):
    EventDataType : Literal['EventData']
class KeyPress(BaseModel):
    pass




if 1:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # import
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    # logging.setLevel(logging.DEBUG)
    '''
    Interace = (frontend_form + EventData_spec+ backend_callback)
    '''


    class CallbackPrototype(BaseModel):
        # name: str
        @classmethod
        def on_recv(self, msg, pmsg, vis):
            # _dp = logging.debug
            _dp = print
            _dp(f'[before_on_recv]{self}{pmsg}')
            _dp(pmsg.json(indent=2))
            # get_event_initor(msg['target'])(msg, pmsg, vis)
            assert isinstance(msg,MarkovComapreEventData),repr(msg)
            assert isinstance(pmsg,VisdomCallbackMessage),repr(pmsg)
            self._on_recv(msg, pmsg, vis)
            _dp(f'[after_on_recv]{self}{pmsg}')
        @classmethod
        def _on_recv(cbk, msg, pmsg, vis):
            self = msg
            env = self.target_env

            vis.text(f'''
            <h2>controller</h2>
            <b>{vis_html_jump_button(self.origin_env)}</b>
            <br/>
            Last Message from {pmsg.eid!r}
            <br/>
            <label>msg.json()</label>
            <br/>
            <textarea>{pmsg.json(indent=2)}</textare>
            ''',win='controller',env=env)
            # raise NotImplementedError(cbk)

    class CallbackTrain(CallbackPrototype):
        name: ClassVar = 'train'
        # @staticmethod
        @classmethod
        def _on_recv(cls, msg, pmsg, vis):
            # assert hasattr(pmsg,'event_data'), f'Unexpected callback {pmsg!r}'
            super()._on_recv(msg,pmsg,vis)
            self = msg
            env = self.target_env

            vis.text(f'''
            <h2>controller2</h2>
            <b>received {cls!r} Message!!!</b>
            <b>{vis_html_jump_button(self.origin_env)}</b>
            <br/>
            Last Message from {pmsg.eid!r}
            <br/>
            <label>msg.json()</label>
            <br/>
            <textarea>{pmsg.json(indent=2)}</textarea>
            ''',win='controller2',env=env)
            # return 0
            # self = msg
            # env = self.target_env
            msg.vis_add_form_with_callback(vis, msg.target_env, cls.name)
            # __name__)
            if 1:
                win = f'controller-{cls.name}'
                # buf =''

                # pmsg.json(indent=2)}</textare>
                # ''',win='controller2',env=env)
                from markov_lm.util_base import toml_to_argv, dict_to_argv
                from markov_lm.nlp.train import conf_main_loop,argv_to_conf
                # import plotly.graph_objects as go
                buf = ['']
                # buf[] +=
                class stdout(object):
                    @staticmethod
                    def write(v):
                        buf[0] = f'{v}\n' + buf[0][:200]
                        vis.text(f'''
                        <h2>{win}</h2>
                        <textarea style="height:400px; width: 500px;">{buf[0]}</textarea>''',win=win,env=env)
                stdout.write(f'\n[{datetime.now().isoformat()}]{cls}.Started')
                conf1 = conf_main_loop(argv_to_conf(dict_to_argv(self.model_session_config_1)),'train', stdout)
                # buf += f'\n[{datetime.now().isoformat()}]{cls}.Ended'

    class CallbackCompare(CallbackPrototype):
        name: ClassVar = 'compare'
        # name = 'compare'
        # @staticmethod
        @classmethod
        def _on_recv(cbk, msg, pmsg, vis):
            # assert hasattr(pmsg,'event_data'), f'Unexpected callback {pmsg!r}'
            super()._on_recv(msg,pmsg,vis)
            self = msg
            env = self.target_env
            msg.vis_add_form_with_callback(vis, msg.target_env, cbk.__name__)
            if 1:
                from markov_lm.Model_NLM import U
                import pandas as pd
                import numpy as np
                import torch
                from markov_lm.util_base import toml_to_argv, dict_to_argv
                from markov_lm.nlp.train import conf_main_loop,argv_to_conf
                import plotly.graph_objects as go

                conf1 = conf_main_loop(argv_to_conf(dict_to_argv(self.model_session_config_1)),'load')
                conf2 = conf_main_loop(argv_to_conf(dict_to_argv(self.model_session_config_2)),'load')

                dataset = conf1.dataset
                dataloader = conf1.dataloader
                # vis = conf1.vis


                dataset.test()
                item = next(iter(dataloader))
                # rng = torch.get_rng_state().numpy()

                from jinja2 import Environment, BaseLoader
                def jr(template, data):
                    rtemplate = Environment(loader=BaseLoader).from_string(template)
                    data = rtemplate.render(**data)
                    return data
                def jr_table(tab,):
                    template = '''
                    <table border=1 style="{{BORDER_STYLE}}">
                    <tbody>
                    {% for tabrow in tab %}
                        <tr>
                        {% for tabrowcol in tabrow %}
                            <td style="{{BORDER_STYLE}}">{{tabrowcol}}</td>
                        {% endfor %}
                        </tr>
                    {% endfor %}
                    </tbody>
                    </table>
                    '''
                    buf = jr(template, dict(tab=tab,BORDER_STYLE = 'border:0.5px solid grey; text-align:center; padding: 2px;'))
                    return buf



                def get_data(item=item):
                    generator = torch.Generator(device=conf1.device)
                    # rng = generator.get_state()
                    # with torch.no_grad():
                    seed = generator.seed()
                    loss = []
                    for i, conf in enumerate([conf1,conf2]):
                        model = conf.model
                        generator = generator.manual_seed(seed)
                        item = model.add_target_notnull(item)
                        T = item['target'].shape[-1]
                        t1, t2 = model.corrupt_target(item['target'], item['target_notnull'],generator)
                        generator = generator.manual_seed(seed)
                        loss += [model._loss(item,'masked_loss_per_loc', generator = generator ).reshape((-1,T))]

                    # loss1,loss2 = map((lambda x:) ,[conf1,conf2])
                    # loss1,loss2 = map((lambda x:x.model._loss(item,'loss', generator )) ,[conf1,conf2])
                    # mat = loss_sum = U.N(torch.stack(loss,dim=-1).sum(2).mean(1))
                    mat = loss_sum = U.N(torch.stack(loss,dim=-1).sum(1))
                    wordize = np.vectorize(dataset.tgt_wordize)
                    G = conf1.dataset.graph_dim

                    # T = t1.shape[-1]
                    t1 = t1.reshape((-1,T))
                    t2 = t2.reshape((-1,T))

                    xw1 = wordize(U.N(t1).clip(0,G-1))
                    xw2 = wordize(U.N(t2).clip(0,G-1))
                    # loss1 = U.N(loss1)
                    # loss2 = U.N(loss2)
                    import collections
                    cls = collections.namedtuple('_fret','loss_sum loss t1 t2 xw1 xw2')

                    return cls(
                        loss_sum = U.N(loss_sum),
                        loss = list(map(U.N,loss)), t1 = U.N(t1), t2=U.N( t2 ),
                        xw1=xw1,xw2=xw2,
                        )
                    # 1 = loss1,loss2=loss2)
                xd = get_data()

                xdiff = xd.loss_sum[:,1] - xd.loss_sum[:,0]

                if 1:
                    key = 'very Negative xdiff'
                    target = item['target']
                    # xdiff = xdloss[]
                    # loss2 - loss1
                    # sel =  xdiff < -1
                    DIFF_BELOW = -20
                    sel =  (xd.loss[1] - xd.loss[0]).sum(-1) < DIFF_BELOW


                    text = xd.xw2
                    t0 = xd.xw1
                    t1 = xd.xw2
                    t2 = xd.xw2
                    t3 = t2.copy()
                    t3[:] = '-'

                    B = len(text)
                    z0 = (xd.xw2!=xd.xw1)
                    z1 = (xd.loss[0])
                    z2 = (xd.loss[1])
                    z3 = z2*0
                    z0 = z0*(z1.max()+z2.max())/2.

                    ZMIN,ZMAX=0,1
                    YMAX = 30

                    z    = np.stack([z0,z1,z2,z3], 1)
                    text = np.stack([t0,t1,t2,t3], 1)
                    z    = z[sel]
                    text = text[sel]

                    BB   = z.shape[0]*z.shape[1]
                    z    = z.reshape((BB,-1))
                    text = text.reshape((BB,-1))

                    z = z.clip(ZMIN,ZMAX)

                    if YMAX>=0:
                        z = z[:YMAX]
                        text = text[:YMAX]

                    # text = np.tile(text[:,None],(1,3,1)).reshape((3*BB,-1))

                    # z  =[sel]

                    # z = z[sel]
                    # text = text[sel]

                    title = f'{key} {sel.shape} {z.shape} {text.shape}'
                    key

                    if 1:
                        fig = go.Figure(data=go.Heatmap(
                                            z=z,
                                            text=text,
                                            xgap=1,
                                            ygap=1,
                                            texttemplate="%{text}",
                                            textfont={"size":10}))
                        # title = f'{key}, len:{len(x)}'
                        fig['layout'].update(title=title)
                        vis.plotlyplot(fig, env=env,win=key)

                key = 'test_loss_scatter'
                mat = xd.loss_sum
                x,y = mat.T
                MAX = int(mat.max())
                MIN = 0
                vis.scatter( mat, env=env, win = key,opts=dict(title=key,xtickmin=MIN,xtickmax=MAX, ytickmin=MIN,ytickmax=MAX,markersize=5,textlabels= list(range(len(mat)))))

                key ='test_loss_boxplot'
                vis.boxplot( mat, env=env, win = key,opts=dict(title=key))
                # ,xtickmin=MIN,xtickmax=MAX, ytickmin=MIN,ytickmax=MAX))

                key =f'test_loss_diff_histogram'
                vis.histogram( mat.T[1] - mat.T[0], env=env, win = key,opts=dict(title=key+f' ts:{datetime.now().isoformat()}'))



                # vis.text(jr_table(x), win=key,env=env)
                # loss2 + margin < loss1
                key ='Evaluation Dialogue'
                html_buffer = ''''<input></input>'''
                vis.text(html_buffer,win=key,env=env)

            # 'compare train'.split())
    class MarkovComapreEventData(BaseModel):
        EventDataType: Literal['MarkovComapreEventData']='MarkovComapreEventData'
        # EventDataType: 'MarkovComapreEventData'
        model_session_config_1: dict
        model_session_config_2: dict
        target_env: str
        origin_env: str='testdev'
        # callback_list: list = 'compare train'.split()
        callback_list: ClassVar  = [CallbackCompare,CallbackTrain]
        # 'compare train'.split()

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



        def vis_add_form_with_callback(self, vis, env, win, ):
            callback_list = self.callback_list
            # del callback_target
            '''
            Create a form that would sends this message back
            to visdom to trigger a backend callback
            '''

            msg = self
            def get_js_cbk(callback_target,env=env,rapp=rapp, ):
                str_assignment = '''
                const _v = {}
                '''

                str_assignment += f'''
                _v.target= '{callback_target}';
                _v.env = '{env}'
                '''
                js_cbk1 =  str_assignment + '''

                const data = new FormData(this.parentElement);
                const dataValue = Object.fromEntries(data.entries());
                const msg = {"cmd":"forward_to_vis","data":{"target": _v.target, "eid":_v.env, "event_type":"SubmitForm", event_data: dataValue}}
                console.log(msg);

                %s.sendSocketMessage(msg);
                console.log('injected')
                '''%rapp
                return js_inject_prep(js_cbk1)
                # return js_cbk1



            style = 'border:2px solid black;'

            v = ''
            k = 'EventDataType'
            v += f'''<input type='hidden' name="{k}" value="{msg.__dict__[k]}" />'''

            k = 'target_env'
            v += f'<input type="text" name="{k}" value="{msg.__dict__[k]}" style="{style}"></input>'

            v += '<br/>'

            k = 'model_session_config_1'
            v += add_textarea(k, toml.dumps(msg.__dict__[k]))
            v += '<br/>'

            k = 'model_session_config_2'
            v += add_textarea(k, toml.dumps(msg.__dict__[k]))
            v += '<br/>'

            # xx1 = toml.loads(default_model1)

            k = 'EventDataType'
            vd = 'MarkovComapreEventData'
            css = 'border: 0px solid black;'
            btns = ''
            for cbk in callback_list:
                btns += f'<button onclick="javascript:{get_js_cbk(cbk.name)}" style="height:40px; font-size:30px; ">{cbk.name}</button>'
                btns +='<br/>'
            win_out = vis.text(
            f'''
            <div class="visext-window" style="{css}">
                <form action="javascript:void(0);">
                {btns}
                <br/>
                {v}
                </form>
            </div>
            '''
            ,env=env,win=win)
            return win,win_out

        @staticmethod
        def make_callback( iface_recv, vis, env):
            '''
            Return a callback to be executed on receiving the message
            '''
            def _on_recv( event,  win='reply_win', vis=vis):
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
                logging.info(f'[callbackMsgRecv]{msg!r}')
                if isinstance( msg.event_data, MarkovComapreEventData):
                    # import pdb; pdb.set_trace()
                    env = msg.eid
                    hb = f'''
                    [{datetime.now().isoformat()}] Callback recved: {repr(msg)[:50]}
                    <u><b>{vis_html_jump_button(msg.event_data.target_env,'Click To see result:'+msg.event_data.target_env)}</b></u>

                    <div><textarea>{(msg.event_data.json())}</textarea></div>
                    <br/>
                    <br/>
                    Calculating....

                    '''
                    vis.text(hb, win=win, env=env)

                    import time
                    t0 = time.time()
                    iface_recv(msg.event_data, msg, vis)
                    # msg.event_data.on_recv(msg,vis)
                    hb += f'''
                    <br/>
                    <br/>

                    [{datetime.now().isoformat()}] Finished calculation! time:{time.time()-t0:.2f}s
                    '''

                    vis.text(hb, win=win, env=env)
                else:
                    raise NotImplementedError(f'Not Implemented for {msg.event_data!r}')
                    # print(msg)

            return _on_recv

    cls = MarkovComapreEventData
    cls._example_value =  cls(**json.loads(cls._example)['event_data'])




    class VisdomCallbackMessage(BaseModel):
        eid: str
        event_type: str
        target: str
        pane_data: Optional[PaneData]  ### intermediaste
        event_data: Union[EventData, MarkovComapreEventData] = Field(default={},discriminator='EventDataType')

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


    # import logging
    # from markov_lm.core_models import VisdomCallbackMessage, MarkovComapreEventData
    class InterfaceCompare(object):
        '''
        创建前端表单来接受socket请求
        '''
        # @staticmethod
        target_win = win = 'box1'
        target_env = env = 'testdev'
        # callback_target= f'{env}/{win}'
        callback_target= f'compare'
        # callback_target = 'compare train'.split()
        # @classmethod
        # def bind_visdom(self, vis):
            # return interface_bind(self, vis)
        @classmethod
        def bind_visdom(I, vis, env='testdev'):
            '''
            Needs to implement two
            '''

            '''
            Create a frontend window
              - This allows frontend to send data back to server
            '''
            I.init_vis_form(vis, env, I.__class__.__name__)

            '''
            register backend callback
               - this performs computation given the message
               - and renders to output
            '''
            for cbk in MarkovComapreEventData.callback_list:
                callback = MarkovComapreEventData.make_callback(cbk.on_recv, vis, env)
                vis.register_event_handler(callback, cbk.name)
            # vis.register_event_handler(callback, 'compare')
            return 0

        @classmethod
        def init_vis_form_default(self, vis, env=None, win=None):
            if env is None:
                env = self.env
            if win is None:
                win = self.win
            self.init_vis_form(vis, env, win)

        @classmethod
        def init_vis_form(self, vis, env, win):
            from markov_lm.util_base import toml_to_argv, dict_to_argv
            target_env = env
            target_win = win


            default_model1 = '''
            loglr = -4
            LOAD = "100_"
            batch_size = 100

            [model]
            embed_dim=60
            model_name = "DLM142"
            kernel_size = 3
            window_size = 1
            depth = 12
            p_null = 0.05
            '''
            k  = 'target_env'
            vd = 'compare0003'
            form_target_env = vd
            msg = MarkovComapreEventData(
                target_env = form_target_env,
                model_session_config_1=default_model1,
                model_session_config_2=default_model1)

            msg.vis_add_form_with_callback(vis, env, win)
