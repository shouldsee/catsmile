'''
Custom pydantic DataModels
'''
import logging
import json,toml

from datetime import datetime
from typing import List, Optional, Union,Literal
from pydantic import BaseModel,validator
from pydantic import Field

from markov_lm.util_base import dset
from markov_lm.util_base import toml_to_argv, dict_to_argv
from markov_lm.util_base import js_inject_prep, vis_html_jump_button, add_textarea, rapp


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



def interface_bind(I, vis, env='testdev'):
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
    callback = I.make_callback(vis, env)
    vis.register_event_handler(callback, I.callback_target)
    return 0

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

    class MarkovComapreEventData(BaseModel):
        EventDataType: Literal['MarkovComapreEventData']='MarkovComapreEventData'
        # EventDataType: 'MarkovComapreEventData'
        model_session_config_1: dict
        model_session_config_2: dict
        target_env: str
        origin_env: str='testdev'

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
        def on_recv(self, pmsg, vis):
            # _dp = logging.debug
            _dp = print
            _dp(f'[before_on_recv]{self}')
            _dp(pmsg.json(indent=2))
            self._on_recv(pmsg, vis)
            _dp(f'[after_on_recv]{self}')
        def _on_recv(self, pmsg, vis):
            # pcli_argv_1 =
            if 1:
                from markov_lm.Model_NLM import U
                from markov_lm.util_base import toml_to_argv, dict_to_argv
                import pandas as pd
                import numpy as np
                import torch
                from markov_lm.nlp.train import conf_main_loop,argv_to_conf

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


                conf1 = conf_main_loop(argv_to_conf(dict_to_argv(self.model_session_config_1)),'load')
                conf2 = conf_main_loop(argv_to_conf(dict_to_argv(self.model_session_config_2)),'load')

                dataset = conf1.dataset
                dataloader = conf1.dataloader
                # vis = conf1.vis


                dataset.test()
                item = next(iter(dataloader))
                # rng = torch.get_rng_state().numpy()

                generator = torch.Generator(device=conf1.device)
                # rng = generator.get_state()
                # with torch.no_grad():
                seed = generator.seed()
                loss1,loss2 = map((lambda x:x.model._loss(item,'loss', generator= generator.manual_seed(seed) )) ,[conf1,conf2])
                # loss1,loss2 = map((lambda x:x.model._loss(item,'loss', generator )) ,[conf1,conf2])

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

                key =f'test_loss_diff_histogram'
                vis.histogram( mat.T[1] - mat.T[0], env=env, win = key,opts=dict(title=key+f' ts:{datetime.now().isoformat()}'))


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

            # callback_interface=
            ci = InterfaceCompare
            self.vis_add_form_with_callback(vis, self.target_env, ci.__name__, ci.callback_target)


        def vis_add_form_with_callback(self, vis, env, win, callback_target):
            '''
            Create a form that would sends this message back
            to visdom to trigger a backend callback
            '''

            msg = self
            str_assignment = '''
            const _v = {}
            '''
            str_assignment += f'''
            _v.target= '{callback_target}';
            _v.env = '{env}'
            '''
            injected_js =  str_assignment + '''

            const data = new FormData(this.parentElement);
            const dataValue = Object.fromEntries(data.entries());
            const msg = {"cmd":"forward_to_vis","data":{"target": _v.target, "eid":_v.env, "event_type":"SubmitForm", event_data: dataValue}}
            console.log(msg);

            %s.sendSocketMessage(msg);
            console.log('injected')
            '''%rapp
            injected_js = js_inject_prep(injected_js)



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
            win_out = vis.text(
            f'''
            <div class="visext-window" style="{css}">
                <form action="javascript:void(0);">
                <button onclick="javascript:{injected_js}">submit</button>
                <br/>
                {v}
                </form>
            </div>
            '''
            ,env=env,win=win)
            return win,win_out

    cls = MarkovComapreEventData
    cls._example_value =  cls(**json.loads(cls._example)['event_data'])



    # import logging
    # from markov_lm.core_models import VisdomCallbackMessage, MarkovComapreEventData
    class InterfaceCompare(object):
        '''
        创建前端表单来接受socket请求
        '''
        # @staticmethod
        target_win = win = 'box1'
        target_env = env = 'testdev'
        callback_target= f'{env}/{win}'
        @classmethod
        def bind_visdom(self, vis):
            return interface_bind(self, vis)
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

            msg.vis_add_form_with_callback(vis, env, win, self.callback_target)





        @classmethod
        def make_callback(self, vis, env):
            def _callback(event,  win='reply_win', vis=vis):
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
                    msg.event_data.on_recv(msg,vis)
                    hb += f'''
        <br/>
        <br/>

        [{datetime.now().isoformat()}] Finished calculation! time:{time.time()-t0:.2f}s
                    '''

                    vis.text(hb, win=win, env=env)
                else:
                    print(msg)

                # BaseMessage()
            return _callback


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
