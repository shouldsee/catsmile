import toml  ### very lightweight package

def dset(x,key=None):
    '''
    Simple decorator that registers items into dict
    '''
    def wrapper(v,key=key):
        if key is None:
            key = v.__name__
        x[key] = v
        return v
    return wrapper




'''
Object level methods
'''
def register_object_method(self):
    '''
    将某个动态方法挂载到一个object上
    '''
    def decorator(method):
        def wrapper(*args, **kwargs):
            return method(self,*args, **kwargs)
        setattr(self, method.__name__, wrapper)
        return wrapper
    return decorator

if 1:
    '''
    # js console hacker
    在button的onclick方法里注入socket通信方法
    // rapp means (ReactAPP)
    '''

    rapp = 'app._reactRootContainer._internalRoot.current.child.stateNode'
    js_head = 'rapp = app._reactRootContainer._internalRoot.current.child.stateNode; '


    def js_inject_prep(injected_js):
        injected_js = injected_js.replace('\"','\'')
        injected_js = ';'.join(injected_js.splitlines())
        return injected_js

    def vis_html_jump_button(_target_env, text=None,js_head=js_head):
        if text is None:
            text = _target_env
        return f'''<button onclick="javascript:{js_inject_prep(js_head+'rapp.selectEnv([%r])'%_target_env)};">{text}</button>'''

    style_of_input_box = "border:2px solid black; width:250px;height:150px;"
    def add_textarea(k,default, style = style_of_input_box):
        return f'''
        <label>{k}</label>
        <br/>
        <textarea name="{k}" style="{style}">{default}</textarea>
        '''

def dict_to_argv(x, prefix='', out=None):
    '''
    serialise a dict to cmdline argv
    '''
    if out is None:
        out = []
    for k,v in x.items():
        if prefix:
            prefixx = f'{prefix}.{k}'
        else:
            prefixx = k
        if isinstance(v,(str,int,float)):
            v = str(v)
            out.append(f'--{prefixx}')
            out.append(v)
        elif isinstance(v,dict):
            dict_to_argv(v, prefixx,out)
        else:
            raise NotImplementedError(f'{type(v)},{repr(v)}')
    return out



def dict_to_argv(x, prefix='', out=None):
    '''
    serialise a dict to cmdline argv
    '''
    if out is None:
        out = []
    for k,v in x.items():
        if prefix:
            prefixx = f'{prefix}.{k}'
        else:
            prefixx = k
        if isinstance(v,(str,int,float)):
            v = str(v)
            out.append(f'--{prefixx}')
            out.append(v)
        elif isinstance(v,dict):
            dict_to_argv(v, prefixx,out)
        else:
            raise NotImplementedError(f'{type(v)},{repr(v)}')
    return out

def toml_to_argv(x,k=None):
    xx = toml.loads(x)
    if k is not None:
        xx=xx[k]
    return dict_to_argv(xx)



if __name__ == '__main__':
    pass
