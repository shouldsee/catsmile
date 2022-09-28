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
