from transformers import AutoTokenizer, AutoModel
import torch
from collections import OrderedDict
import os

def fws(k,L,sep=' '):'''Formating: fill string k with white space to have length L'''; return (str(k)+sep*(L-len(k)))[:L]


def recur_detach(xdict,name,output,detach=True,sep='/'):
    '''
    Recursively detach the Tensor and store into xdict.

    '''
    if isinstance(output,tuple):
        for k,v in enumerate(output):
            recur_detach(xdict, name+sep+str(k), v, detach,sep)
    elif isinstance(output,torch.Tensor):
        if detach:
            output = output.detach()
        xdict[name] = output
    elif isinstance(output,dict):
        for k,v in output.items():
            recur_detach(xdict,name+sep+str(k),v, detach,sep)
    else:
        print(name,output.__class__)
        assert 0,'Unknown type %s'%output.__class__
        # output=output.detach()
    return output



class tbuf_cls(object):
    '''
    tbuf contains callback methods that
    can be hooked to construct a html
    table
    '''
    def __init__(s):
        s.tbody = ''
        s.tr = ''
        s.td = ''
        s.thead = ''
        s.is_head = 1
    def end_thead(s):
        s.thead = s.tbody
        s.tbody = ''
        s.is_head = 0
        return
    def add_row(tbuf,):
        tbuf.tbody += f'<tr>{tbuf.tr}</tr>'
        tbuf.tr=''
        return
    def add_elem(tbuf,v):
        if tbuf.is_head:
            tbuf.tr += '<th>%s</th>\n'%v
        else:
            tbuf.tr += '<td>%s</td>\n'%v




    def get_table(tbuf):
        html_style = '''


table thead tr th {
  /* you could also change td instead th depending your html code */
  background-color: #CCCCCC;
  position: sticky;
  z-index: 100;
  top: 0;
}

table tbody tr {
    background-color: #EEEEEE;
}
thead > :last-child th
{
    position: sticky;
    top: 30px; /* This is for all the the "th" elements in the second row, (in this casa is the last child element into the thead) */
}

thead > :first-child th
{
    position: sticky;
    top: 0px; /* This is for all the the "th" elements in the first child row */
}
            '''

        x = f'''
        <body style="background-color: #777777;">
            <style>
                {html_style}
            </style>

            <table border='1'>
                <thead>
                    {tbuf.thead}
                </thead>
                <tbody>
                    {tbuf.tbody}
                </tbody>
            </table>
        </body>
        '''
        return x



def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
    return grad

def jacobian(y, x):
    """Compute dy/dx = dy/dx @ grad_outputs;
    for grad_outputs in [1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]"""
    jac = torch.zeros(y.shape[0], x.shape[0])
    for i in range(y.shape[0]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1
        jac[i] = gradient(y, x, grad_outputs = grad_outputs)
    return jac

import itertools
from tqdm import tqdm
def _obs_jacobian():
    '''
    TOOOOOOOOO Slow!
    Needs: CONFIG_DETACH = 0
    Calculates: full jacobian between two layer
    '''
    out = []
    it = list(itertools.product(*[range(_x) for _x in y.shape]))
    for xind in tqdm(it):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[xind] = 1
        out.append(torch.autograd.grad(y,x,retain_graph=True,grad_outputs=grad_outputs)[0].reshape(-1))



def add_hook_for_output_tensor(model,tokenizer,CONFIG_EXPAND_LEVEL,CONFIG_DETACH, CONFIG_PRINT_CLASS):
    '''
    Adds to model a method  `model.get_tensors(sent,xdict)`
    which returns a dict of tensors outputted at each nn.Module with maximum depth of CONFIG_EXPAND_LEVEL.
    '''
    xdict = OrderedDict()

    def get_forward_hook(name,xdict=xdict):
        def hook(model, input, output):
            recur_detach(xdict,name,output,CONFIG_DETACH,'/')
            # xdict[name] = output
        return hook

    '注入灵魂'
    for k,v in model.named_modules():
        if k.count('.')<=CONFIG_EXPAND_LEVEL:
            if CONFIG_PRINT_CLASS:
                print(fws(k,60) ,v.__class__);
            v.register_forward_hook(get_forward_hook(k))

    def get_all_tensors(sent):
        '''
        Convert a sentence to a dict of internal representations
        '''
        xdict.clear()
        inputs = tokenizer(sent, return_tensors="pt")
        outputs = model(**inputs)
        return xdict.copy()
    model.get_all_tensors = get_all_tensors

    return model



def main():
    PKL = __file__+'.temp.pkl'
    if os.path.exists(PKL):
        tokenizer,model = torch.load(PKL)
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        torch.save((tokenizer,model),PKL)
        '''
        -rw-r--r--  1 root root 1.3K 4月  17 17:51 c9007.py
        -rw-r--r--  1 root root 419M 4月  17 17:51 c9007.py.temp.pkl
        '''

    '''
    Extra methods for tokenizers
    '''
    def get_tok(sent,tokenizer=tokenizer): return tokenizer.convert_ids_to_tokens(tokenizer(sent,return_tensors='pt')['input_ids'][0])
    def pad_sentence(sent1,sent2,tokenizer=tokenizer):
        #### fill sentence to the same length
        ldiff = tokenizer.f_get_tok(sent1).__len__()- tokenizer.f_get_tok(sent2).__len__()
        if ldiff<0:
            sent1,sent2=sent2,sent1
        else:
            for _ in range(abs(ldiff)):
                sent2 = sent2+' blah'
        return sent1,sent2

    tokenizer.f_get_tok = get_tok
    tokenizer.f_pad_sentence = pad_sentence

    disp = BaseDisplay()
    main_task1(model,tokenizer,disp)
    # main_task2(model,tokenizer,disp)


class BaseDisplay(object):
    # def __init__(self)
    c1 = 30
    c2 = 20
    c3p= 5
    tbuf = tbuf_cls()
    buf = ['']

    def xa(self,v):
        '''
        table element callback (TEC)
        '''
        self.buf[0]+=v
        self.tbuf.add_elem(v)
    def xf(self,):
        '''
        end-of-row callback(EORC)
        '''
        print(self.buf[0])
        self.buf[0]=''
        self.tbuf.add_row()



def main_task2(model,tokenizer,disp):
    ## Init output buffer
    c1 = disp.c1
    c2 = disp.c2
    c3p= disp.c3p
    tbuf = disp.tbuf
    buf = disp.buf
    xa = disp.xa
    xf = disp.xf

    get_tok = tokenizer.f_get_tok


    ### Add runtime hooks


    CONFIG_EXPAND_LEVEL = 3
    CONFIG_DETACH = 1


    def add_hook_to_inject_noise(model,tokenizer,CONFIG_EXPAND_LEVEL,CONFIG_DETACH, CONFIG_PRINT_CLASS):
        '''
        Objective is to calculate a position-wise correlation matrix.
        Adds to model a method  `model.get_tensors(sent,xdict)`
        which returns a dict of tensors outputted at each nn.Module with maximum depth of CONFIG_EXPAND_LEVEL.
        '''
        xdict = OrderedDict()

        def get_forward_hook(name,xdict=xdict):
            def hook(model, input, output):
                # if naem
                recur_detach(xdict,name,output,CONFIG_DETACH,'/')
                # xdict[name] = output
            return hook

        for k,v in model.named_modules():
            if k.count('.')<=CONFIG_EXPAND_LEVEL:
                if CONFIG_PRINT_CLASS:
                    print(fws(k,60) ,v.__class__);
                v.register_forward_hook(get_forward_hook(k))

        def get_all_tensors(sent):
            '''
            Convert a sentence to a dict of internal representations
            '''
            xdict.clear()
            inputs = tokenizer(sent, return_tensors="pt")
            outputs = model(**inputs)
            return xdict.copy()
        model.get_all_tensors = get_all_tensors

        return model

    model = add_hook_for_output_tensor(model,tokenizer,CONFIG_EXPAND_LEVEL,CONFIG_DETACH,1)




    sent1 = "I've come to the Google headquarter in Los Angeles on the west coast"
    sent2 = "I've come to the Google headquarter in Los Angeles on the china coast"


    sent1 = "I've come to the Google headquarter in LA, USA"
    sent2 = "I've come to the Google headquarter in Shanghai, China"

    sent1 = "I've come to the Google head quarter in LA, USA"
    sent2 = "I've come to the Microsoft head quarter in LA, USA"

    sent1 = "I've come to the Google head quarter in USA, which is a 12-floor ai startup"
    # sent1 = "I've come to the beautiful head quarter in LA, which is a 12-floor structure"
    # sent2 = "I've come to the ugly head quarter in LA, which is a 12-floor structure"
    # sent2 = "I've come to the Microsoft head quarter in china, which is a 12-floor ai startup"
    # sent2 = "I've come to the Microsoft head quarter in USA, which is a 12-floor ai startup"
    sent2 = "I've come to the Google head quarter in china, which is a 12-floor ai startup"
#    sent2 = "I've come to the China head quarter in LA, which is a 12-floor structure"


    sent1,sent2 = tokenizer.f_pad_sentence(sent1,sent2)

    x1 = model.get_all_tensors(sent1)
    x2 = model.get_all_tensors(sent2)


    [xa(c1*' '), xa(c2*' ')] + [ xa(fws(x,c3p)) for x in get_tok(sent1)] + [xf()]
    [xa(c1*' '), xa(c2*' ')] + [ xa(fws(x,c3p)) for x in get_tok(sent2)] + [xf()]
    tbuf.end_thead()

    CONFIG_INT_HIDE = 100
    for k in x1:
        v1 = x1[k]
        v2 = x2[k]
        xa(fws(k,c1))
        xa(fws(tuple(v1.shape),c2))
        if v1.shape.__len__()==3:
            '''
            用L1范数计算每个位置的产生的响应偏移, 进行适当的标准化,并取整数,并隐藏阈值以下的数值
            '''
            xb = (v2-v1).abs()
    #        xb = xb.sum(-1)
            '求平均消去嵌入维度'
            xb = xb.mean(-1)
            # xb = xb / xb.mean(-1,keepdims=True); xb=xb*100
            xb = xb*1000
            xb = xb.numpy().astype(int)[0]
            for xbb in xb:
                v = ''
                if xbb > CONFIG_INT_HIDE: v = xbb.__str__()
                xa(fws(v,c3p))
        else:
            xa('%.0f'%(v2-v1).abs().sum().item())
        xf()

        if k.startswith('encoder.layer.') and k.endswith('/0') and k.count('.')==2:
            xa('-'*(c1+c2))
            xf()

    xbuf = tbuf.get_table()
    with open(__file__+'.html','wb') as f:
        f.write(xbuf.encode())

def main_task1(model,tokenizer,disp):
    ## Init output buffer
    c1 = disp.c1
    c2 = disp.c2
    c3p= disp.c3p
    tbuf = disp.tbuf
    buf = disp.buf
    xa = disp.xa
    xf = disp.xf

    get_tok = tokenizer.f_get_tok


    ### Add runtime hooks


    CONFIG_EXPAND_LEVEL = 3
    CONFIG_DETACH = 1
    model = add_hook_for_output_tensor(model,tokenizer,CONFIG_EXPAND_LEVEL,CONFIG_DETACH,1)




    sent1 = "I've come to the Google headquarter in Los Angeles on the west coast"
    sent2 = "I've come to the Google headquarter in Los Angeles on the china coast"


    sent1 = "I've come to the Google headquarter in LA, USA"
    sent2 = "I've come to the Google headquarter in Shanghai, China"

    sent1 = "I've come to the Google head quarter in LA, USA"
    sent2 = "I've come to the Microsoft head quarter in LA, USA"

    sent1 = "I've come to the Google head quarter in USA, which is a 12-floor ai startup"
    # sent1 = "I've come to the beautiful head quarter in LA, which is a 12-floor structure"
    # sent2 = "I've come to the ugly head quarter in LA, which is a 12-floor structure"
    # sent2 = "I've come to the Microsoft head quarter in china, which is a 12-floor ai startup"
    # sent2 = "I've come to the Microsoft head quarter in USA, which is a 12-floor ai startup"
    sent2 = "I've come to the Google head quarter in china, which is a 12-floor ai startup"
#    sent2 = "I've come to the China head quarter in LA, which is a 12-floor structure"


    sent1,sent2 = tokenizer.f_pad_sentence(sent1,sent2)

    x1 = model.get_all_tensors(sent1)
    x2 = model.get_all_tensors(sent2)


    [xa(c1*' '), xa(c2*' ')] + [ xa(fws(x,c3p)) for x in get_tok(sent1)] + [xf()]
    [xa(c1*' '), xa(c2*' ')] + [ xa(fws(x,c3p)) for x in get_tok(sent2)] + [xf()]
    tbuf.end_thead()

    CONFIG_INT_HIDE = 100
    for k in x1:
        v1 = x1[k]
        v2 = x2[k]
        xa(fws(k,c1))
        xa(fws(tuple(v1.shape),c2))
        if v1.shape.__len__()==3:
            '''
            用L1范数计算每个位置的产生的响应偏移, 进行适当的标准化,并取整数,并隐藏阈值以下的数值
            '''
            xb = (v2-v1).abs()
    #        xb = xb.sum(-1)
            '求平均消去嵌入维度'
            xb = xb.mean(-1)
            # xb = xb / xb.mean(-1,keepdims=True); xb=xb*100
            xb = xb*1000
            xb = xb.numpy().astype(int)[0]
            for xbb in xb:
                v = ''
                if xbb > CONFIG_INT_HIDE: v = xbb.__str__()
                xa(fws(v,c3p))
        else:
            xa('%.0f'%(v2-v1).abs().sum().item())
        xf()

        if k.startswith('encoder.layer.') and k.endswith('/0') and k.count('.')==2:
            xa('-'*(c1+c2))
            xf()

    xbuf = tbuf.get_table()
    with open(__file__+'.html','wb') as f:
        f.write(xbuf.encode())

if __name__ == '__main__':
    main()
