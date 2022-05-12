from collections import OrderedDict
import os
import torch
def fws(k,L,sep=' '):'''Formating: fill string k with white space to have length L'''; k=str(k); return ((k)+sep*(L-len(k)))[:L]


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
