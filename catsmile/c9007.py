from transformers import AutoTokenizer, AutoModel
import torch
from collections import OrderedDict
import os

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



def main():
    CUDA = 1
    PKL = __file__+'.temp.pkl'
    device = torch.device('cuda:0' if CUDA else 'cpu')
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
    model =model.to(device)
    # tokenizer=tokenizer.to(device)

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
    # main_task1(model,tokenizer,disp)
    # main_task2(model,tokenizer,disp)
    main_task4(model,tokenizer,disp)
    # main_task3(model,tokenizer,disp)


class BaseDisplay(object):
    def __init__(s):
        s.c1 = 30
        s.c2 = 20
        s.c3p= 5
        s.tbuf = tbuf_cls()
        s.buf = ['']

    def xa(self,v):
        '''
        table element callback (TEC)
        '''
        self.buf[0]+=str(v)
        self.tbuf.add_elem(v)
    def xf(self,):
        '''
        end-of-row callback(EORC)
        '''
        print(self.buf[0])
        self.buf[0]=''
        self.tbuf.add_row()

from pprint import pprint

def main_task3(model,tokenizer,disp):
    '''
    目标: 观察 Q^T K 和 V 正交性,也就是考察 $||Q^T K V ||_F$
    '''
    xp = model.named_parameters()
    xp = dict(xp)
    n_att = model.config.num_attention_heads
    att_dim = model.config.hidden_size // n_att
    hidden_dim = model.config.hidden_size
    for id in range(12):
        # att_dim = model.config.n_att
        f1,f2,f3,f4,f5,f6=0,0,0,0,0,0
        for ia in range(12):
            xq = xp[f'encoder.layer.{id}.attention.self.query.weight'].T.view((-1,n_att,att_dim))[:,ia]
            xk = xp[f'encoder.layer.{id}.attention.self.key.weight'].T.view((-1,n_att,att_dim))[:,ia]
            xv = xp[f'encoder.layer.{id}.attention.self.value.weight'].T.view((-1,n_att,att_dim))[:,ia]

            # xq = xp[f'encoder.layer.{id}.attention.self.query.weight'].T.view((-1,att_dim,n_att))[:,:,ia]
            # xk = xp[f'encoder.layer.{id}.attention.self.key.weight'].T.view((-1,att_dim,n_att))[:,:,ia]
            # xv = xp[f'encoder.layer.{id}.attention.self.value.weight'].T.view((-1,att_dim,n_att))[:,:,ia]

            # xqb = xp['encoder.layer.11.attention.self.query.bias']
            # xkb = xp['encoder.layer.11.attention.self.key.bias']
            # xvb = xp['encoder.layer.11.attention.self.value.bias']
            # ([print(x.square().sum().item()) for x in [xqb,xkb,xvb]])

            xker = xq.matmul(xk.T)
            # xpd = xker.matmul( xv)

            # xker = xker / xker.std(dim=1,keepdims=True)
            # xv = xv/xv.std(0,keepdims=True)
            #
            # (xker[:20,:10]*100).int()
            # (xv[:20,:10]*100).int()

            # f5 = ( xker* xker.T).square().sum()
            # f6 += ( xk).square().sum()
            # f5 += ( xq).square().sum()
            # # f4 += xker.square().sum()
            # f4 += (xker.matmul( xv)).square().sum()
            # f3 += (xker.T.matmul( xv)).square().sum()
            # # f3 = (xker.matmul( xv)).square().sum()
            # # f3 = (xker.T.matmul( xv)).square().sum()
            # f1 += (xker).square().sum()
            # f2 += (xv).square().sum()

            f6 += ( xk).square().mean()
            f5 += ( xq).square().mean()
            # f4 += xker.square().sum()
            f4 += (xker.matmul( xv)).square().mean()
            f3 += (xker.T.matmul( xv)).square().mean()
            # f3 = (xker.matmul( xv)).square().sum()
            # f3 = (xker.T.matmul( xv)).square().sum()
            f1 += (xker).square().mean()
            f2 += (xv).square().mean()

        y=(xker.matmul(xv)*1000); y = y.int()[:10,:10];y
        y=(xv*1000); y = y.int()[:10,:10];y

        ([print(x.item()*100000) for x in [f1,f2,f3,f4,f5,f6]])
        print()

    f1,f2,f3,f4,f5,f6=0,0,0,0,0,0
    xq = torch.normal(0,0.04,(hidden_dim,att_dim))
    xk = torch.normal(0,0.04,(hidden_dim,att_dim))
    xv = torch.normal(0,0.04,(hidden_dim,att_dim))
    xker = xq.matmul(xk.T)
    # f6 += ( xk).square().sum()
    # f5 += ( xq).square().sum()
    # # f4 += xker.square().sum()
    # f4 += (xker.matmul( xv)).square().sum()
    # f3 += (xker.T.matmul( xv)).square().sum()
    # # f3 = (xker.matmul( xv)).square().sum()
    # # f3 = (xker.T.matmul( xv)).square().sum()
    # f1 += (xker).square().sum()
    # f2 += (xv).square().sum()

    f6 += ( xk).square().mean()
    f5 += ( xq).square().mean()
    # f4 += xker.square().sum()
    f4 += (xker.matmul( xv)).square().mean()
    f3 += (xker.T.matmul( xv)).square().mean()
    # f3 = (xker.matmul( xv)).square().sum()
    # f3 = (xker.T.matmul( xv)).square().sum()
    f1 += (xker).square().mean()
    f2 += (xv).square().mean()


    ([print(x.item()*10000) for x in [f1,f2,f3,f4,f5,f6]])
    print()


import hashlib
import shutil
from collections import defaultdict
# class _Tree():
#     @staticmethod
#     def tree(): return defaultdict(_Tree.tree)
# tree = _Tree.tree

def tree(): return defaultdict(tree)
# _Tree.tree)

if 0:
    x = tree()
    print(x[1][2][3])
    torch.save([x],'test.pkl')
    x = torch.load('test.pkl')
    print(x)

# class Tree(defaultdict):

#     def __getstate__(self):
#         pass
#     def __setstate__(self):
#         pass



def tree(): return defaultdict(tree)



def add_hook_to_inject_noise(
    xdict,
    model,
    tokenizer,
    xl,
    jreq,
    sigma,
    n_sample=100,mean=0.,

CONFIG_EXPAND_LEVEL=3,CONFIG_DETACH=1, CONFIG_PRINT_CLASS=0):
    '''
    Objective is to calculate a position-wise correlation matrix.
    '''
    # xdict = OrderedDict()

    # sigma = 0.02
    # mean = 0.
    # n_sample = 100
    handles = []
    def is_monitor_layer(k):
        return k == 'embeddings.dropout' or k.startswith('encoder.layer.') and k.endswith('output') and k.count('.')==3


    i = [-1 ]
    joutput = [None]
    def get_forward_hook(name,xdict=xdict):
        def hook(_model, input, output):
            k = name
            # if name == noise_injection_layer:

            '''
            如果到达第j层,就把j存起来
            '''
            if is_monitor_layer(k):
                '''
                如果不是第一层,
                就储存jacobian fnorm 待用
                '''
                i[0] += 1
                k = i[0]
                if k<jreq:
                    return


                if k==jreq:
                    ### no harvest
                    ### store
                    joutput[0] = output
                    ### perturb and forward

                elif k>jreq:
                    msdy = (output[1:] - output[:1]).square().sum(-1).mean(0)
                    msdy = msdy / sigma**2
                    # import pdb; pdb.set_trace()
                    xdict[jreq][k][xl] = msdy.detach()
                    ### harvest
                    ### replace
                    output = joutput[0]
                #perturb and forward


                xd = torch.normal(mean,sigma,(n_sample,1,output.shape[-1]))
                loc = (torch.arange(output.shape[1]) == xl)[None,:,None]
                loc2= (torch.arange(xd.shape[0]) > 0) [:,None,None]
                output = (xd*loc*loc2).to(model.device) + output

                # xdict['v1'] = output
                # xdict[k] = output

            return output
        return hook

    for k,v in model.named_modules():
        if k.count('.')<=CONFIG_EXPAND_LEVEL:
            if CONFIG_PRINT_CLASS:
                print(fws(k,60) ,v.__class__);
            h = v.register_forward_hook(get_forward_hook(k))
            handles.append(h)

    def get_all_tensors(sent):
        '''
        Convert a sentence to a dict of internal representations
        '''
        xdict.clear()
        inputs = tokenizer(sent, return_tensors="pt")
        ### The slowness is inherent to the model
        # inputs['input_ids'] = inputs['input_ids'].repeat((10,1))
        outputs = model(**inputs)
        return

    model.get_all_tensors = get_all_tensors
    model.remove_hooks = lambda handles= handles: [h.remove() for h in handles]
    return model

def main_task4(model,tokenizer,disp):
    '''
    目标: 观察计算图是否可以近似认为驻定.
    通过计算Jocobian的扰动的F范数来确定.
    12层的BERT有12^2种替换方式定义,和12个jacobian,因此一共12^3计算方式.
    定义:
      $v[j][k][\text{'fnorm'}] = ||J_k(v_j)||_F^2 - ||J_k(v_i)||_F^2 $
      $v[j][k][x][y] = ||J_k(v_j)[x][y]||_F^2 $

      注意到 i和j是交换的,所以仅计算i<j即可
    也就是第k层的J矩阵在面对i隐含层和j隐含层的差矩阵时的Jacobian的差异.

    ## 思路:
      - 每次固定噪声注入位置 x
      - 考虑把j替换到所有k>=j
      - 然后每次在到达下一层位置时,收获响应函数用于估计 $J_k(v_j)[x][y]$ , 重新注入噪声
      -



    '''

    v = tree()

    c1 = disp.c1
    c2 = disp.c2
    c3p= disp.c3p
    tbuf = disp.tbuf
    buf = disp.buf
    xa = disp.xa
    xf = disp.xf
    get_tok = tokenizer.f_get_tok




    # sent1 = ''
    # sent1 = "and again we've come to the super huge Google head quarter in USA, which is a 12-floor ai startup"
    sent1 = "Beijing is the super duper big capital of China"
    xdict_list = defaultdict(lambda:[])
    tok1 = get_tok(sent1)
    sigma = 0.01
    n_sample =100
    # noise_injection_layer = 'embeddings.dropout'
    # noise_injection_layer = 'encoder.layer.6.output'
    md5 = hashlib.md5(sent1.encode('utf-8')).hexdigest()

    '''
    '''

    PKL = f'{__file__}.{md5}.{sigma}.{n_sample}.pkl'

    if os.path.exists(PKL):
        xdict = torch.load(PKL)
    else:
        xdict = tree()
        it = list(itertools.product( range(tok1.__len__()), range(12)))
        for i,j in tqdm(it) :
            # for j in range(12):
                model = add_hook_to_inject_noise(xdict, model,tokenizer, i, j, sigma, n_sample)
                inputs = tokenizer(sent1, return_tensors="pt").input_ids.to(model.device)
                outputs = model(inputs)
                model.remove_hooks()
                # break

    # xdict[0][1][0]

        torch.save(xdict,PKL+'.temp')
        shutil.move(PKL+'.temp',PKL)
    CONFIG_INT_HIDE = 50
    xd = xdict
    for j,xdd in xd.items():
        for k,xddd in xdd.items():
            [xa(c3p*' ') for _ in range(4)] + [ xa(fws(x,c3p)) for x in get_tok(sent1)] + [xf()]
            for x,xdddd in xddd.items():
                # if j==2 and k==3:
                if j+1==k:
                # if k==8:
                    xa(fws(j,c3p))
                    xa(fws(k,c3p))
                    xa(fws(x,c3p))
                    xa(fws(tok1[x],c3p))

                    # for v in (xdddd*100000).int().cpu().numpy():
                    for v in (xdddd*10).int().cpu().numpy():
                        if v <CONFIG_INT_HIDE:
                            v = ''
                        xa(fws(v,c3p))
                    xf()
    with open(__file__+'.task4.html','w') as f:
        f.write(tbuf.get_table())
        # body())

    import pdb; pdb.set_trace()

def main_task2(model,tokenizer,disp):
    '''
    目标: 在每个位置注入噪声,观察不同层层的响应函数.
    '''
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


    # CONFIG_EXPAND_LEVEL = 3
    # CONFIG_DETACH = 1


    def add_hook_to_inject_noise(model,tokenizer,
    noise_injection_layer,
    idx,sigma,n_sample,mean=0.,

    CONFIG_EXPAND_LEVEL=3,CONFIG_DETACH=1, CONFIG_PRINT_CLASS=0):
        '''
        Objective is to calculate a position-wise correlation matrix.
        '''
        xdict = OrderedDict()

        # sigma = 0.02
        # mean = 0.
        # n_sample = 100
        handles = []

        def get_forward_hook(name,xdict=xdict):
            def hook(_model, input, output):
                k = name
                if name == noise_injection_layer:
                    # idx = 11
                    xd = torch.normal(mean,sigma,(n_sample,1,output.shape[-1]))
                    loc = (torch.arange(output.shape[1]) == idx)[None,:,None]
                    loc2= (torch.arange(xd.shape[0]) > 0) [:,None,None]
                    output = (xd*loc*loc2).to(model.device) + output

                    xdict[name] = output.detach()
                    # xdict['v1'] = output

                if k.startswith('encoder.layer.') and k.endswith('output') and k.count('.')==3:
                # if name == 'encoder.layer.11.output':

                    xdict[name] = output.detach()
                    # xdict[k] = output

                return output

            return hook

        for k,v in model.named_modules():
            if k.count('.')<=CONFIG_EXPAND_LEVEL:
                if CONFIG_PRINT_CLASS:
                    print(fws(k,60) ,v.__class__);
                h = v.register_forward_hook(get_forward_hook(k))
                handles.append(h)

        def get_all_tensors(sent):
            '''
            Convert a sentence to a dict of internal representations
            '''
            xdict.clear()
            inputs = tokenizer(sent, return_tensors="pt")
            inputs['input_ids']=inputs['input_ids'].to(model.device)
            ### The slowness is inherent to the model
            # inputs['input_ids'] = inputs['input_ids'].repeat((10,1))
            outputs = model(input_ids=inputs['input_ids'])
            return xdict.copy()

        model.get_all_tensors = get_all_tensors
        model.remove_hooks = lambda handles= handles: [h.remove() for h in handles]
        return model





    sent1 = "I've come to the Google headquarter in Los Angeles on the west coast"
    sent2 = "I've come to the Google headquarter in Los Angeles on the china coast"


    sent1 = "I've come to the Google headquarter in LA, USA"
    sent2 = "I've come to the Google headquarter in Shanghai, China"

    sent1 = "I've come to the Google head quarter in LA, USA"
    sent2 = "I've come to the Microsoft head quarter in LA, USA"

    sent1 = "add again and I've come to the super huge Google head quarter in USA, which is a 12-floor ai startup"
    # sent1 = "I've come to the beautiful head quarter in LA, which is a 12-floor structure"
    # sent2 = "I've come to the ugly head quarter in LA, which is a 12-floor structure"
    # sent2 = "I've come to the Microsoft head quarter in china, which is a 12-floor ai startup"
    # sent2 = "I've come to the Microsoft head quarter in USA, which is a 12-floor ai startup"
    sent2 = "I've come to the Google head quarter in china, which is a 12-floor ai startup"
#    sent2 = "I've come to the China head quarter in LA, which is a 12-floor structure"


     # CONFIG_EXPAND_LEVEL,CONFIG_DETACH,1)
    sent1,sent2 = tokenizer.f_pad_sentence(sent1,sent2)
    sent1 = 'Beijing is the second biggest capital of China'


    def parse_single_perturb(x1,tokenizer=tokenizer,model=model,tbuf=tbuf):
        [xa(c1*' '), xa(c2*' ')] + [ xa(fws(i,c3p)) for i,x in enumerate(get_tok(sent1))] + [xf()]
        [xa(c1*' '), xa(c2*' ')] + [ xa(fws(x,c3p)) for x in get_tok(sent1)] + [xf()]
        # [xa(c1*' '), xa(c2*' ')] + [ xa(fws(x,c3p)) for x in get_tok(sent2)] + [xf()]
        tbuf.end_thead()

        CONFIG_INT_HIDE = 10
        for k in x1:

            v = x1[k]
            v = v.detach()

            if v.shape.__len__()==3:
                '''
                用L2范数计算每个位置的产生的响应偏移, 进行适当的标准化,并取整数,并隐藏阈值以下的数值
                '''
                '求平均消去嵌入维度'
                dv2 = (v[1:] -v[0:1]).square().sum(-1).mean(0,keepdims=True)
                dv2 = dv2*1000


                # xb = (v2-v1).abs()
        #        xb = xb.sum(-1)
                # xb = xb.mean(-1)
                # xb = xb / xb.mean(-1,keepdims=True); xb=xb*100
                # xb = xb*1000
                xb = dv2.cpu().numpy().astype(int)[0]

                xa(fws(k,c1))
                xa(fws(tuple(v.shape),c2))
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


    def parse_multiple_perturb(xdict_list,sent1,CONFIG_INT_HIDE = 10,tok=tokenizer,model=model,disp=disp):
        get_tok = tok.f_get_tok

        tbuf = disp.tbuf
        buf = disp.buf
        xa = disp.xa
        xf = disp.xf
        tok1 = get_tok(sent1)

        [xa(c1*' '), xa(c2*' ')] +[xa(c3p*' ')]+ [ xa(fws(i, disp.c3p)) for i,x in enumerate(get_tok(sent1))] + [xf()]
        [xa(c1*' '), xa(c2*' ')] +[xa(c3p*' ')]+ [ xa(fws(x, disp.c3p)) for x in get_tok(sent1)] + [xf()]
        # [xa(c1*' '), xa(c2*' ')] + [ xa(fws(x,c3p)) for x in get_tok(sent2)] + [xf()]
        tbuf.end_thead()


        for k in xdict_list:
            [xa(c1*' '), xa(c2*' ')] +[xa(c3p*' ')]+ [ xa(fws(i, disp.c3p)) for i,x in enumerate(get_tok(sent1))] + [xf()]
            [xa(c1*' '), xa(c2*' ')] +[xa(c3p*' ')]+ [ xa(fws(x, disp.c3p)) for x in get_tok(sent1)] + [xf()]

            vlist = xdict_list[k]
            for i,v in enumerate(vlist):
                if hasattr(v,'detach'): v = v.detach()

                if v.shape.__len__()==3:
                    '''
                    用L2范数计算每个位置的产生的响应偏移, 进行适当的标准化,并取整数,并隐藏阈值以下的数值
                    '''
                    '求平均消去嵌入维度'
                    dv2 = (v[1:] -v[0:1]).square().sum(-1).mean(0,keepdims=True)
                    dv2 = dv2/sigma**2 *10


                    # xb = (v2-v1).abs()
            #        xb = xb.sum(-1)
                    # xb = xb.mean(-1)
                    # xb = xb / xb.mean(-1,keepdims=True); xb=xb*100
                    # xb = xb*1000
                    xb = dv2.cpu().numpy().astype(int)[0]
                    xa(fws(k,c1))
                    xa(fws(tuple(v.shape),c2))
                    xa(fws(tok1[i],c3p))
                    for xbb in xb:
                        v = ''
                        if xbb > CONFIG_INT_HIDE: v = xbb.__str__()
                        xa(fws(v,c3p))
                xf()
            xf()
            # break


    from collections import defaultdict
    import hashlib
    import shutil
    xdict_list = defaultdict(lambda:[])
    tok1 = get_tok(sent1)
    sigma = 0.01
    n_sample = 100
    noise_injection_layer = 'embeddings.dropout'
    # noise_injection_layer = 'encoder.layer.6.output'
    md5 = hashlib.md5(sent1.encode('utf-8')).hexdigest()

    '''
    Observation:
     - information is lost as progressed through layers. injection at `embedding.dropout` induces less
     response at last_hidden_state than injection at `encoder.layer.6.output`. Which means the
     the more distance between layers, the smaller the jacobian F norm.
     - the response at last_hidden_state seems rather sparse.
     - observation:
        - noise at structural tokens "the/a/," not passed to next tokens, somehow absorbed
        - noise at 6th layer "which" indcues more response at 11th layer
          than noise at 0th "which". This means that the
        - noise at other tokens get passed to the next layer
          - tokens like "google" influences its modifiers
        - jacobian norm at layer 7,8,11 layer is probably much lower than other layers.
          - much variation is discarded.
          - other layers propagate variation
          - network alternates between propagation of variation and discarding variation
          - lower layer is very sensitive to ngram


    '''

    PKL = f'{__file__}.{noise_injection_layer}.{md5}.{sigma}.{n_sample}.pkl'

    with torch.no_grad():
        if os.path.exists(PKL):
            xdict_list,x1 = torch.load(PKL)
        else:

            for i in tqdm(range(tok1.__len__())):
                model = add_hook_to_inject_noise(model,tokenizer, noise_injection_layer, i, sigma, n_sample)
                x1 = model.get_all_tensors(sent1)
                for k,v in x1.items():
                    xdict_list[k].append(v)
                model.remove_hooks()
            torch.save((dict(xdict_list),x1),PKL+'.temp')
            shutil.move(PKL+'.temp',PKL)

        with open(__file__+'.html','wb') as f:
            parse_single_perturb(x1)
            xbuf = tbuf.get_table()
            f.write(xbuf.encode())

            disp = BaseDisplay()
            parse_multiple_perturb(xdict_list,sent1,disp=disp,CONFIG_INT_HIDE=100)
            f.write(disp.tbuf.get_table().encode())

        with open(__file__+'.matrix.html','wb') as f:
            disp = BaseDisplay()
            parse_multiple_perturb(xdict_list,sent1,disp=disp)
            f.write(disp.tbuf.get_table().encode())


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
