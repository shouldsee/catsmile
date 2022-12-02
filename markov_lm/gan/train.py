'''
- Author: Feng Geng
- Changelog:
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
from tqdm import tqdm


from markov_lm.Dataset.translation_dataset import ArithmeticTest
from markov_lm.Model_add import AddModelWithBert
from markov_lm.Model_add import AddModelWithAttention

import math
import collections
import shutil


def get_model_test_loss(conf,epoch):
    '''
    For each test entry, obtain model's prediction loss, as a dict{index:loss}
    This is for evaluation of models performance
    '''
    model                = conf.model
    dataset              = conf.dataset
    loss                 = conf.loss
    data_input_transform = conf.data_input_transform
    dataloader           = conf.dataloader

    model.eval()
    dataset.test()
    index = []
    lsl = []
    for tsi,item in enumerate(dataloader):

        item['epoch']       = epoch
        item['batch_index'] = tsi
        item['state']       = 'test'

        item = data_input_transform(item)
        _loss = loss(item).detach()
        if _loss.shape.__len__()>=2:
            assert 0,_loss.shape
            _loss = _loss.mean(tuple(range(1,1+len(_loss.shape[1:]))))
        index.append(item['index'].to(_loss.device))
        lsl.append(_loss)
        # item['losses'])
    index = torch.cat(index,dim=0)
    lsl   = torch.cat(lsl,dim=0)
    st = index.argsort()
    v = torch.stack([index,lsl],dim=1)[st,:]
    return v
        # loss_test_sum +=  float(loss.item())

def parse_checkpoint(sys,):
    if '--auto' in sys.argv:
        LOAD_GET_AUTO = 1
    else:
        LOAD_GET_AUTO = 0

    os.makedirs('Checkpoints') if not os.path.exists('Checkpoints') else None

    if LOAD_GET_AUTO:
        res = glob.glob('Checkpoints/*pkl')
        getLoad = lambda x:int(x.replace('Checkpoints/Checkpoint','').split('.')[0])
        res = sorted(res,key=getLoad)[-1]
        LOAD= getLoad(res)
    else:
        LOAD = '-1'

    if '--LOAD' in sys.argv:
        LOAD = sys.argv[sys.argv.index('--LOAD')+1]
        # LOAD = int(LOAD)
    return LOAD

from markov_lm.c9007_util import recur_detach
from markov_lm.conf_gan import GANConfigPrototype

def init_conf(CUDA,shuffle, AddModelWithAttention=AddModelWithAttention):
    '''
    Runtime subclassing AddModelWithAttention()

    returns a config object that controls training process.
    binds (dataset,model,device)
    '''

    def add_hook_for_output_tensor(model,CONFIG_EXPAND_LEVEL,CONFIG_DETACH, CONFIG_PRINT_CLASS):
        '''

        returns a temp object with a dict of tensors outputted at each nn.Module with maximum depth of CONFIG_EXPAND_LEVEL.
        '''
        class Temp(object):
            xdict = collections.OrderedDict()
            handles = []
            def __init__(self,):
                pass
                # self.xdict= xdict
            def remove_hooks(self):
                # ,handles= handles):
                 [h.remove() for h in self.handles]
        t = Temp()
        handles = t.handles
        xdict = t.xdict


        def get_backward_hook(name,xdict=xdict):
            # def hook(model, input_grad, output_grad):
            def hook(output_grad):
                recur_detach(xdict,name,output_grad,CONFIG_DETACH,'/')
                # xdict[name] = output
            return hook

        '注入灵魂'
        for k,v in model.named_parameters():
            if k.count('.')<=CONFIG_EXPAND_LEVEL:
                if CONFIG_PRINT_CLASS:
                    print(fws(k,60) ,v.__class__);
                h = v.register_hook(get_backward_hook(k))
                # h = v.register_full_backward_hook(get_backward_hook(k))
                handles.append(h)

        return t

    # conf = ConfigPrototype()
    conf = GANConfigPrototype(__file__+'.log.html')
    '''
    Abusing attributes here
    [TBC]
    '''
    # conf.task = 'add'
    # conf.task = 'refill'
    # conf.task = 'ner1'
    conf.task = 'duie-mlm'

    conf.instance= 29

    torch.manual_seed(conf.instance)
    conf.embed_dim     = 50
    conf.device        =  torch.device('cuda:0' if CUDA else 'cpu')
    conf.num_epoch     = 5000
    conf.learning_rate = 0.001
    conf.batch_size    = 60
    add_optimizer      = lambda conf,params:torch.optim.RMSprop( params, lr=conf.learning_rate,)
    # add_optimizer      =  lambda conf,params:torch.optim.RMSprop( params, lr=conf.learning_rate,eps=0.01)

    conf.SAVE_INTERVAL = 5
    conf.tsi_max = -1
    # print(conf.task)
    # conf.task='UNK'

    torch.manual_seed(conf.instance)
    if conf.task=='refill':
        from markov_lm.Dataset.translation_dataset import RefillDataset
        conf.dataset = dataset =RefillDataset(CUDA=CUDA)
        conf.data_input_transform = lambda item:dict(unmasked = item['english'],masked=item['masked'], mask=item['mask'])

        conf.loss = lambda item:conf.model.loss(item)
        conf.grad_loss = lambda item:conf.model.loss(item)
        conf.callback_epoch_start = lambda epoch: conf.dataset.op_extract_and_mask(n_mask=4)
        conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)


    elif conf.task=='add':
        ### This is a random dataset !!!! init after random seed is set
        conf.dataset = dataset = ArithmeticTest(CUDA=CUDA)
        # import pdb;
        conf.data_input_transform = lambda item: dict(unmasked = item['unmasked'],masked=item['masked'], mask=item['mask']);

        conf.loss = lambda item:conf.model.loss(item)
        conf.grad_loss = lambda item:conf.model.loss(item)

        ### test dataset works
        conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)
        # dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)
    elif conf.task in 'ner1 duie-mlm'.split():
        from markov_lm.Dataset.DUIE_NER import DUIE_NER
        ### This is a random dataset !!!! init after random seed is set
        conf.dataset = dataset = DUIE_NER(CUDA=CUDA,task_mode=conf.task)
        conf.data_input_transform = lambda item: item
        conf.loss = lambda item:conf.model.loss(item)
        conf.grad_loss = lambda item:conf.model.loss(item)

        ### test dataset works
        conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)
        if conf.task=='duie-mlm':
            conf.callback_epoch_start = lambda epoch: conf.dataset.op_sample_mask(n_mask=0)
            # conf.
        # dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)
    else:
        raise NotImplementedError(conf.task)
        # op_extract_and_mask
    (conf.dataset[range(5)])



    # '''
    # now specify model
    # '''
    if 0:
        conf.learning_rate = 0.001
        conf.kernel_size = 5
        conf.depth = 12
        conf.embed_dim = 40
        conf.kernel_dim = 10
        conf.use_dropout= 0.5
        conf.use_dense_relu = 13
        conf.use_layernorm = 1
        conf.use_gradnorm = 1
        conf.use_input_image =0
        conf.step_size = 0.05
        conf.iter_per_layer= 100
        assert conf.is_model_set is False
        conf.is_model_set =True

        conf.depth = 12
        conf.iter_per_layer=1
        # conf.learning_rate = 0.0001
        # conf.instance = 4;torch.manual_seed(conf.instance)
        conf.model = model = AddModelWithBert(graph_dim = dataset.graph_dim,
            depth=conf.depth,
            # state_count=conf.state_count,
            embed_dim=conf.embed_dim,device=conf.device,
            iter_per_layer = conf.iter_per_layer,
            mask_token_idx=dataset.mask_token_idx)

    class AddModelWithAttention(AddModelWithAttention):
        '''
        Adds callback to record energy
        '''
        bi = 0
        def callback_step(self,inner,outer,s=conf.s):
            energy= outer[-2].mean()
            k = f'energy_{self.bi}'
            conf.init_s(k,energy)
            s[k].append(energy.item()*1000)

        def callback_end(self,outer,inner,s=conf.s):
            self.bi += 1
        def train(self,*a,**kw):
            super().train(*a,**kw)
            self.bi = 0


    CLS = [AddModelWithAttention]
    def _add_model(conf,cls = CLS):
        conf.model = model = CLS[0](
            graph_dim      = dataset.graph_dim,
            depth          = conf.depth,
            use_dropout    = conf.use_dropout,
            kernel_size    = conf.kernel_size,
            use_dense_relu = conf.use_dense_relu,
            use_layernorm  = conf.use_layernorm,
            use_gradnorm   = conf.use_gradnorm,
            use_input_image= conf.use_input_image,
            embed_dim      = conf.embed_dim,device=conf.device,
            kernel_dim = conf.kernel_dim,
            iter_per_layer = conf.iter_per_layer,
            mask_token_idx = dataset.mask_token_idx,
            n_choice       = conf.n_choice,
            step_size      = conf.step_size)
        return model
    from markov_lm.Model_add import LayerConfig
    conf.kernel_dim = None
        # AddModelWithAttention.callback_step = callback_step
    conf.n_choice = 0
    conf.kernel_dim = 10
    if 1:
        def _add_model(conf):
            conf.model = model = CLS[0](device=conf.device, config=conf.lconf,charset = dataset.charset)
            return model

        conf.learning_rate = 0.001
        # conf.kernel_size = 15
        # conf.depth = 20
        # conf.embed_dim = 50

        conf.lconf = LayerConfig(
            kernel_size = 5,
            depth = 4,
            embed_dim = 40,
            kernel_dim = 10,
            use_dropout= 0.0,
            use_dense_relu = 11,
            use_layernorm = 1,
            use_gradnorm = 1,
            use_input_image =0,
            step_size = 0.5,
            iter_per_layer= 1,
            mask_token_idx = -1,
            graph_dim=dataset.graph_dim
        )
        assert conf.is_model_set is False
        conf.is_model_set =True

        from markov_lm.Model_add import AddModelWithAttentionStacked,AddModelBertInterface
        CLS[0] = AddModelWithAttentionStacked
        from markov_lm.conf_gan import AttentionGAN
        CLS[0] = AttentionGAN
        conf.loss = lambda item:conf.model.loss(item)
        conf.grad_loss = lambda item:conf.model.loss(item)
        conf.callback_epoch_start = lambda epoch:[
        conf.dataset.op_sample_mask(n_mask=0),
        print(f'[last_d_loss]{conf.model.last_d_loss}'),
        print(f'[last_g_loss]{conf.model.last_g_loss}'),
        print(f'[last_unembed_nll]{conf.model.last_unembed_nll}')
        ]
        # conf.loss = lambda

        # conf.is_model_set =True


    _add_model(conf)
    if isinstance(conf.model,AttentionGAN):
        tok2char = conf.dataset.toks2chars
        def callback(model,item,inner):
            # (item,lptok,xexp) = inner
            # lptok
            # xexp = item['']
            # lptok = model.vocab(inner[-1])
            lptok = conf.model.model_d.vocab(inner[-1])
            state = conf.dataset.mode
            # model.alias = 'generator'
            if model.alias=='dis':return
            # state = 'train' if conf.dataset.state=='train'
            _tok = lambda x: '' if x ==0 else x
            for i in range(min(5,len(lptok))):
                # k = f'token_{state}_{i}_word'
                # # print(item['masked'].shape)
                # vlist = item['masked'][i].detach().cpu().numpy().tolist()
                # conf.s[k] = [k,list(item['masked'].shape)] +   vlist
                #
                # k = f'token_{state}_{i}_wordv'
                # vlist = list(map(conf.dataset.charset.__getitem__,vlist))
                # conf.s[k] = [k,list(item['masked'].shape)] +   vlist
                # # [[v] for v in item['masked'][i].detach().cpu().numpy().tolist()]
                # # list(map(_tok, (10*lptok[i].argmax(-1)).detach().cpu().numpy().tolist()))
                #
                # k = f'token_{state}_{i}_unmask'
                # conf.s[k]  = [k,''] + list(map(_tok,(item['unmasked'][i]).detach().cpu().numpy()))

                # space= item['masked'][i]*0
                # v = (1*lptok[i].argmax(-1)).detach()
                # reseq = lambda v: torch.scatter(space,src=v,index=item['mask'][i],dim=0)

                if item['masked'].shape.__len__()<3:
                    vlist = item['masked'][i].detach().cpu().numpy().tolist()
                    k = f'token_{state}_{i}_wordv_{model.alias}'
                    conf.s[k] = [k,list(item['masked'].shape)] +   tok2char(vlist)

                k = f'token_{state}_{i}_pred_{model.alias}'
                conf.s[k] = [k,''] +  tok2char(((1*lptok[i,:,:-1].argmax(-1)).detach()).int())

                # k = f'token_{state}_{i}_pred'
                # conf.s[k] = [k,''] +  list(map(_tok, (1*lptok[i].argmax(-1)).detach().cpu().numpy().tolist()))

                # # k = f'token_{state}_{i}_exp'
                # # conf.s[k]  = [k,''] + list(map(_tok,(xexp[i]).detach().cpu().numpy()))
                # k = f'token_{state}_{i}_ws'
                # conf.s[k]  = [k]
        conf.model.model_g.callback_end_simple = callback
        conf.model.model_d.callback_end_simple = callback
        # conf.model.model_d.callback_end_simple = callback

    '''
    Class name is used to construct checkpoint filenames. manually record important params here, needs a better system.
    [TBC]
    '''
    lconf = conf.lconf
    conf._session_name =  f'S{conf.instance}'
    conf._session_name += f'-task{conf.task}-shuffle{int(shuffle)}'
    conf._session_name += f'-{conf.model.__class__.__name__}'
    conf._session_name += f'-D{lconf.depth}-E{lconf.embed_dim}-K{lconf.kernel_size}-KE{lconf.kernel_dim}-IPL{lconf.iter_per_layer}-'
    conf._session_name += f'DenseRelu{lconf.use_dense_relu}-Layernorm{lconf.use_layernorm}-Dropout{lconf.use_dropout}-Gradnorm{lconf.use_gradnorm}'

    conf._session_name += f'-loglr{math.log10(conf.learning_rate):.1f}'
    # conf._session_name += f'-nchoice{lconf.n_choice}' if conf.n_choice else ''
    conf._session_name += f'-UseInputImage{lconf.use_input_image}'



    conf.model = model = conf.model.to(conf.device)
    params = list(model.parameters())
    print(dict(model.named_parameters()).keys())

    #### using Adam with high learning_rate is catastrophic
    conf.optimizer = add_optimizer(conf,params)
    conf.handler = add_hook_for_output_tensor(model,CONFIG_DETACH=1,CONFIG_EXPAND_LEVEL=4,CONFIG_PRINT_CLASS=0)
    assert conf._session_name is not None,'Please set conf._session_name before finish init!'
    return conf


def main():
    CUDA    = ('--cpu' not in sys.argv)
    conf    = init_conf(CUDA,shuffle=True)
    model   = conf.model
    dataset = conf.dataset

    CKPT = parse_checkpoint(sys,)
    if(CKPT!='-1'):

        epoch = CKPT
        # res = glob.glob(os.path.join("Checkpoints",f"{conf._session_name}_{CKPT}_*.pkl"))
        res = glob.glob(os.path.join("Checkpoints",f"{conf._session_name}_{epoch}*.pkl"))
        assert len(res)==1,['Which one to choose?',res]
        print(f'[LOADING]{res[0]}')

        checkpoint   = torch.load(res[0])
        test_losses  = checkpoint["test_losses"]
        train_losses = checkpoint["train_losses"]
        epoch        = checkpoint["epoch"]
        x            = checkpoint["model"]
        xx = {}
        for k,v in x.items():
            if k in dict(model.named_parameters()):
                xx[k] = v
            else:
                pass
                # print(k)
        x = xx
        STRICT_LOAD = '--nostrict' not in sys.argv
        # print('[strict]',STRICT_LOAD)
        v = ''
        k = '--load_blacklist'
        if k in sys.argv:
            v = sys.argv[sys.argv.index(k)+1]
        BLACKLIST = v.split(',')
        for k in BLACKLIST:
            if k:
                del x[k]

        model.load_state_dict(x,strict=STRICT_LOAD)
        if STRICT_LOAD:
            conf.optimizer.load_state_dict(checkpoint['optimizer'])

    else:
        test_losses = []
        train_losses = []
        epoch = -1


    loss_test_mean = 0
    n_mask = 4
    for _epoch in range(conf.num_epoch):
        # conf.dataset.op_extract_and_mask(n_mask)
        epoch += 1
        loss_train_sum = 0
        loss_test_sum = 0
        curr_seed = torch.seed()
        conf.callback_epoch_start(epoch)

        loss_mat = get_model_test_loss(conf,epoch)
        loss_test_mean = loss_mat[:,1].mean()

        ### needs to random extract tokens into sets and sequences
        if(epoch % conf.SAVE_INTERVAL ==0):
            # target_filename = conf.get_ckpt_name(os.path.join("Checkpoints",f"{conf._session_name}_{epoch}_{loss_test_mean:.5f}.pkl"))
            target_filename = os.path.join("Checkpoints",f"{conf._session_name}_{epoch}_{loss_test_mean:.5f}.pkl")
            torch.save({
                "model"       :conf.model.state_dict(),
                "optimizer"   :conf.optimizer.state_dict(),
                "epoch"       :epoch,
                "train_losses":train_losses,
                "test_losses" :test_losses,
                "loss_mat"    :loss_mat,
                "curr_seed"   :[curr_seed, torch.seed()],
                "model_config":conf.model.__dict__.get('config',{})
            },target_filename)
            linkFile = __file__+'.curr.ckpt.pkl'
            # os.unlink(linkFile) if os.path.exists(linkFile) else None
            # os.link(target_filename,linkFile)
            shutil.copy2(target_filename,linkFile+'.temp')
            shutil.move(linkFile+'.temp',linkFile)
            # loss = cross_entropy(x,y)


        model.train()
        dataset.train()
        model.zero_grad()
        conf.callback_start(epoch,None,None)
        for tri,item in enumerate(tqdm(conf.dataloader)):
            item['epoch']       = epoch
            item['batch_index'] = tri
            item['state']   = 'train'
            item            = conf.data_input_transform(item)
            grad_loss       = conf.grad_loss(item).mean()
            loss            = conf.loss(item).mean()
            loss_train_sum += float(loss.item())
            grad_loss.backward()
            conf.optimizer.step()
            conf.callback_step(epoch,(tri,item),loss)
        conf.callback_end(epoch,(tri,item),loss)

        if hasattr(model,'get_hidden'):
            fs_list = []
            zs = []
            idxs = []
            sents = []
            xsa = []
            for tri,item in enumerate(tqdm(conf.dataloader)):
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


        loss_train_mean = loss_train_sum/(1+tri)
        # loss_test_mean = loss_test_sum/(1+tsi)
        print(f'Epoch: {epoch}')
        print(f'ModelClassName: {conf._session_name}')
        print(f'Training Loss: {loss_train_mean}')
        print(f'Testing Loss: {loss_test_mean}')

        train_losses.append(loss_train_mean)
        test_losses.append(loss_test_mean)

if __name__=='__main__':
    main()
