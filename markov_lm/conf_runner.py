import torch
from dataclasses import dataclass

from markov_lm.c9007_util import tbuf_cls,fws,recur_detach
import shutil

import collections
import os,sys
import torch
from tqdm import tqdm
import glob


def parse_checkpoint(argv,):
    if '--auto' in argv:
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

    if '--LOAD' in argv:
        LOAD = argv[argv.index('--LOAD')+1]
        # LOAD = int(LOAD)
    return LOAD

def conf_parse_all(sys_argv):
    '''
    Parse all arguments from a list
    '''
    CUDA    = ('--cpu' not in sys_argv)
    CKPT = parse_checkpoint(sys_argv,)
    CKPT = CKPT
    STRICT_LOAD = '--nostrict' not in sys_argv
    # print('[strict]',STRICT_LOAD)
    v = ''
    k = '--load_blacklist'
    if k in sys_argv:
        v = sys_argv[sys_argv.index(k)+1]
    BLACKLIST = v.split(',')

    if '--save' in sys_argv:
        v= sys_argv[sys_argv.index('--save')+1]
    else:
        v = 10
    v = int(v)
    SAVE_INTERVAL = v

    k = '--seed'
    _caster = int
    if k in sys_argv:
        v= sys_argv[sys_argv.index(k)+1]
    else:
        v = 29
    v = _caster(v)
    SEED = v


    meta_dict = {}

    k = '--target'
    _caster = int
    if k in sys_argv:
        v= sys_argv[sys_argv.index(k)+1]
    else:
        v = 6000
    v = _caster(v)
    meta_dict['num_epoch']= v

    # k = '--visdom_port'
    # _caster = int
    # if k in sys_argv:
    #     v= sys_argv[sys_argv.index(k)+1]
    # else:
    #     v = 6006
    # v = _caster(v)
    # meta_dict['visdom_port']= v

    k = '--loglr'
    _caster = float
    if k in sys_argv:
        v= sys_argv[sys_argv.index(k)+1]
    else:
        v = -3.
    v = _caster(v)
    meta_dict['loglr']= v


    k = '--batch_size'
    _caster = int
    if k in sys_argv:
        v= sys_argv[sys_argv.index(k)+1]
    else:
        v = 200
    v = _caster(v)
    meta_dict['batch_size']= v


    model_dict = {}
    for i,k in enumerate(sys_argv):
        if k.startswith('--model'):
            # kk = k[len('--model'):]
            assert '.' in k, k
            kk = k.split('.',1)[1]
            # k[len('--model'):]
            v = sys_argv[i+1]
            model_dict[kk] = v
    visdom_dict = meta_dict['visdom'] ={}
    if '--visdom.port' not in sys_argv:
        sys_argv+=['--visdom.port','9002']
    if '--visdom.base_url' not in sys_argv:
        sys_argv+=['--visdom.base_url','/visdom']

    for i,k in enumerate(sys_argv):
        if k.startswith('--visdom.'):
            # kk = k[len('--model'):]
            assert '.' in k, k
            kk = k.split('.',1)[1]
            # k[len('--model'):]
            v = sys_argv[i+1]
            visdom_dict[kk] = v


    return CUDA,CKPT,STRICT_LOAD,BLACKLIST,SAVE_INTERVAL,SEED,model_dict,meta_dict



def get_model_test_loss(conf):
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
    item = None

    if isinstance(dataloader,torch.utils.data.DataLoader):
        'force seq'
        self = dataloader
        dataloader._sampler = dataloader.sampler
        dataloader._batch_sampler = dataloader.batch_sampler
        sampler   = torch.utils.data.sampler.SequentialSampler(dataloader.dataset)
        batch_sampler = torch.utils.data.sampler.BatchSampler( sampler, self.batch_size, self.drop_last)
        super(dataloader.__class__,dataloader).__setattr__('sampler',sampler)
        super(dataloader.__class__,dataloader).__setattr__('batch_sampler', batch_sampler)

# ?    dataloader.sampler =
    # import pdb; pdb.set_trace()
    conf.callback_before_test_all(conf,model,item)
    for tsi,item in enumerate(dataloader):
        item = data_input_transform(item)
        conf.callback_before_test_step(conf,model,item)
        _loss = loss(item).detach()
        if _loss.shape.__len__()>=2:
            _loss = _loss.mean(item.shape[1:])
        if item['index'] is not None:
            index.append(item['index'].to(_loss.device))
        lsl.append(_loss)
        conf.callback_after_test_step(conf,model,item)
    conf.callback_after_test_all(conf,model,item)

    if isinstance(dataloader,torch.utils.data.DataLoader):
        'recover shuffle'
        super(dataloader.__class__,dataloader).__setattr__('sampler', dataloader.sampler)
        super(dataloader.__class__,dataloader).__setattr__('batch_sampler', dataloader.batch_sampler)
        # super(dataloader.__class__,dataloader).__setattr__('sampler', dataloader._sampler)

        # item['losses'])
    lsl   = torch.cat(lsl,dim=0)
    if index:
        index = torch.cat(index,dim=0)
        st = index.argsort()
    else:
        index=  torch.arange(lsl.size(0),device=conf.device)[:]
        # ,None]
        st =index
        # index[:,0]
        # st = range(len())
    v = torch.stack([index,lsl],dim=1)[st,:]

    return v
        # loss_test_sum +=  float(loss.item())
'''
BREAKING CHANGE conf_main_loop
'''
# def conf_main_loop(conf,  CKPT,  STRICT_LOAD,  BLACKLIST,  SAVE_INTERVAL,ret='train'):
def conf_main_loop(conf, ret='train', stdout=sys.stdout):
    def pcbk(s):
        stdout.write(f'{s}\n')
    '''
    A shared main loop for gradient descent
    '''
    CKPT = conf.CKPT
    STRICT_LOAD = conf.STRICT_LOAD
    BLACKLIST=conf.BLACKLIST
    SAVE_INTERVAL = conf.SAVE_INTERVAL
    # ,  SAVE_INTERVAL,
    # CKPT = conf.CKPT
    model = conf.model
    dataset = conf.dataset
    if '--debug' in sys.argv:
        model.debug=1
    else:
        model.debug =0

    sys_argv = sys.argv
    k =  '--LOAD_ABS'
    if k in sys_argv:
        v= sys_argv[sys_argv.index(k)+1]
    else:
        v =None
    LOAD_ABS = v

    if(CKPT!='-1') or LOAD_ABS is not None:
# markov_lm/gmm/Checkpoints/-S29-tasktranslate-mutli30k-de2en-l50-shuffle1-graph-dim38783-model-nameSoftAlignmentModel-window-size0-loss-nameKLD-grad-loss-nameKLD-depth1-beta0.0-n-step50-kernel-size0-embed-dim128-p-null0.0001-submodel-name-loglr-4.0_15_6.10327.pkl
        epoch = CKPT


        if LOAD_ABS is not None:
            res = LOAD_ABS
        else:
            res = glob.glob(os.path.join("Checkpoints",f"{conf._session_name}_{epoch}*.pkl"))
            assert len(res)==1,['Which one to choose?',res]
            res = res[0]
        print(f'[LOADING]{res}')

        checkpoint   = torch.load(res)
        test_losses  = checkpoint["test_losses"]
        train_losses = checkpoint["train_losses"]
        epoch        = checkpoint["epoch"]
        epoch_list   = checkpoint.get('epoch_list',[])
        x            = checkpoint["model"]
        xx = {}
        for k,v in x.items():
            if k in dict(model.named_parameters()):
                xx[k] = v
            else:
                pass
        x = xx
        for k in BLACKLIST:
            if k:
                del x[k]

        model.load_state_dict(x,strict=STRICT_LOAD)
        if STRICT_LOAD:
            conf.optimizer.load_state_dict(checkpoint['optimizer'])

    else:
        test_losses = []
        train_losses = []
        epoch_list = []
        epoch = -1
    if ret=='load':
        return conf


    loss_test_mean = 0
    n_mask = 4
    meta = {}
    model.meta = meta
    for _epoch in range(conf.num_epoch+1):
        # conf.dataset.op_extract_and_mask(n_mask)
        epoch += 1
        conf.epoch = epoch
        loss_train_sum = 0
        loss_test_sum = 0
        curr_seed = torch.seed()
        conf.callback_epoch_start(epoch)

        loss_mat = get_model_test_loss(conf)
        loss_test_mean = loss_mat[:,1].mean().item()
        test_losses.append(loss_test_mean)
        epoch_list.append(epoch)

        model.train()
        dataset.train()
        model.zero_grad()
        conf.callback_start(epoch,None,None)
        for tri,item in enumerate(tqdm(conf.dataloader,file=stdout)):
            item            = conf.data_input_transform(item)
            grad_loss       = conf.grad_loss(item).mean()
            loss            = conf.loss(item).mean()
            loss_train_sum += float(loss.item())
            grad_loss.backward()
            conf.optimizer.step()
            conf.callback_step(epoch, tri,item,loss)
        conf.callback_end(epoch, tri,item,loss)


        meta.update({
            "epoch"       :epoch,
            "train_losses":train_losses,
            "test_losses" :test_losses,
            "epoch_list" : epoch_list,
            "loss_mat"    :loss_mat,
            "curr_seed"   :[curr_seed, torch.seed()],
            "model_config":conf.model.__dict__.get('config',{}),
            "model_cls"   :conf.model.__class__,
            'conf_task'   :conf.task,
        })
        ### needs to random extract tokens into sets and sequences
        if(epoch % SAVE_INTERVAL ==0):
            # target_filename = conf.get_ckpt_name(os.path.join("Checkpoints",f"{conf._session_name}_{epoch}_{loss_test_mean:.5f}.pkl"))
            target_filename = os.path.join("Checkpoints",f"{conf._session_name}_{epoch}_{loss_test_mean:.5f}.pkl")
            # conf.model.target_filename = target_filename
            meta.update({
            "model"       :conf.model.state_dict(),
            "optimizer"   :conf.optimizer.state_dict(),
            })
            # model.meta = meta
            torch.save( meta, target_filename)
            linkFile = __file__+'.curr.ckpt.pkl'
            # os.unlink(linkFile) if os.path.exists(linkFile) else None
            # os.link(target_filename,linkFile)
            shutil.copy2(target_filename,linkFile+'.temp')
            shutil.move(linkFile+'.temp',linkFile)
            conf.callback_checkpoint(conf, conf.model, None)
            # , target_filename)
            # loss = cross_entropy(x,y)
        # assert 0,pcbk



        loss_train_mean = (loss_train_sum/(1+tri))
        # loss_test_mean = loss_test_sum/(1+tsi)
        pcbk(f'Epoch: {epoch}')
        pcbk(f'ModelClassName: {conf._session_name}')
        pcbk(f'Training Loss: {loss_train_mean}')
        pcbk(f'Testing Loss: {loss_test_mean}')

        train_losses.append(loss_train_mean)
