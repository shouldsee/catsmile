
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

# from markov_lm.Dataset.translation_dataset import EnglishToGermanDataset
from markov_lm.Dataset.translation_dataset import RefillDataset
from markov_lm.Dataset.translation_dataset import BertMiddleLayer

import os,sys

import numpy as np
from tqdm import tqdm
from torch import autograd

from markov_lm.util_html import write_png_tag

#import pandas as pd
import sys
import glob
from pprint import pprint
import shutil
from tqdm import tqdm

# device = torch.device('cuda:0' if CUDA else 'cpu')

from markov_lm.Model import cross_entropy



from markov_lm.Model_pretrain import BertTok,BertModel,BertLayer
# from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectSampling



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


class Config(object):
    def __init__(self):
        return

from markov_lm.Model_pretrain import FirstOrderLowRankEnergy,NoAttention
from markov_lm.Model_pretrain import SimpleLowRankEnergy
from markov_lm.Model_pretrain import BertLayer,BertIntermediate,BertOutput

#
CUDA = 1
def init_conf(CUDA,shuffle):
    conf = Config()
    conf.criterion = cross_entropy
    conf.embed_dim = 50
    conf.mixture_count = 8
    conf.state_count = 15
    conf.device =  torch.device('cuda:0' if CUDA else 'cpu')
    conf.num_epoch = 5000
    # model = MixtureOfHMM(graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device)

    conf.learning_rate = 0.0001
    conf.SAVE_INTERVAL = 10
    conf.batch_size = 60
    conf.tsi_max = 10
    conf.mimicLayerIndex = 7

    # conf.dataset = dataset = RefillDataset(CUDA=CUDA)
    conf.dataset = dataset = BertMiddleLayer(conf.mimicLayerIndex,CUDA=CUDA)
    ### test dataset works
    (conf.dataset[range(5)])

    conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)
    # dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)
    # conf.model = model = ExtractionAndMarkovTemplateMatching(graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device)
    conf.optimizer_factory = None
     # torch.optim.RMSprop


    # assert 0
    bconf = BertModel.config

    layer = BertModel.encoder.layer[3] ## cached for 7
    layer.__class__.__name__ = 'layer3'
    f = lambda x: layer(x)[0]

    from markov_lm.Model_pretrain import FirstOrderLowRankEnergyExpAttention
    #
    # layer = BertModel.encoder.layer[8] ## cached for 6
    # layer.__class__.__name__ = 'layer8'
    # f = lambda x: layer(x)[0]
    #
    # # layer = BertModel.encoder.layer[4]
    # # layer.__class__.__name__ = 'layer4'
    # # f = lambda x: layer(x)[0]
    #
    # # layer = BertModel.encoder.layer[11]
    # # layer.__class__.__name__ = 'layer11'
    # # f = lambda x: layer(x)[0]
    #
    layer = BertLayer(BertModel.config,).to(conf.device)
    layer.__class__.__name__ = 'layerX'
    f = lambda x: layer(x)[0]


    # # conf.learning_rate = 0.01
    # K = 15
    # D = 100
    # layer = FirstOrderLowRankEnergy( K,bconf.hidden_size, D ).to(conf.device)
    # f = lambda x: layer(x)
    # layer.__class__.__name__ += f'-K{K}-D{D}'

    # K = 15
    # D = 100
    # layer = FirstOrderLowRankEnergyExpAttention( K,bconf.hidden_size, D ).to(conf.device)
    # f = lambda x: layer(x)
    # layer.__class__.__name__ += f'-K{K}-D{D}'

    #
    # from markov_lm.Model_pretrain import FirstOrderLowRankEnergyWithLimit
    # K = 15
    # D = 100
    # layer = FirstOrderLowRankEnergyWithLimit( K,bconf.hidden_size, D ).to(conf.device)
    # f = lambda x: layer(x)
    # layer.__class__.__name__ += f'-K{K}-D{D}'
    #
    # # # # conf.learning_rate = 0.01
    # # K = 5
    # L = 15
    # D = 102
    # layer = SimpleLowRankEnergy( L,bconf.hidden_size, D ).to(conf.device)
    # f = lambda x: layer(x)
    # layer.__class__.__name__ += f'-L{L}-D{D}'
    # 
    # layer = NoAttention(bconf).to(conf.device)
    # conf.model = model = layer
    # f = lambda x: layer(x)
    # layer.__class__.__name__ += f'-Dataset2'


    # from markov_lm.Model_pretrain import TransAttention
    # # AdditiveAttention(bconf)
    # # conf.learning_rate = 0.01
    # layer = TransAttention(bconf).to(conf.device)
    # conf.model = model = layer
    # f = lambda x: layer(x)
    # layer.__class__.__name__ += f'-3-Dataset2'

    conf.model = model = layer

    '''
    Construct a dynamic model to probe whether it's possible to
    bypass the quadratic attention.

    The first-order attention seeks to perform graph
    message passing on a simplified network where
    each node of the prototype network is extracted from
    the sequence by a softmax average.


    \sum_i A_{ik} = 1
    A_{ik} = \frac{ \exp( \mu_k^T x_i ) }{ \sum_i \exp(\mu_k^T x_i)}
    B_k   = \sum_i A_{ik} x_i
    E = \sum B_{k1} W_{k1,k2} B_{k2}
    '''


    params = list(model.parameters())
    print(list(dict(model.named_parameters()).keys())[:5])
    #### using Adam with high learning_rate is catastrophic
    # conf.optimizer = torch.optim.Adagrad( params, lr=conf.learning_rate)
    conf.optimizer      = torch.optim.RMSprop( params, lr=conf.learning_rate)
    # conf.optimizer      = torch.optim.Adam( params, lr=conf.learning_rate)
    for name,xl in layer.named_children():
        # xl = layer.output.LayerNorm
        if name.endswith('LayerNorm'):
            xl.reset_parameters()
            xl.register_parameter('weight', None)
            xl.register_parameter('bias', None)

#     def lossFunc(x,y,f=f):
#         y = y /(1E-1+y.std(-1,keepdims=True))
#         # x = x /(1E-1+x.std(-1,keepdims=True))
#         yp = f(x)
#         yp = yp /(1E-1+yp.std(-1,keepdims=True))
#
#         # yp = layer(x)[0]
#         # import pdb; pdb.set_trace()
#
#         # loss = -(y-x)*(yp-x)
#         # loss = loss.mean((-1,-2))
#
#         loss = -(y*yp)
#         loss = loss.mean((-1,-2))
# #        f = layer
#         return loss

    def lossFunc(x,y,f=f):
        '''
        Requires f(x) \approx g(x)
        '''
        # y = y /(1E-1+y.std(-1,keepdims=True))
        # x = x /(1E-1+x.std(-1,keepdims=True))
        yp = f(x)
        # yp = yp /(1E-1+yp.std(-1,keepdims=True))

        loss = (y-yp).square()
        # import pdb; pdb.set_trace()

        loss = loss.mean((-1,-2))

#        f = layer
        return loss
    # layer.__class__.__name__ += '-mse'

    # def lossFunc(x,y,f=f):
    #     '''
    #     Requires f'(x) \approx g'(x)
    #     '''
    #     # y = y /(1E-1+y.std(-1,keepdims=True))
    #     # x = x /(1E-1+x.std(-1,keepdims=True))
    #     yp = f(x)
    #     loss = ((y - y[0:1]) - (yp - yp[0:1])).square()
    #     # yp = yp /(1E-1+yp.std(-1,keepdims=True))
    #
    #     loss = (y-yp).square()
    #     # import pdb; pdb.set_trace()
    #
    #     loss = loss.mean((-1,-2))
    #     return loss
    # layer.__class__.__name__ += '-jmse'



    conf.loss = lossFunc
    conf.grad_loss = lossFunc
    return conf


def main():
    CUDA = 1
    conf = init_conf(CUDA,shuffle=True)
    model = conf.model
    dataset = conf.dataset
    dataset
    CKPT = parse_checkpoint(sys,)
    if(CKPT!='-1'):
        epoch = CKPT
        # res = glob.glob(os.path.join("Checkpoints",f"{conf.model.__class__.__name__}_{CKPT}_*.pkl"))
        res = glob.glob(os.path.join("Checkpoints",f"{conf.model.__class__.__name__}_{epoch}*.pkl"))
        assert len(res)==1,['Which one to choose?',res]
        print(f'[LOADING]{res[0]}')

        # fn = res[0]        checkpoint   = torch.load(os.path.join("Checkpoints","Checkpoint"+str(LOAD)+".pkl"))
        checkpoint   = torch.load(res[0])
        test_losses  = checkpoint["test_losses"]
        train_losses = checkpoint["train_losses"]
        epoch        = checkpoint["epoch"]
        x            = checkpoint["model"]
        # import pdb; pdb.set_trace()
        xx = {}
        for k,v in x.items():
            if k in dict(model.named_parameters()):
                xx[k] = v
            else:
                pass
        x = xx
        STRICT_LOAD = '--nostrict' not in sys.argv
        model.load_state_dict(x,strict=STRICT_LOAD)
        if STRICT_LOAD:
            conf.optimizer.load_state_dict(checkpoint['optimizer'])
        # optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        test_losses = []
        train_losses = []
        epoch = -1


    loss_test_mean = 0
    n_mask = 4
    for _epoch in range(conf.num_epoch):
        conf.dataset.jiggle_data()
        # conf.dataset.op_extract_and_mask(n_mask)
        epoch += 1
        loss_train_sum = 0
        loss_test_sum = 0

        ### needs to random extract tokens into sets and sequences
        if(epoch % conf.SAVE_INTERVAL ==0):
            target_filename = os.path.join("Checkpoints",f"{conf.model.__class__.__name__}_{epoch}_{loss_test_mean:.5f}.pkl")
            torch.save({
                "model"       :conf.model.state_dict(),
                "optimizer"   :conf.optimizer.state_dict(),
                "epoch"       :epoch,
                "train_losses":train_losses,
                "test_losses" :test_losses
            },target_filename)
            linkFile = __file__+'.curr.ckpt.pkl'
            # os.unlink(linkFile) if os.path.exists(linkFile) else None
            # os.link(target_filename,linkFile)
            shutil.copy2(target_filename,linkFile+'.temp')
            shutil.move(linkFile+'.temp',linkFile)
            # loss = cross_entropy(x,y)

        model.eval()
        dataset.test()
        for tsi,item in enumerate(conf.dataloader):
            x = item['input']
            y = item['output']

            loss = conf.loss(x,y).mean()
            loss_test_sum +=  float(loss.item())
            if tsi==conf.tsi_max:
                break
                # print(tsi)



        model.train()
        dataset.train()
        model.zero_grad()
        for tri,item in enumerate(tqdm(conf.dataloader)):
            x = item['input']
            y = item['output']

            gradloss = conf.grad_loss(x,y).mean()
            loss     =  conf.loss(x,y).mean()
            # loss.mean()
            loss_train_sum += float(loss.item())
            loss.backward()
            conf.optimizer.step()
            # break


        loss_train_mean = loss_train_sum/(1+tri)
        loss_test_mean = loss_test_sum/(1+tsi)
        print(f'Epoch: {epoch}')
        print(f'ModelClassName: {conf.model.__class__.__name__}')
        print(f'Training Loss: {loss_train_mean}')
        print(f'Testing Loss: {loss_test_mean}')

        train_losses.append(loss_train_mean)
        test_losses.append(loss_test_mean)

if __name__=='__main__':
    main()
