
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

# from markov_lm.Dataset.translation_dataset import EnglishToGermanDataset
# from markov_lm.Dataset.translation_dataset import RefillDataset
# from markov_lm.Dataset.duie import RefillDataset
from markov_lm.Dataset.DUIE_NER import DUIE_NER

import os,sys

import numpy as np
from tqdm import tqdm
from torch import autograd

from markov_lm.util_html import write_png_tag

import pandas as pd
import sys
import glob
from pprint import pprint
import shutil
from tqdm import tqdm

# device = torch.device('cuda:0' if CUDA else 'cpu')

from markov_lm.Model import cross_entropy

# from markov_lm.Model_Refill import RefillModel
# from markov_lm.Model_Refill import RefillModelRNNSwitch
# from markov_lm.Model_Refill import RefillModelRNNAttention
# from markov_lm.Model_Refill import RefillModelRNNAdditive
# from markov_lm.Model_Refill import RefillModelRNNAdditiveDirect
# from markov_lm.Model_Refill import RefillModelRNNGRU
# from markov_lm.Model_Refill import RefillModelNGRAM
# # Additive
# from markov_lm.Model_Refill import RefillModelOld
# from markov_lm.Model_Refill import RefillModelCopy
# from markov_lm.Model_Refill import RefillModelCopyWithRandomFill
# from markov_lm.Model_Refill import RefillModelRNNAdditiveWithPseudoSampling
# from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectMixing
# from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectMixingWithGate
# # RefillModelRNNAdditiveDirectMixingWithGate
# from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectMixingBidirectional
#
# from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectSampling
# from markov_lm.Model_Refill import RefillModelRNNAdditiveSweeping
# from markov_lm.Model_Refill import RefillModelRNNAdditiveSweepingWithResidual
# from markov_lm.Model_Refill import RefillModelMixtureRNNSweepingOldEmission
# from markov_lm.Model_Refill import RefillModelMixtureRNNSweepingOldEmission2
# from markov_lm.Model_Refill import RefillModelMixtureRNNSweepingNewEmission
# from markov_lm.Model_Refill import RefillModelMixtureRNNSweepingOldEmissionDifferentTransition
from markov_lm.Model_NER import RefillModelRNNAdditiveDirectMixingWithAttention
from markov_lm.Model_NER import RefillModelRNNAdditiveDirectMixing
from markov_lm.Model_NER import RefillModelWithFixedEmission
from markov_lm.Model_NER import RefillModelDynamicRNN
# from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectMixingWithAttention
# from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectMixingWithGate

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

CUDA = 1
def init_conf(CUDA,shuffle):
    conf = Config()
    conf.criterion = cross_entropy
    conf.embed_dim = 51
    conf.mixture_count = 8
    conf.state_count = 15
    conf.device =  torch.device('cuda:0' if CUDA else 'cpu')
    conf.num_epoch = 5000
    # model = MixtureOfHMM(graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device)

    conf.learning_rate = 0.001
    conf.SAVE_INTERVAL = 1
    conf.batch_size = 100
    conf.tsi_max = 10

    # conf.dataset = dataset = RefillDataset(CUDA=CUDA)
    conf.dataset = dataset = DUIE_NER(CUDA=CUDA)
    ### test dataset works
    (conf.dataset[range(5)])

    conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)
    # dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)
    # conf.model = model = ExtractionAndMarkovTemplateMatching(graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device)
    conf.optimizer_factory = None
     # torch.optim.RMSprop
    # conf.optimizer_factory = torch.optim.Ada
    # conf.model = model = RefillModelOld(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device)

    '''
    RefillModelCopy:     420_0.9066             ### Just copying sequence
    RefillModelCopyWithRandomFill_400_1.07266   ### Copying sequence except at mask copying from set
    RefillModelRNNSwitch_240_0.078              ### static selector_q.weight
    RefillModelRNNSwitch_100_0.11912            ### static selector_q.weight
    RefillModelRNNSwitch_100_0.21843            ### dynamic select_q.weight
    '''

    conf.model = model = RefillModelRNNAdditiveDirectMixingWithAttention(total_length=dataset.total_length,min_len=dataset.max_len, graph_dim =dataset.vocab_size,mixture_count=conf.mixture_count, state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device, mask_token_idx=-1,)
    conf.model = model = RefillModelRNNAdditiveDirectMixing(total_length=dataset.total_length,min_len=dataset.max_len, graph_dim =dataset.vocab_size,mixture_count=conf.mixture_count, state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device, mask_token_idx=-1,)
    conf.model = model = RefillModelWithFixedEmission(total_length=dataset.total_length,min_len=dataset.max_len, graph_dim =dataset.vocab_size,mixture_count=conf.mixture_count, state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device, mask_token_idx=-1,)
    conf.model = model = RefillModelDynamicRNN(total_length=dataset.total_length,min_len=dataset.max_len, graph_dim =dataset.vocab_size,mixture_count=conf.mixture_count, state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device, mask_token_idx=-1,)


    params = list(model.parameters())
    print(dict(model.named_parameters()).keys())
    #### using Adam with high learning_rate is catastrophic
    # conf.optimizer = torch.optim.Adagrad( params, lr=conf.learning_rate)
    conf.optimizer      = torch.optim.RMSprop( params, lr=conf.learning_rate)
    # conf.optimizer      = torch.optim.Adam( params, lr=conf.learning_rate)
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
                # print(k)
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
            zi= item['idx']
            z = item['z']
            x = item['x']
            y = None

            print(x.shape,z.shape)

            loss = model.loss(zi,x,y,z).mean()
            loss_test_sum +=  float(loss.item())
            if tsi==conf.tsi_max:
                break

        model.train()
        dataset.train()
        model.zero_grad()
        for tri,item in enumerate(tqdm(conf.dataloader)):
            zi= item['idx']
            z = item['z']
            x = item['x']
            y = None
            # x    = item['english']
            # zi   = item['index']
            gradloss = model.grad_loss(zi,x,y,z).mean()
            loss =  model.loss(zi,x,y,z).mean()
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
