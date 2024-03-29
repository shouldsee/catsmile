
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

# from markov_lm.Dataset.translation_dataset import EnglishToGermanDataset
from markov_lm.Dataset.translation_dataset import RefillDataset

import os,sys

import numpy as np
from tqdm import tqdm
from torch import autograd

from markov_lm.util_html import write_png_tag

# import pandas as pd
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
from markov_lm.Model_Refill import RefillModelRNNAdditive
from markov_lm.Model_Refill import RefillModelRNNAdditiveDirect
from markov_lm.Model_Refill import RefillModelRNNGRU
from markov_lm.Model_Refill import RefillModelNGRAM
# Additive
from markov_lm.Model_Refill import RefillModelOld
from markov_lm.Model_Refill import RefillModelCopy
from markov_lm.Model_Refill import RefillModelCopyWithRandomFill
from markov_lm.Model_Refill import RefillModelRNNAdditiveWithPseudoSampling
from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectMixing
from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectMixingWithGate
# RefillModelRNNAdditiveDirectMixingWithGate
from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectMixingBidirectional

from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectSampling
from markov_lm.Model_Refill import RefillModelRNNAdditiveSweeping
from markov_lm.Model_Refill import RefillModelRNNAdditiveSweepingWithResidual
from markov_lm.Model_Refill import RefillModelMixtureRNNSweepingOldEmission
from markov_lm.Model_Refill import RefillModelMixtureRNNSweepingOldEmission2
from markov_lm.Model_Refill import RefillModelMixtureRNNSweepingNewEmission
from markov_lm.Model_Refill import RefillModelMixtureRNNSweepingOldEmissionDifferentTransition
from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectMixingWithAttention
from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectMixingWithKAttention
from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectMixingWithRegressionAttention

from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectMixingWithGate
from markov_lm.Model_Refill import RefillModelCrossRNNAdditiveSweeping
from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectEmission


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
    conf.embed_dim = 50
    conf.mixture_count = 8
    conf.state_count = 15
    conf.device =  torch.device('cuda:0' if CUDA else 'cpu')
    conf.num_epoch = 5000
    # model = MixtureOfHMM(graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device)

    conf.learning_rate = 0.001
    conf.SAVE_INTERVAL = 10
    conf.batch_size = 60
    conf.tsi_max = 10

    conf.dataset = dataset = RefillDataset(CUDA=CUDA)
    ### test dataset works
    (conf.dataset[range(5)])

    conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)
    # dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)
    # conf.model = model = ExtractionAndMarkovTemplateMatching(graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device)
    conf.optimizer_factory = None
     # torch.optim.RMSprop
    # conf.optimizer_factory = torch.optim.Ada
    conf.model = model = RefillModelOld(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
        state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device)

    '''
    RefillModelCopy:     420_0.9066   ### Just copying sequence
    RefillModelCopyWithRandomFill_400_1.07266  ### Copying sequence except at mask copying from set
    RefillModelRNNSwitch_240_0.078    ### static selector_q.weight
    RefillModelRNNSwitch_100_0.11912  ### static selector_q.weight
    RefillModelRNNSwitch_100_0.21843  ### dynamic select_q.weight
    '''
    # conf.model = model = RefillModelCopy(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    # conf.model = model = RefillModelRNNAdditive(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    # conf.model = model = RefillModelNGRAM(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    # conf.model = model = RefillModelRNNAdditiveDirect(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    # conf.model = model = RefillModelRNNAdditiveDirectMixing(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    # conf.model = model = RefillModelRNNAdditiveDirectMixingWithRegressionAttention(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    # # conf.model = model = RefillModelRNNAdditiveDirectMixingWithAttention(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    # #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # conf.model = model = RefillModelRNNAdditiveDirectMixingWithKAttention(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    from markov_lm.Model_Refill import RefillModelRNNConvolve
    # conf.embed_dim =15
    conf.model = model = RefillModelRNNConvolve(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
        state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    model.__class__.__name__ = 'RefillModelRNNConvolveWithHiddenVectorMSE'
    model.__class__.__name__ = 'RefillModelRNNConvolveWithHiddenVector5with-nocore-Mix'

    # from markov_lm.Model_Refill import RefillModelRNNConvolveWithAttention7
    # conf.model = model = RefillModelRNNConvolveWithAttention7(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # # model.__class__.__name__ = 'RefillModelRNNConvolveWithAttentionZero'
    # model.__class__.__name__ = 'RefillModelRNNConvolveWithAttentionAtt6MSE'
    # #
    # # from markov_lm.Model_Refill import RefillModelRNNConvolveWithAttentionSymm
    # # conf.model = model = RefillModelRNNConvolveWithAttentionSymm(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    # #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # # model.__class__.__name__ = 'RefillModelRNNConvolveWithAttentionSymm8'
    # #
    # from markov_lm.Model_Refill import RefillModelRNNConvolveWithLinearAttention
    # conf.model = model = RefillModelRNNConvolveWithLinearAttention(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ = 'RefillModelRNNConvolveWithLinearAttention34MSE'
    #
    # from markov_lm.Model_Refill import RefillModelRNNConvolveWithLSTM
    # conf.model = model = RefillModelRNNConvolveWithLSTM(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ = 'RefillModelRNNConvolveWithLSTM-MSE'
    # #
    #


    # from markov_lm.Model_Refill import RefillModelRNNClusterAndRotate
    # conf.model = model = RefillModelRNNClusterAndRotate(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ = 'RefillModelRNNClusterAndRotateLR'
    # # conf.learning_rate = 0.001
    #
    # from markov_lm.Model_Refill import RefillModelRNNDynamicCluster
    # conf.model = model = RefillModelRNNDynamicCluster(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ = 'RefillModelRNNDynamicCluster'
    #
    # from markov_lm.Model_Refill import RefillModelRNNBigramMixture
    # conf.model = model = RefillModelRNNBigramMixture(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ = 'RefillModelRNNBigramMixture'

    #
    # from markov_lm.Model_Refill import RefillModelRNNConvolveWithMixedEmissionMatrix
    # conf.mixture_count=1
    # conf.model = model = RefillModelRNNConvolveWithMixedEmissionMatrix(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ = f'RefillModelRNNConvolveWithMixedEmissionMatrix-MixedTransition2-V2-K{conf.mixture_count}'


    # from markov_lm.Model_Refill import RefillModelRNNMixture
    # conf.model = model = RefillModelRNNMixture(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ = 'RefillModelRNNMixture'



    # from markov_lm.Model_Refill import RefillModelRNNConvolveWithMixedEmission
    # conf.model = model = RefillModelRNNConvolveWithMixedEmission(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ += '-Mix1'
    #

    # from markov_lm.Model_Refill import RefillModelRNNConvolveWithMixedEmissionAndMixedTransition
    # conf.model = model = RefillModelRNNConvolveWithMixedEmissionAndMixedTransition(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ += '-MSE1'

    # #
    # from markov_lm.Model_Refill import RefillModelRNNConvolve
    # conf.model = model = RefillModelRNNConvolve(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ = 'RefillModelRNNConvolve-DynamicFittingSession2'
    #
    # # from markov_lm.Model_Refill import RefillModelRNNConvolveWithHiddenVectorDynamicFitting
    # # conf.model = model = RefillModelRNNConvolveWithHiddenVectorDynamicFitting(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    # #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # # model.__class__.__name__ = 'RefillModelRNNConvolveWithHiddenVectorDynamicFitting-K5E'
    # # # conf.learning_rate= 0.001
    # #
    # #
    # from markov_lm.Model_Refill import RefillModelRNNConvolveWithHiddenVectorDynamicFittingPaired
    # conf.model = model = RefillModelRNNConvolveWithHiddenVectorDynamicFittingPaired(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # # model.__class__.__name__ = 'RefillModelRNNConvolveWithHiddenVectorDynamicFittingPaired-K5E3WC2M'
    # # model.__class__.__name__ = 'RefillModelRNNConvolveWithHiddenVectorDynamicFittingPaired-K5E4M1'
    # model.__class__.__name__ = 'RefillModelRNNConvolveWithHiddenVectorDynamicFittingPaired-K5E4M2'
    # # conf.learning_rate= 0.001

    #
    # # from markov_lm.Model_Refill import RefillModelRNNConvolveWithLSTMMem
    # from markov_lm.Model_Refill import RefillModelRNNConvolveWithLSTMWithMemory
    # conf.model = model = RefillModelRNNConvolveWithLSTMWithMemory(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ = 'RefillModelRNNConvolveWithHiddenVector17WithLSTMMemory'
    #

    # from markov_lm.Model_Refill import RefillModelWithBert
    # conf.model = model = RefillModelWithBert(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ = 'RefillModelWithBert4'


    # from markov_lm.Model_Refill import RefillModelRNNConvolveWithLSTMWithMemory
    # conf.model = model = RefillModelRNNConvolveWithLSTMWithMemory(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ = 'RefillModelRNNConvolveWithLSTMWithMemory1'
    #
    # from markov_lm.Model_Refill import RefillModelRNNConvolveWithDynamicWeight
    # conf.model = model = RefillModelRNNConvolveWithDynamicWeight(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ = 'RefillModelRNNConvolveWithDynamicWeight6MSE'
    #
    #
    # from markov_lm.Model_Refill import RefillModelRNNConvolveWithHiddenVector
    # conf.model = model = RefillModelRNNConvolveWithHiddenVector(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ = 'RefillModelRNNConvolveWithHiddenVector17-K10'

    # conf.learning_rate = 0.0001
    # from markov_lm.Model_Refill import RefillModelRNNConvolveWithHiddenVectorDynamic
    # conf.model = model = RefillModelRNNConvolveWithHiddenVectorDynamic(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ = 'RefillModelRNNConvolveWithHiddenVector2Dynamic'



    # from markov_lm.Model_Refill import RefillModelRNNConvolveWithLinearAttention
    # conf.model = model = RefillModelRNNConvolveWithLinearAttention(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # model.__class__.__name__ = 'RefillModelRNNConvolveWithLinearAttention4'


    # # conf.learning_rate = 0.0001
    #
    # from markov_lm.Model_Refill import RefillModelRNNConvolveWithAttentionDist
    # conf.model = model = RefillModelRNNConvolveWithAttentionDist(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #

    # from markov_lm.Model_Refill import RefillModelRNNConvolveWithSelection
    # conf.model = model = RefillModelRNNConvolveWithSelection(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    # from markov_lm.Model_Refill import RefillModelRNNConvolveSmall
    # conf.model = model = RefillModelRNNConvolveSmall(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])

    # from markov_lm.Model_Refill import RefillModelRNNConvolveLowRank
    # conf.model = model = RefillModelRNNConvolveLowRank(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    #
    # from markov_lm.Model_Refill import RefillModelRNNConvolveHighRank
    # conf.model = model = RefillModelRNNConvolveHighRank(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    # from markov_lm.Model_Refill import RefillModelRNNConvolveHighRank2
    # conf.model = model = RefillModelRNNConvolveHighRank2(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])

# onvolve
    # from markov_lm.Model_Refill import RefillModelRNNConvolveGrad
    # conf.model = model = RefillModelRNNConvolveGrad(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    # from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectEmission
    # conf.model = model = RefillModelRNNAdditiveDirectEmission(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])

    #
    # conf.model = model = RefillModelRNNAdditiveDirectMixingWithGate(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])

    # conf.model = model = RefillModelRNNAdditiveDirectSampling(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    # #
    # from markov_lm.Model_Refill import RefillModelRNNAdditiveDirectMixingBidirectionalFixedEmission
    # conf.model = model = RefillModelRNNAdditiveDirectMixingBidirectional(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    # conf.model = model = RefillModelRNNAdditiveDirectMixingBidirectionalFixedEmission(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    #
    # conf.model = model = RefillModelRNNAdditiveDirectMixingWithGate(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])

    # from markov_lm.Model_Refill import RefillModelThreeWayRNNAdditiveSweeping2
    # conf.model = model = RefillModelRNNAdditiveDirectMixingWithAttention(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    # conf.model = model = RefillModelRNNAdditiveSweeping(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])

    # conf.model = model = RefillModelThreeWayRNNAdditiveSweeping2(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])

    #
    # conf.model = model = RefillModelRNNAdditiveDirectMixingWithGate(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])


    # conf.model = model = RefillModelRNNAdditiveSweeping(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])

    # conf.model = model = RefillModelRNNAdditiveSweepingWithResidual(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    # conf.mixture_count=3
    # conf.model= model = RefillModelMixtureRNNSweepingOldEmission(graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    #
    #
    # conf.mixture_count=3
    # conf.model= model = RefillModelMixtureRNNSweepingOldEmissionDifferentTransition(graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])

    # conf.mixture_count=3
    # conf.model= model = RefillModelMixtureRNNSweepingOldEmission2(graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # conf.model= model = RefillModelMixtureRNNSweepingNewEmission(graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # conf.model = model = RefillModelRNNAdditiveWithPseudoSampling(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    # conf.model = model = RefillModelRNNGRU(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])

    # conf.model = model = RefillModelCopyWithRandomFill(total_length=dataset.total_length(),min_len=dataset.min_len,graph_dim = dataset.english_vocab_len,mixture_count=conf.mixture_count,
    #     state_count=conf.state_count,embed_dim=conf.embed_dim,device=conf.device,mask_token_idx=dataset.english_vocab['<mask>'])
    ### 180_0.1007
    conf.model = model = model.to(conf.device)
    params = list(model.parameters())
    print(dict(model.named_parameters()).keys())
    #### using Adam with high learning_rate is catastrophic
    # conf.optimizer = torch.optim.Adagrad( params, lr=conf.learning_rate)
    conf.optimizer      = torch.optim.RMSprop( params, lr=conf.learning_rate)
    # conf.optimizer      = torch.optim.Adagrad( params, lr=conf.learning_rate)
    # conf.optimizer      = torch.optim.Adadelta( params, lr=conf.learning_rate)
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
        conf.dataset.op_extract_and_mask(n_mask)
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
            x    = item['english']
            zi   = item['index']
            y    = item['extracted']
            z    = item['masked']
            # print(zi.min())
            loss = model.loss(zi,x,y,z).mean()
            loss_test_sum +=  float(loss.item())
            if tsi==conf.tsi_max:
                break
                # print(tsi)



        model.train()
        dataset.train()
        model.zero_grad()
        for tri,item in enumerate(tqdm(conf.dataloader)):
            x    = item['english']
            zi   = item['index']
            y    = item['extracted']
            z    = item['masked']
            # print(zi.min())
            # z = model.encode(x)
            # y = model.decode(z)
            gradloss = model.grad_loss(zi,x,y,z).mean()
            loss =  model.loss(zi,x,y,z).mean()
            # loss.mean()
            loss_train_sum += float(loss.item())
            loss.backward()
            conf.optimizer.step()
            # break

        if hasattr(model,'get_hidden'):
            fs_list = []
            zs = []
            idxs = []
            sents = []
            xsa = []
            for tri,item in enumerate(tqdm(conf.dataloader)):
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
            torch.save(dict(zs=zs,idxs=idxs,fs=fs,xsa=xsa, sents=sents),__file__+'.'+model.__class__.__name__+'.pkl')


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
