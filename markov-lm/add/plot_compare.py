# f1 = ('Checkpoints/S29-taskduie-mlm-shuffle1-AddModelWithAttention-D12-E40-K5'
# '-KE10-DenseRelu11-Layernorm1-Dropout0.5-Gradnorm1-loglr-3.0-UseInputImage0-1i29_180_4.24001.pkl'
# )
# f2 = ('Checkpoints/S29-taskduie-mlm-shuffle1-'
# 'AddModelWithAttention-D12-E40-K5-KE10-DenseRelu13'
# '-Layernorm1-Dropout0.5-Gradnorm1-loglr-3.0-UseInputImage0-1i29_180_4.52329.pkl')



# f1 = ('Checkpoints/S29-taskduie-mlm-shuffle1-AddModelWithAttentionStacked-D12-E40-K5'
# '-KE10-DenseRelu11-Layernorm1-Dropout0.5-Gradnorm1-loglr-3.0-UseInputImage0-1i29_155_*.pkl'
# )
# f2 = ('Checkpoints/S29-taskduie-mlm-shuffle1-'
# 'AddModelWithAttentionStacked-D12-E40-K5-KE10-DenseRelu13'
# '-Layernorm1-Dropout0.5-Gradnorm1-loglr-3.0-UseInputImage0-1i29_155_*.pkl')


import os
from glob import glob
from markov_lm.util_html import write_png_tag
from collections import defaultdict

import torch
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

# import cPickle as pickle
import pickle
from markov_lm.Model_add import AddModelWithAttention
from markov_lm.Model_add import AddModelWithAttentionStacked

import time
from pprint import pprint
from markov_lm.add.train import ConfigPrototype
from markov_lm.add.train import get_model_test_loss
# from markov_lm.add.train import get_recovered_corrupted_seq
from markov_lm.loss_contrast_seq import get_recovered_corrupted_seq,get_high_score_seq,get_resample_seq
# thr = 0.3
THRESHOLD = 0.3

def get_score(model,seq1):
    '''
    Score a given sequence
    '''
    seq1_embed = model.norm(model.embed(seq1))
    # seq2 = model.norm(seq2)
    out1 = model.forward(dict(masked = seq1))[-1][-1]
    out1 = model.norm( model.project(out1))
    v1 = ((seq1_embed) * (out1)).mean(-1).mean(1)
    return v1

def seq_sample_noise(model,masked,mask):
    '''
    Sample random tokens to replace masked positions
    '''
    repl = torch.randint(model.config.graph_dim,size=mask.shape,device=model.device)
    # repl = 0 * item['mask'] + model.config.mask_token_idx
    seq = torch.scatter(masked,src=repl,index=mask,dim=1)
    return seq

def inference_missing_word():
    '''
    Inference in energy model is hard because must compare all possibilites.
    '''
    pass



def main():
    '''
    We can verify whether a model is good at discriminating a mutated sentence from a real sentence
    '''

    f1 = 'Checkpoints/-S29-taskduie-ce-shuffle1-AddModelWithAttentionStacked-D8-E100-K50-KE11-IPL100-DenseRelu11-Layernorm1-Dropout0.0-Gradnorm0-loglr-4.0-UseInputImage0-1i29_75_0.00001.pkl'

    # f1 = 'Checkpoints/-S29-taskduie-ce-shuffle1-AddModelWithAttentionStacked-D4-E100-K5-KE11-IPL100-DenseRelu1-Layernorm1-Dropout0.49-Gradnorm1-loglr-4.0-UseInputImage0-1i29_310_0.05324*'
    # f1 = 'Checkpoints/-S29-taskduie-ce-shuffle1-AddModelWithAttentionStacked-D4-E100-K5-KE11-IPL100-DenseRelu1-Layernorm1-Dropout0.49-Gradnorm1-loglr-4.0-UseInputImage0-1i29_360*'
    f2 = 'Checkpoints/-S29-taskduie-ce-shuffle1-AddModelWithAttentionStacked-D4-E100-K5-KE11-IPL100-DenseRelu1-Layernorm1-Dropout0.49-Gradnorm1-loglr-4.0-UseInputImage0-1i29_110_0.24106*'
    # f2 = 'Checkpoints/-S29-taskduie-ce-shuffle1-AddModelWithAttentionStacked-D4-E100-K5-KE11-IPL100-DenseRelu1-Layernorm1-Dropout0.49-Gradnorm1-loglr-4.0-UseInputImage0-1i29_500_*'
    # f1= 'Checkpoints/S29-taskduie-mlm-shuffle1-AddModelBertInterface-D4-E100-K5-KE10-IPL1-DenseRelu11-Layernorm1-Dropout0.1-Gradnorm1-loglr-3.0-UseInputImage0-1i29_180_3.93135.pkl'
    # f2 = 'Checkpoints/S29-taskduie-mlm-shuffle1-AddModelWithAttentionStacked-D4-E100-K5-KE10-IPL1-DenseRelu11-Layernorm1-Dropout0.5-Gradnorm1-loglr-3.0-UseInputImage0-1i29_155_4.11078.pkl'
    # f2 = ('Checkpoints/S29-taskduie-mlm-shuffle1-AddModelWithAttentionStacked-D12-E40-K5'
    # '-KE10-DenseRelu11-Layernorm1-Dropout0.5-Gradnorm1-loglr-3.0-UseInputImage0-1i29_155_*.pkl'
    # )
    XLIM = (-1,5)
    YLIM = (-1,5)
    #
    # with open(f1,'rb') as f:f1=pickle.load(f)
    # with open(f2,'rb') as f:f2=pickle.load(f)
    # d0 = time.time()

    def abline(ax,slope, intercept):
        """Plot a line from slope and intercept"""
        # axes = plt.gca()
        x_vals = np.array(ax.get_xlim())
        y_vals = intercept + slope * x_vals
        ax.plot(x_vals, y_vals, 'r--')

    pprint(glob(f1))
    pprint(glob(f2))
    f1n = glob(f1)[-1]
    f2n = glob(f2)[-1]


    import sys
    conf = ConfigPrototype(__file__+'.log.html',is_sorted=True,field_width=[30,10,100,10],section_ender='')
    conf.CUDA = ('--cpu' not in sys.argv)
    dev = torch.device('cpu' if not conf.CUDA else 'cuda:0')
    conf.device = dev
    conf.learning_rate = 0.001
    # conf.kernel_size = 15
    # conf.depth = 20
    # conf.embed_dim = 50

    f1 = torch.load(f1n,map_location=conf.device)
    f2 = torch.load(f2n,map_location=conf.device)
    conf.graph_dim = f1['model']['embed.weight'].shape[0]
    conf.mask_token_idx=-1

    import sys
    # print(time.time()-d0)

    from markov_lm.Dataset.DUIE_NER import DUIE_NER
    conf.batch_size = 60
    conf.shuffle    = shuffle = False
    conf.task       = 'duie-mlm'
    conf.dataset    = dataset = DUIE_NER(CUDA=conf.CUDA,task_mode=conf.task)
    conf.data_input_transform = lambda item: item
    conf.loss       = lambda item:conf.model.loss(item)
    conf.grad_loss  = lambda item:conf.model.loss(item)

    def tok2char(vlist,dataset=conf.dataset):
        vlist = list(map(dataset.charset.__getitem__,vlist))
        return vlist

    def add_masked_words(conf,alias,masked,i,channel='word',transform=tok2char):
        #### adds masked seq as indexes
        k = f'token_{conf.dataset.mode}_{i}_{channel}_{alias}'
        # print(item['masked'].shape)
        vlist = masked.detach().cpu().numpy().tolist()
        conf.s[k] = [k,list(masked.shape)] +   transform(vlist)


    def loss(item,conf=conf):
        ss = get_recovered_corrupted_seq(conf.model,item['unmasked'],item['mask'],method='score')
        lossVal = -ss.log_softmax(-1)[:,0]
        return lossVal

    conf.loss = loss
    conf.grad_loss = loss

    ### test dataset works
    conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)
    if conf.task=='duie-mlm':
        conf.callback_epoch_start = lambda epoch: conf.dataset.op_sample_mask(n_mask=10)



    STRICT_LOAD = '--nostrict' not in sys.argv

    m = f1['model_config'].to_model(conf.device,conf.dataset.charset).to(conf.device)
    m.load_state_dict(f1['model'],strict=STRICT_LOAD)
    m.alias = 'm1'
    m1 = m

    m = f2['model_config'].to_model(conf.device,conf.dataset.charset).to(conf.device)
    m.load_state_dict(f2['model'],strict=STRICT_LOAD)
    m.alias = 'm2'
    m2 = m


    def run_with_monitor(conf,m,MONITOR):
        torch.manual_seed(f2['curr_seed'][0])
        conf.callback_epoch_start(0)
        conf.model = m
        # _add_callback(conf,conf.model,MONITOR)
        ls = get_model_test_loss(conf)
        return ls

    ls2= run_with_monitor(conf,m2,[])
    ls1= run_with_monitor(conf,m1,[])
    conf.callback_end(0,None,0)

    v1 = ls1.detach().cpu()
    v2 = ls2.detach().cpu()
    # v1 = f1['loss_mat'].cpu()
    # v2 = f2['loss_mat'].cpu()
    with open(__file__+'.html','w') as f:
        fig,axs = plt.subplots(1,3,figsize=[12,4])
        axs = axs.ravel()
        ax = axs[0]

        # ax.hist(v1[:,1].ravel(),alpha=0.5,label= f1)
        # ax.hist(v2[:,1].ravel(),alpha=0.5,label= f2)

        x = v1[:,1].numpy()
        y = v2[:,1].numpy()
        ax = axs[0]
        xmin,xmax = x.min(),x.max()
        ymin,ymax = y.min(),y.max()


        xbin = np.linspace(xmin,xmax,30)
        ybin = np.linspace(ymin,ymax,30)
        cts,xb,yb = np.histogram2d(x,y,bins=(xbin,ybin))
        ax.imshow(np.log2(1+cts))
        ax = axs[1]
        ax.scatter(x,y,2)
        abline(ax,1,0)
        ax.set_xlabel(f1n[:40])
        ax.set_ylabel(f2n[:40])

        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)

        ax = axs[2]
        x = v1[:,1]
        y = v2[:,1]
        x,y = (x+y)/2., (y-x)/2.
        ax.scatter(x,y,2)
        abline(ax,0,0)
        thr = THRESHOLD
        abline(ax,0,thr)
        abline(ax,0,-thr)
        ax.set_title(f'total:{len(v1)} {v1[:,1].mean():.2f} {v2[:,1].mean():.2f} {(y>thr).sum()} {(-y>thr).sum()}')
        ax.set_xlabel(f1n[:40])
        ax.set_ylabel(f2n[:40])
        f.write(write_png_tag(plt.gcf()))
        f.write(f'<pre>{v2[y>thr]}</pre>')
        MONITOR1 = v2[y>thr][:,0].int().detach().numpy().tolist()
        MONITOR2 = v2[-y>thr][:,0].int().detach().numpy().tolist()
        f.write(f'Better f1:{f1n}<pre>{MONITOR1}</pre>')
        f.write(f'Better f2:{f2n}<pre>{MONITOR2}</pre>')
    from tqdm import tqdm


    '''
    Plot sentence to a log file specified in ConfigPrototype()
    '''
    for MONITOR in [MONITOR1,MONITOR2]:
        for item in tqdm(conf.dataloader):
            # item =
            def tok2sent(seq):
                return [''.join(tok2char(seq))]
            idx = torch.isin(item['index'],torch.tensor(MONITOR))
            idxes  =item['index'][idx]
            for i in range(sum(idx)):
                ix = idxes[i]
                meseq = item['unmasked'][idx][i]
                add_masked_words(conf,'',meseq,ix,channel='unmaked',transform=tok2sent)

            if not len(idxes):
                continue
            for model in [m1,m2]:
                # meseqs,meidx,ss = get_resample_seq(model,item['unmasked'][idx],item['mask'][idx],method='hardmax')
                meseqs,meidx,ss = get_resample_seq(model,item['unmasked'][idx],item['mask'][idx],method='softmax_sample')
                # add_masked_words(conf,alias,masked,i,channel='word',transform=tok2char)
                # import pdb; pdb.set_trace()
                # print(model.alias,ss.log_softmax(-1)[:,0].mean().item(),meidx.ravel())
                print(model.alias,ss.softmax(-1)[:,0].mean().item(),meidx.ravel())
                # print((10*ss.log_softmax(-1)).int()[:3])

                for i in range(len(meseqs)):
                    ix = idxes[i]
                    meseq = meseqs[i]
                    add_masked_words(conf,model.alias,meseq,ix,channel='lowe',transform=tok2sent)
                    # add_masked_words()        ls2= run_with_monitor(conf,m2,MONITOR)
        ls1= run_with_monitor(conf,m1,MONITOR)


        # dat = conf.dataset[MONITOR[0]-conf.dataset.test_index_start]
        # import pdb; pdb.set_trace()
        # ls2= run_with_monitor(conf,m2,MONITOR)
        # ls1= run_with_monitor(conf,m1,MONITOR)
        '''
        Flush log ouputs
        '''
        conf.callback_end(0,None,0)
        conf._print('*'*50)

if __name__=='__main__':
    main()
