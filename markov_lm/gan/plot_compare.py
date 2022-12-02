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


def main():


    f1= 'Checkpoints/S29-taskduie-mlm-shuffle1-AddModelBertInterface-D4-E100-K5-KE10-IPL1-DenseRelu11-Layernorm1-Dropout0.1-Gradnorm1-loglr-3.0-UseInputImage0-1i29_180_3.93135.pkl'
    f2 = 'Checkpoints/S29-taskduie-mlm-shuffle1-AddModelWithAttentionStacked-D4-E100-K5-KE10-IPL1-DenseRelu11-Layernorm1-Dropout0.5-Gradnorm1-loglr-3.0-UseInputImage0-1i29_155_4.11078.pkl'
    # f2 = ('Checkpoints/S29-taskduie-mlm-shuffle1-AddModelWithAttentionStacked-D12-E40-K5'
    # '-KE10-DenseRelu11-Layernorm1-Dropout0.5-Gradnorm1-loglr-3.0-UseInputImage0-1i29_155_*.pkl'
    # )

    #
    # with open(f1,'rb') as f:f1=pickle.load(f)
    # with open(f2,'rb') as f:f2=pickle.load(f)
    import time
    d0 = time.time()

    def abline(ax,slope, intercept):
        """Plot a line from slope and intercept"""
        # axes = plt.gca()
        x_vals = np.array(ax.get_xlim())
        y_vals = intercept + slope * x_vals
        ax.plot(x_vals, y_vals, 'r--')

    from pprint import pprint
    pprint(glob(f1))
    pprint(glob(f2))
    f1n = glob(f1)[-1]
    f2n = glob(f2)[-1]


    from markov_lm.add.train import ConfigPrototype
    from markov_lm.add.train import get_model_test_loss
    conf = ConfigPrototype(is_sorted=True)
    conf.CUDA = 1
    dev = torch.device('cpu' if not conf.CUDA else 'cuda:0')
    conf.device = dev


    conf.learning_rate = 0.001
    # conf.kernel_size = 15
    # conf.depth = 20
    # conf.embed_dim = 50


    def _add_callback(conf,model,MONITOR):
        # if isinstance(model, AddModelWithAttention):
        if 1:
            print('[attach]')
            def tok2char(vlist):
                vlist = list(map(conf.dataset.charset.__getitem__,vlist))
                return vlist

            def callback(outer,inner):
                (item,lptok,xexp) = inner
                state = conf.dataset.mode
                # state = 'train' if conf.dataset.state=='train'
                _tok = lambda x: '' if x ==0 else x
                for i in range(len(item['index'])):
                # for i in range(min(5,len(lptok))):
                    if int(item['index'][i]) not in MONITOR:
                        # print(i)/
                        continue
                    k = f'token_{state}_{i}_word_{model.alias}'
                    # print(item['masked'].shape)
                    vlist = item['masked'][i].detach().cpu().numpy().tolist()
                    conf.s[k] = [k,list(item['masked'].shape)] +   vlist

                    k = f'token_{state}_{i}_wordv_{model.alias}'

                    conf.s[k] = [k,list(item['masked'].shape)] +   tok2char(vlist)
                    # [[v] for v in item['masked'][i].detach().cpu().numpy().tolist()]
                    # list(map(_tok, (10*lptok[i].argmax(-1)).detach().cpu().numpy().tolist()))
                    k = f'token_{state}_{i}_unmask_{model.alias}'
                    conf.s[k]  = [k,''] + tok2char((item['unmasked'][i]).detach().cpu().numpy())
                    # torch.scatter(item['masked'],src=)
                    space= item['masked'][i]*0
                    v = (1*lptok[i].argmax(-1)).detach()
                    reseq = lambda v: torch.scatter(space,src=v,index=item['mask'][i],dim=0)
                    # vv =
                    k = f'token_{state}_{i}_pred_{model.alias}'
                    conf.s[k] = [k,''] +  tok2char(reseq((1*lptok[i].argmax(-1)).detach()).int())
                    # .cpu().numpy().tolist()))
                    k = f'token_{state}_{i}_exp_{model.alias}'
                    conf.s[k]  = [k,''] + tok2char(reseq((xexp[i])).detach().int())
                    k = f'token_{state}_{i}_ws_{model.alias}'
                    conf.s[k]  = [k]
                    # for k,v in conf.s.items():
                    #     print(k,len(v))

                    # conf._print(*[fws(xx,10) + '  ' for xx in  lptok[i].argmax(-1).detach().cpu().numpy()])
            model.callback_end = callback


    f1 = torch.load(f1n,map_location=conf.device)
    f2 = torch.load(f2n,map_location=conf.device)
    conf.graph_dim = f1['model']['embed.weight'].shape[0]
    conf.mask_token_idx=-1

    # if 1:
    #     conf.kernel_size = 5
    #     conf.depth = 12
    #     conf.embed_dim = 40
    #     conf.kernel_dim = 10
    #
    #     conf.use_dropout= 0.5
    #     conf.use_dense_relu = 11
    #     conf.use_layernorm = 1
    #     conf.use_gradnorm  = 1
    #     conf.use_input_image =0
    #     conf.step_size = 0.05
    #     conf.iter_per_layer= 100
    #     m = AddModelWithAttention(**vars(conf))
    #     m.load_state_dict(f1['model'])
    #     m.alias = 'm1'
    # m1 = m
    # if 1:
    #     conf.learning_rate = 0.001
    #     conf.kernel_size = 5
    #     conf.depth = 12
    #     conf.embed_dim = 40
    #     conf.n_choice = 0
    #     conf.use_dropout= 0.5
    #     conf.use_dense_relu = 13
    #     conf.use_layernorm = 1
    #     conf.use_gradnorm = 1
    #     conf.use_input_image = 0
    #     conf.step_size = 0.05
    #     conf.iter_per_layer= 100
    #     assert conf.is_model_set is False
    #     conf.is_model_set =True
    #     m = AddModelWithAttention(**vars(conf))
    #     m.load_state_dict(f2['model'])
    #     m.alias = 'm2'
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

    ### test dataset works
    conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)
    if conf.task=='duie-mlm':
        conf.callback_epoch_start = lambda epoch: conf.dataset.op_sample_mask(n_mask=10)


    # m = AddModelWithAttentionStacked(device=conf.device,config=f1['model_config'])
    STRICT_LOAD = '--nostrict' not in sys.argv

    m = f1['model_config'].to_model(conf.device,conf.dataset.charset).to(conf.device)
    m.load_state_dict(f1['model'],strict=STRICT_LOAD)
    m.alias = 'm1'
    m1 = m

    m = AddModelWithAttentionStacked(device=conf.device,config=f2['model_config']).to(conf.device)
    # m = f2['model_config'].to_model(conf.device,conf.dataset.charset)
    m.load_state_dict(f2['model'],strict=STRICT_LOAD)
    m.alias = 'm2'
    m2 = m


    # MONITOR = [3543, 3610, 3620, 3801, 4133]
    # MONITOR = [3446, 3523, 3575, 3672, 3681, 3926, 4093, 4193]
    MONITOR = [3375, 3382, 3395, 3397, 3407, 3417, 3423, 3430, 3432, 3435, 3438, 3442, 3453, 3461, 3468, 3473, 3474, 3484, 3490, 3500, 3503, 3504, 3516, 3518, 3524, 3535, 3536, 3537, 3540, 3543, 3551, 3557, 3563, 3566, 3568, 3581, 3583, 3591, 3624, 3628, 3635, 3641, 3647, 3650, 3653, 3666, 3667, 3669, 3670, 3674, 3692, 3699, 3700, 3701, 3718, 3719, 3725, 3733, 3735, 3736, 3750, 3753, 3758, 3761, 3763, 3773, 3774, 3776, 3778, 3783, 3802, 3807, 3810, 3818, 3823, 3830, 3841, 3849, 3852, 3860, 3883, 3903, 3909, 3917, 3918, 3920, 3928, 3930, 3945, 3961, 3963, 3967, 3982, 3985, 3989, 3990, 3998, 4001, 4017, 4023, 4026, 4029, 4031, 4034, 4044, 4056, 4063, 4067, 4072, 4075, 4089, 4090, 4100, 4101, 4128, 4132, 4133, 4142, 4158, 4178, 4180, 4182, 4192, 4203, 4208]
    def run_with_monitor(conf,m,MONITOR):
        torch.manual_seed(f2['curr_seed'][0])
        conf.callback_epoch_start(0)
        conf.model = m
        _add_callback(conf,conf.model,MONITOR)
        ls = get_model_test_loss(conf)
        return ls
    ls2= run_with_monitor(conf,m2,MONITOR)
    ls1= run_with_monitor(conf,m1,MONITOR)
    conf.callback_end(0,None,0)
        # # conf.callback_end(0,None,0)
        #
        # torch.manual_seed(f2['curr_seed'][0])
        # conf.callback_epoch_start(0)
        # conf.model = m1
        # _add_callback(conf,conf.model,MONITOR)
        # ls1 = get_model_test_loss(conf)
        # conf.callback_end(0,None,0)

    # import pdb; pdb.set_trace()

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

        ax = axs[1]
        x = v1[:,1]
        y = v2[:,1]
        ax.scatter(x,y,2)
        abline(ax,1,0)
        ax.set_xlabel(f1n[:40])
        ax.set_ylabel(f2n[:40])

        ax = axs[2]
        x = v1[:,1]
        y = v2[:,1]
        x,y = (x+y)/2., (y-x)/2.
        ax.scatter(x,y,2)
        abline(ax,0,0)
        thr = 0.85
        abline(ax,0,thr)
        abline(ax,0,-thr)
        ax.set_title(f'{v1[:,1].mean():.2f} {v2[:,1].mean():.2f} {(y>thr).sum()} {(-y>thr).sum()}')
        ax.set_xlabel(f1n[:40])
        ax.set_ylabel(f2n[:40])

        f.write(write_png_tag(plt.gcf()))
        f.write(f'<pre>{v2[y>thr]}</pre>')
        MONITOR1 = v2[y>thr][:,0].int().detach().numpy().tolist()
        MONITOR2 = v2[-y>thr][:,0].int().detach().numpy().tolist()
        f.write(f'Better f1:{f1n}<pre>{MONITOR1}</pre>')
        f.write(f'Better f2:{f2n}<pre>{MONITOR2}</pre>')

    for MONITOR in [MONITOR1,MONITOR2]:
        ls2= run_with_monitor(conf,m2,MONITOR)
        ls1= run_with_monitor(conf,m1,MONITOR)
        conf.callback_end(0,None,0)
        conf._print('*'*50)

if __name__=='__main__':
    main()
