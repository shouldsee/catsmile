import os
from glob import glob
from markov_lm.util_html import write_png_tag
from collections import defaultdict

import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

xs = []
# xs = glob('Checkpoints/*Convolve*.pkl')
# xs += glob('Checkpoints/*WithAttention*.pkl')
# xs += glob('Checkpoints/*MSE*.pkl')
# xs += glob('Checkpoints/*NEW2*.pkl')
# xs += glob('Checkpoints/*.pkl')
# xs += glob('Checkpoints/*LSTM*MSE*.pkl')
# # xs += glob('Checkpoints/*MixedEm*MSE*.pkl')
#
# # xs += glob('Checkpoints/*MixedEm*-Mix*.pkl')
# # xs += glob('Checkpoints/*nocore*-Mix*.pkl')
# # xs += glob('Checkpoints/*LSTM*-Mix*.pkl')
#
# # xs += glob('Checkpoints/*HiddenVector*MSE*.pkl')
# # xs += glob('Checkpoints/*HiddenVector*LSTM*.pkl')
# # xs += glob('Checkpoints/*DynamicFitting*.pkl')
# # xs += glob('Checkpoints/*LSTM*.pkl')
# xs += glob('Checkpoints/*DirectMixing*.pkl')
# # xs += glob('Checkpoints/*Cluster*.pkl')
# xs += glob('Checkpoints/*RNNMixture*.pkl')
# xs += glob('Checkpoints/*MixedEmissionMatrix*.pkl')
xs += glob('Checkpoints/*.pkl')
# print(xs)
# xs += glob('Checkpoints/*WithBert*.pkl')
# xs += glob('Checkpoints/*WithLinearAttention*.pkl')
if 1:
    # xs = [xx for xx in xs if 'taskrefill' in xx]
    # xs = [xx for xx in xs if 'taskner1' in xx]
    xs = [xx for xx in xs if 'duie-mlm' in xx]
    YLIM = (None,None)
    YLIM = (0,10)
# YLIM = (N)
if 0:
    YLIM = (None,None)
    # YLIM = (0,1)
# None,None)
    xs = [xx for xx in xs if '-I3-' not in xx]
    xs = [xx for xx in xs if '-I4-' not in xx]
    # xs = [xx for xx in xs if '-I5-' not in xx]
    xs = [xx for xx in xs if 'SimpleUpdate' not in xx]
    # xs = [xx for xx in xs if '-1I'  in xx]
    # xs = [xx for xx in xs if 'DenseRelu1'  in xx]
    xs = [
    xx for xx in xs
    if 0
    # or '-1I18'  in xx
    # or '-1I19' in xx
    # or '-1I20' in xx
    # or '-1I21' in xx

    # or '-1I22' in xx
    # or '-1I23' in xx
    # or '-1I24' in xx
    # or '-1I25' in xx  #### NIGHTMARE SEED

    # or '-1I26' in xx
    # or '-1I27' in xx
    # or '-1I28' in xx
    or '-1I29' in xx.upper()

    # or '-1I24' in xx

    ]
xs = sorted(xs)
ys = defaultdict(lambda:[])
HTML_FILE = __file__+'.html'
# MIN_YS= 0.3
MIN_YS = -200.
import pandas as pd
# MIN_YS = 0.0
with open(HTML_FILE+'.temp','w') as f:
    plt.figure(figsize=[20,8])
    for x in xs:
        x = os.path.basename(x)
        x = x[:-len('.pkl')]
        base,epc,loss = x.split('_')
        loss = float(loss)
        epc = int(epc)
        ys[base].append((epc,loss,x))
    # print(base,epc,loss)


    dfs = []
    for base,ss in sorted(ys.items()):
        ss = sorted(ss)
        # for epc,loss in ss:
        #     # print(base,epc,loss)
        #     pass
        xs,ys,fns = zip(*ss)
        if (ys[-1]<MIN_YS)*(MIN_YS>0) or (ys[-1]>-MIN_YS)*(MIN_YS<0):
            continue
        df = pd.DataFrame(dict(xs=xs,ys=ys),index=fns)
        dfs.append(df)
        # f.write(f'<pre>loss{ys[-1]:.3f}_{base}</pre>')
        # if ys[-1]<MIN_YS:
        #     continue
        ys = np.array(ys)
        # ys = (ys[:-2] + ys[1:-1] + ys[2:])/3.
        # xs = xs[:-2]
        ys = (0 + ys[:-1] + ys[1:])/2.
        xs = xs[:-1]
        if not len(ys):
            continue

        plt.plot(xs,ys,label=f'loss{ys[-1]:.3f}-{base}')
    plt.xlim(10,1000)
    # plt.ylim(0,2)
    plt.ylim(*YLIM)
    plt.hlines(0.060,0,1000,colors='r',linestyles='--')
    # plt.ylim(0,1.5)
    plt.legend()
    f.write(write_png_tag(plt.gcf()))
    df = pd.concat(dfs,axis=0)
    f.write(df.to_html())
    # f.write(f'<pre>{fns[-1]}</pre>')

import shutil
shutil.move(HTML_FILE+".temp",HTML_FILE)
