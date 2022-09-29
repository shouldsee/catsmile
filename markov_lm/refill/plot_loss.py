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
xs += glob('Checkpoints/Refill*.pkl')
xs += glob('Checkpoints/*LSTM*MSE*.pkl')
# xs += glob('Checkpoints/*MixedEm*MSE*.pkl')

# xs += glob('Checkpoints/*MixedEm*-Mix*.pkl')
# xs += glob('Checkpoints/*nocore*-Mix*.pkl')
# xs += glob('Checkpoints/*LSTM*-Mix*.pkl')

# xs += glob('Checkpoints/*HiddenVector*MSE*.pkl')
# xs += glob('Checkpoints/*HiddenVector*LSTM*.pkl')
# xs += glob('Checkpoints/*DynamicFitting*.pkl')
# xs += glob('Checkpoints/*LSTM*.pkl')
xs += glob('Checkpoints/*DirectMixing*.pkl')
# xs += glob('Checkpoints/*Cluster*.pkl')
xs += glob('Checkpoints/*RNNMixture*.pkl')
xs += glob('Checkpoints/*MixedEmissionMatrix*.pkl')
# print(xs)
# xs += glob('Checkpoints/*WithBert*.pkl')
# xs += glob('Checkpoints/*WithLinearAttention*.pkl')
xs = sorted(xs)
ys = defaultdict(lambda:[])
HTML_FILE = __file__+'.html'
MIN_YS= -1.0
YMAX = 1.0
# MIN_YS = -0.5
# MIN_YS = 0.0
with open(HTML_FILE+'.temp','w') as f:
    plt.figure(figsize=[10,10])
    for x in xs:
        x = os.path.basename(x)
        x = x[:-len('.pkl')]
        assert x.count('_')==2,x
        base,epc,loss = x.split('_')
        loss = float(loss)
        epc = int(epc)
        ys[base].append((epc,loss))
    # print(base,epc,loss)

    for base,ss in sorted(ys.items()):
        ss = sorted(ss)
        for epc,loss in ss:
            # print(base,epc,loss)
            pass
        xs,ys = zip(*ss)
        if (ys[-1]<MIN_YS)*(MIN_YS>0) or (ys[-1]>-MIN_YS)*(MIN_YS<0):
            continue
        # if ys[-1]<MIN_YS:
        #     continue
        ys = np.array(ys)
        ys = (ys[:-2] + ys[1:-1] + ys[2:])/3.

        plt.plot(xs[:-2],ys,label=base)
    plt.xlim(10,1000)
    plt.ylim(0,2)
    plt.ylim(0,YMAX)

    plt.hlines(0.060,0,1000,colors='r',linestyles='--')
    # plt.ylim(0,1.5)
    plt.legend()
    f.write(write_png_tag(plt.gcf()))

import shutil
shutil.move(HTML_FILE+".temp",HTML_FILE)
