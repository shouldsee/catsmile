import os
from glob import glob
from markov_lm.util_html import write_png_tag
from collections import defaultdict

import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

xs = glob('Checkpoints/*.pkl')
ys = defaultdict(lambda:[])
with open(__file__+'.html','w') as f:
    plt.figure(figsize=[10,10])
    for x in xs:
        x = os.path.basename(x)
        x = x[:-len('.pkl')]
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
        ys = np.array(ys)
        ys = (ys[:-2] + ys[1:-1] + ys[2:])/3.

        plt.plot(xs[:-2],ys,label=base)
    plt.xlim(10,700)
    plt.ylim(0,0.15)
    plt.legend()
    f.write(write_png_tag(plt.gcf()))
