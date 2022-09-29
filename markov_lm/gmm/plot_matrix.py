import os
from glob import glob
from markov_lm.util_html import write_png_tag
from collections import defaultdict

import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
xs = []

# xs = glob('Checkpoints/*Convolve*.pkl')

# xs += glob('Checkpoints/*WithAttention*.pkl')
# xs += glob('Checkpoints/*MSE*.pkl')
# xs += glob('Checkpoints/*NEW2*.pkl')
# xs += glob('Checkpoints/*.pkl')
xs += glob('Checkpoints/*RefillModelRNNConvolveWithMixedEmissionMatrix-MixedTransition2-V2*.pkl')
# xs += glob('Checkpoints/*MixedEm*MSE*.pkl')

xs = sorted(xs)
ys = defaultdict(lambda:[])
HTML_FILE = __file__+'.html'
# MIN_YS= 0.3
MIN_YS = -0.5
E = 50

device = torch.device('cpu')
# import pdb; pdb.set_trace()
transition = torch.eye(E)

pkl = sorted(xs)[-1]
model = torch.load(pkl,map_location=device)
transition = model['model']['transition.weight']

def new_v(i): return torch.normal(0,1,size=(1,1,E))

pkl2 = 'train.py.RefillModelRNNConvolveWithMixedEmissionMatrix-MixedTransition2-V2-K1.pkl'
v2   = torch.load(pkl2,map_location=device)
xsa   = v2['xsa'].detach()
def new_v(i): return xsa[i,0,][None,None]

from markov_lm.Dataset.translation_dataset import RefillDataset
class _Dataset(RefillDataset):
    def get_sent(self, idx):
        sent = [self.english_vocab_reversed[idxx] for idxx in idx]
        return sent
    def get_token(self, idx):
        sent = [self.english_vocab[idxx] for idxx in idx]
        return sent
xdat = _Dataset(CUDA=False)
ws_tok = xdat.get_token([' '])[0]
embed = model['model']['embed.weight']
print(ws_tok)
model['model']['updater.weight'].T

with open(HTML_FILE+'.temp','w') as f:
    for i in range(10):
        fig,axs = plt.subplots(1,2,figsize=[10,5])
        v = new_v(i)
        # v =
        _norm = lambda x,dim=-1:x/(1.+x.std(-1,keepdims=True))
        v = _norm(v)
        vs = []
        for _ in range(30):
            v = v @ transition
            v = _norm(v)
            vs.append(v)
            # print(v.std(-1))
        vs = torch.cat(vs,dim=1)
        vs = xsa[i:i+1]
        vs = _norm(vs)
        energy = -((vs@transition)*vs.roll(-1,1)).mean(-1).mean(-1)[0]
        # import pdb; pdb.set_trace()
        # vs =

        # ax= plt.gca()
        axs[0].imshow(vs[0])
        axs[0].set_title(f'E={energy:.3f}')
        # f.write(write_png_tag(plt.gcf()))

        # import pdb; pdb.set_trace()
        vs2 = (embed[ws_tok][None] @ model['model']['updater.weight']) * vs[0]
        axs[1].imshow(vs2.sum(-1,keepdims=True))
        f.write(write_png_tag(plt.gcf()))

        plt.close()
    # import pdb; pdb.set_trace()
    # model = torch.load(pkl)

import shutil
shutil.move(HTML_FILE+".temp",HTML_FILE)
