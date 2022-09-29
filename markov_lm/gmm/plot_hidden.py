import torch
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from markov_lm.util_html import write_png_tag
import shutil

from markov_lm.Dataset.translation_dataset import RefillDataset


fn = 'train.py.RefillModelRNNDynamicCluster.pkl'
xd = torch.load(fn)

fs = xd['fs']
zs = xd['zs']

idxsort = zs.argsort()
zs = zs[idxsort]
fs = fs[idxsort]
sents = xd['sents'][idxsort]
xsa = xd['xsa'][idxsort]

# import pdb; pdb.set_trace()
fs = fs.cpu().detach()
# print(fs.shape)
fs = fs[:,0]
from sklearn.decomposition import PCA
model =PCA(n_components=3)
y = model.fit_transform(fs)
# y = model.fit_transform(xsa.cpu().detach().reshape((xsa.shape[0],-1)))
plt.scatter(y[:,0],y[:,1])

class _Dataset(RefillDataset):
    def get_sent(self, idx):
        sent = [self.english_vocab_reversed[idxx] for idxx in idx]
        return sent

xdat = _Dataset(CUDA=False)
# filter(fs[:])
# fs_CUT = 4
fs_CUT = 1.5
# for i,fss in enumerate(fs[:,1]):
for i,fss in enumerate(y[:,0]):
    if abs(fss)>abs(fs_CUT) and fss*fs_CUT>0:
        s = xdat.get_sent(sents[i])
        print(f'[{i}]'+' '.join(s))
# import pdb; pdb.set_trace()

with open(__file__+'.html.temp','w') as f:
    f.write(write_png_tag(plt.gcf()))
    plt.close('all')
    plt.figure()
    plt.scatter(fs[:,0],fs[:,1])
    plt.vlines(fs_CUT,-0.3,0.3)
    f.write(write_png_tag(plt.gcf()))

shutil.move(__file__+'.html.temp',__file__+'.html')
import pdb; pdb.set_trace()
# import pdb; pdb.set_trace()
