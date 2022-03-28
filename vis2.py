from glob import glob
import sys,os

import torch
from train import init_conf
def main():
    conf = init_conf(CUDA=0)

    epoch = sys.argv[sys.argv.index('--LOAD')+1]
    # epoch = int(epoch.)
    res = glob(os.path.join("Checkpoints",f"{conf.model.__class__.__name__}_{epoch}*.pkl"))
    assert len(res)==1,['Which one to choose?',res]

    fn = res[0]
    xx = torch.load(fn)
    conf.model.load_state_dict(xx['model'])
    print(xx['test_losses'][-1])

    from tqdm import tqdm
    model = conf.model
    print(list(dict(xx['model']).keys()))


    self = model

    dbg = 0
    it = list(tqdm(conf.dataloader))[:1]
    # testit
    # it = list(tqdm(conf.dataloader))[:1]

    n_sample=100
    # traj = torch.cat([model.sample_trajectory(model.latent[item['index'],:],conf.dataset.min_len, n_sample) for item in it],dim=0)

    item = it[0]
    # traj,lp = model.sample_trajectory(model.latent[item['index'],:],conf.dataset.min_len, n_sample);trajm = traj.mean(2)
    print('[TRYING]')
    traj = model.sample_trajectory(model.latent[list(range(200))+list(range(100)),:],conf.dataset.min_len, );trajm = traj
    # print((traj[0][:,:10]*10).long())
    i = 0; print((trajm[i][:,:10]*10).long())
    i = 1; print((trajm[i][:,:10]*10).long())
    lat = model.latent
    lat[0]
    print((model.latent[[0,1,801,802],:10]*10).long())
    assert 0
    loss = torch.cat([model.log_prob(item['index'],item['english']) for item in it],dim=0)
    # clup = torch.cat([model.log_prob(item['english']) for item in tqdm(conf.dataloader)],dim=0)
    # clup = torch.cat(clup,dim=0)[:,:,0]
    for i in range(5):
        print(traj[i,:,:].std())
        print((traj[i,:,:10]*10).long())
        print()
        # print((traj[i,:,:10]*10).long())
        # print((traj[i,:,:10]*10).long())
    # print((traj[900,:,:10]*10).long())

    x = clup.exp()
    print(f'Average Label:{x.mean(0)}')
    print(f'Max Label:{x.max(0)[0]}')
    print(clup.max(0)[0])
    # print(x.max(0)[0])
    clu = clup.argmax(dim=1)
    # clu[:,0]
    # xxx = xx[sel][0]
    idx = -4



    idx = 0
    cutoff = -2.5
    print(x.mean(0))
    print(x.max(0)[0])
    print(x.max(0)[0][idx])
    # sel = (x[:,idx]>0.5)
    # sel = (x[:,idx]>0.14)
    sel = (clup[:,idx]>cutoff)
    print(sel.sum())
    def mapper(xi,conf=conf):return conf.dataset.english_vocab_reversed[xi]
    for xxx in xx[sel][:10]: print(':'.join(map(mapper,xxx)))


    for xxx in xx[sel][:20]:
        res = [];
        for xi in xxx:  res.append(conf.dataset.english_vocab_reversed[xi])


    # def log_prob_cluster(self,x):
    if 1:
        item = next(iter(conf.dataloader))
        # self = model
        # x = item['english']
        # z = self.log_prob_chain(x)
        # z = z * x.shape[1]
        # logp = z.logsumexp(dim=2).log_softmax(dim=1)[:,:,0]
        # return logp

    import pdb; pdb.set_trace()
    print(fn)
main()
