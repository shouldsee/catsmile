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
    # print(list(dict(xx['model'])))
    for k,v in dict(xx['model']).items():     print(k);print(  (100*v.reshape(-1)[:10]).long())

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
    traj = model.sample_trajectory(model.latent[list(range(0,50))+list(range(800,850)),:],conf.dataset.min_len, );trajm = traj
    tokens = model.sample_tokens(model.latent[list(range(0,50))+list(range(800,850)),:],conf.dataset.min_len, );
    i = 0
    print()

    n_sample = 20
    xp = tokens.softmax(-1)[:,None].repeat((1,n_sample,1,1))
    xpc = xp.cumsum(-1)
    val,which = (torch.rand(xp.shape[:-1])[:,:,:,None]<xpc).max(-1)
    sample = which
    for xx in  sample[4][2]: print(repr(conf.dataset.english_vocab_reversed[xx]),end=':')
    for xx in  sample[4][3]: print(repr(conf.dataset.english_vocab_reversed[xx]),end=':')
    print()

    for xx in  tokens.argmax(-1)[25]: print(repr(conf.dataset.english_vocab_reversed[xx]),end=':')
    print()

    def pdr(i):
        for xx in  tokens.argmax(-1)[i]: print(repr(conf.dataset.english_vocab_reversed[xx]),end=':')
        print()
    def ptr(i):
        for xx in  conf.dataset.english_sentences_train[i]: print(repr(conf.dataset.english_vocab_reversed[xx]),end=':')
        print()

    # print((traj[0][:,:10]*10).long())
    i = 0; print((trajm[i][:,:10]*10).long())
    i = 49; print((trajm[i][:,:10]*10).long())
    i = 50; print((trajm[i][:,:10]*10).long())
    i = 52; print((trajm[i][:,:10]*10).long())
    i = 51; print((trajm[i][:,:10]*10).long())
    i = 1; print((trajm[i][:,:10]*10).long())
    i = 0; print((trajm[i][:,:10]*10).long())

    lat = model.latent
    lat[0]
    i=0; [ptr(i),pdr(i)]
    i=1; [ptr(i),pdr(i)]
    for i in range(10): [ptr(i),pdr(i)]

    print((model.latent[[0,1,801,802],:10]*100).long())
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
