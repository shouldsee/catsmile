from glob import glob
import sys,os

import torch
from train import init_conf
import numpy as np
def main():
    conf = init_conf(CUDA=0,shuffle=False)

    epoch = sys.argv[sys.argv.index('--LOAD')+1]
    # epoch = int(epoch.)
    res = glob(os.path.join("Checkpoints",f"{conf.model.__class__.__name__}_{epoch}*.pkl"))
    assert len(res)==1,['Which one to choose?',res]

    fn = res[0]
    xx = torch.load(fn)
    conf.model.load_state_dict(xx['model'],strict='--nostrict' not in sys.argv)
    print(xx['test_losses'][-1])

    # for k,v in conf.model.named_parameters():
    #     print(k)
    #     print((v*100).long().reshape(-1)[:10])

    from tqdm import tqdm
    model = conf.model
    # model.sigma = 2
    # print(list(dict(xx['model'])))
    for k,v in dict(xx['model']).items():     print(k);print(  (100*v.reshape(-1)[:10]).long())

    self = model
    conf.dataset.op_extract_and_mask(4)

    print('[TRYING]')
    def idx2word(i): return conf.dataset.english_vocab_reversed[i]
    def seqs_to_printer(seqs_of_seqs,L=10,mapper=conf.dataset.english_vocab_reversed):
        def printer(i):
            for seqs in seqs_of_seqs:
                for xx in  seqs[i]:
                    v = mapper.__getitem__(int(xx))
                    v = (v + ' '*(L-len(v)))[:L]
                    print(v,end=':')
                    # print(repr(mapper.__getitem__(xx)),end=':')
                    # mapper(int(xx))
                print()

        return printer

    def wlen(v,L): v = (v + ' '*(L-len(v)))[:L]; return v
    ### hook callbacks
    def callback(s, xc, out='token'):
        (zi,x,y,z,fs) =s
        (i,sel,xz,xs) = xc
        if out=='traj':
            lptok = conf.model.vocab(fs[0:1,-1:])
        elif out == 'token':
            lptok = fs[0:1,i:i+1]
            sel = sel.exp()
        else:
            assert 0
        xz = z[:,i:i+1]
        # print(sel.shape)
        # print((sel[0,:,:10]*100).long())
        v = idx2word(conf.model.vocab(xz[0:1]).argmax(-1)[0])
        v2= idx2word(lptok.argmax(-1)[0])
        v3= idx2word((int(x[0:1,i:i+1])))
        print(wlen(v,7), end=' : ')
        print(wlen(v2,7),end=' : ')

        print(wlen(v3,7),end=' : ')
        # if v!='<mask>': print();return
        print((sel[0,:,:100]*100).long(),end=' : ')
        # print((xs[0,-1:,:10]*10).long(),end=':')
        print()
    conf.model.callback_init= lambda *a:print('-'*40)
    conf.model.callback_step = callback
    # def callback(s, inner ): print(wlen(idx2word(conf.model.vocab(inner[2][0:1]).argmax(-1)[0]),10),end=' : '); print((inner[1][0,:,:10]*100).long())
    conf.model.callback_step = callback

    outs = []
    xs   = []
    ts   = []
    ts = torch.tensor([],requires_grad=True)
    for i in range(2):
        if i==0:
            # conf.dataset.train()
            conf.dataset.test()
        else:
            conf.dataset.test()
        for item in (conf.dataloader):
            out = model.get_tokens(item['index'],item['english'],item['extracted'],item['masked'])
            xs.append(item['masked'])
            ts = torch.cat([ts,item['english']],dim=0)
            outs.append(out)

    xs  = torch.cat(xs,dim=0)
    xrs = torch.cat(outs,dim=0)
    # ptdr,ptr,pdr = getptdr
    printer = seqs_to_printer([ts,xrs.argmax(-1),xs],L=7)
    for i in range(20):printer(i);print()
    # for i in range(20): ptdr(i);print()
    import pdb; pdb.set_trace()


def obs():
    assert 0
    # print()
    # torch.cat([conf.dataset.english_sentences_train,conf.dataset.english_sentences_test],dim=0),conf.dataset.min_len*2, is_embed=False);
    # model.latent[range(1000),:]
    i = 0
    print()
    lat = model.latent
    lat    = lat / (0.00001 + lat.std(-1,keepdims=True)) *0.113

    # n_sample = 20
    # xp = tokens.softmax(-1)[:,None].repeat((1,n_sample,1,1))
    # xpc = xp.cumsum(-1)
    # val,which = (torch.rand(xp.shape[:-1])[:,:,:,None]<xpc).max(-1)
    # sample = which
    #
    # for xx in  sample[4][2]: print(repr(conf.dataset.english_vocab_reversed[xx]),end=':')
    # for xx in  sample[4][3]: print(repr(conf.dataset.english_vocab_reversed[xx]),end=':')
    # print()
    #
    # for xx in  tokens.argmax(-1)[25]: print(repr(conf.dataset.english_vocab_reversed[xx]),end=':')
    # print()


    # xtokens = model.sample_tokens(lat[0:5,:]+lat[5:10,:],conf.dataset.min_len*2, );
    # xptdr = getptdr(xtokens)


    # print((traj[0][:,:10]*10).long())
    # i = 0; print((trajm[i][:,:10]*10).long())
    # i = 49; print((trajm[i][:,:10]*10).long())
    # i = 50; print((trajm[i][:,:10]*10).long())
    # i = 52; print((trajm[i][:,:10]*10).long())
    # i = 51; print((trajm[i][:,:10]*10).long())
    # i = 1; print((trajm[i][:,:10]*10).long())
    # i = 0; print((trajm[i][:,:10]*10).long())

    lat = model.latent
    lat[0]
    print('[Fitted]')
    for i in range(5): ptdr(i);print()
    print('[NonFitted]')
    for i in range(50,55): ptdr(i);print()

    ## order by l2 loss to latent vectors
    xlat = model.latent.reshape((-1,model.state_count,model.embed_dim))
    xlat    = xlat / (0.00001 + xlat.std(-1,keepdims=True)) *0.113

    # [0]
    def xcnn(i,k=0):
        xx = (xlat[i:i+1] - xlat[:])[:,k:k+1].square().mean(-1)[:,0]
        xst = xx.argsort()
        for  si in xst[:5]: print(si,xx[si]);ptdr(si)
    def replace(x,a,b): x = np.vectorize('{0:03d}'.format)(x); x[x==a]=b; return x
    # def pa(i,k=20,model=model,replace=replace):print(replace((model.atts[i,:k]*100).long().numpy(),'000','   '))
    def pa(i,k=20,model=model,replace=replace):print()
    # print(replace((model.atts[i,:k]*100).long().numpy(),'000','   '))
    # pa(4)
    i = 1; k =4
    xcnn(5)
    xcnn(1)
    xcnn(0)

    assert 0
    print((model.latent[[0,1,801,802],:10]*100).long())
    # loss = torch.cat([model.log_prob(item['index'],item['english']) for item in it],dim=0)
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
'''
Epoch: 999
Training Loss: 1.3184270177568709
Testing Loss: 8.12166859886863

Epoch: 400
Training Loss: 1.39116393668311
Testing Loss: 6.793539957566694

'''
main()
