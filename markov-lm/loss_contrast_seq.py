import torch
def get_high_score_seq(model,unmasked,mask,method='hardmax'):
    return get_recovered_corrupted_seq(model,unmasked,mask,method=method)
def get_resample_seq(model,unmasked,mask,method='softmax_sample'):
    return get_recovered_corrupted_seq(model,unmasked,mask,method=method)

def get_recovered_corrupted_seq(model,unmasked,mask,K=4,method='softmax_sample'):
    '''
    Mix target with corrupted samples and select the
    lowest energy config
    '''
    device = model.device
    seqs = torch.stack([
        unmasked]+[
        seq_sample_noise(model,unmasked,mask) for _ in range(K)]
        # item['mask']),
    ,dim=1)
    B,K,L = seqs.shape
    seqss=  seqs.reshape((seqs.shape[0]*seqs.shape[1],L))
    ss = get_score(model,seqss).reshape((B,K))

    if method=='softmax_sample':
        ssc = ss.softmax(-1).cumsum(-1)
        val,meidx = (torch.rand(ss.shape,device=device)<=ssc).max(-1)
        meidx = meidx[:,None]
    elif method == 'hardmax':
        meidx = ss.argmax(-1,keepdims=True)
    elif method=='score':
        return ss
    else:
        assert 0,method
    meseqs = seqs.gather(index=meidx[:,:,None].repeat((1,1,L)),dim=1)[:,0]
    return meseqs,meidx,ss



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
# def get_energy

def seq_sample_noise(model,masked,mask):
    '''
    Sample random tokens to replace masked positions
    '''
    repl = torch.randint(model.config.graph_dim,size=mask.shape,device=model.device)
    # repl = 0 * item['mask'] + model.config.mask_token_idx
    seq = torch.scatter(masked,src=repl,index=mask,dim=1)
    return seq

if 0:
    model = conf.model
    seq1 = item['unmasked']
    outer,inner = conf.model.forward(dict(masked = seq1 ))
    out1 = inner[-1]

    def get_energy(model,seq1):
        seq1_embed = model.norm(model.embed(seq1))
        # seq2 = model.norm(seq2)
        out1 = model.forward(dict(masked = seq1))[-1][-1]
        out1 = model.norm( model.project(out1))
        v1 = ((seq1_embed) * (out1)).mean(-1).mean(1)
        return v1


    repl = torch.randint(conf.model.config.graph_dim,size=item['mask'].shape,device=model.device)
    # repl = 0 * item['mask'] + model.config.mask_token_idx
    seq2 = torch.scatter(item['masked'],src=repl,index=item['mask'],dim=1)
    # out2 = conf.model.forward(dict(masked = seq2 ))[-1][-1]

    repl = torch.randint(conf.model.config.graph_dim,size=item['mask'].shape,device=model.device)
    seq3 = torch.scatter(item['masked'],src=repl,index=item['mask'],dim=1)

    repl = torch.randint(conf.model.config.graph_dim,size=item['mask'].shape,device=model.device)
    seq4 = torch.scatter(item['masked'],src=repl,index=item['mask'],dim=1)


    beta = 1.0
    v1 = get_energy(model,seq1)
    v2 = get_energy(model,seq2)
    vs = torch.stack([v1,v2,
    get_energy(model,seq3),
    get_energy(model,seq4),
    ],-1)
    lossVal = - vs.log_softmax(-1)[:,0]
    ## hard margin hinge loss
    # margin = 10.
    # lossVal = torch.relu(-v1 + margin + v2)

    ### log-loss soft hinge
    # lossVal = torch.log(1+torch.exp(-v1  + v2))
    # beta = 0.001
    # lossVal = torch.log(1+torch.exp(beta*(-v1  + v2)))
    '''
    #### log-loss is just log_softmax of the positive samples
    #### from-scratch model seems to be overfitting this means the
    the problems proposed by the generator is too easy
    to solve and does not require semantic knowledge to solve.
    Hence it's essential to build harder problems to
    '''

    # lossVal = -v1 + (conf.last_v1_mean - v1).square() +  v2 #+ 0.5*(seq3*out3).mean(-1).mean(1)
    # conf.last_v1_mean *= 0.8
    # conf.last_v1_mean += 0.2*v1.mean().detach()


    # lossVal = -(((seq1) * (out1)).mean(-1).mean(1)).sigmoid()  +  (1.0*((seq2)* (out2) ).mean(-1).mean(1)).sigmoid() #+ 0.5*(seq3*out3).mean(-1).mean(1)
    # lossVal = -(model.embed(seq1) * model.project(out1)).mean(-1).mean(1) + (model.embed(seq2)* model.project(out2) ).mean(-1).mean(1)
    # import pdb; pdb.set_trace()\


    # '''
    # now specify model
    # '''

    class AddModelWithAttention(AddModelWithAttention):
        '''
        Adds callback to record energy
        '''
        bi = 0
        def callback_step(self,inner,outer,s=conf.s):
            energy= outer[-2].mean()
            k = f'energy_{self.bi}'
            conf.init_s(k,energy)
            s[k].append(energy.item()*1000)

        def callback_end(self,outer,inner,s=conf.s):
            self.bi += 1
        def train(self,*a,**kw):
            super().train(*a,**kw)
            self.bi = 0
