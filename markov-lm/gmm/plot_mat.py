
def seq_sample_noise(model,masked,mask):
    '''
    Sample random tokens to replace masked positions
    '''
    repl = torch.randint(model.config.graph_dim,size=mask.shape,device=model.device)
    # repl = 0 * item['mask'] + model.config.mask_token_idx
    seq = torch.scatter(masked,src=repl,index=mask,dim=1)
    return seq,0.


def get_lps(model,masked,mask, beta):
    out1  = model.forward(dict(masked = masked))[-1][-1]
    out1  = model.norm( model.project(out1))
    sel   = torch.gather(out1,index=mask[:,:,None].repeat((1,1,out1.size(-1))),dim=1)

    lps = sel@ model.norm(model.embed.weight).T*beta
    lps = lps.log_softmax(-1)
    return lps

def seq_sample_noise_by_vocab(model,masked,mask, beta):
    '''
    Sample random tokens to replace masked positions
    '''
    # repl = torch.randint(model.config.graph_dim,size=mask.shape,device=model.device)
    # embed = model.embed(sel)
    lps = get_lps(model, masked, mask, beta)
    ps  = lps.exp()
    # ps = ps.softmax(-1)
    # ps = (model.vocab(sel)*beta).softmax(-1)
    p_acc = ps.cumsum(-1)
    _,repl = (torch.rand(ps.shape[:2],device=model.device)[:,None]<p_acc).max(dim=-1)
    forward_lpsel = torch.gather(lps,index=repl.unsqueeze(-1),dim=-1)

    seq = torch.scatter(masked,src=repl,index=mask,dim=1)
    lps = get_lps(model, seq, mask, beta)
    ps  = lps.exp()
    # import pdb; pdb.set_trace()
    rev_lpsel = torch.gather(lps,index=torch.gather(masked,index=mask,dim=1).unsqueeze(-1),dim=-1)
    diff_lpsel = (rev_lpsel - forward_lpsel).sum(1)

    # rev_lpsel = tor
    return seq, diff_lpsel


def sample_from_score(model,unmasked,mask,n_step,beta,n_mut,PRINT_INTERVAL=100):
    '''
    run MHMC using a proposal distribution coupled with a score distribution

    For BERT
    Parallel sampling

    通过MCMC采样来了解模型究竟学到了什么样的句子.

    中国的MMMMM
    日本的MMMMM
    '''
    device=  model.device
    # PRINT_INTERVAL = 100
    # unmasked = item['unmasked']
    # mask = item['']
    # mask = item['mask'][:,:1]
    # mask = mask[:,:1]

    ### higher beta is more stringent
    ### lower beta means more thermo mistakes
    # beta = 10.0

    B,L = unmasked.shape
    if mask is None:
        mask_total = torch.arange(L,device=device)
    else:
        mask_total = mask
    for i in range(n_step):


        ### sample_from_random distribution might be very slow

        mask = mask_total[torch.randint(len(mask_total),(B,n_mut),device=device)]
        '''
        ### Simple Proposal Distribution. Select n_mut random position and replace with random words
        '''

        # sampled,diff_lpsel = seq_sample_noise(model, unmasked, mask)

        '''
        ### use the vocabulary table to aid selection of
        '''

        beta_inner = 0.01
        # beta_inner = beta(i)
        sampled,diff_lpsel = seq_sample_noise_by_vocab(model, unmasked, mask, beta_inner)

        # mask = mask_total[torch.randint(len(mask_total),(B,n_mut),device=device)]
        # repl = torch.randint(model.config.graph_dim,size=mask.shape,device=model.device)
        # sampled = seq_sample_noise(model, unmasked, mask)


        score0  = get_recovered_corrupted_seq(model,unmasked,mask,  K=0,method='score')
        score1  = get_recovered_corrupted_seq(model,sampled, mask,  K=0,method='score')
        xds = score1 - score0
        ### if score1 > score0 always accept
        p_acc = torch.exp(xds*beta(i) + diff_lpsel)
        acc = (torch.rand(xds.shape,device=device)<p_acc).int()
        unmasked = (1-acc) * unmasked + acc*sampled
        if PRINT_INTERVAL > 0 and i%PRINT_INTERVAL==0:
            print(f'[{i}] {score0.mean().item():.4f}')
    scores =  (1-acc)*score0 + acc*score1
    return unmasked,scores
    # item['sample_output'] = unmasked
    # item['sample_scores'] = (1-acc)*score0 + acc*score1
    # return item

#
    # import pdb; pdb.set_trace()
    pass

from markov_lm.loss_contrast_seq import get_recovered_corrupted_seq
from markov_lm.loss_contrast_seq import MyCharset
from pprint import pprint
from glob import glob
from markov_lm.conf_gan import ConfigPrototype
import torch
import sys
def main():
    '''
    We can verify whether a model is good at discriminating a mutated sentence from a real sentence
    '''

    conf = ConfigPrototype(__file__+'.log.html',is_sorted=True,field_width=[30,10,100,10],section_ender='')
    STRICT_LOAD = '--nostrict' not in sys.argv
    conf.CUDA = ('--cpu' not in sys.argv)
    dev = torch.device('cpu' if not conf.CUDA else 'cuda:0')
    conf.device = dev
    conf.learning_rate = 0.001
    # conf.kernel_size = 15
    # conf.depth = 20
    # conf.embed_dim = 50
    f1 = 'Checkpoints/sampleMethod-mixed-MAXL50-TS5-NM1-Mix1-maxPos48-SSK0.0-CK4--S28-taskduie-mlm-shuffle1-AddModelWithAttentionMixed-D1-E50-K11-KE50-IPL1-DenseRelu37-Layernorm1-Dropout0.0-Gradnorm0-loglr-3.0-UseInputImage0-1i28_180*'
    f2 = f1
    # f2 = 'Checkpoints/sampleMethod-mixed-MAXL50-TS50-NM-1-maxPos48-SSK0.5-CK4--S28-taskduie-mlm-shuffle1-AddModelWithAttentionStacked-D5-E50-K12-KE11-IPL1-DenseRelu13-Layernorm1-Dropout0.0-Gradnorm0-loglr-3.0-UseInputImage0-1i28_330_*'

    XLIM = (-1,5)
    YLIM = (-1,5)


    pprint(glob(f1))
    pprint(glob(f2))
    f1n = glob(f1)[-1]
    f2n = glob(f2)[-1]


    f1 = torch.load(f1n,map_location=conf.device)
    f2 = torch.load(f2n,map_location=conf.device)

    attv_weight = f1['model']['layers.0.layers.1.attv.weight']
    atto_weight = f1['model']['layers.0.layers.1.atto.weight']

    kernel_dim = f1['model_config'].kernel_dim
    E = embed_dim = f1['model_config'].embed_dim
    wx = atto_weight.reshape((E*5,-1))[:E*5,:kernel_dim]
    wy = attv_weight.reshape((-1,E))[:kernel_dim,:E]
    wz = (wx@wy).reshape((E,5,E))

    # wmat = (wz*100).int().transpose(1,0).reshape((5,-1))[:5,:10]
    print((wz*100).int().transpose(1,0)[(0,1,3,4),:10,:10])
    assert 0
    # ,:10]

    conf.graph_dim = f1['model']['embed.weight'].shape[0]
    conf.mask_token_idx=-1

    '''
    Load datasets
    '''
    from markov_lm.Dataset.DUIE_NER import DUIE_NER
    conf.batch_size = 60
    conf.shuffle    = shuffle = False
    conf.task       = 'duie-mlm'
    conf.dataset    = dataset = DUIE_NER(CUDA=conf.CUDA,task_mode=conf.task)
    conf.data_input_transform = lambda item: item
    conf.charset = MyCharset(conf.dataset.charset)

    def tok2char(vlist,dataset=conf.dataset):
        vlist = list(map(dataset.charset.__getitem__,vlist))
        return vlist

    def add_masked_words(conf,alias,masked,i,channel='word',transform=tok2char):
        #### adds masked seq as indexes
        k = f'token_{conf.dataset.mode}_{i}_{channel}_{alias}'
        # print(item['masked'].shape)
        vlist = masked.detach().cpu().numpy().tolist()
        conf.s[k] = [k,list(masked.shape)] +   transform(vlist)


    def loss(item,conf=conf):
        model = conf.model
        return 0.

    conf.loss = loss
    conf.grad_loss = loss

    ### character set is

    ### test dataset works
    conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)
    if conf.task=='duie-mlm':
        conf.callback_epoch_start = lambda epoch: conf.dataset.op_sample_mask(n_mask=10)

    m = f1['model_config'].to_model(conf.device,conf.dataset.charset).to(conf.device)
    m.load_state_dict(f1['model'],strict=STRICT_LOAD)
    m.alias = 'm1'
    m1 = m

    m = f2['model_config']
    # m.use_dropout = 0.
    m = m.to_model(conf.device,conf.dataset.charset).to(conf.device)
    m.load_state_dict(f2['model'],strict=STRICT_LOAD)
    m.alias = 'm2'
    m2 = m
    import pdb; pdb.set_trace()

    conf.dataset.test()
    conf.n_step = 1000
    conf.beta = lambda i: (i+1)/200.
    conf.n_mut = 1
    model = m1
    item = next(iter(conf.dataloader))
    # conf.dataset.char2idx('中')
    # import pdb; pdb.set_trace()
    # unmasked  = conf.charset.sent2tok('中国'')


    # unmasked = torch.randint(conf.charset.charset.__len__(),(conf.batch_size,49),device=conf.device).long()
    # unmasked[:,:2] = torch.tensor(conf.charset.sent2tok('中国'))

    unmasked = torch.randint(conf.charset.charset.__len__(),(conf.batch_size,49),device=conf.device).long()
    unmasked[:,:2] = torch.tensor(conf.charset.sent2tok('中国'))
    unmasked[:,2:] = conf.dataset.char2idx('[mask]')

    # unmasked = torch.tensor(unmasked,device=model.device).long()[None,:].repeat((conf.batch_size,1))
    mask_loc = torch.arange(2,unmasked.shape[1],device=model.device)

    unmasked = item['unmasked']
    mask_loc = torch.arange(2,unmasked.shape[1],device=model.device)

    # unmasked
    # mask_loc = None

    m1.beta = lambda i: (i+1)/30.
    m2.beta = lambda i: (i+1)/30.
    # m1.beta = lambda i: 1.
    # m2.beta = lambda i: 1.
    # (i+1)/30.

    for model in [m1,m2]:
        masked = unmasked.clone()
        masked[:,25:]  = conf.dataset.char2idx('[mask]')
        meseqs = masked.clone()
        for i in range(25,masked.shape[1]):

            # mask = torch.topk(unmasked ==  conf.dataset.char2idx('[mask]'),k=25,dim=1)
            xsa = model.forward(dict(masked=meseqs,mask=None))[-1][-1]
            # unmasked[:,25:] = model.vocab(xsa).argmax(-1)[:,25:]
            lps = model.vocab(xsa @ model.emittor.weight)

            ps  = lps.softmax(-1)
            # ps = ps.softmax(-1)
            # ps = (model.vocab(sel)*beta).softmax(-1)
            p_acc = ps.cumsum(-1)
            _,sel = (torch.rand(ps.shape[:2],device=model.device).unsqueeze(-1)<p_acc).max(dim=-1)
            # masked[]
    # forward_lpsel = torch.gather(lps,index=repl.unsqueeze(-1),dim=-1)

        # sel = model.vocab(xsa @ model.emittor.weight).argmax(-1)

            meseqs[:,i:i+1] = sel[:,i:i+1]
        if 0:
            meseqs, ret_scores =sample_from_score(model, unmasked, mask_loc, conf.n_step, model.beta, conf.n_mut)
        # meseqs  = ret_item['sample_output']
        for i in range(len(meseqs)):
            idx = i
            meseq = meseqs[i]
            add_masked_words(conf, model.alias, meseq, idx, channel='final',transform=conf.charset.tok2sent)
    for idx, meseq in enumerate(unmasked):
        add_masked_words(conf, 'm0', meseq, idx, channel='final',transform=conf.charset.tok2sent)
    for idx, meseq in enumerate(masked):
        add_masked_words(conf, 'mm', meseq, idx, channel='final',transform=conf.charset.tok2sent)
    conf.callback_end(0,None,0)
    # assert 0
    # from markov_lm.add.train import get_model_test_loss
    # def run_with_monitor(conf,m,MONITOR):
    #     torch.manual_seed(f2['curr_seed'][0])
    #     conf.callback_epoch_start(0)
    #     conf.model = m
    #     # _add_callback(conf,conf.model,MONITOR)
    #     ls = get_model_test_loss(conf)
    #     return ls


if __name__ == '__main__':
    main()
