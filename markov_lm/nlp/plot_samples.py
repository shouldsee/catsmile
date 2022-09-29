
from markov_lm.loss_contrast_seq import get_recovered_corrupted_seq
from markov_lm.loss_contrast_seq import MyCharset
from pprint import pprint
from glob import glob
from markov_lm.conf_gan import ConfigPrototype
import torch
import sys

import matplotlib.pyplot as plt
from markov_lm.util_html import write_png_tag
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
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size160-model-nameAutoEncoder-beta0.001-loglr-4.0_60_*'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size160-model-nameConv2DMixtureV1-beta0.001-loglr-1.0_90_884.35016*'
    # f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size80-model-nameGlobalMixtureEncoder-loglr-1.0_120_1009.56818*'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size160-model-nameGlobalMixtureEncoder-loglr-1.0_480_*'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameMixedDiffEncoder-beta0.01-loglr-2.0_90_*'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameMixedDiffEncoder-beta0.01-loglr-2.0_30_*'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameMixedDiffEncoder-beta0.01-loglr-2.0_390_*'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameMixedDiffEncoder-beta0.01-loglr-2.0_30_*'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameMixedDiffEncoder-beta1.0-loglr-2.0_30*'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameMixedDiffEncoder-beta0.01-loglr-2.0_90_*'
    # f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameMixedDiffEncoder-beta1.0-loglr-3.0_30*'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size60-model-nameMixedDiffEncoder-beta1.0-loglr-3.0_30*'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size10-model-nameMixedDiffEncoder-beta0.01-loglr-3.0_60*'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size10-model-nameMixedDiffEncoder-beta0.01-loglr-3.0_240*'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size50-model-nameGlobalGradMixtureEncoder-beta0.001-loglr-2.0_30_*'

    # f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size100-model-nameGlobalGradMixtureEncoder-beta0.001-loglr-1.0_60*'
    # f1   = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size100-model-nameGlobalGradMixtureEncoder-beta0.001-loglr-1.0_180_1728.83838.pkl'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameGlobalGradMixtureDiffEncoder-beta0.001-loglr-1.0_30_1650.01611.pkl'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size60-model-nameGlobalMixtureEncoderEOL-beta0.01-loglr-1.0_60*'

    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size60-model-nameGlobalMixtureEncoderEOL-beta0.1-loglr-1.0_60_1738.92273.pkl'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size60-model-nameGlobalMixtureEncoderLEOP-beta0.1-loglr-2.0_60_5260.04102.pkl'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size60-model-nameGlobalMixtureEncoderLEOP-beta0.1-loglr-1.0_30_4639.10449.pkl'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size60-model-nameGlobalMixtureEncoderLEOP-beta10.0-loglr-1.0_30*'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size200-model-nameGlobalMixtureEncoderEOLGrad-beta0.1-loglr-1.0_30_1535.10474.pkl'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size300-model-nameGlobalMixtureEncoderEOL-beta0.1-loglr-1.0_30_1574.79858.pkl'
    f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameGlobalMixtureEncoderDiffEOL-beta1.0-n-step1-loglr-4.0_60_*'

    # f1 = '-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameGlobalMixtureEncoderDiffEOL-beta1.0-n-step1-loglr-4.0_240*'
    # f1 = '-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameGlobalMixtureEncoderDiffEOL-beta1.0-n-step1-loglr-4.0_390*'
    # f1 = '-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameGlobalMixtureEncoderDiffEOLGrad-beta100.0-n-step5-loglr-4.0_30*'
    # f1 ='-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameGlobalMixtureEncoderDiffEOLGrad-beta1.0-n-step5-loglr-1.0_20_981.28198.pkl'
    f1 = '-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size60-model-nameGlobalMixtureEncoderDiffEOLGrad-beta1.0-n-step1-loglr-1.0_25_909.89697.pkl'
    f1 = 'Checkpoints/'+f1.strip()

    # f2 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size200-model-nameGlobalMixtureEncoderEOLGrad-beta0.1-loglr-1.0_60_*'
    # f2 = '-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size160-model-nameConv2DMixtureV1-beta0.001-loglr-2.0_540_*'
    f2 = '-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size160-model-nameGlobalMixtureEncoder-loglr-1.0_480_*'
    f2 = 'Checkpoints/'+f2

    # f2 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size100-model-nameGlobalMixtureEncoderEOLGrad-beta1.0-loglr-1.0_60*'

    # f2 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size60-model-nameGlobalMixtureEncoderEOL-beta0.001-loglr-1.0_60_1193.94812.pkl'

    # f2 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size50-model-nameGlobalGradMixtureEncoder-beta0.001-loglr-1.0_30_*'
    # f2 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size100-model-nameGlobalGradMixtureEncoder-beta0.001-loglr-1.0_240_1050.16431.pkl'
    # f2 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size50-model-nameGlobalGradMixtureEncoder-beta0.001-loglr-1.0_60_*'
    # f1 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size100-model-nameGlobalGradMixtureEncoder-beta0.001-loglr-1.0_60_*'

    # f2 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameMixedDiffEncoder-beta1.0-loglr-3.0_60*'
    # f2 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameMixedDiffEncoder-beta1.0-loglr-3.0_90*'


    # f2 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameMixedDiffEncoder-beta0.01-loglr-2.0_150_*'
    # f2 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameMixedDiffEncoder-beta0.01-loglr-2.0_180_*'
    # f2 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameMixedDiffEncoder-beta0.01-loglr-2.0_270_*'
    # f2 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size30-model-nameMixedDiffEncoder-beta0.01-loglr-2.0_240_*'

    # f2 = 'Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size160-model-nameConv2DMixtureV1-beta0.001-loglr-2.0_540_*'
    # f2 = f1
    # f1,f2 = f2,f1

    XLIM = (-1,5)
    YLIM = (-1,5)


    pprint(glob(f1))
    pprint(glob(f2))
    f1n = glob(f1)[-1]
    f2n = glob(f2)[-1]


    f1 = torch.load(f1n,map_location=conf.device)
    f2 = torch.load(f2n,map_location=conf.device)
    # conf.graph_dim = f1['model']['embed.weight'].shape[0]
    # conf.mask_token_idx=-1

    '''
    Load datasets
    '''
    from markov_lm.gmm.train import attach_dataset_fashion_mnist_compress
    # from markov_lm.Dataset.DUIE_NER import DUIE_NER
    conf.batch_size = 60
    conf.shuffle    = shuffle = False
    attach_dataset_fashion_mnist_compress(conf)

    def loss(item,conf=conf):
        model = conf.model
        return 0.

    conf.loss = loss
    conf.grad_loss = loss

    ### character set is

    ### test dataset works

    from markov_lm.Model_add import AddModelWithAttentionMixed
    m = f1['model_cls'](config=f1['model_config'],device=conf.device,).to(conf.device)
    # m = f1['model_config'].to_model(conf.device,conf.dataset.charset).to(conf.device)
    m.load_state_dict(f1['model'],strict=STRICT_LOAD)
    # m = m.layers[0]
    m.alias = 'm1'
    m1 = m

    # m.use_dropout = 0.
    m = f2['model_cls'](config=f2['model_config'],device=conf.device,).to(conf.device)
    # m = m.to_model(conf.device,conf.dataset.charset).to(conf.device)
    m.load_state_dict(f2['model'],strict=STRICT_LOAD)
    # m = m.layers[0]
    m.alias = 'm2'
    m2 = m

    conf.dataset.test()
    conf.n_step = 1000
    # conf.beta = lambda i: (i+1)/200.
    conf.n_mut = 1
    model = m1
    item = next(iter(conf.dataloader))
    # conf.dataset.char2idx('中')
    # import pdb; pdb.set_trace()
    # unmasked  = conf.charset.sent2tok('中国'')


    # unmasked = torch.randint(conf.charset.charset.__len__(),(conf.batch_size,49),device=conf.device).long()
    # unmasked[:,:2] = torch.tensor(conf.charset.sent2tok('中国'))

    # unmasked = torch.randint(conf.charset.charset.__len__(),(conf.batch_size,49),device=conf.device).long()
    # unmasked[:,:2] = torch.tensor(conf.charset.sent2tok('中国'))
    # unmasked[:,2:] = conf.dataset.char2idx('[mask]')
    #
    # # unmasked = torch.tensor(unmasked,device=model.device).long()[None,:].repeat((conf.batch_size,1))
    # mask_loc = torch.arange(2,unmasked.shape[1],device=model.device)

    images = item['images']
    # mask_loc = torch.arange(2,unmasked.shape[1],device=model.device)

    # unmasked
    # mask_loc = None

    # m1.beta = lambda i: (i+1)/30.
    # m2.beta = lambda i: (i+1)/30.
    # m1.beta = lambda i: 1.
    # m2.beta = lambda i: 1.
    # (i+1)/30.
    # with open(__file__)
    fig,axs = plt.subplots(3,1,figsize=[12,6])
    # ax = axs[0];
    axi=-1
    lx = 1
    ly = 25
    def plot_concat_mat(ax,x,lx,ly,ix=28,iy=28):
        xx = x[:lx*ly].reshape((lx,ly,ix,iy)).permute((0,2,1,3)).reshape((lx*ix,ly*iy)).cpu()
        ax.imshow(xx)

    axi+=1; ax =axs[axi]
    plot_concat_mat(ax,images.detach().cpu(),lx,ly)

    for model in [m1,m2]:

        x = images
        y = model.decode(model.encode(x),ret='rec')

        axi+=1; ax =axs[axi]
        plot_concat_mat(ax,y.detach().cpu(),lx,ly)
        ax.set_title(model.alias)

    conf.f.write(write_png_tag(fig))
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
