import torch
import sys
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from glob import glob
from markov_lm.conf_gan import ConfigPrototype
from markov_lm.util_html import write_png_tag
from markov_lm.gmm.train import attach_dataset_fashion_mnist_compress
STRICT_LOAD = '--nostrict' not in sys.argv

# from markov_lm.gmm.train import attach_dataset_fashion_mnist_compress

def init_conf(CUDA=0):
    conf = ConfigPrototype(__file__)
    conf.CUDA = CUDA
    conf.device = (torch.device('cuda:0' if conf.CUDA else 'cpu'))
    conf.batch_size = 150
    conf.shuffle = 0
    attach_dataset_fashion_mnist_compress(conf)
    return conf



def get_model(conf,f1):
    f1n = glob(f1)[-1]
    f1 = torch.load(f1n,map_location=conf.device)
    m = f1['model_config'].to_model(conf.device).to(conf.device)
    m.load_state_dict(f1['model'],strict=STRICT_LOAD)
    m.alias = 'm1'
    m1 = m
    return m


axi=-1
lx = 1
ly = 25
def plot_concat_mat(ax,x,lx,ly,ix=28,iy=28):
    xx = x[:lx*ly].reshape((lx,ly,ix,iy)).permute((0,2,1,3)).reshape((lx*ix,ly*iy)).cpu()
    res = ax.imshow(xx)
    return res

x = next(conf.dataloader)
y = m.encode(x)
import pdb; pdb.set_trace()

def main():
    # f1 = '-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size61-model-nameGLGAEGradEOL-beta0.01-n-step5-p-null0.0001-loglr-2.0_90'
    # f1 = '-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size61-model-nameGLGAEGradEOL-beta0.01-n-step5-p-null0.0001-loglr-2.0_'
    f1= '-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size61-model-nameGLGAEGradEOL-beta0.01-n-step5-p-null0.0-loglr-2.0'
    f1 += '_150_'
    # f1 = '-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size31-model-nameGLGAEGradEOL-beta0.01-n-step5-p-null0.0001-loglr-2.0_570_1925.72998.pkl'
    # f1 = '-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size60-model-nameGlobalMixtureEncoderDiffEOLGrad-beta1.0-n-step1-loglr-1.0_25_909.89697.pkl'
    f1 = 'Checkpoints/'+f1.strip() + '*'
    conf = init_conf()
    m = get_model(conf,f1)
    with open(__file__+'.html','w') as f:
        def plot_centroid_and_recons(f):

        fig,axs = plt.subplots(1,1,figsize=[12,4])
        ax =axs
        x = m.xa.cpu().detach()
        x = x.T
        plot_concat_mat(ax,x,3,10)
        # ax.imshow()
        f.write(write_png_tag(fig,'height=400'))
        #fig.savefig

        it = iter(conf.dataloader)
        xi = it.__next__()['images']
        xz = m.encode(xi)
        y = m.decode(xz,ret='rec')

        fig,axs = plt.subplots(1,1,figsize=[10,4])
        ax =axs
        x = xi
        x = x.cpu().detach()
        x1 = x
        # plot_concat_mat(ax,x,1,20)

        fig,axs = plt.subplots(1,1,figsize=[12,4])
        ax =axs
        x = y
        x = x.cpu().detach()
        x2 = x
        x = torch.cat([x1[:20],x2[:20]],dim=0)
        plot_concat_mat(ax,x,2,20)
        # ax.imshow()
        f.write(write_png_tag(fig,'height=400'))
        # f.write(write_png_tag(fig,))
        #fig.savefig


        fig,axs = plt.subplots(1,1,figsize=[12,6]); ax = axs
        x = m.xb.cpu().detach()
        x = x.T
        plot_concat_mat(ax,x,1,30,20,1)
        # ax.imshow()
        f.write(write_png_tag(fig,))

        print(xz.min().item(),xz.max().item())

        print(m.xb.min().item(), m.xb.max().item())


    import pdb; pdb.set_trace()
    #plt.plot(m)
if __name__=='__main__':
    main()
