# from markov_lm.gmm.train import attach_dataset_fashion_mnist_compress
from markov_lm.conf_gan import ConfigPrototype
from markov_lm.conf_data import ConfigDatasetInstance
import matplotlib;matplotlib.use('agg')
import matplotlib.pyplot as plt

import sys
import torch

def init_conf(task, CUDA=0):
    conf = ConfigPrototype(__file__)
    conf.CUDA = CUDA
    conf.device = (torch.device('cuda:0' if conf.CUDA else 'cpu'))
    conf.batch_size = 150
    conf.shuffle = 0
    conf.rnd = 0
    ConfigDatasetInstance.attach_task_to_conf(conf, task)
    conf.dataset.train()
    conf.train_data =  next(iter(conf.dataloader))
    conf.dataset.test()
    conf.test_data =  next(iter(conf.dataloader))
    conf.STRICT_LOAD = '--strict' not in sys.argv
    return conf

def get_model(f1n,device,strict):
    # f1n = glob(f1)[-1]
    f1 = torch.load(f1n,map_location=device)
    m = f1['model_config'].to_model(device).to(device)
    m.load_state_dict(f1['model'],strict=strict)
    # STRICT_LOAD)
    m.alias = f1n
    f1['filename']=f1n
    m.meta = f1
    m1 = m
    return m


axi=-1
lx = 1
ly = 25
def plot_concat_mat(ax,x,lx,ly,ix,iy):
    xx = x[:lx*ly].reshape((lx,ly,ix,iy)).permute((0,2,1,3)).reshape((lx*ix,ly*iy)).cpu()
    res = ax.imshow(xx)
    for lyy in range(ly):
        ax.vlines(lyy*iy-0.5, 0-0.5,ix-0.5)
    return res


def plot_fashion_mnist_recon(model,  images, labels, **kw):
    fig,axs = plt.subplots(1,1,figsize=[12,4])
    # ax = axs[0];
    axi=-1
    lx = 2
    ly = 10
    ix,iy=28,28

    ax = axs
    x = images


    if model.config.model_name in 'AutoEncoderDDPMWithFixedAEAndLabelEmbedding'.split():
        lat = model.encode(dict(images=images,labels=labels))
    # if isinstance(model,AutoEncoderDDPMWithFixedAEAndLabelEmbedding):
    else:
        try:
            lat = model.encode(x,ret='rec')
        except Exception as e:
            lat = model.encode(x)

    y = model.decode(lat,ret='rec')
    plot_concat_mat(ax, torch.cat([images[:ly],y[:ly]],dim=0).detach().cpu(),lx,ly,ix,iy)
    mse = (y-x).square().mean()
    ax.set_title(f'{model.alias} \n mean_rec={mse}')
    return fig


def plot_fashion_mnist_perp(model, images, labels,**kw):
    fig,axs = plt.subplots(2,1,figsize=[12,7])
    # axs = torch.ravel( axs )
    # ax = axs[0];
    axi=-1
    lx = 2
    ly = 20
    ix,iy=28,28
    n_sample = 100

    ax = axs
    x = images

    if model.config.model_name in 'AutoEncoderDDPMWithFixedAEAndLabelEmbedding'.split():
        lat = model.encode(dict(images=images,labels=labels))
    # if isinstance(model,AutoEncoderDDPMWithFixedAEAndLabelEmbedding):
    else:
        try:
            lat = model.encode(x,ret='rec')
        except Exception as e:
            lat = model.encode(x)


    if isinstance(lat,tuple):
        lat,shape = lat
    else:
        shape = None
    z1,z2 = lat[0:2, None]
    mix = torch.linspace(0,1,n_sample)
    delta = mix[1]-mix[0]
    lat = z1 * mix[:,None] + z2 * (1-mix[:,None])
    if shape is not None:
        shape = (len(lat),) + shape[1:]
        # lat = lat.reshape((len(lat),shape[1:]))
        lat = lat,shape

    y = model.decode(lat,ret='rec')

    ax =axs[0]
    plot_concat_mat(ax, torch.cat([images[:ly],y[::(n_sample//ly)][:ly]],dim=0).detach().cpu(),lx,ly,ix,iy)
    # mse = (y-x).square().mean()
    mse = -1.
    ax.set_title(f'{model.alias}\n mean_rec={mse}')

    ax =axs[1]
    diff = (y[1:] - y[:-1]).square().mean(-1).sqrt() / delta.square().sqrt()
    diff = diff.detach().cpu()

    ax.plot(diff,'x--')
    # ax.set_ylim(-20000,20000)
    # ax.set_ylim(0,diff.max()*1.1)
    ax.set_ylim(0,400)
    ax.grid(1)
    ax.set_title(f'sqrt_mse={diff.mean()} sqrt_mse_std={diff.std()}')

    return fig


from markov_lm.util_html import write_png_tag
# def plot_translation_attention(model,  source, target, buf, dataset,**kw):
def plot_translation_attention(model,  source, target, target_len, source_len, buf, dataset,**kw):
    # , labels, **kw):
    fig,axs = plt.subplots(1,1,figsize=[12,4])
    # fig0=
    # ax = axs[0];
    axi=-1
    lx = 1
    ly = 5
    # ix,iy= 25,25
    ix,iy= 19,19
    # 40

    ax = axs
    # x = images

    output_logit, att_weight = model.forward(dict(source=source,target=target,source_len=source_len,target_len=target_len,**kw))
    mat = att_weight[:ly, :ix, :iy].detach().cpu()
    iy = min(iy, mat.shape[2])
    ix = min(ix, mat.shape[1])
    # assert 0,mat[:ly].shape
    plot_concat_mat(ax, mat,lx,ly,ix,iy)
    # mse = (y-x).square().mean()
    mse = 0
    ax.set_title(f'{model.alias} \n mean_rec={mse} \n {mat.shape} ({lx*ix,ly*iy})')
    buf.write(f'<pre>{target[0:3, :30]}</pre><br/>')
    buf.write(f'Source<pre>{source[0:3, :30]}</pre><br/>')
    buf.write(write_png_tag(fig))
    # plt.close(fig)

    for i in range(ly):
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        # i = 2
        zmat = att_weight[i,:ix,:iy].detach().cpu()
        #
        src= source[i]
        tgt = target[i]
        # import pdb; pdb.set_trace()

        # im =ax.matshow( zmat,vmin=-4,vmax=0)
        im =ax.matshow( zmat)
        # vmin=-4,vmax=0)
        plt.sca(ax)
        plt.colorbar(im)

        # xlab,wordize = tgt,dataset.tgt_vocab.wordize
        xlab,wordize = tgt[:iy], dataset.tgt_wordize
        plt.xticks(range(len(xlab)), [ wordize(x) for x in xlab],rotation= 45)

        xlab,wordize = src[:ix], dataset.src_wordize
        # xlab = src
        ax.grid(1)
        plt.yticks(range(len(xlab)), [ wordize(x) for x in xlab], rotation = 0)
        buf.write(write_png_tag(fig))
        plt.close(fig)

    model.log_param(buf, plt)
    x = getattr(getattr(model,'mapping',None),'weight',None)
    buf.write(x.__repr__())
    # .att.weight)
        # '.__repr__())
    return None


from markov_lm.util_html import write_png_tag
# def plot_translation_attention(model,  source, target, buf, dataset,**kw):
def plot_translation_attention(model,  source, target, target_len, source_len, buf, dataset,**kw):
    '''
    Needs a plot to indicate the loss at different positions

    Needs to compare a model that predict object from the subject. SVO decompostion. instead of just
    predict the next word. 
    '''
    # , labels, **kw):
    fig,axs = plt.subplots(1,1,figsize=[12,4])
    # fig0=
    # ax = axs[0];
    axi=-1
    lx = 1
    ly = 5
    # ix,iy= 25,25
    ix,iy= 19,19
    # 40

    ax = axs
    # x = images

    output_logit, att_weight = model.forward(dict(source=source,target=target,source_len=source_len,target_len=target_len,**kw))
    mat = att_weight[:ly, :ix, :iy].detach().cpu()
    iy = min(iy, mat.shape[2])
    ix = min(ix, mat.shape[1])
    # assert 0,mat[:ly].shape
    plot_concat_mat(ax, mat,lx,ly,ix,iy)
    # mse = (y-x).square().mean()
    mse = 0
    ax.set_title(f'{model.alias} \n mean_rec={mse} \n {mat.shape} ({lx*ix,ly*iy})')
    buf.write(f'<pre>{target[0:3, :30]}</pre><br/>')
    buf.write(f'Source<pre>{source[0:3, :30]}</pre><br/>')
    buf.write(write_png_tag(fig))
    # plt.close(fig)

    for i in range(ly):
        fig,ax = plt.subplots(1,1,figsize=[10,10])
        # i = 2
        zmat = att_weight[i,:ix,:iy].detach().cpu()
        #
        src= source[i]
        tgt = target[i]
        # import pdb; pdb.set_trace()

        # im =ax.matshow( zmat,vmin=-4,vmax=0)
        im =ax.matshow( zmat)
        # vmin=-4,vmax=0)
        plt.sca(ax)
        plt.colorbar(im)

        # xlab,wordize = tgt,dataset.tgt_vocab.wordize
        xlab,wordize = tgt[:iy], dataset.tgt_wordize
        plt.xticks(range(len(xlab)), [ wordize(x) for x in xlab],rotation= 45)

        xlab,wordize = src[:ix], dataset.src_wordize
        # xlab = src
        ax.grid(1)
        plt.yticks(range(len(xlab)), [ wordize(x) for x in xlab], rotation = 0)
        buf.write(write_png_tag(fig))
        plt.close(fig)

    model.log_param(buf, plt)
    x = getattr(getattr(model,'mapping',None),'weight',None)
    buf.write(x.__repr__())
    # .att.weight)
        # '.__repr__())
    return None
