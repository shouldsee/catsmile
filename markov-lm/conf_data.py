import torch
from dataclasses import dataclass

from markov_lm.c9007_util import tbuf_cls,fws,recur_detach
import shutil

import collections


def get_multi30k_dataset(conf, fix_length, dataset_name,root=None):
    from markov_lm.Dataset.translation_dataset import DIR
    import torchtext
    import random
    random.seed(conf.rnd)
    if root is None:
        root = DIR
    ret = torchtext.datasets.Multi30k.download(root)

    get_field = lambda :torchtext.data.Field(lower=True, include_lengths=True, batch_first=True,fix_length=fix_length, init_token='<start>', eos_token='<end>')
    src = get_field()
    tgt = get_field()
    # tgt = torchtext.data.Field(lower=True, include_lengths=True, batch_first=True,fix_length=fix_length, )
    dataset = torchtext.datasets.Multi30k(root+'/multi30k/train',('.de','.en'),(src,tgt))
    # dataset = torchtext.datasets.Multi30k.splits(root+'/multi30k/train',('.de','.en'),(src,tgt))
    dataset_train,dataset_test  = dataset.split(0.8)
    # , rain='_train',test='_test')
    # torchtext.datasets.Multi30k.splits(path=None,root=root,exts=('.de','.en'),fields=(src,tgt),train='_train',test='_test')
    # for dataset in [dataset_train, dataset_test]:
    src.build_vocab(dataset, max_size=40000)
    tgt.build_vocab(dataset, max_size=40000)
    # import pdb; pdb.set_trace()
    conf.dataset = dataset
    dataset.offset_list = [dataset.fields['src'].vocab.__len__(),dataset.fields['trg'].vocab.__len__()]
    dataset.graph_dim = sum(dataset.offset_list)

    dataset.data_dim = fix_length
    dataset.mode='train'
    dataset.train = lambda : setattr(dataset,'mode','train')
    dataset.test = lambda : setattr(dataset,'mode','test')
    dataset.src_wordize = lambda v: src.vocab.itos.__getitem__(v)
    dataset.tgt_wordize = lambda v: tgt.vocab.itos.__getitem__(v-dataset.offset_list[0])


    class IterMaker(object):
        def __iter__(self):
            return self.my_iter()
        def __len__(self):
            if dataset.mode=='test':
                return dataset_test.__len__()//conf.batch_size
            else:
                return dataset_train.__len__()//conf.batch_size
        @classmethod
        def get_iter(cls,dataset_curr,dataset=dataset):
            it = torchtext.data.BucketIterator(dataset=dataset_curr, batch_size=conf.batch_size,shuffle=conf.shuffle, device=conf.device)
            # if dataset.mode=='train':
            for x in it:
                # import pdb; pdb.set_trace()
                src, src_len = x.src
                trg, trg_len = x.trg
                yield {'source': src,
                        'target':trg+dataset.offset_list[0],
                        'source_len':src_len,
                        'target_len':trg_len,
                        'has_start_token':1,
                        'index':None}
            yield None
        @classmethod
        def my_iter(cls):
            train_iter= cls.get_iter(dataset_train)
            test_iter = cls.get_iter(dataset_test)
            while True:
                if dataset.mode=='train':
                    v = next(train_iter)
                    # .__next__()
                elif dataset.mode=='test':
                    v = next(test_iter)
                if v is not None:
                    yield v
                else:
                    break


    conf.dataloader = dataloader = IterMaker()
    return dataset, dataloader


def get_ptb_dataset(conf, fix_length, dataset_name,root=None):
    from markov_lm.Dataset.translation_dataset import DIR
    import torchtext
    import random
    random.seed(conf.rnd)
    if root is None:
        root = DIR
    # cls = torchtext.datasets.Penn_Treebank
    cls = torchtext.datasets.PennTreebank
    ret = cls.download(root)

    # get_field = lambda :torchtext.data.Field(lower=True, include_lengths=True, batch_first=True,fix_length=fix_length, init_token='<start>', eos_token='<end>')
    get_field = lambda :torchtext.data.Field(lower=True, include_lengths=True, batch_first=True,fix_length=False, init_token='<start>', eos_token='<end>')
    trg = get_field()
    # tgt = get_field()
    # tgt = torchtext.data.Field(lower=True, include_lengths=True, batch_first=True,fix_length=fix_length, )
    # dataset = cls(root+'/multi30k/train',('.de','.en'),(src,tgt))
    # dataset_train = cls(ret+'ptb.train.txt',trg)
    dataset_train = cls(ret+'ptb.train.txt',trg)
    # dataset_train = cls.iters(path=ret+'ptb.train.txt',text_field=trg, batch_size=conf.batch_size, bptt_len=fix_length,
    #     device=conf.device)
        # , root='.data', vectors=None, **kwargs)
    dataset_test = cls(ret+'ptb.test.txt',trg)
    dataset = dataset_train
    # dataset = torchtext.datasets.Multi30k.splits(root+'/multi30k/train',('.de','.en'),(src,tgt))
    # dataset_train,dataset_test  = dataset.split(0.8)
    # , rain='_train',test='_test')
    # torchtext.datasets.Multi30k.splits(path=None,root=root,exts=('.de','.en'),fields=(src,tgt),train='_train',test='_test')
    # for dataset in [dataset_train, dataset_test]:
    # src.build_vocab(dataset, max_size=40000)
    trg.build_vocab(dataset_train,dataset_test, max_size=40000)
    # import pdb; pdb.set_trace()
    conf.dataset = dataset
    dataset.offset_list = [dataset.fields['text'].vocab.__len__()]
    dataset.graph_dim = sum(dataset.offset_list)

    dataset.data_dim = fix_length
    dataset.mode='train'
    dataset.train = lambda : setattr(dataset,'mode','train')
    dataset.test = lambda : setattr(dataset,'mode','test')
    # dataset.src_wordize = lambda v: src.vocab.itos.__getitem__(v)
    dataset.tgt_wordize = lambda v: trg.vocab.itos.__getitem__(v-dataset.offset_list[0])


    class IterMaker(object):
        def __iter__(self):
            return self.my_iter()
        def __len__(self):
            if dataset.mode=='test':
                return dataset_test.examples[0].text.__len__()//conf.batch_size//fix_length
            else:
                return dataset_train.examples[0].text.__len__()//conf.batch_size//fix_length
        @classmethod
        def get_iter(cls,dataset_curr,dataset=dataset):
            it = torchtext.data.BucketIterator(dataset=dataset_curr, batch_size=conf.batch_size,shuffle=conf.shuffle, device=conf.device)
            # if dataset.mode=='train':
            buf = next(iter(it))
            trg = buf.text[0][0]

            chunk_size = (conf.batch_size*fix_length)
            for xi in range( trg.__len__()//chunk_size ):
                yield {
                        'source': None,
                        # 'target':trg+dataset.offset_list[0],
                        'target': trg[xi*chunk_size : (xi+1)*chunk_size].reshape(conf.batch_size,fix_length),
                        # +dataset.offset_list[0],
                        'source_len':None,
                        'target_len':torch.ones([conf.batch_size],device=conf.device,dtype=torch.long)*fix_length,
                        # trg_len,
                        'has_start_token':1,
                        'index':None}
            # return
            yield None
            # # buf
            # # import pdb; pdb.set_trace()
            # for x in it:
            #     # import pdb; pdb.set_trace()
            #     # src, src_len = x.src
            #     trg, trg_len = x.text
            #     yield {
            #             'source':None,
            #             # 'target':trg+dataset.offset_list[0],
            #             'target':trg,
            #             # +dataset.offset_list[0],
            #             'source_len':None,
            #             'target_len':trg_len,
            #             'has_start_token':1,
            #             'index':None}
            # yield None

        @classmethod
        def my_iter(cls):
            train_iter= cls.get_iter(dataset_train)
            test_iter = cls.get_iter(dataset_test)
            while True:
                if dataset.mode=='train':
                    v = next(train_iter)
                    # .__next__()
                elif dataset.mode=='test':
                    v = next(test_iter)
                if v is not None:
                    yield v
                else:
                    break
    # assert 0, dataset.examples.__len__()

    conf.dataloader = dataloader = IterMaker()
    return dataset, dataloader



class ConfigDataset(object):

    # @staticmethod
    def attach_task_to_conf(self,conf,task):
        conf.task = task
        CUDA= conf.CUDA



        conf.data_input_transform = lambda item: item
        conf.loss = lambda item:conf.model.loss(item)
        conf.grad_loss = lambda item:conf.model.grad_loss(item)
        if 0:
            pass
        elif conf.task == 'fashion-mnist-compress':
            self.attach_dataset_fashion_mnist_compress(conf)
        elif conf.task=='refill':
            self.attach_dataset_refill(conf)
        elif conf.task == 'translate-mutli30k-de2en-l50':
            conf.dataset,conf.dataloader = get_multi30k_dataset(conf, 50,conf.task)
        elif conf.task == 'translate-multi30k-de2en-l20':
            conf.dataset,conf.dataloader = get_multi30k_dataset(conf, 20,conf.task)
        elif conf.task == 'translate-ptb-l20':
            conf.dataset,conf.dataloader = get_ptb_dataset(conf, 20,conf.task)
        elif conf.task == 'translate-ptb-l100':
            conf.dataset,conf.dataloader = get_ptb_dataset(conf, 100,conf.task)

                # my_iter()
                # get_iter()

        elif conf.task == 'translate-german-english':
            from markov_lm.Dataset.translation_dataset import GermanToEnglishDatasetRenamed
            conf.dataset = dataset = GermanToEnglishDatasetRenamed(CUDA=CUDA)
            conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=conf.shuffle)

        elif conf.task == 'translate-wmt14-de2en-5k':
            from markov_lm.Dataset.translation_dataset import WMT14
            conf.dataset = dataset = WMT14(B=5000,source='de',target='en',CUDA=CUDA)
            conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=conf.shuffle)

        elif conf.task == 'translate-wmt14-de2en-50k':
            from markov_lm.Dataset.translation_dataset import WMT14
            conf.dataset = dataset = WMT14(B=50000,source='de',target='en',CUDA=CUDA)
            conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=conf.shuffle)
        elif conf.task == 'translate-wmt14-de2en-20k':
            from markov_lm.Dataset.translation_dataset import WMT14
            conf.dataset = dataset = WMT14(B=20000,source='de',target='en',CUDA=CUDA)
            conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=conf.shuffle)
            # EnglishToGermanDatasetRenamed

        elif conf.task=='add':
            ### This is a random dataset !!!! init after random seed is set
            conf.dataset = dataset = ArithmeticTest(CUDA=CUDA)
            conf.data_input_transform = lambda item: dict(unmasked = item['unmasked'],masked=item['masked'], mask=item['mask']);
            conf.loss = lambda item:conf.model.loss(item)
            conf.grad_loss = lambda item:conf.model.loss(item)
            ### test dataset works
            conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=conf.shuffle)
            # dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)

        elif conf.task in 'ner1 duie-mlm duie-ce'.split():
            from markov_lm.Dataset.DUIE_NER import DUIE_NER
            ### This is a random dataset !!!! init after random seed is set
            conf.dataset = dataset = DUIE_NER(CUDA=CUDA,task_mode=conf.task,method='text',thin_sep=conf.thin_sep,max_len=conf.max_len)
            conf.data_input_transform = lambda item: item

            if conf.task == 'duie-ce':
                from markov_lm.loss_contrast_seq import get_recovered_corrupted_seq
                from markov_lm.add.plot_samples import sample_from_score
                # def loss(item,conf=conf):
                #     '''
                #     Naive corruption
                #     '''
                #     ss = get_recovered_corrupted_seq(conf.model,item['unmasked'],item['mask'],method='score',sample_method=conf.sample_method)
                #     lossVal = -ss.log_softmax(-1)[:,0]
                #     return lossVal

                def grad_loss(item,conf=conf):
                    '''
                    More difficult corruption that yields similar score
                    Instead of asking the model to discriminate between data and corrupted data
                    ask it to discriminate data and its extrapolation of data
                    so that at the end the model cannot tell between the data and the extrapolation.

                    At the end, the model should learns to extrapolate between data points and forget
                    everything that is not included in data.
                    '''
                    if conf.sample_method =='mixed':
                        lossVal = 0.
                        ss = get_recovered_corrupted_seq(conf.model,item['unmasked'],None,method='score',sample_method='shuffle',K=conf.contrast)
                        lossVal += -ss.log_softmax(-1)[:,0]
                        ss = get_recovered_corrupted_seq(conf.model,item['unmasked'],None,method='score',sample_method='simple',K=conf.contrast)
                        lossVal += -ss.log_softmax(-1)[:,0]
                    elif conf.sample_method =='nll':
                        lossVal = 0.
                        # ss = get_recovered_corrupted_seq(conf.model,item['unmasked'],item['mask'],method='score',sample_method='shuffle',K=conf.contrast)
                        # lossVal += -ss.log_softmax(-1)[:,0]
                        ss = get_recovered_corrupted_seq(conf.model,item['unmasked'],item['mask'],method='score',sample_method='simple',K=0)
                        # los
                        lossVal += -ss[:,0]
                    elif conf.sample_method=='mcmc':
                        '''
                        MCMC goes to unwanted modes
                        explicitly request MCMC to go back to samples
                        '''

                        corrupt = seq_sample_noise(model,item['unmasked'],item['mask'])
                        sampled,ss = sample_from_score(model,corrupt, None, n_step = 10, beta=lambda i: 10, n_mut=1,PRINT_INTERVAL=0)
                        item['unmask']

                        sampled,ss = sample_from_score(model,item['unmasked'], None, n_step = 10, beta=lambda i: 10, n_mut=1,PRINT_INTERVAL=0)
                        s1 = get_recovered_corrupted_seq(model,sampled, item['mask'], K=0,method='score')
                        s0 = get_recovered_corrupted_seq(model,item['unmasked'], item['mask'], K=0,method='score')
                        lossVal = -(s0-s1)[:,0]

                    else:
                        ss = get_recovered_corrupted_seq(conf.model,item['unmasked'],item['mask'],method='score',sample_method=conf.sample_method)
                        # ss = get_recovered_corrupted_seq(conf.model,item['unmasked'],item['mask'],method='score',sample_method='simple',K=24)
                        # conf.sample_method)
                        lossVal = -ss.log_softmax(-1)[:,0]
                    return lossVal

                loss = grad_loss
                conf.loss = loss
                conf.grad_loss = grad_loss
                conf.last_v1_mean = 0.
            else:
                conf.loss = lambda item:conf.model.loss(item)
                conf.grad_loss = lambda item:conf.model.loss(item)

            ### test dataset works
            conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)
            if conf.task in 'duie-mlm duie-ce':
                conf.callback_epoch_start = lambda epoch: conf.dataset.op_sample_mask(n_mask=conf.n_mask)
            # dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)
        else:
            raise NotImplementedError(conf.task)

    @staticmethod
    def attach_dataset_refill(conf):
        CUDA= conf.CUDA
        shuffle = conf.shuffle
        from markov_lm.Dataset.translation_dataset import RefillDataset
        conf.dataset = dataset =RefillDataset(CUDA=CUDA)
        conf.data_input_transform = lambda item:dict(index=item['index'], extracted=item['extracted'],unmasked = item['english'],masked=item['masked'], mask=item['mask'])
        conf.loss = lambda item:conf.model.loss(item)
        conf.grad_loss = lambda item:conf.model.grad_loss(item)
        conf.callback_epoch_start = lambda epoch: conf.dataset.op_extract_and_mask(n_mask=4)
        conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)

    @staticmethod
    def attach_dataset_duie(conf):
        conf.n_mask = 1
        conf.thin_sep = 5
        conf.max_len = 50

        return

    @staticmethod
    def attach_dataset_fashion_mnist_compress(conf):
        '''
        Image recover for fashion mnist
        '''
        conf.batch_size;


        from markov_lm.Dataset.fashion_mnist import fashion_mnist
        conf.dataset = fashion_mnist(CUDA=conf.CUDA)
        conf.dataloader = torch.utils.data.DataLoader(conf.dataset, batch_size=conf.batch_size, shuffle=conf.shuffle)
        conf.data_input_transform = lambda item:[item.__setitem__('epoch',conf.epoch),item][-1]
        conf.loss = lambda *a,**kw: conf.model.loss(*a,**kw)
        # conf.loss = lambda *a,**kw: conf.model.grad_loss(*a,**kw)
        conf.grad_loss = lambda *a,**kw: conf.model.grad_loss(*a,**kw)
        # conf.grad_loss = lambda *a,**kw: conf.model.loss(*a,**kw)
        # conf.grad_loss = lambda : conf.model.loss
        #dict(unmasked = item['english'],masked=item['masked'], mask=item['mask'])
        # import pdb; pdb.set_trace()
ConfigDatasetInstance = ConfigDataset()
