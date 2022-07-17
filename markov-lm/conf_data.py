import torch
from dataclasses import dataclass

from markov_lm.c9007_util import tbuf_cls,fws,recur_detach
import shutil

import collections


class ConfigDataset(object):

    # @staticmethod
    def attach_task_to_conf(self,conf,task):
        conf.task = task
        CUDA= conf.CUDA
        if 0:
            pass
        elif conf.task == 'fashion-mnist-compress':
            self.attach_dataset_fashion_mnist_compress(conf)
        elif conf.task=='refill':
            self.attach_dataset_refill(conf)
        elif conf.task=='add':
            ### This is a random dataset !!!! init after random seed is set
            conf.dataset = dataset = ArithmeticTest(CUDA=CUDA)
            conf.data_input_transform = lambda item: dict(unmasked = item['unmasked'],masked=item['masked'], mask=item['mask']);
            conf.loss = lambda item:conf.model.loss(item)
            conf.grad_loss = lambda item:conf.model.loss(item)
            ### test dataset works
            conf.dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle)
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
