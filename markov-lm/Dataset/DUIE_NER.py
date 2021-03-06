# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import time


FILES = '''
data/duie_dev.json/duie_dev.json
data/duie_test2.json/duie_test2.json
data/duie_train.json/duie_train.json
data/duie_sample.json/duie_sample.json
data/duie_schema/duie_schema.json
'''.strip().splitlines()


['data/duie_dev.json/duie_dev.json',       10524275]
['data/duie_sample.json/duie_sample.json', 5055097]
['data/duie_schema/duie_schema.json',      4559]
['data/duie_test2.json/duie_test2.json',   23336504]
['data/duie_train.json/duie_train.json',   86803162]

{'spo_list': [{'object': {'@value': '顾漫'},
               'object_type': {'@value': '人物'},
               'predicate': '作者',
               'subject': '何以笙箫默',
               'subject_type': '图书作品'},
              {'object': {'@value': '匪我思存'},
               'object_type': {'@value': '人物'},
               'predicate': '作者',
               'subject': '来不及说我爱你',
               'subject_type': '图书作品'}],
 'text': '《步步惊心》改编自著名作家桐华的同名清穿小说《甄嬛传》改编自流潋紫所著的同名小说电视剧《何以笙箫默》改编自顾漫同名小说《花千骨》改编自fresh果果同名小说《裸婚时代》是月影兰析创作的一部情感小说《琅琊榜》是根据海宴同名网络小说改编电视剧《宫锁心玉》，又名《宫》《雪豹》，该剧改编自网络小说《特战先驱》《我是特种兵》由红遍网络的小说《最后一颗子弹留给我》改编电视剧《来不及说我爱你》改编自匪我思存同名小说《来不及说我爱你》'}


{'spo_list': [{'object': {'@value': '孙鹏'},
               'object_type': {'@value': '人物'},
               'predicate': '父亲',
               'subject': '孙安佐',
               'subject_type': '人物'},
              ],
 'text': '吴宗宪遭服务生种族歧视, 他气呛: '
         '我买下美国都行!艺人狄莺与孙鹏18岁的独子孙安佐赴美国读高中，没想到短短不到半年竟闹出校园安全事件被捕，因为美国正处于校园枪击案频传的敏感时机，加上国外种族歧视严重，外界对于孙安佐的情况感到不乐观 '
         '吴宗宪今（30）日录影前谈到美国民情，直言国外种族歧视严重，他甚至还被一名墨西哥裔的服务生看不起，让吴宗宪气到喊：「我是吃不起是不是'}

{'object_type': {'@value': '学校'}, 'predicate': '毕业院校', 'subject_type': '人物'}

{'text': '歌曲《墨写你的美》是由歌手冷漠演唱的一首歌曲'}

{'spo_list': [{'object': {'@value': '冰火未央'},
               'object_type': {'@value': '人物'},
               'predicate': '作者',
               'subject': '邪少兵王',
               'subject_type': '图书作品'}],
 'text': '《邪少兵王》是冰火未央写的网络小说连载于旗峰天下'}
import torch
import re
import json
from pprint import pprint
import hashlib
import shutil
DIR = os.path.dirname(os.path.realpath(__file__))


import sys


class DUIE_NER(torch.utils.data.Dataset):

    def __init__(self,CUDA=False,max_len=50,thin_sep=50,task_mode = 'mlm', method='text'):
        super().__init__()
        self.max_len = max_len
        self.thin_sep = thin_sep
        self.device = torch.device('cuda:0' if CUDA else 'cpu')
        xd  = self.preprocess()
        xd['charset'].insert(0,'[sep]')
        xd['charset'].insert(1,'[ent]')
        xd['charset'].insert(2,'[mask]')
        self.char_size = xd['charset'].__len__()
        self.vocab_size = xd['charset'].__len__() + xd['entity_type'].__len__()
        xd['char2idx'] = {v:k for k,v in enumerate(xd['charset'])}
        xd['ent2idx'] = {v:k+self.char_size for k,v in enumerate(xd['entity_type'])}
        self.idx2char  = xd['charset'].__getitem__
        self.char2idx  = xd['char2idx'].__getitem__
        self.ent2idx   = xd['ent2idx'].__getitem__
        xd['charset'].extend(xd['entity_type'])

        self.task_mode = task_mode
        # self.char2idx = xd['char2idx']
        # xd['char2idx'].update(xd['ent2idx'])
        # self.vocab2idx = xd['ent2idx'].__getitem__
        xx = xd['data'][0]
        xxx= xx['entity_list'][0]
        # thin_sep = 5



        self.src, self.target,self.length = self.get_cached_data(xd,max_len,thin_sep, method)

        B,L = self.src.shape[:2]

        ### z is masked sequence
        ### x is the target
        self.test_ratio  = 0.2
        self.test_index_start = int(B*(1-self.test_ratio))

        self.total_length = B
        # print(self.x.shape, self.y.shape)
        self.xd = xd
        self.charset = xd['charset']
        self.graph_dim      = len(self.xd['charset'])
        self.mask_token_idx = self.char2idx('[mask]')

        if self.task_mode == 'duie-mlm':
            self.op_sample_mask(10)
        else:
            self.masked = self.src
            self.unmasked = self.target
            self.mask = torch.arange(self.src.shape[1],device=self.device)[None,:].repeat((B,1))
            pass

        self.train()

    def toks2chars(self,vlist):
        vlist = [self.charset[vv]for vv in vlist]
        return vlist

    def op_sample_mask(self,n_mask):
        B,L = self.src.shape[:2]
        # n_mask = 10
        v = self.src[:,:-1].long()

        if n_mask == -1:
            n_mask = torch.randint(1,L-1,(1,))[0].item()

        # self.mask = m = torch.randint(L-1,(B,n_mask),device=self.device).long().sort(dim=1)[0]
        device= self.device
        self.length = self.length.clip(0,L-1,)
        bar = torch.arange(1,L+1,device=device).unsqueeze(0).repeat((B,1)).unsqueeze(-1)/self.length[:,None,None]
        self.mask = m = (torch.rand((B,1,n_mask),device=self.device)<bar).max(1)[1]
        # self.mask = m = torch.randint(self.length,(B,n_mask),device=self.device).long().sort(dim=1)[0]

        # self.m = m = (1+((v==vt).max(1)[1]))[:,None].long()
        self.masked = torch.scatter(v, src=torch.ones(v.shape,device=self.device).long()*self.mask_token_idx, index=m,dim=1).long()
        self.unmasked = v
        self.summer = torch.gather(self.unmasked,index=self.mask,dim=1)!=self.char2idx('[sep]')
        # item['summer']  = torch.gather(item['unmasked'],index=item['mask'],dim=1)!=self.char2idx('[sep]')

        # print(m.shape,self.masked.shape,v.shape)
        # import pdb; pdb.set_trace()
    def test(self):
        self.mode = "test"
    def train(self):
        self.mode = "train"

    def __len__(self):
        if self.mode=='train':
            return self.test_index_start
        else:
            return self.src.__len__() - self.test_index_start

    def __getitem__(self,idx):
        if self.mode=='train':
            idx = idx
        else:
            idx = self.test_index_start+idx
        item = dict(
            index       = idx,
            unmasked    = self.unmasked[idx],
            masked      = self.masked[idx],
            mask        = self.mask[idx],
            summer = self.summer[idx],
            )

        return item

        # import pdb; pdb.set_trace()
        ### simple binary classificaiton model without instance boundary
    def get_cached_data(self,xd,max_len,thin_sep,method):
        # self.max_len = max_len = 50
        OUTPUT_PICKLE = __file__+f'.get_cached_data.{max_len}.{thin_sep}.{method}.pkl'
        if os.path.exists(OUTPUT_PICKLE):
            x,y,l = torch.load(OUTPUT_PICKLE,map_location=self.device)
        else:
            outv = []
            outy = []
            outl = []
            for xx in xd['data'][::thin_sep]:
                if method=='concat':
                    for xxx in xx['entity_list']:
                        entidx= self.ent2idx(xxx['type'])
                        v = list(map(self.char2idx, xx['text'][:max_len-1])) + [self.char2idx('[sep]')]*(max_len-1-len(xx['text'])) +  [self.ent2idx(xxx['type'])]
                        y = [0]*max_len
                        # entok = self.char2idx('[ent]')
                        entok = self.ent2idx(xxx['type'])
                        for span in xxx['span_list']:
                            span    = list(span)
                            span[0] = min(span[0],max_len)
                            span[1] = min(span[1],max_len)
                            y[span[0]:span[1]]=[entok]*(span[1]-span[0])
                        if sum(y)==0:
                            continue
                        outv.append(v)
                        outy.append(y)
                        outl.append(len(xx['text']))
                elif method =='text':
                    v = list(map(self.char2idx, xx['text'][:max_len-1])) + [self.char2idx('[sep]')]*(max_len-1-len(xx['text']))
                    y = 0
                    outv.append(v)
                    outy.append(y)
                    outl.append(len(xx['text']))

                    # outm.append()

            x = torch.tensor(outv).to(self.device).long()
            y = torch.tensor(outy).to(self.device).long()
            l = torch.tensor(outl).to(self.device).long()
            # mask = torch.arange(x.shape[1]).
            with open(OUTPUT_PICKLE+'.temp','wb') as f: torch.save((x,y,l),f)
            shutil.move(OUTPUT_PICKLE+'.temp',OUTPUT_PICKLE)
        return x,y,l
        # import pdb; pdb.set_trace()
        # xd['data'][0]['text']
        # sents = []
    @staticmethod
    def preprocess_template():
        #### Separate the schema from the tokens
        #### so that one hot vector can be constructed
        OUTPUT_PICKLE = __file__+'.output.template.pkl'
        if os.path.exists(OUTPUT_PICKLE) and '--force' not in sys.argv:
            xd = torch.load(OUTPUT_PICKLE)
        else:
            print('[Init] Dataset')
            fn  = os.path.join(DIR,'data/duie_schema/duie_schema.json')
            with open(fn,'rb') as f:
                schema = map(json.loads,f.readlines())
                schema = list(schema)
            xd={}
            xd['object_type'] = list(sum([ list(x['object_type'].values()) for x in schema],[]))
            xd['subject_type']= list([x['subject_type'] for x in schema])
            xd['entity_type'] = list(set(xd['object_type']) | set(xd['subject_type']))
            from pprint import pprint
            # pprint((xd['entity_type']))
            charset = set()
            xd['data'] = []
            for fn in '''
    data/duie_dev.json/duie_dev.json
    data/duie_train.json/duie_train.json
    data/duie_sample.json/duie_sample.json'''.strip().splitlines():
                with open(os.path.join(DIR,fn.strip()),'rb') as f:
            # fn = 'data/duie_dev.json/duie_dev.json'
                    valBuffer = ''

                    suc = 0
                    total = 0
                    fail = 0
                    xd['list'] = lst = []
                    for xl in tqdm(f.readlines()):
                        xll = json.loads(xl)
                            #     # print(typ,val)
                        try:
                            # it  = re.finditer(val,xll['text'])
                            # it  = [x.span() for x in it]
                            # lst = out['entity_dict'].setdefault(typ,[])

                            it  = (re.finditer('之前的一个',xll['text']))
                            it  = [x.span() for x in it]
                            xs  = it
                            it  = re.finditer('每一个',xll['text'])
                            it  = [x.span() for x in it]
                            ys  = it
                            # if len(xs) and len(ys):
                            if len(ys):
                                idx = ys[0][0]
                                # if xs[0][0] < ys[0][0]:
                                lst.append(
                                    dict(
                                    # xs=xs[0][0],
                                    ys=ys[0][0],
                                    text1=xll['text'][ max(0,idx-10):  idx+10 ],
                                    # text=xll['text']
                                    ))
                            # lst.extend(it)
                            suc+=1
                        except Exception as e:
                            raise e
                            fail+=1
                        total += 1
                        valBuffer +=xll['text']+'\n'
                    charset =  charset | set(valBuffer)
                    print(total,suc,suc*100//total)

                        # print(len(charset))
                    xd['charset']=list(charset)
                    with open(__file__+'.output.template.txt','w') as f: f.write(valBuffer)
                    with open(OUTPUT_PICKLE+'.temp','wb') as f: torch.save(xd,f)
                    shutil.move(OUTPUT_PICKLE+'.temp',OUTPUT_PICKLE)
                    pprint(lst[:20])
                    import pdb; pdb.set_trace()
                    return

    def preprocess(self):
        #### Separate the schema from the tokens
        #### so that one hot vector can be constructed
        OUTPUT_PICKLE = __file__+'.output.pkl'
        if os.path.exists(OUTPUT_PICKLE):
            xd = torch.load(OUTPUT_PICKLE)
        else:
            print('[Init] Dataset')
            fn  = os.path.join(DIR,'data/duie_schema/duie_schema.json')
            with open(fn,'rb') as f:
                schema = map(json.loads,f.readlines())
                schema = list(schema)
            xd={}
            xd['object_type'] = list(sum([ list(x['object_type'].values()) for x in schema],[]))
            xd['subject_type']= list([x['subject_type'] for x in schema])
            xd['entity_type'] = list(set(xd['object_type']) | set(xd['subject_type']))

            # pprint((xd['entity_type']))
            charset = set()
            xd['data'] = []
            for fn in '''
    data/duie_dev.json/duie_dev.json
    data/duie_train.json/duie_train.json
    data/duie_sample.json/duie_sample.json'''.strip().splitlines():
                with open(os.path.join(DIR,fn.strip()),'rb') as f:
            # fn = 'data/duie_dev.json/duie_dev.json'
                    valBuffer = ''

                    suc = 0
                    total = 0
                    fail = 0
                    for xl in tqdm(f.readlines()):
                        xll = json.loads(xl)
                        out = {'entity_list':[],'entity_dict':{}, 'text':xll['text'],'md5':hashlib.md5(xll['text'].encode()).hexdigest()}
                        xd['data'].append(out)
                        for xx in xll['spo_list']:
                            for xxx in xx['object_type'].values():
                                assert xxx in xd['entity_type']
                            for xk in set(xx['object']) & set(xx['object_type']):
                                typ = xx['object_type'][xk]
                                val = xx['object'][xk]
                                # print(typ,val)
                                try:
                                    it  = re.finditer(val,xll['text'])
                                    it  = [x.span() for x in it]
                                    lst = out['entity_dict'].setdefault(typ,[])
                                    lst.extend(it)
                                    suc+=1
                                except Exception as e:
                                    # raise e
                                    fail+=1
                                total += 1
                        out['entity_list'] = [{'type':k, 'span_list':v } for k,v in out.pop('entity_dict',{}).items()]

                        assert xx['subject_type'] in xd['entity_type']
                        # txt = txt | set(xll['text'])
                        valBuffer +=xll['text']
                    charset =  charset | set(valBuffer)
                    # print(valBuffer)
                    print(total,suc,suc*100//total)
                # print(len(charset))
            xd['charset']=list(charset)
            with open(__file__+'.output.json','w') as f: json.dump(xd,f,indent=2,ensure_ascii=False)
            with open(OUTPUT_PICKLE+'.temp','wb') as f: torch.save(xd,f)
            shutil.move(OUTPUT_PICKLE+'.temp',OUTPUT_PICKLE)
        return xd

if __name__=='__main__':
    DUIE_NER.preprocess_template()
    DUIE_NER()
    # DUIE_NER()
