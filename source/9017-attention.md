#! https://zhuanlan.zhihu.com/p/545545719

# 9017 机器翻译中的注意力和质控 Attention And Quality Control in Machine Translation

[CATSMILE-9017](http://catsmile.info/9017-attention.html)


```{toctree}
---
maxdepth: 4
---
9017-attention.md
```

## 前言

- 背景与动机: 
    - 往期实验中难以复现transformer的优越性
    - 所以转向了更为直观的NMT翻译中的注意力机制
- 结论: 
    - WMT14不干净,不要轻信数据集的整洁性,质控要勤.
    - 数据蒸馏和数据质控很重要!
    - 我们目前无法从给定的数据中,学到数据中并不存在的事实. 即便天王老子来了,也不可能无中生有.
    - PS: 要是在这么脏的数据集上能训练出接近对角的注意力矩阵才是见鬼了吧.
    - `业务 -> 数据 -> 模型` 在这个链条上,信息是逐渐减少的. 模型不一定能捕捉到数据的所有特征,数据也不能捕捉到业务的所有特征.
    - `业务设计 <- 数据结构设计 <- 模型设计` 这是上式,在设计空间的对偶,应当先对业务进行设计,再决定用什么样的数据结构去捕捉业务信息,
    再决定用什么样的模型去自动分析这些数据结构. 可以说,很多灌水的深度学习的工作,就是没有尊重这个依赖链条,过多地纠结在了模型设计里.
    - 当然,后续机器学习的发展方向之一,就是逐渐把这些设计过程集成起来,
    变成一套统一的方法论,但是目前似乎还没有令人满意的
- 完成度: 中,缺代码和公式
- 备注: 
    - 本来这个周末是想研究一下插入/删除/反序操作的toy model的,
    周六改了半天RNN,周日调了一天WMT14.醉了
- 关键词: WMT14/数据质量控制/NMT
- 后续目标/展望方向:
    - [TBC,清洗一下WMT14,或者找个干净点的数据集]
    - [TBC,做一下消融实验证明注意力机制的优越性]
    - [TBC,检查注意力机制对于删除/反序问题的效果]
    - 关注ConvS2S对Transformer的高效替换
    - Transformer/大模型的效果那么好,是不是因为其对数据的洁净度/结构化要求更低呢?
- 相关篇目
- CHANGLOG:
    - 20220724 加入WMT14质控相关内容


### 材料与结论: WMT14数据集不是很干净

我画了半天alignment图,说怎么看起来这么奇葩,原来WMT14 En-De 数据集似乎不是很干净

```bash 
# See https://nlp.stanford.edu/projects/nmt/
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de
```

### 例子: WMT14里面一个对齐得比较好的例子

![这两句在数据集里对齐得不错,所以注意力矩阵很漂亮](./9017-p2.png)


### 例子: WMT14里面对齐得比较差的例子

![这两句在数据集本身对齐得很糟糕,所以注意力矩阵很差.line4003](./9017-p3.png)

WMT14EN2DE的line4003的对齐确实很糟糕

- line4003 Cereals , fruit , ham , cheese and scrambled eggs . Coffee came pre ##AT##-##AT## made in a flask - not what one expects in a hotel of this calibre .
- line4003 Beim abendlichen Aufenthalt im Außenbereich war der Service teuer .

### 例子: 另一个WMT14对齐失败的例子

#### train.en

- line3994 The 102 rooms , suites and junior suites display elements of contemporary and classical architectural design , along with Florentine furniture and decorations .
- line3995 Hotel Albani Firenze in located in the heart of the city , in a quiet area very close to both the Florence Congress Centre ; Fortezza da Basso , Palazzo dei Congressi and Centro Affari are less then 5 minutes ’ walk away .
- line3996 The main tourist attractions can be reached on foot . The Santa Maria Novella train station is a quick 150 metre walk .

#### train.de

- line3994 Das Hotel verfügt über 102 Zimmer und Suiten , die sowohl moderne als auch klassische Elemente widerspiegeln und die perfekte Verbindung aus modernem Design und florentinischem Zauber bieten .
- line3995 Das Hotel Albani Firenze befindet sich in ruhiger Umgebung im Herzen der Stadt .
- line3996 Sowohl das Kongresszentrum als auch die wichtigsten touristischen Attraktionen sind schnell und bequem erreichbar .

可以看到 3995 和 3996 的英德句子信息并不对称,英文3995有5 minutes away 的信息但是在德文3995里面压根没有出现

### 方法: nano 使用小技巧

- 用nano打开你的txt文件
- Ctrl+Shift+"-"=Ctrl+_ = 跳转到第X行
- Alt+C 显示行数状态栏

### 方法: 质量控制模型

下面的图都是用`markov_lm.Model_NLP.SoftAlignmentModel`画的

<https://github.com/shouldsee/catsmile/tree/master/markov-lm>

#### markov_lm.Model_NLP.SoftAlignmentModel

为了确保模型尽量简单,我没有使用RNN, 直接创建了  简单来说就是对于每个句子对,考虑一个自由的对齐矩阵,然后优化最优对齐情况下的对数似然即可.[TBC,补一点公式,push代码]

从计算形式的角度来讲,唯一和Transformer相似的地方在于logsumexp的使用,使得导数中会出现softmax项

#### markov_lm.Model_NLP.AlignmentModel

和SoftAlignmentModel类似,区别在于仅考虑了最优对齐,max替代了logsumexp

#### markov_lm.Model_NLP.Seq2SeqWithAttention

从 https://github.com/graycode/nlp-tutorial 里面的seq2seq模型魔改
而来, 原始模型里面居然用for-loop计算了Attention矩阵根本没法并行化,
我也是非常醉. 解码器损失函数用了TeacherForcing/AutoRegression的形式.

Adapted from <https://github.com/graykode/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention).py>

#### markov_lm.Model_NLP.Seq2SeqWithNoAttention

用来做消融实验的类


### 备注和讨论:

今天按照Bahdanuau2014的图去搜了一下复现方法,发现一篇有趣的
pervasive Attention with 2D convolution (Elbayad2018)
用 7M 的参数量复现了 59M Transformer的 BLEU, 不过FLOPS计算量差不多. 手动捋了一下WMT14,发现attention图要复现出来并不是那么trivial.而且Bahdanuau2014居然到处都找不到code,好像是因为其依赖Theano的Groundhog框架已经挂掉几年了

[Google image search](https://www.google.com.hk/imgres?imgurl=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1284%2F1*Zd5VeDGHmNBQo-bwa7bOIg.png&imgrefurl=https%3A%2F%2Ftowardsdatascience.com%2Fend-to-end-attention-based-machine-translation-model-with-minimum-tensorflow-code-ae2f08cc8218&tbnid=3eBlnkQdTTewVM&vet=12ahUKEwij66johZH5AhVZxosBHbshDtMQMygOegUIARDVAQ..i&docid=8LILCvSJCcQH9M&w=642&h=647&q=machine%20translation%20attention%20matrix%20github&client=ubuntu&ved=2ahUKEwij66johZH5AhVZxosBHbshDtMQMygOegUIARDVAQ#imgrc=3eBlnkQdTTewVM&imgdii=HTU8Baqht8I9UM)

![Source: Tu2016](./9017-p1.png)

ConvS2S 看起来蛮有吸引力的. attn2d也可以考虑作为接下来的复现目标

### 画廊: 

![Clues:这乱糟糟的](./9017-p4.png)

### 代码:

torchtext的数据加载挺魔幻的,正在尝试理解

关于`torchtext>=0.8.0`后的BucketIterator语法的迁移 [Issue969](https://github.com/pytorch/text/issues/969)


在0.8.0版本[data/iterator.py](https://github.com/pytorch/text/blob/release/0.8/torchtext/data/iterator.py)下有调用链条

`data.BucketIterator -> data.pool -> data.batch`

但是实际上把token转化成index的代码是 Field.numericalize ,但是扒源码没有能够很好地看出在哪里调用的,但是反正肯定是哪里用了

example from doc

```python
# continuing from above
mt_dev = data.TranslationDataset(
    path='data/mt/newstest2014', exts=('.en', '.de'),
    fields=(src, trg))
src.build_vocab(mt_train, max_size=80000)
trg.build_vocab(mt_train, max_size=40000)

train_iter = data.BucketIterator(
    dataset=mt_train, batch_size=32,
    sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg))
    )
```

example adapted 这里抛弃了iterator,直接用了更底层的方法. 这样会比较容易Adapt进目前的markov_lm架构里.

```python
    src = tgt = torchtext.data.Field(lower=False, include_lengths=False, batch_first=True,fix_length=fix_length)
    root = DIR
    ret = torchtext.datasets.Multi30k.download(root)
    m30k = torchtext.datasets.Multi30k(root+'/multi30k/train',('.de','.en'),(src,tgt))
    src.build_vocab(m30k, max_size=80000)
    tgt.build_vocab(m30k, max_size=40000)
    src.numericalize([m30k[0].src])
```

## 参考

- Bahdanau-Bengio2014: Neural Machine Translation by Jointly Learning to Align and Translate <https://arxiv.org/abs/1409.0473.pdf>
    - Github: <https://github.com/lisa-groundhog/GroundHog/tree/master/experiments/nmt>
    - Polarion的博客<http://polarlion.github.io/nmt/2016/06/06/ground-show-alignment.html>


Tu2016 Modeling Coverage for Neural Machine Translation <https://arxiv.org/abs/1601.04811>

- Elbayad2018 Pervasive Attention: 2D Convolutional Neural Networks for Sequence-to-Sequence Predictiton <https://aclanthology.org/K18-1010.pdf>
    - <https://github.com/elbayadm/attn2d>

Gehring2017 ConvS2S: Convolutional Sequence to Sequence Learning <https://arxiv.org/abs/1705.03122>

Nvidia OpenSeq2Seq <https://github.com/NVIDIA/OpenSeq2Seq>

WMT14 by Stanford NLP <https://nlp.stanford.edu/projects/nmt/>