#! https://zhuanlan.zhihu.com/p/555931754
# 9022-复合性语言模型 Compositional Language Model

[CATSMILE-9022](http://catsmile.info/9022-comp-lm.html)

## 前言

- 结论:
- 背景与动机: 
    - CATSMILE-9020 太臃肿了, 单独开文记录新思路. 9020大致探测了哪些思路是不work的,做了一些数据集,得到了一些思路,希望能确保9022的构造性强一些.
- 备注: 
- 关键词: 
- 后续目标/展望方向:
- 相关篇目:
    - [CATSMILE-9020](./9020-language-model) 
- CHANGLOG:
    - 20220820 INIT


### 9020 总结

#### 一些不work的东西清单:

- 手写的RNN: 太慢,没有CUDA加速
- 手写的Transformer-RNN: 太慢,不知道CUDA加速有没有用
- LSTM离散序列先验的VAE: 收敛到了无聊的隐变量序列上, 可能直接指望RNN学到合理的序列变化是困难的. 

#### Work的东西

- 各类RNN-VAE, 无论是单隐变量的,还是分布式隐变量,都work. 不过可能高斯先验不是很合理.


### CLM

CLM的目标是自动分词. 我们尝试在VAE的框架内,构造一个具有层次的语言模型, 分为以下三个部件

1. Segmentator 分词器 $q_e(z|x)$
    - 最简单的分词器就是 `x.split()`
1. Auto-regressive Decoder 解码器
    - 具有一个RNN形式
    - 对于输入segment,使用KV矩阵构造一个arbitrary的非线性lookup变换,得到segment vector. 
    - 在输出时,倒过来使用KV矩阵,解码出具体的低级符号.
    - 自回归地计算 $q(z_1,z_2,\dots,z_S)$ ,经过强制concat得到观测序列的.
    - 在使用KV矩阵时,需要注意segment长度的处理. c2s过程中,必须pad到同样长度. s2c过程中,可以截断s到c一样的长度. 

### DLM100: Refactoring DLM28

奇怪的是,无论基于character进行训练,还是基于word进行训练,得到
的似然居然是差不多的?

- 重要的可能bug: 直接对数据取词表是会隐式地泄露数据给模型的. 比如训练集中没有出现'dog',测试集中出现了'dog',如果在统一建立词表时检索了测试集而加入了'dog'令牌,那么就相当于把这个组合泄露给了模型.考虑更加极端的情况,如果模型把整句长序列视为一个待预测令牌,那么在测试时,模型只需要做选择题而不是写作题.这是不科学的,因为这个长序列压根可能在训练集里没有出现过,模型也不应该被告知这个序列的存在性.
    - 简单来说,OOV词语的概率估计需要谨慎
    - 应用char-based rnn的模型一般没有这种问题
    - 应用DLM101的kv-pair lookup也可以规避这个bug


### DLM101=CLM001: 使用一个最简单的`x.split(' ')`分词器

字符到词语的转换是一个令人头疼的问题. 一个heuristic是采取KV注意力来取出键值,但是这可能会造成很多间断点.

实验记录:
- 对DLM101使用不同的字母进行分词,得到的结果是用空白字符' '分词效果最好.这是符合直觉的. 在前几个字母里,使用a/i分词的效果好于b/c.

### DLM102:

DLM101给出了一个进行segmentation-concatenation的框架. 但是KV注意力的性质可能并没有那么好. 考虑到KV的主要作用是把字符串编码成词向量,再从词向量解码出字符串,这听起来似乎可以用一个RNN完成,于是我们尝试往里面扔一个RNN来做这个编码解码.

即便只把编码器换成RNN,效果都已经比KV好很多了...但是这仍然意味着模型可以做自回归建模

### DLM104: Use CNN to encode words

CNN似乎不会有很强的surprise效应. 这侧面说明了token层面的自回归是一个很奇特的框架. 

在这个框架里,我们直接取了一个唯一且可逆的parsing,根据VAE框架,这是一个特例.因为parsing的可逆,确保了重要性采样是成立的(?).这里的隐变量其实只有parsing的位置: 在给定观测序列时, parsePosition的位置是确定的,映射到唯一的parsed序列. 在给定parsed序列时,直接concat,就映射到唯一的观测序列,不存在一对多或多对一的情况.

$$\begin{aligned}
ELBO &= E_{q_e(z|x)}[\log {q(z)q_r(x|z)\over q_e(z|x)}] 
\\&= \log q(z)
\end{aligned}$$

对于一个具有一定随机性的parser,就需要应用VAE的ELBO技巧. 采样出一个parsing,映射到一个parsed序列,不过concat的唯一性还是一样的. 这时候对梯度的估算就要用到REINFORCE技巧.

$$\begin{aligned}
&\nabla_m ELBO 
\\&= E_{q_e(z|x)} \nabla_m \left[ \begin{aligned} & \log q_e(z|x) sg( \log { q(z) q_r(x|z) \over q_e(z|x)} - c) 
\\ &\dots + \log q(z) q_r(x|z)  \end{aligned}\right] 
\\ &\text{c is the estimated baseline to reduce variance}
\end{aligned}$$

### Thoughts:

RNN的自回归建模从某种程度上是一种逐步的建模,这是通常的神经网络并不能很好地做到的一个点. 但是RNN的parsing是一种比较trivial的parsing. 可以认为,目前DDPM等分段模型,某种程度上也是一种类似RNN的自回归建模. 这两者的共性在于,parsing过程,或者生成中间态的过程都是预设好的/不可训练的.那么从概念上讲,这个中间态最好是能够由模型自行编码生成.

### Refs:

### [TBC,Transformer为啥这么慢??]

层次模型似乎不是很多

- Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models <https://arxiv.org/abs/1507.04808.pdf>

VAE

- Diverse Text Generation via Variational Encoder-Decoder Models with Gaussian Process Priors <https://arxiv.org/abs/2204.01227.pdf>

- Pointer Generator For summarization <https://aclanthology.org/P17-1099.pdf

- Hierarchical Representation in Neural Language Models: Suppression and Recovery of Expectations <http://aclanthology.lst.uni-saarland.de/W19-4819.pdf>

- Stochastic RNN SRNN: Sequential Neural Models with Stochastic Layers <https://arxiv.org/abs/1605.07571>

## 参考

基于句法树对RNN有过一些改造得到的模型据说效果还不错 ON-LSTM.

- GRU network <https://arxiv.org/abs/1412.3555.pdf>

- pytorch GRU cpp source
    - calls torch::gru in this file <https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/nn/modules/rnn.cpp>
    - cpp的逻辑太难扒了,找不到`torch::gru`的具体实现...倒是有找到vulkan后端的实现,但是rnn为啥会直接调用cuda底层代码呢?不知道是不是有啥加速特性 <https://github.com/pytorchpytorch/blob/9ec8d64d0c464d0b23b564bd10869bb2819d223b/aten/src/ATen/native/vulkan/ops/Gru.cpp>
    - 搜了一下,有人尝试过直接手写gru实现 https://github.com/emadRad/lstm-gru-pytorch

- GBZhou2016 MGU: Minimal Gated Unit for Recurrent Neural Networks <https://arxiv.org/abs/1603.09420>
