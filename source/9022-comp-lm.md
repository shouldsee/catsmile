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

## DLM106:

DLM104加上了ConcatenationToken

## DLM107:

DLM106 with a VAE loss

$$
q_r(x|z) = 1
\\ q(z) \text{ from RNN}
\\ q(z|x) \text{ from parsing}
$$

## DLM108: DLM107 with GRU segment encoders

DLM107 and DLM108 的实验结果说明Tokenizer倾向于输出一个
确定的全空或者全1的tokenization,并利用其中一个RNN来进行自回归. 从VAE的角度来讲,生成模型倾向于不使用或者一直使用segmentationToken, 而其ELBO并没有显著好于一个RNN

### Thoughts:

RNN隐式建模了很多玩意儿.实际上RNN做生成任务效果是挺好的,并没有performance的问题.接下来会更多地做拆解RNN的工作,而不把RNN作为一个子模块.

### DLM110:

其实ngram是比RNN更为基础的一个模型.如果用embed-unembed范式去建模一个ngram,就要牵扯到概率的分解形式.最简单的bigram模型需要用到前一邻来为embedding提供梯度. 但是bigram对上下文的丢失,会让模型不知道怎么处理 'bag' 'dag' 'tag' 'tab' 'pad' 'dad' 等等 'xax' 的概率. 这并不能通过在字母间插入复杂的混合隐变量来实现,因为基于markov性bigram建模的方法压根就不允许模型获得更多的信息. 一个比较简单的词模型,需要绕过RNN,实现一个自编码的建模过程. 也就是,要用一个层级式的生成过程, 配合一个解码器, 来模拟词语的建构. 其实词语无非就是很简单的一个词表,但是要模拟出实体/词语的边界,却并不简单. 我们需要一个把空间切块的过程,并且让词语尽量地不要跨越区块. 在区块内,我们使用一个转子来寻找词向量里的下一个投影. 也就是说,我们用几个词向量, 来表征反复出现的连续字符串. 比如我允许模型取10个vmf向量,并且允许它进行几个动作:下一词(且下一字符),同一词下一字符. 模型当然有可能把所有信息都压缩到第一个vmf向量里.所以建模的方式必须鼓励模型去合并更小的信息单元.

但是问题在于,如何确保一个局部的结构能够降低数据的似然? 如果
vmf向量只是存储第x位是y字符,那么这某种程度上只是一个kmer counter,用kmer的概率来替代1mer的概率.

假定charVector固定,那么每一个词语可以对应到char上面的一条路径,如果这条路径满足马尔科夫性质,那就可以浓缩到一个Attention矩阵里.

### DLM111: CNN模型

最简单粗暴的方法,当然是不用RNN模型,直接搞一个CNN检测器. 这样可以在先验上动一些脑筋.

### DLM112: Simple VAE

我们知道VAE-ELBO具有简单的IS形式. 对于确定性很高的encoder,其编码分布的熵也就越小,损失的ELBO也就越多. 

$$\begin{aligned}
 E_{q_e(z|x)}[ \nabla_m\log q_e(z|x)] 
&= \int_z  q_e(z|x) \nabla_m \log q_e(z|x).dz
\\&= \int_z  \nabla_m  q_e(z|x).dz
\\&= 0
\end{aligned}$$

考虑一个trivial的生成模型: 从高斯q(z)里解码出离散token,同时用token嵌入的扩散定义一个编码器. 这将作为研究CNN模型的基础.


$$\begin{aligned}
ELBO &= E_{q_e(z|x)}[\log {q(z)q_r(x|z)\over q_e(z|x)}] 
\\ &= - D_{KL}(q_e(z|x)||q(z)) + E_{q_e(z|x)}[\log q_r(x|z)]
\end{aligned}$$



$$\begin{aligned}
&\nabla_m ELBO 
\\&= E_{q_e(z|x)} \nabla_m \left[ \begin{aligned} & \log q_e(z|x) sg( \log { q(z) q_r(x|z) \over q_e(z|x)} - c) 
\\ &\dots + \log q(z) q_r(x|z)  \end{aligned}\right] 
\\&= - \nabla_m  D_{KL}(q_e(z|x)||q(z)) 
\\&\dots+ E_{q_e(z|x)} \nabla_m \left[ \begin{aligned} & \log q_e(z|x) sg( \log { q_r(x|z) } - c) 
\\ &\dots +  \log q_r(x|z)  \end{aligned}\right] 
\\ &\text{c is the estimated baseline to reduce variance}
\end{aligned}$$

注意: DLM112在采样数为1的时候是不work的,可能跟采样估计效率有关


$$\begin{aligned}
\log q_e(z|x) = - 0.5 \beta^2 (x_i-\mu_i)^2 +\log \beta - 0.5 \log (2\pi)
\end{aligned}$$

结果: 模型倾向于把先验分布到空间的各个地方. 这可能是由高斯先验的性质所确定的. 而随着先验和后验不断地稀释,采样效率越來越低,导致模型的梯度估算的方差越來越大,不再能实现梯度下降.

一个比较简单的办法是,用一个高斯混合的先验,来实现高斯的方差.用了高斯混合以后,虽然ppl降低是快了一些,但是还是会退化成一个均匀分布,所以我手动把标准差差clip在2.5以下. 最近做的实验都是以negative logppl-per-sequence(NLPPS)作为表征,DLM114终于收敛到一个正常的ppl. 只能说高斯先验的方差真的需要控制.

### tasktranslate-multi30k-de2en-chardata-l100

NLPPS: Negative Log-Perplexity Per Sequence 

| model_desc | mutated_pos | epoch | NLPPS_E20  | NLPPS_E30  |  NLPPS_E40  |
|------------|-------------|-------|---------|---------|---------|
| DLM100        | -1          | 20 | 76  |   | |
| DLM123,D3,K30,E32,W5| -1       | 20 | 149 | 142 | 136|
| DLM123,D3,K30,E32,W1| -1          | 20 | 152 | 147 | 139|
| DLM123,D5,K30,E32,W5| -1       | 20 | 152 | 145 | 136|
| DLM124,D5,K30,E32,W1| -1       | 20 | 147 | 144 | 142|
| DLM121,K30,E32| -1          | 20 | 155 | 150 | 144|
| DLM121,K30    | -1          | 20 | 245 | 161 | 145|
| DLM119,K30 | -1          | 20 | 191 | 162 | 156|
| DLM120,K30 | -1          | 20 | 226 | 170 | 159|
| DLM118,K30 | -1          | 20 | 286  | 180|162|
| DLM117,K30 | -1          | 20 | 322  |   | 163|
| DLM122,K30 | -1          | 20 | 307 | 199 | 167|
| DLM114,K30 | -1          | 20 | 193  |   | |
| DLM115,K30 | -1          | 20 | 199  |   | |
| DLM112     | -1          | 20 | 310  |   | |

看这张表可以发现自回归的RNN模型还是很猛,直接干到了76. 相比之下最简单的GMMVAE在193,加了卷积的VAE能够提到136. 这说明用卷积去捕捉局部结构可能确实是可行的.

### DLM115

让我继续瞪着ELBO看一会...

- q(x|z) 用简单的unembed
- q_e(z|x) 用embed + noise
- q(z) 用个bigram看看?
- bigram其实在这里应该属于RNN的方法,没啥大意思

$$\begin{aligned}
ELBO &= E_{q_e(z|x)}[\log {q(z)q_r(x|z)\over q_e(z|x)}] 
\\ &= - D_{KL}(q_e(z|x)||q(z)) + E_{q_e(z|x)}[\log q_r(x|z)]
\end{aligned}$$

### DLM116:

考虑一个不太一样的noise形式,更加接近demasking的模型. 这个noise以一定概率把当前token替换成一个mask token. 但是其生成模型应当有一个尽量简单的形式,一个办法是考虑从全`|mask|`里生成目标序列.不过这个模型压根不需要encoder,因为生成过程比较简单. 也就是说如果写不出一个生成模型,那么去凑编码器也是很困难的. 通常目前生成模型的隐变量最多也就是语序了,很少有见对NER或者对词汇进行生成式建模的.




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
