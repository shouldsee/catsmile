#! https://zhuanlan.zhihu.com/p/517156488

# 9008-20220526：对于BERT模型隐式分布的一些思考

<http://catsmile.info/9008-bert-latent-dist.html>

## 引言

最近看了iclr2022上用bert做mcmc的 {footcite:t}`gdb2022-bertmcmc`，用了raw logits作为衡量序列合理度的分数，
在单mask作为proposal的基础上用分数做了acceptance修正，与gibbs相比在
bleu上有较大的提升。这令人思考，bert所对应的这个pdf到底有啥性质？

用bert分数在给定序列附近可以采样到一个分布，那么这样的一个分布究竟具有什么
样的性质？bert间接的通过mlm构造了分数来定义了这个分布，那么如果能直接用变
分方法直接近似这个分布不是更加简单？相比之下，gpt可以近似地看做是对于序列
的分布的一个变分近似，但是链式计算的teacher forcing现象也让scoring不是
那么简单。

根据一条朴素的假设:不同网络架构的优化性质反映的是损失函数的构造,我们继续考察损失函数接近
输出层的性质.

实际业务->损失函数->网络架构

以BERT为例,MLM损失函数使得transformer成为最常用的有效架构.MLM的提出是为了运用无监督学习来
学习大量语料背后隐藏的统计分布,

NLP下游任务->语料建模->MLM损失函数->Transformer架构

机器学习的授权链条具有强烈的线性性,对于网络架构的选择基本上基于终端任务的表达能力,
通过恰当设计损失函数,可以将特定任务交给社区来动用大量研究员的人力和算力来寻求可衡量的
损失函数的降低. 但是我们必须时刻对损失函数的构造保持清醒,因为损失函数直接决定我们对
架构的选择性,也就是,我们总是选择让损失函数最小化的那个架构,而不是选择让实际业务最优化
的那个架构. 如果损失函数本身和实际业务之间出现了断层,那么架构的更迭将不必导致业务的优化.

以MLM为例子,一个令人困惑的问题是Transformer系列模型惊人的参数量.如果我们能够证明
这些惊人的参数量都实际上都是被损失函数的信息瓶颈所限制,那么对损失函数进行恰当的改造,
就可以大幅度降低模型的参数量.本文将对MLM做一些理论思考来考虑这种改造的可能性和必要性.

## 举例

mlm运算和生物信息学的msa有着紧密的联系，对于相似的句子，可以抽象出一个其附
近的mlm分布，也就把句式作为句子空间上的一个delta函数后，符合这个delta函数
的样本空间上的那个分布。比如（甲甲）的首都是（乙乙）就只关心满足第三到第六个
字符合模板A的那些序列，比如下面就是一些合法序列。

```
中国的首都是北京。
法国的首都是巴黎。
英国的首都是伦敦。
模板A:
甲甲的首都是乙乙。
```

那么bert显然还要处理横轴上的插入和删除，也就是说我们认为如下模板B，所约束的
分布，应该和模板A所约束的分布是有联系的。

```
模板B:
甲甲甲的首都是乙乙
```

bert的内部是基于qkv注意力和pe的，要理解其对于delta函数的表征，应当直接描述
中间四个字对于bert内部表征的约束。（的首都是）应当对bert产生一个激发，并造
成甲甲和乙乙之间的一个联系，使得满足kg约束的甲甲和乙乙具有较低能量，或者较
高的概率，但这仍然是通过分布的性质去理解的。对于bert的理解难点在于，表述其
内部表征，难于其外部性质，虽然大多数神经网络也都是这样难以描述。或者我们把问
题转化为，为啥bert的架构能够让其高效地表征这样一些模板A，B附近的概率分布？
解答了这个问题，就可以探讨只把模板映射到其附近概率分布的模型应该长成什么样子。

基于模板的思考，跟prompt研究是有相关性的，其本源在于对于序列空间上的分布的
描述。对于 $Seq \in Length^{Vocab}$ 的序列空间，需要高效地分治这个相空间来获得可计算的
分布。模板化的思考把原始相空间分割成满足不同模板的子空间，这也是mlm建模的
重要抽象意义。

多mask的mlm建模实际上在分布的角度看并不严密，因为其输出是基于每个mask独立
计算的，也就是说p（甲甲乙乙）=p（甲）p（甲）p（乙）p（乙）这显然是不可能的
，因为p（中国北京）>p(中国北黎）。听说spanBERT专门采样了连续mask，可
以深入思考一下如何表征给定模板后的p（甲甲乙乙）。

## 理论实验1: 输出层对多掩码位的表征特性

让我们考虑spanbert的连续mask形式，也就是说（甲甲乙乙）在模板中占据连续的
位置，那么对于符合这个模板的序列的分布p（甲甲乙乙），如何计算其概率呢？目
前我还没听说过比较系统的解决方案。显然直接做postion-wise的分割是过于粗
暴的，怎么说都得套一个hmm或者mrf，鉴于都是神经网络，用mrf一般更顺手。由
于外部模板已经建立，所以直接对甲甲乙乙内部的mrf尝试进行二阶建模。

先考虑极端情况，甲甲乙乙退化成单token甲，那么mrf只需要一个unary potential
就可以被完全描述，这个时候损失函数和单mask的mlm是一致的。

考虑简单序列，其能量需要写成 $p(甲乙)\propto \exp(\beta e(甲乙))，e(甲乙)=\phi_1(甲)+\phi_2(乙)+h(甲，乙)$ ，
那么模型需要输出一个 $h(甲，乙)$ 来描述其约束关系，这个是一般的bert模型输出层不具备的。对于更长的序列，其二阶项也就更多，通过position encoding
和qkv attention，理论上可以把这些参数传导到这些位置的一阶项上.
比较难描述的是h（甲，乙），一种办法
是在每个位置上计算动态的Q矩阵和K矩阵，然后用（Q甲）点乘（K乙）得到其数值。

那为了算个二阶项还得引入动态的QK岂不是很熟悉？不如考虑直接用
qkv attention加nonlinear layer是吧，这就很有意思了，那不就是bert本身
对于输入数据的处理形式吗？也就是说，单层的bertlayer，可以用来在每个位置上
计算了一个特殊的动态的mrf能量函数，做avg pooling就可以得到总的能量值。
或者说，单层的qkv attention ffn取出的表示和原始表示卷积后，可以用来做一个
能量函数，来构造p（甲甲乙乙）。但是这样又来了一个问题，就是这个根据外部的模板
而确定的参数要怎么表示。一个自然的想法是沿用bert本身的形式，让模板和待定序列产
生交互，这样最后得到的能量就是含有模板信息的，但是有一个问题就是模板那边交互后引出
的能量要不要包含进来作为能量的一部分？这个可能只有做实验才能解答了，可以选择的方式
有，

1. 计算模板和待定的总能量（简洁且一致）
2. 仅计算待定序列上的能量（也就是把模板能量给mask掉）
3. 把模板的前向计算给停掉，使得其能量停在一个定值上。

个人猜测1或者2是更好一点的计算方式。

还有一种更奇葩的引入交互项的动态性的办法，是把目标向量或者初始向量给动态化，
也就是把输入向量看成是参数而不是待定序列，然后把待定序列放到输出的位置上，
听起来有点像unilm啊，这样做的好处，是可以把能量给动态化，坏处是无法表示
二阶项了（只能借助一阶项）。所以还是算了吧，待定序列必须放在bert的输入层，
那么在计算输出能量的时候，到底是采用一个线性层直接计算，还是把输入挪过来
做内积，目前不是很清晰。原始bert是把mask放在输入，然后把candidate放在
输出的，但是从能量的角度看这可能不是必须的，但是为了跟bert接近，可以是
开始的一种方式。首先要验证的就是mask和待定序列所引导出来的能量是否具有相似性。

倒过来思考，如果在输出层能量并不能表示为和待定序列的内积，会引出什么问题呢？
这目前不是很好回答，但是可以简单地认为这种架构会向rbm靠拢，而rbm最大的问题
就是不好训练。因此我们可以认为，采用bert这种形式的好处在于更加接近mrf的形式，
从而确保了损失函数易于训练。考虑一个极简的能量输出层，直接输出跟原点的mse，
我们要求符合约束的句子有着更小的mse，而不符合约束的有更大的mse。这样最终获得
的输出层，描述的应该是各项约束的满足程度，也就是一个挑错机器。但是还是回到能
量模型本身的问题上，实在是损失函数没那么好写了。在模板附近，p（模板，甲甲乙乙）
应当挑选出那些符合数据的元素赋予高概率，然后对不符合数据的元素赋予低概率，
也就是通过构造正负样本来进行对比学习。

### 条件独立与modes

甲和乙在给定隐变量时条件独立等价于下式

$$P(甲乙|h) = P(甲|h)P(乙|h)$$

条件独立有着至少两种不同的理解方式. 比如h指向"北京"上的一个$\delta$分布,
那么 $p(甲=北) = p(乙=京) = 1 = p(甲乙=北京)$ 是满足条件独立要求的.
但是 $p(甲=张) = p(甲=李) = 0.5,   p(乙=三) = p(乙=四) =0.5$ 在条件
独立的情况下就会导出 $p(张三)=p(李三)=p(张四)=p(李四)$.这也就是为什么说
softmax输出限制了多mask的MLM表现力的原因.如果要简单地把这种表现力加回去
的话,就要简洁地表示这些位置之间的组合上的分布,最简单的办法还是搞一个链式的
的HMM来确保概率可以直接显式写出,但是,感觉好难啊,因为输出的隐藏状态是以单向量
表示的,要找到一个合适的对向量进行解码的模型并不trivial. position-wise的
向量需要表征出跨position的约束,怎么看都是很别扭的,可能最简单的办法还是用QK
trick,每个position预测前向Q和后向向量K,驻定向量V,然后做一个线性的解码.
这样从左到右分解确实是可行的,但是又会绕回RNN的老问题,有可能导致长程约束
的丢失,所以还是用QKV Attention的形式做能量学习可能能更好地解决这个问题,
但是冒出来了新的问题就是怎么去算partition function. 

如果直接扩展链式HMM但保留配分函数的简便性,应当找到一个树状模型,来对联合概率
进行分解.那么这个树状模型是需要从模板中计算出来的.但是直接用注意力做edge 
prediction是很难保证不出现loop并且保证spanning tree的.也就是说,如何
高效地找到一个可以被normalise的,同时又能表示二体和长距作用的分布形式,
仍然是一个open question.如果借用结构生物学MSA和contact prediction
的思想的话,potts model或者pairwise gaussian模型可能是一个比较简单
的近似,但是需要计算逆二阶项目的行列式,而且需要有一步unembed的操作.

那么还是考虑用QKVAttention和FFN来做,那么最后其实如果目标是降低
计算量,同时又加入二阶项,那不如就在最后一层把token填上然后再跑一层
算一下能量.那么这样我们可以发现, 零层的网络最多只有一阶的项目,
一层的网络具有二阶的项目,以此类推,要获得越高阶的作用量,就越要早将
信息注入网络.

```python

def score_simple(model, seqs):
    '''
    model.submodel[0] is a hugging face pre-trained bert-base-chinese
    model is a wrapper to call bert forward layer
    seqs is "input_ids" returned by transformers.AutoTokenizer.from_pretrained("bert-base-chinese")
    
    mask no position, score with inner product
    '''

    ### returns the final layer hidden_states

    embedded = model.submodel[0].embeddings.word_embeddings(seqs)
    xsa = model.forward(dict(masked=seqs))[-1][-1]
    ll_per_pos  = (xsa * embedded[:,1:-1]).sum(-1)
    ll = ll_per_pos.mean(-1)
    return ll

def score_raw(model, seqs):
    '''
    mask each position, then score each position with inner product
    '''

    ### returns the final layer hidden_states
    xm = model.submodel[0]
    xt = model.tok[0]
    embedded = xm.embeddings.word_embeddings(seqs)
    # embedded = model.submodel[0].embeddings(seqs)
    # embedded.repeat()
    B,L = seqs.shape[:2]
    rseqs = seqs[:,None].repeat((1,L,1))
    xe = (torch.eye(L,device=model.device).long()).unsqueeze(0)
    mrseqs = rseqs * (1 - xe) + xe * xt.mask_token_id
    # import pdb; pdb.set_trace()
    mrseqs = mrseqs[:,1:-1]
    xsa = model.forward(dict(masked=mrseqs.reshape(-1,L)))[-1][-1].reshape((B,L-2,L-2,-1))

    xsaa = torch.gather(xsa, index=torch.arange(L-2,device=model.device)[None,:,None,None].repeat(B,1,1,xsa.shape[-1]),dim=2)[:,:,0]
    ll_per_pos  = (xsaa * embedded[:,1:-1]).sum(-1)
    ll = ll_per_pos.mean(-1)
    return ll

def score_norm(model, seqs):
    '''
    mask each position, then score each position with normalised log_softmax
    '''

    ### returns the final layer hidden_states
    xm = model.submodel[0]
    xt = model.tok[0]
    embedded = xm.embeddings.word_embeddings(seqs)
    # embedded = model.submodel[0].embeddings(seqs)
    # embedded.repeat()
    B,L = seqs.shape[:2]
    rseqs = seqs[:,None].repeat((1,L,1))
    xe = (torch.eye(L,device=model.device).long()).unsqueeze(0)
    mrseqs = rseqs * (1 - xe) + xe * xt.mask_token_id
    # import pdb; pdb.set_trace()
    mrseqs = mrseqs[:,1:-1]
    xsa = model.forward(dict(masked=mrseqs.reshape(-1,L)))[-1][-1].reshape((B,L-2,L-2,-1))

    xsaa = torch.gather(xsa, index=torch.arange(L-2,device=model.device)[None,:,None,None].repeat(B,1,1,xsa.shape[-1]),dim=2)[:,:,0]
    ll_per_pos  = model.target_energy((xsaa @ xm.embeddings.word_embeddings.weight.T).log_softmax(-1),seqs[:,1:-1])
    ll = ll_per_pos.mean(-1)
    return ll


def score_4mask(model, seqs):
    '''
    mask the spaces to be filled.
    '''
    device = model.device
    masks = torch.tensor([0,1,1,0,0,0,0,1,1,0],device=device).long()
    ### returns the final layer hidden_states
    xm = model.submodel[0]
    xt = model.tok[0]
    embedded = xm.embeddings.word_embeddings(seqs)
    B,L = seqs.shape[:2]

    rseqs = seqs
    xe = masks[None,:]
    mrseqs = rseqs * (1 - xe) + xe * xt.mask_token_id
    mrseqs = mrseqs

    xsa = model.forward(dict(masked=mrseqs))[-1][-1]
    ll_per_pos  = (xsa * embedded[:,1:-1]).sum(-1)
    ll = ll_per_pos.mean(-1)

    return ll
'''
--------------------
score_simple
0.352      中国的首都是北京
-0.345     中国的首都是上海
0.066      中国的首都是巴黎
0.156      法国的首都是巴黎
0.009      日本的首都是东京
-0.031     日本的首都是大阪
-0.327     日本的首都是上海
0.313      韩国的首都是上海
0.573      韩国的首都是首尔
-0.606     上海的首都是上海
-0.160     法本的首都是巴京
--------------------
score_raw
0.145      中国的首都是北京
-0.304     中国的首都是上海
-0.353     中国的首都是巴黎
0.215      法国的首都是巴黎
0.279      日本的首都是东京
0.331      日本的首都是大阪
-0.133     日本的首都是上海
-0.205     韩国的首都是上海
0.878      韩国的首都是首尔
-0.659     上海的首都是上海
-0.196     法本的首都是巴京
--------------------
score_norm
0.181      中国的首都是北京
-0.289     中国的首都是上海
-0.443     中国的首都是巴黎
0.087      法国的首都是巴黎
0.178      日本的首都是东京
0.316      日本的首都是大阪
-0.035     日本的首都是上海
-0.142     韩国的首都是上海
0.846      韩国的首都是首尔
-0.594     上海的首都是上海
-0.105     法本的首都是巴京
--------------------
score_4mask
0.025      中国的首都是北京
-0.032     中国的首都是上海
-0.068     中国的首都是巴黎
0.051      法国的首都是巴黎
-0.001     日本的首都是东京
-0.194     日本的首都是大阪
-0.038     日本的首都是上海
0.060      韩国的首都是上海
0.097      韩国的首都是首尔
-0.043     上海的首都是上海
0.144      法本的首都是巴京
'''
```

![Image](https://pic4.zhimg.com/80/v2-253ffa31dfeafd097fc7286345583a3f.png)



## 理论实验2: 输出层的单峰分布引起的次生吸引


对MLM损失函数进行细致的考察. 我们可以发现,在single mask的情况下,这个模型有个
糟糕的瓶颈:在其输出层预测mask词时,先输出一个隐藏固定维数向量h,再将h与词表
w进行內积对照,输出一token y. 这样的一步操作,限制了词表w的形式,必须是某种意义上的
聚集的,也就是如果两个token在词表w中接近,那么它们在不同情景下的输出的概率也会比较接近.
在word2vec研究时代,曾经关注过一词多义的问题,那么由于输出层词表w的限制,也会造成一种
隐形的词义限制.

这种输出分布有一个问题,假设我们有一个template `我的姓是[mask]`,
训练MLM来预测mask,那么我们需要模型对于所有符合约束的词语赋予同等的概率. 
这可以填入`张`,也可以填入`李`,`胡`,`王`.那么我们希望输出的隐藏
表示,是关注w词表中的`姓名空间`.从投影的角度来考虑,合理token的w在h的投影
下应当有较大的值,而不合理词有较小的值.这也就是说,我们要求`张李胡王`聚集在h1的附近,
同样地,我们希望`王帝侯爵`聚集在h2的附近.`桃李果杏`聚集在h3的附近,如果我们把这些
支持向量隐藏起来,那么在词语空间里,我们就会发现`张李胡王`之间存在相互吸引,`王帝侯爵`之间
存在相互吸引,由于吸引的传递性,那么就会在`张李胡|王|帝侯爵`中产生次生吸引,也就是说,不同
的语境之间,会产生一种隐性的约束,这是由于$p(y|h) = {1 \over Z}\exp (-h\cdot y)$的
单峰形式所决定的. 从序列构造的角度上来看,单峰分布并不能很好地进行语法约束的描述,因为
语法约束天生具有一种多峰的性质,也就是`张李胡王`应当具有相等的概率,而单峰分布显然是做不到这一点的.
在传统的MLM目标里,softmax输出层会对近义词替换进行排斥,我认为这可能进一步加深了对模型
非必要的一些限制:当你预测到`我的姓是[张]`时,你会对`我的姓是[李]`产生很强的排斥,因为这种
排斥可以提高输出`[张]`的概率,也就是说,在同一个batch里面如果只有`[张]`没有`[李]`时,模型会
倾向于大幅降低`[李]`的概率,而不是其他无关词语的概率,使得`[张李]`等概率更加困难,需要在大量的
训练中才能够达到平衡.

## 理论结论2

单掩码情况下,softmax输出分布容易让多个意思相近的句子之间产生互相排斥的梯度,因为模型对于
template并没有记忆,无法聚合多个类似template情况下不同填入词的统计量来直接指导正确的logit,
而必须通过模型内部构造的改变使得这种互斥梯度无法发生作用的一个位置上.

这可以类比图像分类的问题中,对于相似样本的处理.在MLM里,相似样本之间天生容易影响对方的梯度,
而更进一步的存在"相同"样本,具有相同的template但却有不同的填入词,存在多个标签的可能性.
但是瞬态的softmax对非目标标签产生了push-down的作用.其根源在于,单峰的打分方式,并不能很好
描述多个可能性的分布,解决这个问题的一种方法是使用softmax的混合分布来构造一个多峰分布,或者
完全抛弃softmax这种评价方式.我认为,softmax-mlm可以认为是一种特殊的对比学习,可以通过
保留其对比学习的能量形式,改造能量函数的计算方式,从而避免禅寺

## 尝试构建损失函数

为确保模型具有统计意义,考虑如下分布 $P_{data}$ 描述语料中出现过的token序列, $P_{corrupt}$描述注入噪音的对比分布
 $$P_{data}(x) = {1 \over |data|} \delta_{data}(x) $$ 

$$\delta_{data}(x) = 1\ if\ x \in data, \delta_{data}(f_c(x)) = 1\ if\ f(x,i,k) \in data$$

$$P_{corrupt}(x) \propto \max_{i,k}(\delta_{data}(f(x,i,k)))\cdot(1-\delta_{data}(x)))$$

其中$f(x,i,k)$定义为将i位置替换为token k

那么训练目标就变为找到一个分类器来区分$P_d$和$P_c$,这有一点类似于对抗学习的思想,但是生成器$Generator$是完全
基于语料的,其原因是因为直接凭空生成语料太困难了. 那么损失函数可以用标签预测的交叉熵来表示, 令二分logit为$\log Q(x)$表征x来自$P_{data}$的概率

$$loss(\theta)= {1 \over |B|} \sum_b IsData(x_b) \log Q(x_b|\theta) + (1-IsData(x_b))\log( 1- Q(x_b|\theta))$$

在微观上进行对比学习,对每一个序列对,进行交叉熵训练. 注意到$P_{corrupt}$需要判断$\delta_{data}(x)$,这就造成
直接通过采样近似噪音样本的时候$x \sim P_{data},\ y \sim P_{corrupt}(\cdot |x)$需要判断$\delta_{data}(y)$
,而直接遍历数据表计算是很困难的,要结合一些数据结构才能高效判断.目前先进行简单近似.


## 结论

那么，本文的结论就是，要做到更加高效地表述待定序列的分布，那待定序列的
candidate必须在输入层就进入模型，否则就会受到输出层独立结构的约束。
这跟prompt思想是不谋而合的，prompt也是通过在输入层注入序列约束，
等效地把模型约束在了这个相空间附近。这样的模型是不能用mask范式进行训练的，
需要用对比学习构造正负样本的nce进行适配。我个人预测，bert的历史地位应当
在于将对比学习的思想引入序列上复杂分布的表示，其后将因为模型自身的表达能力
受限而让位于其他表达能力更强的对比学习模型。很期待看到能量模型ebm思想
对这些模型改造的结果。

对比地看,EBM的问题可能在于速度不那么快, 也就是说
LNM(locally noarmlised models,任意采用softmax预测token的都可以归入此类)
的意义在于在局部对EBM进行了快速近似.

<table border=1>
<tr>
<td>模型</td>
<td>BERT</td>
<td>GPT</td>
<td>EBM</td>
</tr>

<tr>
<td>速度</td>
<td>快 O(DKL^2)</td>
<td>快 O(DKL^2/2)</td>
<td>慢 O(VDKL^2)</td>
</tr>


<tr>
<td>损失函数</td>
<td>masked token prediction</td>
<td>next token prediction</td>
<td>maxmise cross entropy between isData label and predictor </td>
</tr>

<tr>
<td>采样</td>
<td>raw energy-based MCMC</td>
<td>Beam-search / recurrent sampling</td>
<td>energy based MCMC</td>
</tr>


</table>


搜了一下相关文献，发现 {footcite:t}`deng2020-resebm` 采用的确实是暴力把输出层
hidden state直接pool成一个能量函数，所以其实把nce比较的分布换一下又
可以水一篇了。

那么假设ebm的formulation是可行的，我们要怎么从这个模型里抽取信息呢？
这个模型又是怎么表示实体，语法和知识的呢？这都仍然是未知的问题。
ebm的generation还好，encoding就比较复杂了，一个自然的扩展是把
ebm扩展到更高层级的结构，比如句子层级。同时对于标签也要用ebm的手法
做normalisation的classification。不过，看来使用ebm的好处是后
续的损失构造都比较有原则性，可以用简单的概率论解决，这样就把深度学
习的黑盒性质关在了这个能量函数里，同时允许对能量函数进行不同角度的
考察，来对更好地研究其黑盒性质。



residual ebm看着挺有趣的，似乎是一种通用残差模块

## 题外话

今天到bengio和lecunn的谷歌结果兜了一圈，bengio最近在搞gflownet，
lecunn批判了半天gpt-3然后撸了一个opt也着实是分裂的很。看来学界和业界的氛
围还是有区别。我这篇随想能写出来也是多亏了前人经验，在二狗组里讨论到msa和能
量模型的mcmc后，顺手查了bert的mcmc，没想到真有人做过挺多东西。看来做研究还
是得多交流换脑子，而且要跨领域迁移方法。

## 其他

bert是个混杂了knowledge和semantic rules的语言模型，mlm的目标是拟合
masked to original的映射，在单mask的情况下退化成为拟合mask上的
normalised counts。（实际一般不用单mask） ，bert的中间表示非常的简单
粗暴，就是跟token seq长度一样长的vector序列。其类似于知识图谱补全的性
质，可以用rescal的框架来理解。对于alternative hypothesis的捕捉，
transformer模型整体来讲都使用了qkv expectation，这样的好处是确
保了hypothesis的表示是简洁的，又或者说，一句话一般只有一个意思，
对于无法消除的歧义，qkv expectation是没有办法表示的，不过好在不影响大部分任务。

例句：
北京有一百万常住人口。
这些常住人口每天的消费额是很大的。

## 参考

```{footbibliography}
```

[Exposing the Implicit Energy Networks behind Masked Language Models via Metropolis--Hastings​](https://arxiv.org/abs/2106.02736)

[Residual Energy-Based Models for Text Generation​](https://arxiv.org/abs/2004.11714)
