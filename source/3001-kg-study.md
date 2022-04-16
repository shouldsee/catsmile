# 3001-知识图谱和NLP句子表示-20220416

*(本文脱离了CATSMILE的严格定义,Markdown天然爱用#来表征段落,可能需要研究一下commmonMark来定义数据格式)* 静态站: http://www.catsmile.info/3001-kg-study.html

## 前言

类似BERT的预训练NLP模型天然可以通过MLM的形式回答有关知识的提问,
但是知识在这种基于注意力的模型中到底是如何被表示的,还没有非常明确的
结论.本文希望通过讨论目前已有的研究,梳理出我们在PLM(Pretrained Language Model)和KG(Knowledge Graph,知识图谱)模型中对于"知识"的定义..

## 传统KG视角对实体关系的定义

TransE, 基于线性关系的定义 $H + \mu_R = T$,可以被扩展成为一个高斯形式,也就是假设 $(H-T) \sim Gaussian(\mu_R,\Sigma_R)$ ,也就是一个双线性的能量函数

$$
\begin{align}
p(H,T|R) &= \exp(-(H-T+\mu_R)^T\Sigma_R^{-1}(H-T+\mu_R)) \\
E(H,T|R) = \log p(H,T|R) & = -(H-T+\mu_R)^T\Sigma_R^{-1}(H-T+\mu_R)
\end{align}
$$

我们可以看到,这个双线性的能量函数关于H,T,R并不是完全对称的,除非把$\Sigma$ 设成共享参数.那在共享的情况下,我们考虑的其实是和向量$H+(-T)+\mu_R$的长度,并且希望和向量尽量靠近原点,也就是三个向量构成一个循环. **!TODO!** 考虑关于HT的对称性.

考虑对R进行反转,$H = T - \mu_R$,也就是说这里定义的关系都是对称的. 这种关系还满足可加性$(H_2 = H_1 + R_1, H_3 = H_2 + R_2) \rightarrow H_3 = H_1 + T_1 + T_2$, 这种可加性在单位球面上可能不是一件好事?不过具有可加性意味着关系可以通过简单的加法进行组合,这可以通过实验去看一看.**!TODO!**如果具有可加性成立,那么构成循环的多个关系,就可以形成一个自指的关系,比如 $R(同事)+R(老板)+R(下属)\approx0$

### 其他能量函数,二阶MRF

如果考虑更加神经网络味道(接近RNN的)的能量函数,可以用一阶马尔科夫场的能量函数 **!REF!**

$$
E= H^T W_1 T +  T^T W_2 R + R^T W_3 H \\
$$

其中最大条件似然后可以得到

$$
\begin{align}
T(R,H) = W_2 R + W_1^T H \\
R(T,H) = W_2^T T +W_3 H  
\end{align}
$$

又或者可以考虑张量积的形式,对H,T,R三阶作用进行打分.

### 其他能量函数,內积函数 **!REF!**

这样写的话跟vmf和单位球关系更接近一点...但是在几何上,如果要使用单位球,总归不太优雅.

$$
\begin{align}
E(H,T,R) &= - (H-T)^T \mu_R \\
         &=  - H^T \mu_R  + T^T \mu_R
\end{align}
$$

### 其他能量,把R放到关联矩阵里的MRF

其实这个应该是最符合MRF的直观感觉的模型,就是可能要做一点Low-Rank近似,不然R的复杂度就变成平方了,或者说就是跟三阶MRF作用量一个意思了.

$$
\begin{align}
E(H,T,R) = T^T W_R H
\end{align}
$$

用lowRank对$W_R$进行一个拆解$W_R=RR^T$后得到下式子**TODO**需要跑一跑实验看看效果.

$$
\begin{align}
E(H,T,R) = T^T R R^T H
\end{align}
$$



## 预训练语言模型中的知识

### 基于论文团簇的盘点 

[K-BERT搜索结果](https://www.connectedpapers.com/main/06a73ad09664435f8b3cd90293f4e05a047cf375/K%20BERT%3A-Enabling-Language-Representation-with-Knowledge-Graph/graph)


1. EaE
    - 解决的问题: 引入外部知识,在模型参数中捕获陈述性的知识
    - 解决的方法: 更改BERT架构对实体进行显式的建模
    - **TODO**损失函数的形式

1. SpanBERT
    - 解决的问题: NA
    - 解决的方法: 使用连续的遮照Mask

1. ELECTRA
    - 解决的问题: MLM太难了,数据利用效率不高,容易学废
    - 解决的方法: 通过把生成目标改写成选择目标,降低任务难度.
    可以认为是对比学习技巧的应用

1. [KEPLER](https://www.semanticscholar.org/paper/KEPLER%3A-A-Unified-Model-for-Knowledge-Embedding-and-Wang-Gao/56cafbac34f2bb3f6a9828cd228ff281b810d6bb)
    - 解决的问题: 引入外部知识
    - 解决的方法: 在MLM基础上通过增加KE(Knowledge Embedding)知识嵌入损失来进行微调. 也就是通过特定令牌来抽取一个知识表示$<s>$.可以认为是在BERT上面通过微调外接了一个实体/关系表征器.

1. K-Adapater
    - 解决的问题: 引入外部知识
    - 解决的方法: 不做微调,直接构造外部适配器
    - 备注: 分久必合,合久必分

1. LUKE
    - 解决的方法: 引入了实体层级的Attention?

1. FastBERT
    - 解决的问题: BERT太慢
    - 解决的方法: 用模型蒸馏去近似BERT

1. [K-BERT](https://www.semanticscholar.org/paper/K-BERT%3A-Enabling-Language-Representation-with-Graph-Liu-Zhou/06a73ad09664435f8b3cd90293f4e05a047cf375)
    - 解决的问题: 引入外部知识
    - 解决的方法: 直接把Token抄到Sequence里

1. 其他模型 NEZHA, ERNIE COLAKE ROBERTA

![Image](https://pic4.zhimg.com/80/v2-dd4a9ddefad950ece2b4bf48e7575f7f.png)

PS: 中文NLP的论文圈子有点小额...

### 我感兴趣的问题

1. BERT有知识,因为把实体Mask掉后,BERT按照定义是能够恢复这个实体的. BERT到底是如何表征一个知识的?

1. 退一步考虑,BERT需要先表征实体才能表征知识,那么BERT也必然能够表征实体.那么BERT到底是怎么表征一个实体的?
    - 更简单地说,字符级别的BERT,如何表征一个词语?

1. BERT对于序列的插入和删除,敏感度如何?

1. 如何做到区分位置和语义? ROPE的直观含义是啥?

1. LSTM作为一个特殊的注意力函数需要写一些文档.

1. 为什么BERT必须要很深?

1. 如何更直观地架构模型?

1. 如何在模型中引入外部知识?

### BERT如何表征一个实体

- 句子A: 我来到了上海市浦东新区.
- 句子B: 我来到了北京市浦东新区.
- 句子C: 我来到了龙游市浦东新区.
- 句子D: 我来到了MM市浦东新区.

通过替换XX市,我们可以表述出不同的实体.BERT针对实体表征的部分应该做出响应.我需要定位到对这个扰动响应最大的那些神经元.也就是做一些神经解剖实验(doge).**!TODO!**

对于实体的表征是一个模型的重要能力,在字符级别的实体表征类似于单词.
我们可以通过探究KG表征和PLM表征之间的相似性,如可加性,来考虑
模型对于实体表征的能力.通过更加显式地建模实体及关系,应该能够得到一个
更加简洁的模型.

### 情感分析 Sentiment Analysis

错误事实和正确事实之间应该有边界,应该已经有实验测试过BERT对于真相和谎言的反应,比如有关假新闻的判断.这是一个信念问题,BERT内源的思想钢印长啥样? [BERT测谎](https://ieeexplore.ieee.org/document/9206937) [CP结果](https://www.connectedpapers.com/main/26dbd656ba82dcb763dd79ccb66b3ac3852a8498/Building-a-Better-Lie-Detector-with-BERT%3A-The-Difference-Between-Truth-and-Lies/graph) [2015开放领域欺诈](https://www.semanticscholar.org/paper/Experiments-in-Open-Domain-Deception-Detection-P%C3%A9rez-Rosas-Mihalcea/5d37364ceeb34010be818ad997746f1336356665)

立场检测最近似的任务应该是Sentiment Analysis情感分析,对于好恶,BERT应该有感知.问题是,如何教会BERT用黑话并且开玩笑呢?

- 句子A: 爱丁堡大学有着很强的遗传学系.
- 句子B: 爱丁堡从来都是洋大人的地盘.

- 相关论文
  - Dataset: [short-jokes](https://github.com/amoudgl/short-jokes-dataset)
  - Model: [ColBERT](https://arxiv.org/pdf/2004.12765.pdf)
  - Humor Detection: A Transformer Gets the Last Laugh
  - Dataset: Puns
    - 检测幽默感
    - 冷笑话生成器

## 其他参考

## Traditional Knowledge Graph

Loss function: Supervised against a triplet datasets that specifies true
triplets

Tool for searching related papers:
 - <https://www.connectedpapers.com/>
 - Google Scholar



## softmax bottleneck

A rather practical problem in word represntation

- [mixtape and sigmoid tree decomp](https://proceedings.neurips.cc/paper/2019/file/512fc3c5227f637e41437c999a2d3169-Paper.pdf)
- [mixture of softmax](https://arxiv.org/pdf/1711.03953.pdf)

## Unsupervised HMM

- [Ke 2016, Unsupervised Neural Hidden Markov Models](https://arxiv.org/pdf/1609.09007.pdf)
- [PCFG: Compound Probabilistic Context-Free Grammarsfor Grammar Induction.](https://aclanthology.org/P19-1228.pdf)
- Viterbi and BW revisited https://nlp.stanford.edu/courses/lsa352/lsa352.lec7.6up.pdf
- HMM in protein alignment https://www.aaai.org/Papers/ISMB/1995/ISMB95-014.pdf
- Sequence level training: https://arxiv.org/pdf/1511.06732.pdf

## MRF, Junction Tree

- CMU slides: https://www.cs.cmu.edu/~epxing/Class/10708-07/Slides/lecture6-JT-annotation.pdf
- Freiburg Slides: https://ml.informatik.uni-freiburg.de/former/_media/teaching/ws1314/gm/10-random_fields.handout.pdf
- Toronto MRF in denoising: https://ml.informatik.uni-freiburg.de/former/_media/teaching/ws1314/gm/10-random_fields.handout.pdf
- RNN seems interpretable with ICM https://en.wikipedia.org/wiki/Iterated_conditional_modes
- More ICM (Iterated Conditional Modes)  https://www.cs.rpi.edu/~stewart/sltcv/handout-07.pdf
- Besag 1986 on ICM  Besag, J. E. (1986), "On the Statistical Analysis of Dirty Pictures", Journal of the Royal Statistical Society, Series B, 48 (3): 259–302, JSTOR 2345426


## Attention!

-  NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE https://arxiv.org/pdf/1409.0473.pdf

## Unsupervised Knowledge Graph

- KG to bias LM: KELM: augment pretraining corpus with KG
  https://ai.googleblog.com/2021/05/kelm-integrating-knowledge-graphs-with.html
  Sample sentences from KG (using a random walk？)  

- KG from LM: KG extraction from BERT by evaluating attention seqs.
  https://arxiv.org/abs/2010.11967

- Visual Storytelling: Convert pictures into natural languages 看图说话.

- NotInteresting,read: Review of KG refinement http://www.semantic-web-journal.net/system/files/swj1167.pdf

- Important! Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems
  https://arxiv.org/pdf/1508.01745.pdf
 - SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient
 - https://ojs.aaai.org/index.php/AAAI/article/view/10804/10663
 - Deep Reinforcement Learning for Dialogue Generation
 - https://arxiv.org/abs/1606.01541
 - A Diversity-Promoting Objective Function for Neural Conversation Models
 - https://arxiv.org/abs/1510.03055

- Quite Weird: Neural Text Generation from Structured Data with Application to the Biography Domain
  https://arxiv.org/pdf/1603.07771.pdf

- ToRead: Controlling Linguistic Style Aspects in Neural Language Generation
  https://arxiv.org/pdf/1707.02633.pdf

- InterestingDirection: KG and Recommendation system
  https://arxiv.org/pdf/2003.00911.pdf

- Fundmental LM: Language Models 1996 https://aclanthology.org/J96-1002.pdf
  Found through WIKI https://en.wikipedia.org/wiki/Language_model
  CRF2001: https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers

- LM Review: 2019 https://arxiv.org/abs/1906.03591

- Sentence Rep: Sanjeev Arora on random walk and sentence rep:
  https://aclanthology.org/W18-3012.pdf

- NLP text Generation notes: https://zhuanlan.zhihu.com/p/162035103

- zhihu KB MemNN: https://zhuanlan.zhihu.com/p/163343976

- LSTM for drawing Deepmind: https://arxiv.org/pdf/1502.04623.pdf

- char-RNN2015: Andrej Karpathy LSTM https://karpathy.github.io/2015/05/21/rnn-effectiveness/
  http://vision.stanford.edu/pdf/KarpathyICLR2016.pdf

- Samuel R. Bowman在Generating Sentences from a Continuous Space中使用VAE进行文本生成。这种方法可以对句子间插值。
  https://arxiv.org/abs/1511.06349

- zhihu Notes on NLG: https://zhuanlan.zhihu.com/p/188446640

- conferences: ACL EMNLP NAACL

- GoolgeNN SENNA 2011: https://www.jmlr.org/papers/volume12/collobert11a/collobert11a
  Representation Learning 2012 https://arxiv.org/pdf/1206.5538.pdf
  Schizophrenia detection?? https://www.sciencedirect.com/science/article/abs/pii/S0165178121004315
