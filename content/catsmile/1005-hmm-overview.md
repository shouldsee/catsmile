---
title: "1005: HMM简介"
draft: false
mathjax: true
---
- 函数全称: Hidden Markov Model 隐马尔科夫模型(隐马万岁！)
- 函数解决的问题/不使用的后果: 无法有效地进行序列对齐，比如POS-tagging，Needle-Wuensch算法。
- 函数解决改问题的原理: 通过引入隐藏状态，避免对于观测状态的直接建模。
- 函数可能存在的问题: 算起来麻烦。
- 函数在某一场景下的具体形式:
   - 观测序列 $O_{t}$
   - 隐藏序列 $X_{t}$, $x_{ik}=1 \ if X_{it}=k$
   - 转移矩阵 $A_{ij}=P(X_{t}=j|X_{t-1}=i)$
   - 发射模型 $P(O_t|X_t)=f(O_t,X_t)$   
- 函数的具体计算方法
   - 一般有如下阶段，训练，推断。其中训练分为:
     - Baum-Welch训练: 一种EM算法，目标是优化所有可能的隐藏序列下，给定观测序列集合的似然。
     - Viterbi训练: 一种拟EM算法，目标是优化最可能的隐藏序列下，给定观测序列集合的似然。
     - 梯度训练: 一种梯度算法，可以优化BW目标或者Viterbi目标。
   - 推断分为:
     - Forward-Backward算法: 可以计算位置$t$的分布，以及$t,t+1$转移的分布，以及
- 函数备注
  - 一般认为BW训练比Viterbi效果更好，但是详细区别有待考证。BW相比与Viterbi，类似于GMM相比于
  Viterbi的关系。
- 函数参考信源
  - Stanford LSA352 https://nlp.stanford.edu/courses/lsa352/lsa352.lec7.6up.pdf
