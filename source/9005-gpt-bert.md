---
title: 9005-GPT和BERT对比-无监督语言模型
date: 2022-04-01T11:39:04+08:00
mathjax: true
---

# 9005: GPT和BERT对比-无监督语言模型

- 不使用的后果: 无法从大型语料库中对语言进行无监督学习

- 直观介绍: MLM尝试从上下文中预测MaskedToken的实际值,目标是最小化
  $L=-\sum_{t} x_{tk} \log \pi_{tk}$

  对于未masked的token,

  $$y_{tk}=\pi_{tk} \in \{1,0\}$$

  因此贡献为0. 对此进行推广,

  $$\pi_{tk}=f(g(x_{tk})),g(x)=\text{perturb}(x)$$

  如果没有掩码扰动,
  那么L等于0. 我们也可以认为我们要求 $L(f(g(x_{tk})))<L(g(x_{tk}))$
  其中f代表BERT施加的修复,g代表掩码扰动, 也就是最小化dL

- $$
  \begin{aligned}
  dL &= L(f(g(x_{tk}))) - L(g(x_{tk})) \\
     &= -\sum_{t} x_{tk} \log f_{tk} +  \sum_{t} x_{tk} \log g_{tk}
  \end{aligned}
  $$

- 注意到f和g只在掩码位置有差别,因此可以简化为掩码m上的和,进一步令$g_{mk}$在掩码位置
  为均匀分布或MaskToken上的delta,dL可以近似MLM目标,也就是
  我们希望bert模型,能够把扰动后的句子,拉回原来的句子.

- $$
  \begin{aligned}
  dL &= -\sum_{m} x_{mk} \log f_{mk} +  \sum_{t} x_{mk} \log g_{mk}
  \end{aligned}
  $$

- 形式:
- MLM Masked Language Model
