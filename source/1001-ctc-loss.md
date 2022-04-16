---
title: "1001: CTC Loss 结构解析"
date: 2022-03-24T13:39:01+08:00
draft: false
mathjax: true
---


(1001-ctc-loss)=
# 1001: CTC Loss 结构解析

- CTC全称：Connectionist Temporal Classification 连接派时域分类（？）
- CTC解决的问题：对齐问题alignment/空格问题
- 不使用CTC的后果：标签空间 $\set{i}$ 和迭代空间 $\set{j}$ 强制耦合，模型对于标签空间的标注会很敏感。
- CTC如何规避这个后果：通过加入免费的（跳过空白）这个操作，CTC只对标签空间的序列顺序变更敏感(A--B -> B-- A)，对绝对位置不敏感，因为绝对位置的变更无法穿透B操作。CTC将所有的顺序相同的标签序列归入同一个等价类，并通过求和来测量这些等价类的概率。也就是说，CTC在B操作后的空间计算概率，而不在有空格的空间计算概率。
- CTC的可能问题：CTC不能直接处理需要表示空格的场景。
- CTC的具体场景和形式:
  - 监督向量 $l_{bik}=1$ , $\text{if}. L_{bi} = k   l_{bik}=1 . \text{if}. L_{bi} = k$
  - 数据向量 $x_b$ ，一般为高维数据有维度c此处省略
  - 结构操作B：跳过空格(-)连接标签
  - 一般的标签比对损失模型:

      \begin{align}
      \text{loss} &=  \sum_{_{bik}}  l_{bik} \log  \pi_{bik} \\   
      &= \sum_{bi} \log P(\Pi_{bi}  = L_{bi} |x_{b},\theta) \\  
      &= \sum_{b} \log P(\{ \Pi_{bi} \}  = \{ L_{bi} \} |x_{b},\theta) \\
      \text{where} \ \pi_{bik}&= P(\Pi_{bi}  = k |x_{b},\theta)
      \end{align}

    可以看出，在做交叉熵计算的时候，已经做出了，给定隐变量后，标签生成是条件独立的假设。
    其中b为batch，i为监督向量位置索引，k为监督向量维度索引，这个交叉熵比较常见，CTC的主要贡献在于重写了后面的概率P


      \begin{align}
      P( \{\Pi_{bi} \} |x_{b},\theta)
      &= \sum_{\{z_{bjk}\}\in B^{-1}(\{\Pi_{bi} \})}  P( \{\Pi_{bi} \}\},\{{z_{bjk}}\}|x_b,\theta) \\
      &= \sum_{\{z_{bjk}\}\in B^{-1}(\{\Pi_{bi} \}\})}  P(\{{z_{bjk}}\}|x_b,\theta) P( \{\Pi_{bi} \} |x_{b},\theta)
      \\ &= \sum_{\{z_{bjk}\}\in B^{-1}(\{\Pi_{bi} \})}  P( \{\Pi_{bi} \}\},\{{z_{bjk}}\}|x_b,\theta)
      \\ & =\sum_{\{z_{bjk}\}\in B^{-1}(\{\Pi_{bi} \}\})}  P(\{{z_{bjk}}\}|x_b,\theta)
      \end{align}

     （由于 $\Pi = B^{-1}(Z)$ 是多对一映射，可以省略 $\Pi$ ）
      由于穷举序列 $\{z_{jk}\}$   的复杂度为指数级别 $O(K^J)$，
      直接暴力计算这个混合过程是低效的，因此CTC的贡献之一，是在（给定隐变量后，标签生成是条件独立的）
      假设下，也就是序列的每个位置可以单独采样
      $P(\{{z_{jk}}\}_{b}|x_b,\theta) =\prod_{j }  P({\{{z_k}\}_{bj}}|x_b,\theta)$ ，
      应用前向-后向思想（可能源自HMM,动态规划）给出了一个快速计算这个混合概率的方法，复杂度为 $O(JK)$ 。

- 具体的forward和backward计算方法：[TBC] 待添加
- 备注：看起来这个forward和backward的迭代并不是固定的，因此计算图取决于具体的监督向量，针对不同的目标Pi，计算这个混合分布的方法不同。
- 参考：
    - 白裳：一文读懂CRNN+CTC文字识别
    - Graves et al. 2006
