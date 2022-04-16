---
title: 9004-RNN生成实验
date: 2022-03-31T11:39:04+08:00
mathjax: true
---

# 9004: RNN生成实验


## 目标:

测试一个架构上更接近HMM的RNN生成模型，避免tokenwise生成过程

## Sentence As Function Decomposition


"Function decomposition" is coined to differentiate from matrix decomposition.
Here the target parameters are not matrices, but functions.

sentence = observed token sequence: `$Y_{bt} = k  <=> y_{btk} = 1$`
sentence generation

projects a vector $z$ from $R^N$ to $Y^{TK}$

### Objective

finds parameters that minimise NLL `$(z_b,w) = \text{argmin}_{z_b,w} (-\sum_{btk} y_{btk} \log f_{tk}(z_b,w))$`

### Structure of $f(z,w)$

`$$
\begin{aligned}
g_{t+1}(z,w)&=w1 \cdot g_{t}(z,w) + w2 \\
f_{t} &= \text{softmax}_k(w3 \cdot g_t(z,w) + w4)
\end{aligned}
$$`


### Optimisation

Use RMSprop. Adagrad doesn't work
