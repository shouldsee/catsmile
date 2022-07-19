#! https://zhuanlan.zhihu.com/p/542457081
---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 1028-从KL散度到海氏距离 From KL divergence to Hellinger Distance

[CATSMILE-1028](http://catsmile.info/1028-hellinger-kl.html)


```{toctree}
---
maxdepth: 4
---
1028-hellinger-kl.md
```

## 前言

- 目标: 阐述海氏距离和KL散度之间的关系
- 背景与动机:
  - 为计算独热向量的方差找一个简单,对称的工具
- 结论: 
  - 如果要计算方差,还是得用 $D_H^2$ 平方海氏距离
- 完成度: 
- 备注: 
- 关键词: 
- 展望方向:
- 相关篇目:
  - [CATSMILE-1019](./1019-vaml)
- CHANGLOG:
  - 20220716 INIT

对于机器学习中的离散数据,我们一般用独热向量表示.通常这意味着使用交叉熵损失约束对应的数据.在K分类上,记预测值为 $x_k$ , 数据值为 $y_k$ ,我们有负KL损失项

$$\begin{align}
- D_{KL}(y||x) &= c - CE(y,x) \\
&= c + \sum_k y_k \log x_k
\end{align}$$

但这个框架有个问题,就是对于独热向量间,无法计算对称的距离.如果硬要计算
对称距离,就得加上labelSmoothing避免对0取log,这就需要引入额外参数了.

我们考虑用HellingerDistance来替代KLDistance

$$\begin{align}
D_H(y||x)^2 &= {1\over 2 }\sum_k(\sqrt{y_k} -\sqrt{x_k} )^2 \\
- D_H^2(y||x)  &= -1 + \sum_k \sqrt{y_k}\sqrt{x_k}  \\ 
- D_H^2(y||x) &= -1 + \sqrt{x_k}
\end{align}$$

我们可以比较一下两者的区别,对于独热向量来说,其实正好差一个指数函数

$$\begin{align}
- D_H^2(y||x) &= -1 + \sqrt{x_k} \\
              &= -1 + \exp {1\over 2}\log  x_k \\
              &= -1 + \exp \left(c - {1\over 2}  D_{KL}(y||x) \right) 
\end{align}$$

所以平时的交叉熵损失也可以写成一个变化过的SquaredHellingerDistance.
我们甚至可以说,交叉熵是利用了 $\log(1+x)$ 进行了变换的一个目标. 它的好处应该在于数值更加稳定

$$\begin{align}
L(m) = \log ( 1 - D_H^2(y||x)) \\
     = c-{1\over 2}  D_{KL}(y||x) 
\end{align}$$

因此,我们可以由此得出一个在非独热的,归一化的向量上也可以应用的推广,对数海氏距离,LogHellingerDistance.这其实可以用来替代余弦距离,也就是:(非对数的关系在Sohangir2017有过讨论)

$$\begin{align}
- D_{LH}(y||x) &=  \log ( 1 - D_H^2(y||x))  \\
&=  \log ( \sum_k \sqrt{y_k x_k})   \\
\end{align}$$

### 特例: 伯努利分布之间的散度

可以想见,对数化的损失会更加平缓一些

$$\begin{align}
1 - D_H^2(y||x) &= \sum_k \sqrt{y_k}\sqrt{x_k} \\
&= \sqrt{p_x p_y } + \sqrt{(1-p_x)(1-p_y)} 
\end{align}$$

```{code-cell}
print(2 + 2)

from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt
import numpy as np
print(1)
plt.ion()
plt.plot((0,0),(1,1))
plt.show()

```

### 特例: 高斯嵌入空间的H散度

假设我们有一个嵌入空间来计算  $x_k$ , 那么由于LogHellingerDistance
和KLDivergence在独热向量上的等价性,我们可以直接观察KL散度的梯度

$$\begin{align}
x_k &= \text{softargmax}_k( -\beta || z - u_k||^2 )\\
-D_{LH}(y||x) &=  - {1\over 2}D_{KL}(y||x) \\
          &=   {1\over 2}\log {\exp( -\beta || z - u_k || ^2) \over \sum_k \exp( -\beta || z - u_k || ^2)}  \\
          &=  - {1\over 2}\beta || z - u_k || ^2  - {1\over 2}\log \sum_k \exp( -\beta || z - u_k || ^2)  \\
- D_H(y||x)^2 &= -1 + \exp (-D_{LH}(y||x))\\
&= -1 + \sqrt{\exp( -\beta || z - u_k || ^2) \over \sum_k \exp( -\beta || z - u_k || ^2) }
\end{align}$$ 

从这个角度看,套一个log似乎能让导数有一些良好的性质?

我们知道,对于归一化因子难以计算的情况,使用一阶导数会更加好一些.但是头疼的地方在于空间的每个点上的归一化因子是不一样的,所以这个技巧看起来并不能直接施展...让我们再看一下 $\log(1+x)$ 对于梯度有啥影响,感觉有一点自适应的意思,让更大概率的项目,梯度更加小. 让小概率的项目,梯度更加大

$$\begin{align}
p_k &= (1+x_k) \\
f(x) &= \sum_k {\partial \over \partial m} \log (1 + x_k) \\
&= \sum_k  {1\over 1 + x_k} {\partial \over \partial m} x_k \\
&= \sum_k  {1\over p_k} {\partial \over \partial m} p_k \\
\end{align}$$

### 讨论

$-D_{LH}(y||x)$ 在独热向量之间为负无穷小.但是在计算嵌入向量的时候可以转化为简单的平方距离

$1-D_{H}^2(y||x)$ 在不同的独热向量之间为0.在同样的独热向量之间为1.

结论:如果要计算方差,还是得用 $D_H^2$ 平方海氏距离

### 其他: 平方(根)散度

平方散度相比HellingerDistance来说不能用到概率的归一性,计算起来并不是很方便,仅放在这里做参考

$$\begin{align}
D_{MSE}(y||x) = \sqrt {\sum_k (y_k-x_k)^2}
\end{align}$$

[TBC,差两张图]

## 参考

- 低维流形假设, Pope2021: <https://arxiv.org/abs/2104.08894.pdf>

- Sohangir2017: Improved sqrt-cosine similarity measurement  <https://journalofbigdata.springeropen.com/articles/10.1186/s40537-017-0083-6#Abs1>

- <https://en.wikipedia.org/wiki/Divergence_(statistics)>

- <https://en.wikipedia.org/wiki/Hellinger_distance>

- Zhihu358895758-张戎 <https://zhuanlan.zhihu.com/p/358895758>