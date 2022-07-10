#! https://zhuanlan.zhihu.com/p/530856390
# 1022: 局部线性生成式自编码器 (LLGAE) Locally Linear Generative Autoencoder

[CATSMILE-1022](http://catsmile.info/1022-llgae.html)


```{toctree}
---
maxdepth: 4
#caption: mycap
numbered: 0
---
1022-llgae.md
```

## 前言

- 目标: 
- 背景与动机: 注意到全局线性模型的一个问题, 就是如果实际的流形仅仅是局部线性,那全局线性
假设就是有问题的. 因此 $W$ 需要也成为隐变量 $z$ 的函数. 这样可以结合离散隐变量和连续隐变量的优点,
更高效地编码数据
- 结论: 
- 备注: 
  - LGAE对应PCA, LLGAE对应混合的PCA
- CHANGELOG:
  - 20220629 加入具体操作形式,和fashion_mnist结果
- 关键词: 
- 展望方向:
  - 有点糊,考虑chain一个CNN提高分辨率 


### 生成式自编码器模型的一般形式

自编码器是一个可以方便地用来定量研究数据压缩的框架.和压缩感知是很相关
的思考角度,考虑流形上的所有样本 $x\in \{X\}$, 自编码器需要对任意样本
进行(编码->解码)两步操作,来保持恢复流形结构所需要的信息.这种信息的保持
一般用一个度量函数来衡量. 一般地,流形可以用一个向量空间上的测度来表示 $p(x)$, 

在 [CATSMILE-1014](./1014-gae) 我们通过KL变分负损失有下界定义了GAE,
一个只需要解码器的自编码器模型.并使用均匀隐变量先验

$$\begin{align}
L_{GAE}(m) &=  E_{p_{data}(x)}[  l_m(z_0,t) ] \\ 
&\leq  E_{p_{data}(x)}[  \max_z q_w(z) q_r(x|z)] \\
&\leq -D_{KL}( p_{data}(x) || q_w(z) q_r(x|z)) +c \\
\end{align}
$$

### 特例: 局部线性的生成式自编码模型 LLGAE 


这里我们考虑一个特殊的具体的解码器形式, r是模型的一部分,用m替换, 我们用经典的低维线性子空间定义一个L2近似

$$
\begin{align}
\max_z \log { q_{r}(x|z) q_w(z)  }   &= \max_z \log \exp G(m,x,z)  \\
 &\propto \max_z -{1\over 2}||x - \mu_{m}(z) - W_{m}(z) b(z) || ^2  \\ 
 &\geq \max_{k(z)} \max_{b(k(z))}  -{1\over 2}||x - \mu_{m}(k(z)) - W_{m}(k(z)) b(k(z)) || ^2  \\ 
 &=  \max_{k(z)} \max_{b(k(z))} l_m(x,k(z),b(k(z)))\\
 &\geq  \max_{k(z)} l_m(  x,k(z),b(k(z)),t)\\
\end{align}
$$

实际操作中,我们只需要写出一个合理的 $l_m(z_0,t)$ 函数来近似求解 $\max_z$ ,这里的z可以分解为两个函数 $k(z),b(k(z))$,对于每个组分 $k(z)$, 我们都获取参数 $\mu,W$ ,并设置 $b(k(z_0))=0$ 来初始化, 然后利用梯度上升估算最大值的下界

$$\begin{align}
b(k,0)  &= b(k(z_0)) = \vec 0 \\
b(k,t) &=  b(k,{t-1})  + \beta_m {\partial \over \partial b  }l_m(  x,k, b, 0) |_{ b = b(k,{t-1})} \\
&= b(k,{t-1}) +  \beta_m  W_{mk}^T (x-\mu_{mk}-W_{mk}b(k,t-1))
\\
\max_{k(z)} l_m(  x,k(z),b(k(z)), t) &= \max_{k} l_m(x,k,b(k,t)) \\
k(z,t)&= \max_k  -{1\over 2}||x - \mu_{mk} - W_{mk} b(k,t) || ^2 \\
\end{align}
$$

至此, $k(z,t)$ 和 $b(k,z,t)$ 都已经解出,也就对初始函数估算了一个最大似然后验 $z_t$ ,我们可以用这个后验对应的
$l_m(x,k(z_t),b(k(z_t)))$ 作为优化目标,继续对m求导,来优化参数 $\mu_{mk},W_{mk}$

### 特例: 线性的生成式自编码模型 LGAE 

在 $|K| = 1$ 时, $\max_k$ 退化成identity函数,于是我们有

$l_m(x,k(z_t),b(k(z_t))) = l_m(x,k=1,b(t))  =-{1\over 2}||x - \mu_{m} - W_{m} b(t) || ^2$

注意这里的损失函数基本对应一个简单的自编码器,区别仅在于 $l_m$ 关于t是迭代的

### 讨论

注意到LLGAE需要对每个可选组分同时进行优化,复杂度是 $O(NKE)$,对每个样本,
需要做K*E次z上的梯度下降.这或许可以做一些采样或者嵌入算法进行加速,来减小$K$
与降噪自编码器DAE不同,LLGAE的目标是利用压缩瓶颈来探测数据结构. 模型的设计理念是,
在给定比特率的情况下,恢复效果越好,模型对于数据的理解就越合理.

LLGAE简单地通过引入混合变量 $k(z)$ 来允许模型从不同k个子模型中选择最佳点估计.因为k(z)之间是没有关系的,
所以模型必须对每个 $k(z)$ 都尝试解出 $b(k,t)$ 才能找到最优解码. 在解出这个编码的过程中, $k(z)$ 和 $b(k,t)$
都被赋予了直观意义: $k(z)$ 表征的是数据的类别 $b(k,t)$ 表示数据在这个类别中的属性. 这就允许属性只需要对局部
数据负责,而不需要对全局数据都有效,从而加强了模型的表达能力.

后续对于LLGAE的扩展可以有很多方向, 核心还是在于发现模型的不足之处, 比如没有使用卷积先验, 无法表征
物体的位置. 位置的表征是一个更消耗计算量的问题,因为模型需要针对每个位置假设都进行检验,并且按照一定的依据
组成更大的表征, 还要考虑多个物体的分布式表征,目前还没有做深入尝试

### 模型效果

fashion mnist 上平均L2 768

![markov-lm/gmm/Checkpoints/-S28-taskfashion-mnist-compress-shuffle1-depth1-graph-dim784-embed-dim20-iter-per-layer-1-kernel-size11-model-nameLLGAE-beta0.01-n-step5-p-null0.0-loglr-3.0_540_768.06268.pkl 
](./1022-llgae-p1.png)



## 草稿

### 特例: 局部线性的生成式自编码模型 LLGAE 

注意到全局线性模型的一个问题, 就是如果实际的流形仅仅是局部线性,那全局线性
假设就是有问题的. 因此 $W$ 需要也成为隐变量 $z$ 的函数. 一个简单的办法
是建立一个插值映射 $W(x) = f_k(x) W_k$.这个映射定义在$X$上比在$z$上要更加方便,因为$z$是线性空间内部的坐标,原则上不可能反过来定义线性空间本身.考虑高斯核形式的 $f_k(x)$ ,并加入偏移量 $\mu_k$ ,
得到

$$
\begin{align}
f_k(x)  &\propto \exp -(x-\mu_k)^2  \\
\log p(x|m,z) &\propto -||x - \mu_k(x) - W(x)_k z || ^2 \\
&= -||x -  \sum_k \text{softargmax}(-(x-\mu_k)^2) (\mu_k + W_k z)||^2
\end{align}
$$

注意这里使用了平滑处理,实际上把一个离散隐变量给消去了.如果联立
两个隐藏变量 $k,z$ 就会出现一个层级结构

$$
\begin{align}
L(x,m,k,z) &= \log p(x|m,k, z) \\
 &\propto - ||x - \mu_k(x) - W(x)_k z || ^2  \\ 
 &\leq \max_{\mu_k,W_k}\max_k \max_z - ||  x - \mu_k - W_k z|| 
\end{align}
$$

对于z上的优化,可以使用梯度下降.对于k上的优化,直接取最优值即可. 
为了公平和其他降维算法比较,也可以考虑用高斯核嵌入空间参数化 $f_k(w)$. 

注意到LLGAE需要对每个可选组分同时进行优化,复杂度是 $O(NKE)$,对每个样本,
需要做K*E次z上的梯度下降.这或许可以用一些采样或者嵌入算法进行加速,来减小$K$

## 参考

[CATSMIE-1014](/1014-gae)
