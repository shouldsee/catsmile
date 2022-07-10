#! https://zhuanlan.zhihu.com/p/536401527
# 1026: 梯度对齐的常用方法 (Recent methods in gradient matching)

[CATSMILE-1026](http://catsmile.info/1026-recent-sm.html)


```{toctree}
---
maxdepth: 4
---
1026-recent-sm.md
```

## 前言

- 目标: 梳理不同的GM技巧,都是为了高效准确地优化Fisher散度从而拟合概率模型
- 背景与动机: 
- 结论: 
  - Fisher散度的目标也可以理解为最小化对数似然比函数的局部方差, 
而KL散度的目标是最大化对数似然比的局部期望值
- 备注: Score matching (SM) 实在有点不雅, 而且Score本身隐含标量的意思,
后续我尽量以Gradient Matching (GM) 指代这些以Fisher divergence为目标的的模型. 但是对于已经常用的词语,估计很难修改了
- 关键词: 
- 相关篇目:
  - [CATSMILE-1024](./1024-ddpm-math)
  - [CATSMILE-1025](./1025-dsm)
- 展望方向
- CHANGELOG

统计学习中的变分方法一般通过最小化目标分布和变分分布之间的一个泛函,把概率建模问题转化为一个最优化问题,因此需要一个散度泛函来衡量近似的效果. 从最大似然框架对应着KL散度为目标的拟合,而Fisher散度是一个正在逐渐复兴的,跟KL散度有一定差别的变分目标,一般对应梯度对齐(Gradient matching, or score matching)模型的目标. Fisher散度的优势在于不需要对数概率函数,只需要其梯度即可,避免了引入归一化因子,让模型变得更加灵活了.

需要注意的是,梯度对齐本身并不是终极目标,实际的目标一般是无条件采样,有条件采样,自编码,或者似然计算. 从梯度对齐到解决这些实际问题,中间还有一些过程. Song2021可以说是给这些中间过程提供了一个很好的理论基础

容易证明Fisher散度是非负的,同时也是非对称的

$$\begin{align}
- D_{F}(p(x)||q(x)) &= -E_{p(x)}[{1\over 2}|| \nabla_x \log p(x) - \nabla_x  \log q(x) ||^2] \\
&= -E_{p(x)}[{1\over 2}|| \nabla_x (\log p(x) -  \log q(x)) ||^2] \\ 
&= -E_{p(x)}[{1\over 2}|| \nabla_x \log {p(x)\over q(x)}  ||^2] 
\end{align}$$

简单对比KL损失函数,感觉上Fisher损失是一个二阶的形式,感觉可以深挖一下背后的直觉. 对于任意位置,如果梯度的范数接近于零,那么意味着向任何方向都不能够改变  $\log p(x) -\log q(x)$ ,也就是说两个能量面是完全平行的. 这和KL目标有很大的区别,因为KL只要求 $\log q(x)$ 要尽量高于 $\log p(x)$

$$\begin{align}
- D_{KL}(p(x)||q(x)) 
&= E_{p(x)}[\log q(x) - \log p(x)]\\
&= E_{p(x)}[\log {q(x)\over p(x)}]
\end{align}$$


至于说为啥要用梯度的L2范数,我在[CATSMILE-1006](./1006-jacobian-matrix)中考察过Jacobian导数矩阵对于足够小的单位高斯噪音的响应, 结论是输入层的高斯扰动会造成一个和Jacobian的L2范数相等的输出扰动. 这里我们的梯度其实就是一个Jacobian的特例,所以在相空间上的局部扰动,投射到 $\log {p(x) \over q(x)}$ 时造成的局部方差就由局部的Fisher散度所刻画. 

$$\begin{align}
G_f(x) &= \frac{ Var_e(f(x+e))}{E_e(|e|^2)}= \\
&= \frac{E_e(|f(x+e)-f(x)|^2)}{E_e(|e|^2)} \\
&= \frac{1}{I} \sum_{i,k}  [J_f(x)_{ki}]^2
\end{align}$$

从这个角度讲,Fisher散度的目标也可以理解为最小化对数似然比函数的局部方差, 
而KL散度的目标是最大化对数似然比的局部期望值. 在这个意义上Fisher散度是一种二阶方法

### "是我眼瞎了? 还是梯度隐身了?" [TBC]

确实很奇怪,因为真实数据分布通常是未知的,通常SGD里的抽样算法给出的都是数据上采样出来的离散样本, 也可以理解为数据上的狄拉克分布. 问题在于,这玩意压根没有梯度啊? 怎么求导怎么算积分呢? 

$$
p_{data}(x) = {1 \over |B|} \sum_{b=1}^B \delta(x_b) 
$$

个人猜测这个可以通过加噪声后取无限小的极限来进行观察.因此对数据分布进行噪声增广,是改善问题性质的一个很好的办法,但是加多少噪声也是一个问题,得到的分布类似于一个KDE高斯核估计. 这个东西的梯度真是真的挺好计算了. 但是加多少噪声,得到什么样的分布,还要看看现有的这些方法,包括ScoreSDE,ScoreODE,NSCN等等的具体操作\

$$\begin{align}
p_{data,\sigma}(x) &= {1 \over |B|} \sum_{b=1}^B {1\over \sigma \sqrt{2\pi} }\exp(-{||x_b -x ||^2\over  2 \sigma^2})\\
&= {1 \over |B|} \sum_{b=1}^B  \mathcal{N}(x-x_b|0,\sigma^2 I )
\end{align}$$

### [TBC] 不同的GM方法技巧

1. ISM: implicit Score matching
    - 消去了数据分布的对数似然项目
    - Hyvarien?
1. ESM: Explicit Score Matching
    - 还不清楚
    - Hyvarien? Vincent2021
1. DSM: Denoising Score Matching
    - 可以看成是一个用某种方法增广数据后的简单无偏采样估计方法
    - Vicent2011
1. SSM: Sliced Score matching
    - 用随机投影近似ISM中Hession矩阵的迹
    - Song2019
1. FSM: Finite-difference score matching
    - 用有限差分近似ISM
    - Pang2020 

梯度对齐模型考虑的是用L2损失拟合对数概率在相空间上的梯度,形式上类似一个fisher divergence. 在文献中经常出现用DSM把ESM替换掉的情况,容易造成
读者断片.

$$\begin{align}
L_{ESM}(m) &= -\int p(x) ||s_m(x) - \nabla_x \log p(x)||^2.dx \\
     &= - E_{p(x)}[||s_m(x) - \nabla_x \log p(x)||^2] \\
L_{DSM}(m) &= -E_{p(x,y)}[||s_m(x) - \nabla_x \log p(x|y)||^2] \end{align}
$$

$$\begin{align}
P(A) P(B|A) = P(A,B) = P(B) P(A|B)
\end{align}$$

### 从梯度对齐模型中进行采样的方法

最近比较新的相关采样算法如下.

1. SMLD: Denoising Score Matching with Langevin Dynamics
    - Song2019 NCSN 采用的算法
    - 被总结成一个Predictor=Identity+Corrector=Model的算法
    - 初始化从 $x_T = N(x|0,\sigma^2_{max} I)$ 中进行采样, 
  然后在每个时间步t,进行M步的Langevin Dynmics. $x_{t-1} = f_{\theta_t}(x_t,M)$
    - $Var(x_T)$ 没有上界,所以又叫方差爆炸形式
2. DDPM: Denoising Diffusion Probabilistic Model
    - Ho2020 的采样
    - 被总结成一个Predictor=Model+Corrector=Identity的算法
    - 从 $x_T = N(x|0,\sigma^2 I )$ 初始化,然后在每个时间步t进行
  反向过程采样 $x_{t} = N(x| \mu_\theta(x_{t+1}), \Sigma_{\theta}(x_{t+1}))$
    - $Var(x_T) \approx Var(x_0)$, 因此又叫方差守恒形式
1. Predictor-Corrector
    - Song2021
    - 大致结合了SMLD和DDPM的两个步骤,每一步反向过程采样后,接M步的Langevin形式的MCMC.
    - PC是解SDE的通用技巧,不过不知道引入到这里有啥好处
1. Probability Flow, scoreODE
    - Chen2018,Song2021
    - 通过提取backward process的确定性形式,进行高效采样
    - 具体原理在Song2021.4.3和Appendix.D中有详细描述
    - 正向过程是扩散 -> 反向过程也是扩散 -> 扩散在概率空间上对应一个ODE.
    - 用随机投影近似了必要的计算.
    - 用ODE积分器可以加速采样
    - 我们可以这样理解scoreODE的结论,对于符合条件的随机逆向过程,存在一个
    确定性的逆向过程,使得两者的边缘分布等价. (在联合分布层面上,显然是不等价的). 这样的一个结论,意味着我们可以直接把原始空间内的采样,转化成一个
    ODE积分问题.

### neural ODE adjoint

$$
\nabla_t z = f_m(z,t)\\
L(z(t_1)) = L(z(t_0) + \int_{t_0}^{t_1} f_m(z(t),t).dt) \\ 
a(t) = \nabla_{z(t)} L \\
\nabla_t a(t) = \nabla_t \nabla_{z(t)} L = {\partial \over \partial t} ({\partial \over \partial z(t)}   L)\\
=- \nabla_{z(t)}^T L \nabla_z f_m(z,t) 
$$

### Instantaneous change of variable 

consider a distribution morphed by an ODE. According to Chen2018,
we have the 

$$\begin{align}
\nabla_t z &= f_m(z,t) \\
\nabla_t \log p_t(z) &= -tr(\nabla_z f_m(z,t)) \\
\log p_T(z(T)) -  \log p_0(z(0)) &= \int_0^T -tr(\nabla_z f_m(z(t),t)) dt \\
\end{align}$$ 

$$
\sum_z p(z(t))= 1\\
\nabla_t \log p_t(z) = \\
\int_z p_t(z) = 1 \\
\int_z \exp \log p_t(z) = 1 \\
\nabla_t \int_z \exp \log p_t(z) = 0 \\
\int_z  p_t(z) \nabla_t \log p_t(z) = 0 \\
$$

### 待添加 [TBC]

Bigdeli2019:DSM的奇特应用方式,同时拟合Fisher散度和KL散度?[TBC]
KexueFM-5716 <https://kexue.fm/archives/5716>

## 参考

- A connection between score matching and denoising autoencoders, Vincent 2011 <https://www.semanticscholar.org/paper/A-Connection-Between-Score-Matching-and-Denoising-Vincent/872bae24c109f7c30e052ac218b17a8b028d08a0>
- Fisher and Jensen–Shannon divergences: Quantitative comparisons
among distributions. Application to position and momentum atomic
densities, Antolin2009 <http://www.ugr.es/~angulo/papers/PDFS/JCP2009.pdf>

- Score-Based Generative Modeling through Stochastic Differential Equations, Song 2021. <https://arxiv.org/abs/2011.13456>

- Efficient Learning of Generative Models via Finite-Difference Score Matching <https://arxiv.org/abs/2007.03317>

- Learning Generative Models using Denoising Density
Estimators Bigdeli2019 <https://arxiv.org/abs/2001.02728.pdf>

- Neural ordinary differential equation, Chen2018 <https://arxiv.org/abs/1806.07366>

## 草稿

话说回来,KL散度是由Gibbs不等式保证非负的,也可以考虑其平方形式. 不过这个梯度看起来形式不咋地

$$\begin{align}
- D_{KLS}(p(x)||q(x)) 
&= E_{p(x)}[{1\over 2}(\log q(x) - \log p(x))^2]\\
&= E_{p(x)}[{1\over 2}\log {q(x)\over p(x)} \log {q(x)\over p(x)}]\\
- \nabla_m D_{KLS}(p(x)||q(x))  &= E_{p(x)}[ (\log p(x) - \log q(x)) \nabla_m \log q(x) ] 
\end{align}$$

不过如果看看把Fisher的散度求梯度和范数顺序变化看看会怎么样?

$$\begin{align}
- D_{F2}(p(x)||q(x)) 
&= -E_{p(x)}[ \nabla_x {1\over 2}|| \log {p(x)\over q(x)}  ||^2] \\
&= -E_{p(x)}[ \log {p(x)\over q(x)}  \nabla_x \log {p(x)\over q(x)} ] \\ 
&= -E_{p(x)}[ (\log {p(x) - \log q(x))}  (\nabla_x  \log {p(x)} - \nabla_x\log q(x) ) ] \\ 
&= c -E_{p(x)}[ \log p(x) \nabla_x\log q(x) - \log q(x)  \nabla_x  \log {p(x)} \\
&+ \log q(x)\nabla_x\log q(x) ) ] \\ 
\end{align}$$


