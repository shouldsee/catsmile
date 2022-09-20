# 1016: 环参数化 Toroidal Parametrisation

[CATSMILE-1016](http://catsmile.info/1016-torus.html)

## 前言

- 目标: 构建一个循环的参数空间
- 关键词: 参数化,激活函数
- 动机: 
  - LayerNorm可以确保向量模长有限,但是梯度总会指向相空间以外的方向.
  我们考虑一种循环的参数空间,希望能用来确保相空间的稳定性
 
对于d维的相空间 $x_d$, Layernorm一般尝试确保L2范数固定,但是我们不能
保证梯度指向单位球的切线方向,因此每次计算都需要重新norm

$$
\sum_d x_d^2 = c
$$

确保值空间有界还可以用各种有界函数,比如取余数,但是存在导数突变的问题.
因此这类有界函数的输出不能直接进入梯度,而是需要引导出一个距离函数,再
使用RBF径向基函数进行激活.

$$
s(x) =x \  \text{ mod } 1 
$$

考虑这个空间上的度量函数.

$$
\begin{align*}
d_e(x,y) &= \min( (x_e-(y_e-1))^2,(x_e-(y_e))^2,(x_e-(y_e+1))^2 \\
d(x,y) &= \sum_e d_e(x,y) \\ 
f(x,y) &= \exp(-\sum_e d_e(x,y))
\end{align*}
$$

## 例子: 向量夹角的概率

利用超球面面积，和夹角的sin，可以求出n维空间单位球上随机向量的夹角为

$$\begin{aligned}
p(\theta) 
&= {\Gamma(n/2)\over \Gamma((n-1)/2)} {\sin^{n-2}(\theta) \over \sqrt{\pi}}
\\ &= {\Gamma(n/2)\over \Gamma((n-1)/2)} {\cos^{n-2}(\theta - \pi / 2) \over \sqrt{\pi}}
\\
p(\psi) 
&= {\Gamma(n/2)\over \Gamma((n-1)/2)} {\cos^{n-2}(\psi) \over \sqrt{\pi}} \psi \in [-\pi/2,\pi/2]
\end{aligned}$$

采样的时候直接用invertcdf进行对称的采样，然后进行平移。头疼的是，cdf需要写一些积分。对于分数的幂次，可能没有closed form解。那其实用一些分段技巧进行近似采样就够用了。

## 参考

- Jerkwin哲科文20130318 <https://jerkwin.github.io/2013/03/18/%E7%A9%BA%E9%97%B4%E4%B8%AD%E4%B8%A4%E9%9A%8F%E6%9C%BA%E5%90%91%E9%87%8F%E9%97%B4%E5%A4%B9%E8%A7%92%E7%9A%84%E6%A6%82%E7%8E%87%E5%AF%86%E5%BA%A6%E5%88%86%E5%B8%83/>