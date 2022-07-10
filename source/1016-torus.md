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

