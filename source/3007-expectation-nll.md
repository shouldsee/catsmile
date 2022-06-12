#! https://zhuanlan.zhihu.com/p/527022595

# 3007: 混合模型和期望函数

[CATSMILE-3007](http://catsmile.info/3007-expectation-nll.html)

## 前言

- 目标: 辨析期望的似然,和似然的期望之间的关系
- 结论: EOL是几个模型的下界.
- 备注: $p$ 符号在下文中被严重滥用了....谁有比较好的符号建议吗?

我们知道,混合模型,可以理解为测度空间上的特殊卷积, 比如高斯混合模型,可以认为是离散的点密度,用高斯核进行热扩散.
因此混合模型的pdf一般可以写成子pdf的期望,这种形式也可以理解为(概率的期望).相对应的,在大部分使用softargmax,和sigmoid
注意力的神经网络里,实际上在候选值之间进行了插值(即期望),再计算似然,可称为(期望的似然).

个人猜测,EOP更加接近于神经科学中population encoding的编码形式,而LOE是目前神经网络采取的形式.目前之所以采取LOE,
是因为在gpu架构上比较好实现,而EOP并不是那么好实现

我从直觉上的猜测是,似然的期望,可以更好地处理离散相空间的突变,而期望的似然,可以更好地确保模型的连续性.

## 概率的期望的对数 LEOP, log expectation of probability

$$
\begin{align}
p(x) &=  \sum_z p(x|c) p(c) \\
p(x) &=  \sum_k p(x|c=k) p(c=k) \\
\log p(x) &= \log \sum_k p(x|c_k) p(c_k) \\
\log p(x) &= \log E_{p(c)\sim z }[ p(x|c) ] \\
LEOP(x,z) &=  \log \sum_k p(x|c_k) p(c_k)
\end{align}
$$

从 $z_k$ 中恢复出 x. 我们可以计算此时期望的似然


## 似然的期望 EOL, expectation of likelihood

这里似然指对数概率,是概率期望的下界

$$
\begin{align}
p(x) &=  \sum_z p(x|c) p(c) \\
p(x) &=  \sum_k p(x|c=k) p(c=k) \\
\log p(x) &= \log \sum_k p(x|c_k) p(c_k) \\
\log \sum_k p(x|c_k) p(c_k)  &\geq \sum_k p(c_k) \log p(x|c_k) \\
EOL(x,z) &= \sum_k p(c_k) \log p(x|c_k)\\
(凸函数性质)\  LEOP(x,z) &\geq EOL(x,z)
\end{align}
$$

如果考虑简单的高斯分布 

$$
\begin{align}
\log p(x|c_k) = -(x-A_k)^2 \\
EOL(x,z) = \sum_k p(c_k) \log p(x|c_k)\\
= \sum_k - p_z(c_k) (x-A_k)^2
\end{align}
$$

## 期望的似然 LOE, likelihood of expectation

考虑一个确定性的解码器,从隐态中恢复出 $\bar x= g(z)$. 简单的一个操作是再次利用 $p(z_k)$ 直接在原型上(prototype, $A_k$ ),计算期望来进行解码

$$
\begin{align}
\bar x &= E_{p(c_k)\sim z}[ A_k ] = \sum_k p(c_k) A_k \\
p(x|\bar x) &= \exp [ -(x-\bar x )^2] \\ 
\log p(x|\bar x) &\propto -(x-\bar x )^2 \\
LOE(x,z)&=-(x- \sum_k p_z(c_k) A_k )^2
\end{align}
$$

由于 $f(A) =-(x-A)^2$ 是个上凸函数,所以由凸函数性质再次有

$$
\begin{align}
f(E(A)) &\geq E(f(A)) \\
-(x-E(A))^2 &\geq -E[(x-A)^2] \\
-(x- \sum_k p_z(c_k) A_k)^2 &\geq \sum_k -p_z(c_k) (x-A_k)^2 \\
LOE(x,z) &\geq EOL(x,z) 
\end{align}
$$


## 对比

$$
\left \{
\begin{align}
LEOP(x,z) &=  \log \sum_k p(x|c_k) p(c_k) \\
EOL(x,z) &= \sum_k p(c_k) \log p(x|c_k)\\
LOE(x,z) &=-(x- \sum_k p_z(c_k) A_k )^2
\end{align}
\right.
$$

联立两个不等式,可以发现是不能传递的,也就是EOL同时是以上这几个模型的下界.

$$
LOE(x,z) \geq EOL(x,z) \leq LEOP(x,z)
$$

可以发现,LOE模型,相比EOP,可以认为是插入了一个中间变量 $\bar x$

在EOP中,模型的似然

$$
\begin{align}
\log p(x|z) &= \log  E_{p(c)\sim z }[ p(x|c) ] \\ 
&= \log \sum_k p(x|c_k,z) p(c_k|z) 
\end{align}
$$

其中第一个p可以是高斯分布,第二个p是分类分布(categorical)

在LOE中,模型的似然

$$
\begin{align}
p(\bar x) &= \delta(\bar x, \sum_k p(c_k|z) A_k  ) \\
\log  p(x|z) &= \log  \int  \partial \bar x .p(\bar x) p(x|\bar x, z) \\
&= \log  \int  \partial \bar x . \delta(\bar x, \sum_k p(c_k|z) A_k  )\cdot p(x|\bar x, z) \\
&= \log  p(x|\bar x=\sum_k p(c_k|z) A_k , z=z) 
\end{align}
$$

最后这个p可以就是一个高斯分布. 我们可以看到,LOE在相空间$A_k$上对  $p(c_k|z)$  求取了期望,而EOP在概率空间 $p(x|c_k,z)$ 上对  $p(c_k|z)$ 求取期望.

LOE模型的使用,意味着在相空间$A_k$上进行插值必须是有意义的,也就是使用一个连续的相空间