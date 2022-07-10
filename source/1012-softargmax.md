#! https://zhuanlan.zhihu.com/p/526932672

# 1012: 软极参函数的求导: Differentiating softargmax (or so-called softmax)
 
[CATSMILE-1012](http://catsmile.info/1012-softargmax.html)

(实际开写的时候,并不会按照数据格式去填表单...目前基本上就是划分成(前言,具体计算,特例,总结展望,参考)几个大块)

## 前言

- 名称: 软极参函数,softargmax,注意经常被误称为softmax函数.实际上softmax应当对应logsumexp函数,
而softargmax才对应常见的求和后为1的那个 $f(k) = {e^{H(k)} /(\sum_{k}e^{H(k)}})$
- 目标: 示范softargmax的求导过程
- 备注: 
  - argmax勉强翻译为(极参函数),也就是找到取极值的参数的函数.
  - softargmax求导一直没有看到比较好的实战演练,这次实在没法绕过去了就手算了一次,不过结果还是挺好看的.
  - softargmax求导比logsumexp求导要复杂许多.
  - 在自动微分时代,一些炼丹师不再那么重视手算导数了,但是对于导数的分析,对于分析神经网络应该还是有很核心的作用的
  - 直接用梯度对模型做编码,效果并不一定很好. 有些模型的初始化对后续影响很大


## 具体计算

$$
\begin{align}
{\partial \over \partial z_d}  p_k(z) &= {\partial \over \partial z_d}  { \exp {h_k(z)} \over \sum_k \exp {h_k(z)} } \\
(微分恒等)\dots&= { [\sum_k \exp {h_k(z)}] \exp {h_k(z)} {\partial \over \partial z_d}  h_k(z) -[\sum_k \exp {h_k(z)} {\partial \over \partial z_d} h_k(z)]  \exp {h_k(z)}    \over (\sum_k \exp {h_k(z)})( \sum_k \exp {h_k(z)})  }  \\
(提因子)\dots&= {\exp h_k(z) \over ( \sum_k \exp {h_k(z)}) }{ [\sum_k \exp {h_k(z)}] {\partial \over \partial z_d}  h_k(z) -[\sum_k \exp {h_k(z)} {\partial \over \partial z_d} h_k(z)]     \over (\sum_k \exp {h_k(z)})  }  \\
(裂项)\dots&= {\exp h_k(z) \over ( \sum_k \exp {h_k(z)}) }
( {\partial \over \partial z_d}  h_k(z) - 
{
 [\sum_k \exp {h_k(z)} {\partial \over \partial z_d} h_k(z)]     \over \sum_k \exp {h_k(z)}  }  
)
\\
(常数入和)\dots&= {\exp h_k(z) \over ( \sum_k \exp {h_k(z)}) }
( {\partial \over \partial z_d}  h_k(z) - 
\sum_k {
 \exp {h_k(z)}     \over \sum_k \exp {h_k(z)}   }  {\partial \over \partial z_d} h_k(z)  
)\\
(换符号)\dots&= p_k(z)
[ {\partial \over \partial z_d}  h_k(z) - 
\sum_k { p_k(z)  }  {\partial \over \partial z_d} h_k(z)  
]\\
(换下标)\dots&= p_k(z)
[ {\partial \over \partial z_d}  h_k(z) - 
\sum_j { p_j(z)  }  {\partial \over \partial z_d} h_j(z)  
]\\
\end{align}
$$


上式就是softargmax单独求导的核心等式,需要整理一下才能得到jacobian.可以看到,softargmax对于梯度有标准化的作用,因为直接对k求和得到0.(因为 $p_k$ 求和固定为1,当然导数为0啊),

$$
\begin{align}
\sum_k
{\partial \over \partial z_d}  p_k(z) &= \sum_k p_k(z)
[ {\partial \over \partial z_d}  h_k(z) - 
\sum_j { p_j(z)  }  {\partial \over \partial z_d} h_j(z) ] \\
&=  \sum_k p_k(z)
[ {\partial \over \partial z_d}  h_k(z) ] - [\sum_k  p_k(z)] \sum_j { p_j(z)  }  {\partial \over \partial z_d} h_j(z)  \\
&=  \sum_k p_k(z)
[ {\partial \over \partial z_d}  h_k(z) ] - 1 \cdot \sum_j { p_j(z)  }  {\partial \over \partial z_d} h_j(z) \\
&=0
\end{align}
$$

对于一个$t_k$关于$p_k$的期望函数,可以有

$$
\begin{align}
{\partial \over \partial z_d}  \sum_k p_k(z) t_k &= \sum_k t_k {\partial \over \partial z_d}  p_k(z) \\
&= \sum_k t_k   p_k(z)
[ {\partial \over \partial z_d}  h_k(z) - 
\sum_j { p_j(z)  }  {\partial \over \partial z_d} h_j(z)  
] \\
&= \sum_k [t_k   p_k(z)
 {\partial \over \partial z_d}  h_k(z)] -\sum_k 
\sum_j t_k   p_k(z)  p_j(z)   {\partial \over \partial z_d} h_j(z)  
] \\
&= \sum_k [t_k   p_k(z)
 {\partial \over \partial z_d}  h_k(z)] -\sum_k 
\sum_j t_j   p_j(z)  p_k(z)   {\partial \over \partial z_d} h_k(z)  
] \\
&= \sum_k [t_k   p_k(z)
 {\partial \over \partial z_d}  h_k(z)] -\sum_k 
[\sum_j t_j   p_j(z) ] p_k(z)   {\partial \over \partial z_d} h_k(z)  
] \\
&= \sum_k (t_k - \sum_j t_j   p_j(z) )   p_k(z)
 {\partial \over \partial z_d}  h_k(z)
 \\
\end{align}
$$

这个形式的好处在于,梯度的下标只在一项里出现 ${\partial \over \partial z_d}  h_k(z)$,是统一的.之所以能做到这点,
是因为在k上进行了求和,不然就没法把下标如此整理了


### 特例:交叉熵

再有我们常见的交叉熵cross-entropy也可以归纳为一个$t_k$,假设有独热性质 $\sum_k y_k=1$ 

$$
\begin{align}
CE(z,y) &= \sum_k y_k \log p_k \\
{\partial \over \partial z_d} CE(z,y) &= \sum_k y_k {1 \over  p_k }{\partial \over \partial z_d} p_k  \\ 
&=\sum_k t_k {\partial \over \partial z_d}  p_k(z)  \\
t_k &= {y_k \over  p_k } \\
{\partial \over \partial z_d} CE(z,y) &= \sum_k t_k {\partial \over \partial z_d}  p_k(z) \\
&= \sum_k (t_k - \sum_j t_j   p_j(z) )   p_k(z)
 {\partial \over \partial z_d}  h_k(z) \\ 
&= \sum_k ({y_k\over p_k} - \sum_j {y_j \over p_j}    p_j(z) )   p_k(z)
 {\partial \over \partial z_d}  h_k(z) \\ 
&= \sum_k ({y_k\over p_k} - \sum_j y_j    )   p_k(z)
 {\partial \over \partial z_d}  h_k(z) \\ 
&= \sum_k (y_k - p_k(z) \sum_j y_j    )   
 {\partial \over \partial z_d}  h_k(z) \\ 
&= \sum_k (y_k - p_k(z) \cdot 1    )   
 {\partial \over \partial z_d}  h_k(z) \\ 
\end{align}
$$

对于 $y_k=1$ 的位置,会有正流从$h_z$反传,否则对于 $y_k=0$, 只有负流

### 特例:L2损失混合自编码器 

考虑一个从L2混合模型引出的自编码函数

$$
\begin{align}
NLL(z,x) &=  \sum_e - ( x_e  - \sum_k { \exp ({h_k(z)}) \over \sum_k \exp ({h_k(z)}) } A_{ke} )^2\\
  &= \sum_e - (x_e - \sum_k p_k(z) A_{ke})^2 \\
p_k(z) &= softargmax(h_k(z),k)\\
h_k(z)   &= \sum_d -(z_d - c_{kd})^2
\end{align}
$$

通过对复合函数求导,我们可以观察softmax的jacobian matrix是如何作用于导数上的.尽管我们知道JVP能够很好描述
反向传播的梯度变化,但是在实际操作中对于矩阵求导的要求还是比较高,因此我从求和式子入手,把简单的方法先写出来,以资参考.

$$
\begin{align}
{\partial NLL(z,x) \over \partial z_d} &= \sum_e 2 (x_e - \sum_k p_k(z) A_{ke})) {\partial \over \partial z_d} [\sum_k p_k(z) A_{ke}] \\ 
&= \sum_e 2 (x_e - \sum_k p_k(z) A_{ke})) \sum_k [ {\partial \over \partial z_d} p_k(z)] A_{ke}
\end{align}
$$ 

注意到原先的 $NLL(z,x)$的导数也可以写成 $\sum_k t_k {\partial \over \partial z_d}  p_k(z)$的形式,

$$
\begin{align}
{\partial NLL(z,x) \over \partial z_d}  &= \sum_e 2 (x_e - \sum_k p_k(z) A_{ke})) \sum_k [ {\partial \over \partial z_d} p_k(z)] A_{ke} \\ 
&=  \sum_k \sum_e [2 (x_e - \sum_k p_k(z) A_{ke})) A_{ke}] {\partial \over \partial z_d} p_k(z)  \\ 
&=\sum_k t_k {\partial \over \partial z_d}  p_k(z) \\

t_k &= \sum_e 2 (x_e - \sum_k p_k(z) A_{ke})) A_{ke} \\
{\partial \over \partial z_d} h_k &= 2(c_{kd} -z_d ) 
\end{align}
$$ 

$$
\begin{align}
{\partial NLL(z,x) \over \partial z_d}  &=\sum_k t_k {\partial \over \partial z_d}  p_k(z) \\
&= \sum_k (t_k - \sum_j t_j   p_j(z) )   p_k(z)
 {\partial \over \partial z_d}  h_k(z)\\
&= \sum_k 2(t_k - \sum_j t_j   p_j(z) )   p_k(z)
 (c_{kd} - z_d)\\
\\ 
\end{align}
$$

注意到这里 $t_k$ 的形式较为复杂,直接写出来会比较杂乱,因此在实际计算中,我们只需要存储 $t_k$ 这个k维向量即可,
并通过这个算子反传到 $z_d$ 上,这就大致是梯度反传的思想.

## 特例：　EOL函数

$$\begin{align}
EOL(x,z)
  &= \sum_k p(c_k) \log p(x|c_k)\\
{ \partial \over  \partial z_d} EOL(x,z)
  &=  
 \sum_k  \log p(x|c_k) { \partial \over \partial z_d}  p(c_k) \\
&=\sum_k t_k {\partial \over \partial z_d}  p_k(z) \\

t_k &=   \log p(x|c_k) = -\sum_e (x_e-A_{ke})^2\\


\end{align}$$



## 矩阵形式 (待整理)




## 参考

```{footbibliography}
```



