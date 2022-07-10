#! https://zhuanlan.zhihu.com/p/529733047
# 1017: 高斯隐变量自编码器 GLGAE Gaussian Latent Generative Autoencoder

[CATSMILE-1017](http://catsmile.info/1017-glgae.html)

## 前言

- 目标: 记录高斯自编码器的关键参数
- 关键词: 参数化,激活函数
- 动机: 
  - 单独整理, 1013写的太乱了
- 应用:
  - 非线性嵌入
  - 数据压缩
- 评价:
  - GLGAE的动机还是不够明确的,因为引入一个连续隐变量z去近似
  k上的分布,纯粹是为了连续而连续. 在两个连续的领域x和z之间
  强行插入了一个离散的domain k, 必然造成x和z的概率距离变大.
  这纯粹是为了把离散编码连续化的一种尝试.但实际上我用一个一维线段
  可能都比再套一个高斯混合模型更加有效率. 纯粹是个toy model,
  看不出啥实际好处
- CHANGELOG:
  - 增加变分近似视角
  - 到处漏log,改了一些typo

## 负损失函数

考虑参数集$A_k, B_k$,使得隐变量和观测之间通过k个组分
通信,那么编码分布解码后的损失可以定义为期望的解码概率.

$$
\begin{align*}
L_2(z,x,\{m_k\},b) &= \log E_{q(k|z)} p(x|k)\\
&=  \log \sum_k q(k|z)p(x|k) \\
&=  \log \sum_k { \exp(-||B_k-z||^2) 
\over b + \sum_k \exp(-||B_k-z||^2)}
\exp(- ||x- A_k ||^2)
\end{align*}
$$


## 推导过程

考虑高斯混合生成分布

$$
p(x|k) = \exp(- ||x- A_k ||^2)
$$


$$
\begin{align*}
p(x) &=  \sum_k p(x,k) \\
     &=  \sum_k p(k|x) p(x) \\
p(k|x) &= {p(x,k) \over \sum_k p(x,k) } \\
&= {p(x|k)p(k) \over \sum_k p(x|k)p(k) }\\
&= {p(x|k) \over \sum_k p(x|k) } \\ 
&= { \exp(- ||x- A_k ||^2) \over \sum_k \exp(- ||x- A_k ||^2) }
\end{align*}    
$$

考虑用连续参数z指定的分布 $q(k|z)$ 去近似 $p(k|x)$,
目的是施加正则化,限制 $p(k|x)$ 的信息量.然后考虑优化
后验分布上的似然

$$
\begin{align*}
q(k|z)  &= { \exp(-||B_k-z||^2) 
\over b + \sum_k \exp(-||B_k-z||^2)}\\
L_1(z,x,\{m_k\}) &=  \log E_{p(k|x)}[p(x|k)] \\
&=  \log \sum_k p(k|x) p(x|k) \\
&=  \log\sum_k q(k|z) {p(k|x)\over q(k|z)} p(x|k) \\
&\geq  KL( q(k|z)  || p(k|x) ) + E_{q(k|z)}[ \log p(x|k)] \\
&\geq  E_{q(k|z)}[ \log p(x|k)] \\
&= \sum_k q(k|z) (- ||x- A_k ||^2) \\
\end{align*}    
$$

当然比较令人困惑的是这个初始的定义,为啥非要算似然的期望.
能不能在变分分布上直接计算期望呢?从思考上也没有什么大问题呀,一个好的隐编码分布,应当最大化观测概率的期望.当然我们发现,这个期望的下界恰好和 $L_1$ 的下界是一样的

$$
\begin{align*}
L_2(z,x,\{m_k\}) &=  \log E_{q(k|z)}[p(x|k)] \\
&=  \log \sum_k q(k|z)p(x|k) \\
&\geq E_{q(k|z)}[\log p(x|k)]
\end{align*}
$$

从自编码的角度讲,实际上我们想要达到的就是 $\max L_2$,比如 $\max_{z_b} L_2(z_b,x_b)$ 就意味着找到 $x_b$ 的最佳编码.取下界,说不定效果是可以平滑一下参数空间.
考虑 $L_2$ 的具体形式下的编码过程

$$
\begin{align*}
L_2(z,x,\{m_k\}) &=  \log \sum_k q(k|z)p(x|k) \\
&=  \log \sum_k { \exp(-||B_k-z||^2) 
\over b + \sum_k \exp(-||B_k-z||^2)}
\exp(- ||x- A_k ||^2)
\end{align*}
$$

对隐变量求导

$$
\begin{align*}
{\partial \over \partial z}L_2 &= {\partial \over \partial z}  \log \sum_k q(k|z)p(x|k) \\
&=  \sum_k {p(x|k) \over \sum_k q(k|z) p(x|k)} {\partial \over \partial z}  q(k|z)
\\
&=  \sum_k {p(x|k) \over \sum_k q(k|z) p(x|k)} {\partial \over \partial z}  \exp(-||B_k-z||^2)  
\\
&=  \sum_k {p(x|k) \over \sum_k q(k|z) p(x|k)}  \exp(-||B_k-z||^2)  \cdot 2(B_k - z)
\\
&=  2 \sum_k {p(x|k) q(k|z)\over \sum_k q(k|z) p(x|k)}    \cdot (B_k - z)
\end{align*}
$$

把梯度设置为零 ${\partial \over \partial z}L_2=0$ 就给出了一个交替优化的形式.如果是下界的话,那就对应把
$p(x|k)\rightarrow \log p(x|k)$ 做一个对数替换

$$
\left \{
\begin{align*}
z &= {\sum_k p(x|k) q(k|z) B_k \over
\sum_k {p(x|k) q(k|z)} }
\\ 
q(k|z)  &= { \exp(-||B_k-z||^2) 
\over b + \sum_k \exp(-||B_k-z||^2)}
\\
p(x|k) &= \exp(-||A_k-x||^2)
\end{align*}
\right.
$$

之前在1013的时候直接迭代似乎找不到比较好的结构,主要是因为取了一个不恰当的下界,导致梯度比较混乱. 观察这里的迭代子式我们可以发现隐变量z并不会跑出$B_k$构成的凸锥之外,也因此不会产生1013所描述的push-away问题.

## 变分近似视角

高斯混合的解码分布 $p(x|k)$ 引导出一个编码分布 $p(k|x)$ ,通过再一个解码分布 $q(k|z)$ 引导出一个编码
分布 $q(z|k)$ 那么我们就具有 $q(x,k,z)$ 的联合分布了. 这里还要加一个条件,那就是希望z的先验是均匀的.
这样可以把x转化到一个均匀分布, $q(z)\propto 1$

$$
x|k \leftarrow k \rightarrow z|k
$$

$$\begin{align*}
q(x,k,z) &= q(z) q(k|z) p(x|k) \\
L(m) &= -KL(p_{data}(x)||q(x)) \\
&= \int p_{data}(x) \log q(x) dx + H(p_{data})\\
&= \int p_{data}(x) \log \sum_z \sum_k q(z) q(k|z) p(x|k) dx + H(p_{data})
\end{align*}
$$

而实际上 $z$ 上的积分是不太好求的,因此我们用点估计的下界来近似

$$
\log \sum_z \sum_k 1\cdot q(k|z) p(x|k)
\geq \max_z \log \sum_k q(k|z) p(x|k)
$$

于是,优化的过程就转化为

$$
\begin{align*}
L(m) &= \int p_{data}(x) \log \sum_z \sum_k q(z) q(k|z) p(x|k) dx + H(p_{data}) \\
&\geq  \int p_{data}(x) \max_z \log \sum_k q(k|z) p(x|k) dx + H(p_{data}) \\
&\leq  \max_m \int p_{data}(x) \max_z \log \sum_k q(k|z) p(x|k) dx + H(p_{data})\\ 
&= \max_m \int p_{data}(x) \max_z L_2(z,m,x) dx + H(p_{data})
\end{align*}
$$

这样就解释了之前比较令人困惑的构造了 $l_2(x,m,z)= \log E_{q(k|z)}[p(x|k)]$, 原来是来自点估计的要求
