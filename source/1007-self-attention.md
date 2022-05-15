(self_attention)=

# 1007: Self Attention 自注意力机制 20220428

- 函数全称: Self Attention 自注意力

- 函数解决的问题/不使用的后果: 无法训练BERT,GPT等现代模型.

- 函数解决改问题的原理: 通过引入归一化的注意力机制,将模型动态化.

- 函数可能存在的问题: 关于序列长度的$O(L^2)$计算量
- 函数在某一场景下的具体形式:
    - 考虑QKV注意力,其中Q,K,V都是 $R^\text{embed_dim}\rightarrow R^\text{hidden_dim}$ 的线性变换
    - $$
    \begin{align}
    a_{ij}(x) &= \exp(x_i^T Q^T K x_j) \\
    y_{i}(x)  &=  V \sum_j x_j \frac { a_{ij} }{ \sum_j( a_{ij} )}
    \end{align}
    $$
- 函数的具体计算方法: TBC
- 函数备注:
    - data as code: 用动态图而不是静态图去处理数据,vanilla rnn和vanilla cnn可以认为是典型的静态图模型
    - 传统的模型试图用一个静态的模型去对数据进行统一的编码. Transformer区别于传统模型就在于
    让数据自己对自己进行编码. 数据既是被编码的客体,也是指引编码的主体.而实际宏模型只保留得出动态编码的映射.    
    - 传统模型: $z=f(x)$
    - 自指模型: $z=f_x(x)$

    - 特别地,Transformer模型考虑了一种特殊的稀疏模型形式.对于每个数据节点(H),通过一个二阶的注意力头(R)关联到一个伴随节点(T),
    再从伴随节点上传递消息. $dH = V T$,也即最大化 $E=H V T$

    - 因为数据即模型,因此在对节点做修改的时候,也隐式地修改了节点间的连接图,除非存在正交关系 $|| (Q^T K V) ||_F \approx 0$.
    观察BERT的权重, 结果:main_task_3这并不成立. BERT并没有对各层进行重用.而且通过残差连接,可能可以避免矩阵crosstalk产生的影响.
    但是这种neo-stationarity只在距离相近的层内成立
      - 以0为output.embedding,i为encoder.layers.i-1.output
      - 第8层,也即layer7可以兼容第4层的输入,但是对第3层及以下的兼容就差一些.

    - 如果只考虑纯粹SelfAttention,那么层间节点间Jacobian的F范数应该只受到注意力的影响,在没有隐藏节点的情况下,这意味者信息不能被取舍.
    同时也意味者Jacobian的F范数可以捕捉到变量间的瞬态关联性.

    - 在关联性确定了以后,信息可以在给定的图上进行传播. 在类似BERT的结构里,这个图是动态计算的,每生成一层新的,就计算一次图.
    这样的好处是图有更强的动态性. 另一种办法是直接把图完全计算出来, 然后直接在得到的计算图上做前向传播.这种思考的好处,是
    可以把BERT看成是一个动态图模型,假设计算可以分解成12层.且每层的Jacobian可以用QKV来表示出来. 这也可能可以解释为什么
    要加入ResidualConnection,其好处就在于可以确保计算图的计算都能收到第一层的信息.

- 函数参考信源
    - 自注意力机制首次于"Attention is all you need"得到大规模推广
    - <https://mcbal.github.io/post/transformers-from-spin-models-approximate-free-energy-minimization/>
    - Reviews    
      - Efficient Transformers: A Survey <https://www.semanticscholar.org/paper/Efficient-Transformers%3A-A-Survey-Tay-Dehghani/7e5709d81558d3ef4265de29ea75931afeb1f2dd>
      - A Survey of Transformers <https://arxiv.org/abs/2106.04554>
    - Performer,  Set Transformer,
    - A Structured Self-attentive Sentence Embedding: sentiment analysis<https://www.semanticscholar.org/paper/A-Structured-Self-attentive-Sentence-Embedding-Lin-Feng/204a4a70428f3938d2c538a4d74c7ae0416306d8>
      - Star Transformer<https://arxiv.org/abs/1902.09113>
      - ETC: Encoding Long and Structured Inputs in Transformers <https://arxiv.org/pdf/2004.08483.pdf>
      - Longformer <https://arxiv.org/abs/2004.05150>
      - Big Bird

## 可选方向

ETC-源码解析

<https://arxiv.org/pdf/2004.08483v5.pdf>

<https://github.com/google-research/google-research/tree/master/etcmodel>


## Relay nodes

Construct a dynamic model to probe whether it's possible to
bypass the quadratic attention.

The first-order attention seeks to perform graph
message passing on a simplified network where
each node of the prototype network is extracted from
the sequence by a softmax average.


$$
\begin{align}
\sum_i A_{ik} &= 1 \\
A_{ik}        &= \frac{ \exp( \mu_k^T x_i ) }{ \sum_i \exp(\mu_k^T x_i)} \\
B_k           &= \sum_i A_{ik} x_i \\
E             &= \sum B_{k1} W_{k1,k2} B_{k2} \\
E             &= \sum_{k2} - (\sum_{k1} W_{k1,k2}B_{k1} -  B_{k2})^2 \\
\end{align}
$$

## Simple KED model

Named due to complexity $O(K E D)$ K for hidden node size, E for embedding size,
D for hidden interaction size

$$
\begin{align}
\sum_i A_{ik} &= 1 \\
A_{ik}        &= \frac{ \exp( \mu_k^T x_i ) }{ \sum_i \exp(\mu_k^T x_i)} \\
B_k           &= \sum_i A_{ik} x_i \\
\text{(just for intuition)}
E             &= B^T L^T R B \\
{\partial{E} \over \partial{x_i}} &=\sum_k { \partial E \over \partial B_k}  {\partial B_k \over \partial x_i}
\\ { \partial E \over \partial B_k} &\triangleq  L^T R B
\end{align}
$$


## QK注意力的局限性

注意到QK注意力自应用伊始就伴随着几个关键的辅助组件

- 位置表征:关于位置编码和位置嵌入的讨论
- 层标准化: LayerNorm
- 非线性前向模型: FeedForwardNonlinearFunction

## 猜测一: 层标准化的必要性来自于QK关系函数的一些不好的性质

考虑对于注意力  $A(x,y)$ 的如下计算

$$
\begin{align}
x,y \in R^E \\
f(x,y) &= x^T Q^T K y / c \\
A(x,y) &= {\exp(f_{x,y}  \over \sum_{y} \exp f_{x,y} }
\end{align}
$$

其中$f$如果作为一个分布的引导函数,会引导出一个不能标准化的病态分布.

$$
\begin{align}
Z &= \int_{y \in R^E} \exp f(x,y) \partial y \\
  &= \int_{y \in R^E} \exp x^T Q^T K y / c. \partial y \\
  &= \infty \\
\end{align}
$$

同时对于缩放具有线性倍增的性质,这对于希望在语义向量里进行嵌入显然是极为不利的.

$$
f(x,ky) = kf(x,y)
$$

对于这种病态性,常见的处理是加上一个正则惩罚项,考虑如下变形$f2$

$$
\begin{align}
x,y \in R^E \\
f(x,y) &= x^T Q^T K y  - \frac{1}{2} y^T y  \\
\end{align}
$$

通过令导数为零可以解出全局极大值,也因此可以找到一个对应的多元高斯分布

$$
\begin{align}
\frac{\partial f}{\partial y}    &= K^T Q x - y \\
\text{(set gradient to zero)  }  0 &= K^T Q x \\
\text{argmax}_y f(x,y)  &= K^T Q x
\end{align}
$$

通过这样的一个形式,可以避免用LayerNorm来消除线性倍增,因为二次项的增长压倒了线性项 $f2(x,ky)\neq kf2(x,ky) $
但是这样的一个形式仍然不能满足关系抽取的一些性质. 最大的问题在于,这个函数对于y上的所有扰动都非常敏感 $f2(x,y)$
也就是说,y是不太自由的. 为了给y留出一些自由度,考虑用$K y$来替代 $y$, 这样$f3$就对于 $kernel(K)$ 里的扰动不再敏感

$$
\begin{align}
f(x,y) &= x^T Q^T K y  - \frac{1}{2} y^T K^T K y  \\
\frac{\partial f}{\partial (K y)}    &= Q x - K y \\
\text{(set gradient to zero)  }  0 &= Q x - K y \\
  K y &= Q x
\end{align}
$$

这样看起来导出的概率函数在$Ky=Qx$时取到极大值,也就是$Ky$服从关于$x$的高斯分布.注意到如果$y^T K^T K y=c$保持恒定,
那么$f3$退化成$f1$.

$f3$的形式目前具有如下属性

$$
\begin{align}
\text{(标准化常数有界)  } & Z &= \int_{y \in R^E}  \exp f(x,y) \partial y \neq \infty \\
\text{(对于某些方向的y的扰动具有不变性)  } & f(x, y + \partial y) &= f(x,y)\text{ if }\partial y \in  \text{kernel}(K)
\end{align}
$$

## 猜测二: 良好的关系表征应当允许简单的空间嵌入

但是作为注意力使用,仍无法简单地表征位置,比如说右一,右二. 必须借助于类似ROPE([kexue.fm archive](https://kexue.fm/archives/8265))
的形式才能表征这类关系.但ROPE分数$x^T Q^T (R_m^T R_n) K y$也存在一些问题.

- 在原位的注意力核 $R_0^T R_0=I,m=n=0,x=y$,分数也就等于 $x^T Q^T K x$,
- 在长程衰减上具有奇怪的震荡并不平滑
- ROPE的本质是把 $x^T W y$的联络矩阵$W$重新参数化为关于位置$m,n$的乘积形式,但并没有提供一个简单地表征"右边第一个"的简单的$W$
这来自于PostionalEncoding和PostionalEmbedding范式的区别.
- 在ROPE里把幅角$\theta$设为0,就能够恢复与空间无关的注意力头.ROPE的单参数形式,意味着对于m,n是不交换的 $R_m^T R_n\neq R_n^T R_m$,
因为 $(m-n)\theta \neq (n-m) \theta$.

用矩阵乘法来构造符合结合律的联络矩阵是非常构造性的, 因为这样产生的$Q,K$矩阵一般是不能相等的,
也就是意味在逻辑上这种关系近似于物体x的Q等于物体y的K,那么如果Q等于K,这种关系就显然被自指关系
$x=y$所满足$Qx=Ky$,因此,为了避免自指,我们必须用x的Q等于物体y的K来表示他们的关系,也因此
不能使用一个简单的投影空间来表述空间关系.

要解决这种矛盾性,最简单的办法就是修改 $Qx = Ky$ 这个式子,让极大值在另一个点取到,比如说 $Qx + b= Ky $

$$
\begin{align}
{\partial f_4 \over \partial {K y}}  &= Qx - Ky + b \\
f_4(x,y)                             &=  x^T Q^T  K y - \frac{1}{2} y^T K^T K y + b^T K y \\
                                     &=  (x^T Q^T -   \frac{1}{2} y^T K^T + b^T)  K y  
\end{align}
$$


进一步如果假设f属性在x和y的表征是相同(比如空间位置都是对应向量的第一个元素$x_1,y_1$,可以约束$Q=K$.
此时极值点的约束关系类似于[TransR](https://www.semanticscholar.org/paper/Learning-Entity-and-Relation-Embeddings-for-Graph-Lin-Liu/994afdf0db0cb0456f4f76468380822c2f532726)
的损失函数(见[zhihu archvie](https://zhuanlan.zhihu.com/p/147542008)).

$$
\begin{align}
{\partial f_5 \over \partial {K y}}  &= Kx - Ky + b \\
f_5(x,y)                             &=  x^T K^T  K y - \frac{1}{2} y^T K^T K y + b^T K y \\
                                    &=  (x^T K^T -   \frac{1}{2} y^T K^T + b^T)  K y  \\
                                    &=  y^T K^T (K x + b -  \frac{1}{2} K y)
\end{align}
$$

我把基于$f5$的注意力称为$\text{kattention}$.注意到这个函数关于$x,y$是不交换的,
这是因为目前只考虑了x为定值的情况,根据简单的对称性应该有


$$
\begin{align}
f_6(x,y)   &= - \frac{1}{2} x^T Q^T Q x + x^T Q^T  K y - \frac{1}{2} y^T K^T K y + b^T K y  + c^T Q x\\
&= - \frac{1}{2} x^T Q^T ( Q x -  K y ) - \frac{1}{2} ( y^T K^T - x^T Q^T ) K y + b^T K y  + c^T Q x \\
&= - \frac{1}{2} x^T Q^T ( Q x -  K y ) + \frac{1}{2} (  Q x - K y )^T K y + b^T K y  + c^T Q x \\
&= - \frac{1}{2} (Q x - K y )^T ( Q x -  K y )  + b^T K y  + c^T Q x \\
\end{align}
$$



## Kattention

Knowledge-model-based attention, 基于知识模型的注意力函数

kattention注意力头具有多个参数

$$
\begin{align}
E & \text{ for embedding dimension}\\
D &\text{ for relational dimension}\\
Q &\in R^{E \times D} \\
K &\in R^{E \times D} \\
b &\in R^D
\end{align}
$$

具体计算如下

$$
\begin{align}
E(x,y) &= \text{kattention}(Q,K,b)(x,y) \\
       &=  x^T Q^T  K y - \frac{1}{2} y^T K^T K y + b^T K y \\
       &=  (Q x -   \frac{1}{2} K y + b)^T  K y  \\
E_{ij} &= E(x_i,x_j) \\
A_{ij} &= \text{softmax}_j(\beta E_{ij}) \\
       &= \frac{ \exp( \beta E_{ij}) }{\sum_j \exp( \beta E_{ij})}  \\
       &=  \frac{ \exp \beta (x_i^T Q^T  K x_j - \frac{1}{2} x_j^T K^T K x_j + b^T K x_j )}
       {\sum_j \exp \beta (x_i^T Q^T  K x_j - \frac{1}{2} x_j^T K^T K x_j + b^T K x_j)  }
       \\
       &=  \frac{\exp(\beta(Q x_i -   \frac{1}{2} K x_j + b)^T  K x_j )}
        { \sum_j \exp(\beta(Q x_i -   \frac{1}{2} K x_j + b)^T  K x_j )}
       \\
y_{ik} &= \sum_{j} A_{ij} x_j
\end{align}
$$

接下来就是实验的时间了.实际计算中,应该还是采用因子分解的式子会比较简单.

- Kattention看起来像是Mean-Squared-Error在线性空间上的推广额.

(timestamp 20220427)

## Transformer and model mixing

Transformer uses a dynamic model to predict the missing word. In practice, selecting the best word is much easier than a
predicting the exact word.  Here the words candidates are model candidates. For a transformer, the
candidates are words.
