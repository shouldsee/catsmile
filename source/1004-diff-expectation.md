#! https://zhuanlan.zhihu.com/p/554548651

# 1004: 期望函数的梯度计算 Gradient of Expectation

- 目标: 讨论期望函数的梯度计算的一般原理
- 背景与动机: 
    - 不使用的后果：哦天哪，你没法往神经网络里塞期望函数了，所以强化学习没戏了，AlphaGo/AlphaZero/AlphaStar集体阵亡。
    - 求期望是计算KL的必要过程.我们常常会对期望函数求导
    - 提高神经网络的表达能力，将其热力学化
    - 一个经典的例子是VAE中的重参数化的目标就是计算一个期望函数作为损失
- 结论: 
- 备注: 
- 关键词: 
- 展望方向:
    - [TBC,补充具有DAG形式的结构上的期望计算]
- CHANGELOG:
    - 20220816 做DLM的时候加入了一个恒等式子



### 一个对期望求导的恒等式

最近求离散序列上的期望的导数的时候碰上了期望函数不能之间扔进pytorch的问题,顺手推了一下期望函数的导数.

$$\begin{align}
&\nabla_m E_{q(z)}[v(z)] 
\\& = \nabla_m \sum_z q(z) v(z)
\\&= \sum_z \nabla_m [ q(z) v(z)]
\\&= \sum_z (\nabla_m [ q(z)]  v(z) + q(z)\nabla_m v(z) )
\\&= \sum_z (q(z)\nabla_m [ \log q(z)]  v(z) + q(z)\nabla_m v(z) )
\\&= \sum_z (q(z)\nabla_m [ \log q(z)]  \text{stopGrad} [v(z)] + q(z)\nabla_m v(z) )
\\&= \sum_z (q(z)\nabla_m [ \log q(z)]  sg[v(z)] + q(z)\nabla_m v(z) )
\\&= \sum_z (q(z)\nabla_m [ \log q(z) sg(v(z)) ] + q(z)\nabla_m v(z) )
\\&= \sum_z q(z)\nabla_m \left[ \log q(z) sg(v(z)) +  v(z) \right]
\\&= E_ {q(z)} \nabla_m \left[ \log q(z) sg(v(z)) +  v(z) \right]
\end{align}$$

### 特例:高斯分布

对于 $z=\mu+ \sigma\epsilon$ 形式的重参数化,易证明我们需要如下的梯度修正项.用到了重参数化的各位都可以试试看,原则上这可以确保方差的梯度更加精确.我自己推的时候也常常把这项漏掉以为是零了.这个项表征的主要是采样出来的密度,因为 $\sigma$ 越大,密度越稀薄,所以越大的奖励,就对应 $\epsilon$ 附近的越高的局部密度,也就偏好越小的方差. 换句话说,有点类似于 log-jacobian

[TBC,加入高维模式]

$$
\nabla_m \log q(z) 
=  \nabla_m ( -\log \sigma + c + {||z-\mu||^2 \over 2 \sigma^2}  )\\
=  \nabla_m ( -\log \sigma + {1 \over 2}  )\\
=  \nabla_m  (-\log \sigma)  \\
$$


## 参考
  - Attention，MCMC，REINFORCE
  - <https://towardsdatascience.com/attention-in-neural-networks-e66920838742>
  - <http://proceedings.mlr.press/v37/xuc15.pdf>
- 天津包子馅儿：强化学习进阶 第六讲 策略梯度方法
  - <https://hal.archives-ouvertes.fr/hal-02968975/file/Generalized_Stochastic_Backpropagation.pdf>
  - <http://proceedings.mlr.press/v89/xu19a/xu19a.pdf>
