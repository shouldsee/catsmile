---
title: "1003: 高斯随机变量的重参数化技巧(Reparametrisation)"
date: 2022-03-24T13:39:01+08:00
draft: false
---

不使用的后果：假设噪音 \epsilon\epsilon 恒为0，那么隐变量空间的连续性可能会变差，更难进行差值[TBC Needs Evidence]
最近受VAE启发在考虑随机激活函数(stochastic activation function)的可能性，然后就碰到了随机变量无法求微分这个梗。确切地说是采样算符(sampling operator)无法求微分。还好大牛们留的代码给了我一点提示。
回忆对于确定性的神经网络y = f(A) = f(wx+b)y = f(A) = f(wx+b)，和损失函数E(y)E(y)，可得如下反向传播关系:
\begin{aligned} \frac{d E}{d x} = {{d E}\over{d y}} {d y \over d A} {d A \over d x} \\ = {{d E}\over{d y}} \, f'(A) \, w \end{aligned} % \frac{\delta}{\delta }\begin{aligned} \frac{d E}{d x} = {{d E}\over{d y}} {d y \over d A} {d A \over d x} \\ = {{d E}\over{d y}} \, f'(A) \, w \end{aligned} % \frac{\delta}{\delta }
简洁起见令{\delta_x}=\frac{d E}{d x}{\delta_x}=\frac{d E}{d x}
{\delta_x} = {\delta_y} \, f'(A) \, w{\delta_x} = {\delta_y} \, f'(A) \, w
考虑完经典情况后，接下来考虑当ff并不是一个确定函数而是随机变量比如f(A)\sim N(A,1)f(A)\sim N(A,1)，这个时候该如何处理f'(A)f'(A)？答案藏在在重参数化里。考虑重写f(A)为
f(A) = A + \epsilon \\ \epsilon \sim N(0,1)f(A) = A + \epsilon \\ \epsilon \sim N(0,1)
此时可直接上微分算子得
f'(A)= \frac{d A}{d A} + \frac{d \epsilon}{d A} = (1 + 0) = 1f'(A)= \frac{d A}{d A} + \frac{d \epsilon}{d A} = (1 + 0) = 1
总结:反向传播算法并不必须应用在完全Deterministic的神经网络上。对于含有随机激活函数的网络应用反向传播可以自然地导出一个随机的梯度下降过程。
参考

    本文是阅读该博客的笔记。
