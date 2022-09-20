
# 1033: [TBC] L2损失函数-正态分布

- Wrapped Normal Distribution (循环正态分布)


不使用的后果：你可能没法造很多很多很多很多神经网络了。大部分的神经网络可以认为在中间层隐式地采用了L2损失（？[TBC]）
正态分布的基本性质

    X \sim NormalX \sim Normal的主要意义在于P(x) \propto e^{-z^2}P(x) \propto e^{-z^2}，其中变量z是x的拉伸变换。z=\frac{x}{k}z=\frac{x}{k}
    概率密度函数的归一化要求E[1]=1E[1]=1，这提供了第一个约束条件。
    同样地，对二阶矩的考察E[X^2]-E^2[X]=\sigma^2E[X^2]-E^2[X]=\sigma^2提供了第二个约束条件。
    如果只考虑拉伸变换，则隐含了E[X]=0E[X]=0。显式地考虑E[X]=\muE[X]=\mu，给出了第三个约束条件。

引理:\int^{+\infty}_{-\infty}e^{-z^2}dz=\sqrt{\pi}\int^{+\infty}_{-\infty}e^{-z^2}dz=\sqrt{\pi}

1. 显式化正比关系
P(x) =c e^{-z^2}~~~,z=\frac{x}{k}P(x) =c e^{-z^2}~~~,z=\frac{x}{k}
2. 归一化约束
\begin{aligned} E[1]=1 \\ \int P(x)=1 \\ \int^{+\infty}_{-\infty} ce^{-z^2}dx = 1 \\ \int^{+\infty}_{-\infty} ce^{-z^2}d(kz) = 1 \\ ck\int^{+\infty}_{-\infty} e^{-z^2}d(z) = 1 \\ ck\sqrt{\pi}=1 \\ alt. ~~c=\frac{1}{\sqrt{\pi}} \frac{1}{k} \end{aligned}\begin{aligned} E[1]=1 \\ \int P(x)=1 \\ \int^{+\infty}_{-\infty} ce^{-z^2}dx = 1 \\ \int^{+\infty}_{-\infty} ce^{-z^2}d(kz) = 1 \\ ck\int^{+\infty}_{-\infty} e^{-z^2}d(z) = 1 \\ ck\sqrt{\pi}=1 \\ alt. ~~c=\frac{1}{\sqrt{\pi}} \frac{1}{k} \end{aligned}
3. 二阶矩约束:
\begin{aligned} E[X^2]-E^2[X]=\sigma^2 \\ but~E[X]=0 \\ E[X^2]=\sigma^2 \\ \int c e^{-z^2} x^2.dx = \sigma^2 \\ c\int (\frac{de^{-z^2}}{dz}\cdot\frac{1}{-2z}) (kz)^2.d(kz) = \sigma^2 \\ \frac{ck^3}{-2}\int (\frac{de^{-z^2}}{dz}) z.dz = \sigma^2 \\ \frac{ck^3}{-2}\left\{[e^{-z^2}z]^ {+\infty}_{-\infty} - \int ^ {+\infty}_{-\infty} e^{-z^2}.dz \right\} = \sigma^2 \\ \frac{ck^3}{-2} (0-(-\sqrt{\pi}))= \sigma^2 \\ \frac{ck^3}{2} \sqrt{\pi}= \sigma^2 \\ But~~ ck\sqrt{\pi}=1 \\ \frac{k^2}{2}=\sigma^2 \end{aligned}\begin{aligned} E[X^2]-E^2[X]=\sigma^2 \\ but~E[X]=0 \\ E[X^2]=\sigma^2 \\ \int c e^{-z^2} x^2.dx = \sigma^2 \\ c\int (\frac{de^{-z^2}}{dz}\cdot\frac{1}{-2z}) (kz)^2.d(kz) = \sigma^2 \\ \frac{ck^3}{-2}\int (\frac{de^{-z^2}}{dz}) z.dz = \sigma^2 \\ \frac{ck^3}{-2}\left\{[e^{-z^2}z]^ {+\infty}_{-\infty} - \int ^ {+\infty}_{-\infty} e^{-z^2}.dz \right\} = \sigma^2 \\ \frac{ck^3}{-2} (0-(-\sqrt{\pi}))= \sigma^2 \\ \frac{ck^3}{2} \sqrt{\pi}= \sigma^2 \\ But~~ ck\sqrt{\pi}=1 \\ \frac{k^2}{2}=\sigma^2 \end{aligned}
4. 一阶矩约束:
\begin{aligned} & \left \{ \begin{aligned} E[X]&=\mu \\ P(X&=x) = ce^{-z^2}\\ z&= \frac{x-\mu}{k} \end{aligned} \right . \\ &E[X] = \int xP(x).dx \\ &E[X] = \int (kz+\mu)ce^{-z^2}.d(kz+\mu) \\ &E[X] = \int (kz)ce^{-z^2}.d(kz) + \int (\mu) ce^{-z^2} .d(kz) \\ &E[X] = 0 + \mu \int ce^{-z^2} .d(kz) \\ &E[X] = 0 + \mu \int P(x) .d(x) \\ &E[X] = \mu \end{aligned}\begin{aligned} & \left \{ \begin{aligned} E[X]&=\mu \\ P(X&=x) = ce^{-z^2}\\ z&= \frac{x-\mu}{k} \end{aligned} \right . \\ &E[X] = \int xP(x).dx \\ &E[X] = \int (kz+\mu)ce^{-z^2}.d(kz+\mu) \\ &E[X] = \int (kz)ce^{-z^2}.d(kz) + \int (\mu) ce^{-z^2} .d(kz) \\ &E[X] = 0 + \mu \int ce^{-z^2} .d(kz) \\ &E[X] = 0 + \mu \int P(x) .d(x) \\ &E[X] = \mu \end{aligned}
综上，正态分布是以P(x) \propto e^{-z^2}P(x) \propto e^{-z^2}为核心的，用一阶矩和二阶矩参数化的一种一维概率分布。


