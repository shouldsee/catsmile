#! https://zhuanlan.zhihu.com/p/517507882
# 8100: zhihu vscode 测试

项目 <https://github.com/niudai/VSCode-Zhihu>



$$
\begin{aligned}

P(x) &= \int P(x \mid \mu) P(\mu) d\mu\\
&=\int_{-\infty}^{\infty} \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{2}}\frac{1}{b \sqrt{2 \pi}} e^{-\frac{1}{2}\left(\frac{\mu-a}{b}\right)^{2}} d\mu\\
&=\frac{1}{2 \sigma b \pi}\int_{-\infty}^{\infty} e^{-\frac{1}{2}\frac{(x-\mu)^2b^2+(\mu-a)^2\sigma^2}{\sigma^2 b^2}} d\mu\\
&=\frac{1}{2 \sigma b \pi}\int_{-\infty}^{\infty}exp\left(-\frac{1}{2}\left(\frac{\left(\mu-\frac{xb^2+a\sigma^2}{\sigma^2+b^2}\right)^2}{\frac{\sigma^2b^2}{\sigma^2+b^2}}+\frac{(x-a)^2}{\sigma^2+b^2}\right)\right)d\mu\\
&=\frac{e^{-\frac{1}{2}\frac{(x-a)^2}{\sigma^2+b^2}}}{2 \sigma b \pi}\frac{\sigma b}{\sqrt{\sigma^2+b^2}}\sqrt{2 \pi}\int_{-\infty}^{\infty}\frac{1}{\frac{\sigma b}{\sqrt{\sigma^2+b^2}}\sqrt{2 \pi}}exp\left(-\frac{1}{2}\left(\frac{\left(\mu-\frac{xb^2+a\sigma^2}{\sigma^2+b^2}\right)^2}{\frac{\sigma^2b^2}{\sigma^2+b^2}}\right)\right)d\mu\\
&=\frac{1}{\sqrt{\sigma^2+b^2}\sqrt{2 \pi}}e^{-\frac{1}{2}\frac{(x-a)^2}{\sigma^2+b^2}}\int_{-\infty}^{\infty} \mu\sim N\left(\frac{xb^2+a\sigma^2}{\sigma^2+b^2},\frac{\sigma b}{\sqrt{\sigma^2+b^2}}\right)d\mu\\
&=\frac{1}{\sqrt{\sigma^2+b^2}\sqrt{2 \pi}}e^{-\frac{1}{2}\frac{(x-a)^2}{\sigma^2+b^2}}

\end{aligned}
$$

$$
L = \left[\begin{aligned} 
    &1{\color{red}{ [0,0,0,0]} }  \\
    &2{\color{red}[0,1,0,0]} \\
    &3{\color{red}[1,1,0,1]} \\ 
    &4{\color{red}[1,2,0,0]} \\
    &5{\color{red}[2,2,0,0]} \\
    &6{\color{green}[2,2,0,1]} \\
    &7{\color{red}[3,3,0,2]} \\ 
    &8{\color{red}[3,3,1,2]} \\
    &9{\color{red}[4,3,1,2]} 
    \end{aligned}
    \right]\\
$$

$$
x=1 \\
y=2
$$

$$\begin{align}
x=1 \\

\\

y=2
 \\
z = 3
\end{align}$$

$$
\begin{aligned}

P(x) &= \int P(x \mid \mu) P(\mu) d\mu\\
&=\int_{-\infty}^{\infty} \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{2}}\frac{1}{b \sqrt{2 \pi}} e^{-\frac{1}{2}\left(\frac{\mu-a}{b}\right)^{2}} d\mu\\
&=\frac{1}{2 \sigma b \pi}\int_{-\infty}^{\infty} e^{-\frac{1}{2}\frac{(x-\mu)^2b^2+(\mu-a)^2\sigma^2}{\sigma^2 b^2}} d\mu\\
&=\frac{1}{2 \sigma b \pi}\int_{-\infty}^{\infty}exp\left(-\frac{1}{2}\left(\frac{\left(\mu-\frac{xb^2+a\sigma^2}{\sigma^2+b^2}\right)^2}{\frac{\sigma^2b^2}{\sigma^2+b^2}}+\frac{(x-a)^2}{\sigma^2+b^2}\right)\right)d\mu\\
&=\frac{e^{-\frac{1}{2}\frac{(x-a)^2}{\sigma^2+b^2}}}{2 \sigma b \pi}\frac{\sigma b}{\sqrt{\sigma^2+b^2}}\sqrt{2 \pi}\int_{-\infty}^{\infty}\frac{1}{\frac{\sigma b}{\sqrt{\sigma^2+b^2}}\sqrt{2 \pi}}exp\left(-\frac{1}{2}\left(\frac{\left(\mu-\frac{xb^2+a\sigma^2}{\sigma^2+b^2}\right)^2}{\frac{\sigma^2b^2}{\sigma^2+b^2}}\right)\right)d\mu\\
&=\frac{1}{\sqrt{\sigma^2+b^2}\sqrt{2 \pi}}e^{-\frac{1}{2}\frac{(x-a)^2}{\sigma^2+b^2}}\int_{-\infty}^{\infty} \mu\sim N\left(\frac{xb^2+a\sigma^2}{\sigma^2+b^2},\frac{\sigma b}{\sqrt{\sigma^2+b^2}}\right)d\mu\\
&=\frac{1}{\sqrt{\sigma^2+b^2}\sqrt{2 \pi}}e^{-\frac{1}{2}\frac{(x-a)^2}{\sigma^2+b^2}}

\end{aligned}
$$
