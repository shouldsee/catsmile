(jacobian_matrix)=

# 1006: Jacobian Matrix And its determinant 雅克比行列式及其行列式 20220418


- 函数全称: Jacobian Matrix  雅克比行列式
- 函数解决的问题/不使用的后果:
  - 无法表示向量对向量的微分,无法简洁地表示反向传播
  - 无法计算变量代换(Change of Variables)后的概率密度函数(PDF)
    - 对于变量数增大或减小的场景,可以考虑计算$\text{pdet}(J)=\sqrt{\det(J.J^T)}$ See [gh issue](https://github.com/tensorflow/probability/issues/139)
- 函数解决改问题的原理: 是微分算子的高维自然扩展
- 函数可能存在的问题:
  - 算起来可能会慢且复杂。空间复杂度就为 $O(IJ)$
  - 无法链式累积误差? $\det(J J^T)$ 在哪些情况下为0? 如果 $J \in R^{m \times n},m>n$那么必然有$\det(J J^T)=0$
- 函数在某一场景下的具体形式:
  - 考虑简单的线性变换

  $$
  \begin{align}
  \vec{y} &= f(x) = W \vec{x} + \vec{b} \\
  J &= \begin{bmatrix}
  \frac{ \partial y_1}{\partial x_1} & \frac{ \partial y_1}{\partial x_2} & \cdots & \frac{ \partial y_1}{\partial x_I} \\
  \frac{ \partial y_2}{\partial x_1} & \frac{ \partial y_2}{\partial x_2} & \cdots & \frac{ \partial y_3}{\partial x_I} \\
  \vdots & \vdots & \vdots & \vdots\\
  \frac{ \partial y_J}{\partial x_1} & \frac{ \partial y_J}{\partial x_2} & \cdots & \frac{ \partial y_J}{\partial x_I} \\  
  \end{bmatrix} \\
  y_j    &= \sum_i w_{ji} x_i \\
  j_{ji} &= \frac{ \partial y_j }{\partial x_i } = w_{ji} \\
  J      &= W   
  \end{align}
  $$

  - 考虑维数不匹配的形式

  $$
  \begin{align}
  y &= \mu^T x \\
  J &= \mu^T \\
  \text{(J is a row vector by defintion )} \\
  \text{pdet}(J) &= \sqrt{ \det( \mu^T \mu )} = |\mu|
  \end{align}
  $$
- 函数的具体计算方法
   - 通过自动微分工具在计算图上进行反向传播
- 函数备注
- 函数参考信源
  - <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>
  - <https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968/14>
  - <https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html>

- 通过随机向量构造对Jacobian Response的随机近似
  - 给定线性近似 $y=f(x)=J \cdot x$, 考虑微小扰动通过函数后产生的映射函数 $e \sim N(0, \sigma^2 I)$
  - 考虑变量替换 $ (y-y_0) = (f(x_0+e) - y_0)=(J x_0 + J e - y_0) = (J e)$  
  - $E(Je) = J E(e) = 0$
  -
    $$
    \begin{align}
    E(|J e|^2) &= E((J e)^T J e) \\
    &= E(e^T J^T J e) \\
    &= E(e^T K e )  \\
    &= E(\sum_{ij} e_i e_j K_{ij} ) \\
    &= \sum_{ij}E(e_i e_j ) K_{ij} \\
\text{(since off-diagonal elements are zero for e) }
    &= \sum_{i} \sigma^2 K_{ii}   \\
    \\
    K &= J^T J = \sum_{k} J_{ki} J_{kj} \\
\text{(this is the sum of 2-norm of the row vectors )}   
     K_{ii} &= \sum_{k} J_{ki} J_{ki}  \\
    &= \sum_{i} \sigma^2 K_{ii}  \\
    \\
    E(|J e|^2) &= \sum_{i} \sigma^2 (\sum_{k} J_{ki} J_{ki})\\
    \text{(this is the squared Frobenius norm scaled)}   
    E(|J e|^2) &= \sigma^2 \sum_{i,k}  J_{ki}^2\\
    \end{align}
    $$
  - 这说明,在线性情况下,注入高斯分布微扰得到的响应扰动的MSE大小$E_e(|f(x+e)-f(x)|^2)/E_e(|e|^2)$
  由$J$的[F范数,FrobeniusNorm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm)完全刻画,
  这个比Jacobian的行列式要好算很多.

  $$
  G_f(x) = \frac{E_e(|f(x+e)-f(x)|^2)}{E_e(|e|^2)}= \frac{1}{I} \sum_{i,k}  [J_f(x)_{ki}]^2
  $$
  - 考虑链式法则是否成立 $z = g(y) = g[f(x)] = h(x), h = g f$
  - 应用变量替换...是无法证明链式法则成立的,因为分子分母无法抵消.

  $$
  \begin{align}
  G_h(x) & =  \frac{E_e(|h(x+e)-h(x)|^2)}{E_e(|e|^2)} \\
         & \neq \frac{E_c(|g(y+c)-g(y)|^2)}{E_c(|c|^2)} \cdot \frac{E_e|f(x+e)-f(x)|^2)}{E_e(|e|^2)}
  \end{align}
  $$

  - 其实可以从Jacbian的乘积的角度去理解,复合函数意味着Jacobian具有结合律,那就可以应用FrobeniusNorm本身的性质,求得一个上界.
  - 此处直接引用MathExchange上基于[柯西不等式](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality)
  的[应用结果(by hermes)](https://math.stackexchange.com/a/1393667/416294).证明了积矩阵的范数平方小于因子矩阵的范数
  平方之积,并在因子矩阵的行向量和列向量线性相关的那些元素处取等号. 取对数后,我们可以转化为一个求和的关系  
  - 因此,我们可以通过在反向传播的路径上积累Jacobian的F范数的对数,来估计出最终复合Jacobian范数的上界.



  $$
  \begin{align}
  J_h(x)                  & =   J_g(f(x)) \cdot J_f(x) \\
  ||J_h(x)||^2_F          &\leq ||J_g(f(x))||^2_F \cdot ||J_f(x)||^2_F \\
  \log(||J_h(x)||^2_F )   &\leq \log(||J_g(f(x))||^2_F)+ \log(||J_f(x)||^2_F)
  \end{align}
  $$


- 函数参考:
  - Google jacobian noise injection <https://www.google.com/search?q=jacobian+noise+injection+in+neural+network>
  - Jacobian正则化 <https://arxiv.org/pdf/1908.02729.pdf>
  - Contractive Autoencoder (2011) <https://icml.cc/2011/papers/455_icmlpaper.pdf>
  - Gaussian Noise Injection <https://papers.nips.cc/paper/2020/file/c16a5320fa475530d9583c34fd356ef5-Paper.pdf>
  - Sensitivity and Generalisation <https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/46649.pdf>
  - Stabilizing Equilibrium Models by Jacobian Regularization <https://arxiv.org/pdf/2106.14342.pdf>

[pytoch post by pascal](https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968/14)

```python
'''
REF: https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968/14
'''

import torch
def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
    return grad

def jacobian(y, x):
    """Compute dy/dx = dy/dx @ grad_outputs;
    for grad_outputs in [1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]"""
    jac = torch.zeros(y.shape[0], x.shape[0])
    for i in range(y.shape[0]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1
        jac[i] = gradient(y, x, grad_outputs = grad_outputs)
    return jac

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)



x = torch.tensor([1., 2., 3.], requires_grad=True)
w = torch.tensor([[1., 2., 3.], [0., 1., -1.]])
b = torch.tensor([1., 2.])
y = torch.matmul(x, w.t()) + b # y = x @ wT + b => y1 = x1 + 2*x2 + 3*x3 + 1 = 15, y2 = x2 - x3 + 2 = 1
dydx = gradient(y, x)  # => jacobian(y, x) @ [1, 1]
jac = jacobian(y, x)
div = divergence(y, x)
```
