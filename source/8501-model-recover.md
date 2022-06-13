---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


#! https://zhuanlan.zhihu.com/p/519465520

# 8501-20220524:模型复现与分布式建模

静态站:<http://catsmile.info/8501-model-recover.html>

## 目标

- [x] 讨论模型复现的意义
  - 为什么要复现模型? 为了通过提高交换效率进而提高研发效率,
  - 对比源码更改和参数更改,引出`MyExpResult`的自然定义
  - 对机器学习实验进行抽象分工,让机器的归机器,人类的归人类
  - 机器比人脑更适合储存大量的实验数据,前提是有恰当的数据抽象
  - [ ] 抽象以后的架构可以加入可重复性检测
- [ ] 讨论可以达到模型复现的工具
- [ ] 讨论目前已经存在的相关框架
  - kaggle
  - 天池
  - paper with code

## 为什么要复现模型?

在机器学习,尤其是深度学习炼丹学大行其道的今天,我们经常会陷入和合成化学一样的困境:A在X主题上发了文章,B跑过来问A,为什么你不用我的模型/分子B1来探究问题X?A说,
我实在手头按你的配方合成不出B1啊,从你那边要B1的时间运费都太高了,于是我就只好
在我能够合成出的A1上面研究X问题了.

这个问题在深度机器学习也存在类似的情况,复现模型A的工程成本,可能对于研究者来说
并不是探究问题X的最快路径.直接针对问题X来构造模型B,可能快得多.作为这种局部
资源最优化的结果,是模型在社区中呈现散乱和割裂的特征.在Github上找到的仓库,
很大概率在克隆到本地后,无法直接运行,经常出现依赖无法解析,数据无法下载,或者
模型高度依赖一个未被保存的随机数.这反映了深度学习非常实验性的一面:在实验科学如生物或化学里,拿到实验流程,并不代表能完整复刻实验结果.然而,对于一门基于代码和数据仓库的学问来说,我们完全有机会提高它的可重复性.

在我个人进行的模型实验中,经常涉及到python源码的更改,我相信这也是大部分DL
研究人员的现状.DL研究的一大实验方向就是通过人为地设计python源码来提高模型的
某个评价指针.我发现,在PyTorch实验语境下,通过恰当地抽象实验对象,对于提高
实验条理和避免重复炼丹有着至关重要的意义,接下来我会通过几个例子来举例说明.

让我们看一个非常简单的需求, 网络X含有可变参数fn,fn表征的是网络内部某一层的
激活函数. 在实验的过程中,我们可能会将fn替换成relu,elu,gelu等来观察评价指标的变化.让我们定义一些局部变量


考虑`main.py`

```python
config  # 实验环境,如优化器optimiser,随机数rng
dataset # 数据源,一般分为训练集和测试集
model   # 模型,一般可以抽象为一个可变函数
grad_loss    # 为模型优化提供梯度的主损失函数

other_loss   # 其他损失函数
model.device # 设备管理

config.alpha = 0.001

def get_test_metric(metric,config,dataset,model,i):
    '''
    这是一个非常常见的函数,也就是我们希望观察模型演化i个循环后
    测试集上评价指标的变化.
    '''
    pass

GTM = get_test_metric
```

如果要记录GTM在不同超参数和步数下的结果,最简单的办法就是修改main.py源码中
对应的参数定义,然后重新运行main.py到指定的步数,得到输出.但是这产生一个问题
就是main.py的源码在每次实验的过程中将逐渐偏离其初始代码,假设一开始实现的是
一个glm线性模型,在逐步实验的过程中,第一步加入了L1正则项,第二步加入了BatchNorm
,第三步加入了relu非线性函数.那么如果第一步得到了一个较好的评价0.902,第三步
得到了较差的评价0.602.那么如果要迅速复现0.902的结果,最好的情况是
研究员对当时的代码备份了一个版本,否则,就得通过逐步调整的模式,再找到复现0.902
的那个逻辑.

通过以上例子我们可以看出,对于源码进行直接改动的开发习惯,尽管在当前可能是最快
地验证一个评价函数的办法,长期来看却可能造成源码的丢失,一种办法是在版本控制层面
去把源码和评价函数对应起来,使得某个commit或者branch的源码一经运行就能复现一个数字.但是这样意味着很难直接比较不同架构的性质,也没法简单地同时运行多个模型架构.我们可以看到,这个例子的核心问题在于源码在不同实验阶段的状态没有得到保存,
它只具有一个瞬时态,也就是当前运行的这一个版本,t-1和t+1的版本,只保存在研究员
的脑子里.众所周知人脑并不是一个好的储存器,所以至少需要一个机制把这些不同阶段
的版本也储存起来.

最简单的办法,当然是把模型看作是一个生成器作用在一系列参数上的结果.这样我们就能
够在存储结果时,把参数作为元数据加入结果表,从单纯的(main.py,'83ecdf',loss=0.902)变成(main.py,activation='elu',loss=0.902).建立了模型到
评价函数的对应关系后,就可以迅速地在不同参数生成的模型之间进行切换,而不改变源码了.

以上这个例子,为我们引入`MyExpResult(评价, 参数, 损失值)`提供了足够的动机.缺乏元标注的单纯的评价指标是难以整理和比较的,因为实数轴本身不可能蕴藏任何有趣的结构,机器学习更感兴趣的是(参数,数据,损失值),也就是不同参数和数据下的损失值.因此在实验的时候,
最好确保实验的结果能被表达成这种形式,否则就容易陷入无穷无尽的源码管理里.

接下来,将对'(生成器,参数) -> 模型'进行探讨

## 参数管理

把模型分解成生成器和参数的动机很简单:参数需要存到最终的`MyExpResult`里,而生成器服从源码管理.通过恰当地分离,可以确保`MyExpResult`只需要保存一个指向生成器的指针,而不需要存储生成器的源码本身.

'模型 <- (生成器,参数)'

### 对模型交互和储存进行抽象

建模人员跟模型进行的交互通常是迭代的,比如反复地梯度下降就是一种迭代,MCMC采样也是一种迭代.
我们可以以梯度下降作为假象过程对这种交互进行一定的抽象. 梯度下降一般由数据上的一个遍历器,
和模型上的一个梯度函数所组成.

模型和数据储存需要确保session的可重复性.

## 目前已经存在的相关框架

### MLOps

<https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning>

### 竞赛类平台

各类机器学习竞赛其实某种程度上完成了 `ExpResult(评价标准(数据集,函数),评价结果, 程序(生成器,参数))`的一种构造,在这些流程里,参赛选手负责上传(可能)可以复现的程序,以及在测试集上生成的评价. 在这些架构中,生成器的定义是隐式的,有可能是python3.7可执行的任何程序,或者是tpu执行程序,而参数可以是程序源码. 这种生成器
的问题在于参数空间过于庞大且离散,以至于很难在上面分析到底是哪些参数给出了更好
的效果.