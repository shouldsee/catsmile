# 9021-序列的采样 Sampling a sequence

[CATSMILE-9021](http://catsmile.info/9021-seq-sampling.html)

## 前言

- 结论:
- 背景与动机: 
    - [CATSMILE-9021](./9021-seq-sampling)
- 备注: 
- 关键词: 方法论/技术报告
- 后续目标/展望方向:
- 相关篇目:
    - [CATSMILE-9021](./9021-seq-sampling)
- CHANGLOG:
    - 20220814 INIT

在尝试构建语言模型的时候,我尝试对一个CharacterLevel模型进行了采样,但是呢,采样出来的句子,其概率都低得令人发指.反复调试,修掉一个大bug以后(torch.rand的shape写错导致max算子完全无法采样). 

<del>相比之下, 基于VAE的采样即便损失函数弱于RNN,其生成效果也比RNN好. 这令我非常疑惑,为啥明明是优化了生成概率,但是采样效果却如此之糟糕呢?</del> 仔细查了半天,发现是采样时少了一道std_norm/layernorm.....


## 参考

