
# 9008: BERT近似


BERT采用的近乎Transformer+Dense+Dropout

通过噪声注入,我们可以发现,在BERT的不同层次中,具有不同的Jacobian性质.

- 前3层可以用LocalConnection近似.
- 中间3层具有更有意思的实体表示的性质
- 后3层对标识符的扰动非常敏感,可能跟NSP损失有关.

由于Transformer可以看成是把一个序列变成一个计算图的动态模型,我们考虑直接近似这个动态模型.
这样的好处是直接用bert-likeness作为一个训练指标,避免大规模pretrain的计算代价.
