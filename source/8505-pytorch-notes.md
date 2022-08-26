#! https://zhuanlan.zhihu.com/p/556183119
# 8505-PyTorch小抄

[CATSMILE-8505](http://catsmile.info/8505-pytorch-notes.html)

```{toctree}
---
maxdepth: 4
---
8505-pytorch-notes.md
```

## 前言

- 目标:
- 背景与动机:
    - 整理一些常用的索引相关的函数
- 结论: 
- 完成度: 
- 备注: 
- 关键词: 
- 展望方向:
- 相关篇目
- CHANGLOG:

### `torch.gather` 数字索引聚合

这个在取embedding的时候非常常用,取embedding可以用方括号或者torch.gather.一般印象里用方括号会快一点,但是gather更加灵活一点.

```python
import torch

x = torch.arange(50).reshape((5,10))
x = torch.normal(0,1,(5,10))

idx = torch.ones((1,10)).long()
y = torch.gather(x,index=idx,dim=0)


print(x)
print(idx)
print(y)

idx2 = torch.arange(10).long()[None] % 5
y2 = torch.gather(x,index=idx2,dim=0)

print(idx2)
print(y2)

```

```
>>> print(x)
tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])
>>> print(idx)
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
>>> print(y)
tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
>>> print(idx2)
tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]])
>>> print(y2)
tensor([[ 0, 11, 22, 33, 44,  5, 16, 27, 38, 49]])
```

### `torch.scatter(input, dim, index, src)` 数字索引散射

scatter是gather的逆运算,可以做一些类似convolution的放置操作.

### `torch.Tensor.masked_scatter(mask, tensor)` 布尔索引散射

### `torch.masked_select(input, mask, *, out=None)` 布尔索引聚合






