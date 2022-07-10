#! https://zhuanlan.zhihu.com/p/532306429
# [TBC] 9012: 视觉注意力的递归模型 Recurrent Models of Visual Attention 

[CATSMILE-9012](http://catsmile.info/9012-visual-attention)


```{toctree}
---
maxdepth: 4
#caption: mycap
numbered: 0
---
9012-visual-attention.md
```

## 前言

- 目标: 梳理基于bbox的循环视觉注意力
- 背景与动机: 
- 结论: 
- 备注: 
  - 一转眼,已经过了8年了....神经网络从听,读,慢慢过渡到了说,写.真是一个令人感慨的时代
- 更新时间:
- 关键词: 
- 展望方向:


视觉注意力已经有了各种各样的函数形式,我们从2014的CNN-RNN经典文章出发,
尝试理解视觉注意力的实现模式和模型特点. 2014-2015这波多模态基于CNN和RNN
的融合,主要的训练目标是图片的文字描述对齐(caption alignment), 可以理解为
给定图片生成文字的一个objective. 后续在OpenAI手里加上Transformer后就
演变成了CLIP这样的工作. 比较经典的是 Mnih 2014对于LSTM+RNN的结合, 
Regional CNN 对于图像的切分.

## 参考

MULTIPLE OBJECT RECOGNITION WITH
VISUAL ATTENTION 2014,<https://arxiv.org/abs/1412.7755.pdf>

Recurrent Models of Visual Attention 2014<https://arxiv.org/abs/1406.6247.pdf>

Deep Visual-Semantic Alignments for Generating Image Descriptions <https://arxiv.org/abs/1412.2306>