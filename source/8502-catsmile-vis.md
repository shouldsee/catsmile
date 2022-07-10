# 8502: CATSMILE模型可视化


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


[CATSMILE-8502]:<http://catsmile.info/8502-catsmile-vis.html>


## 前言

- 目标: 定义CATSMILE对于模型的渲染模式
- 关键词: 
- 动机: 
- 应用:
- 备注:
- CHANGELOG:

模型和数据渲染是了解模型的一个重要途径,且常常和具体数据集/具体模型高度耦合.
这里我希望把可视化抽象成一个可以扩展和维护的接口,来对散落各处的脚本和文件加以整理.


比如对于FASHION_MNIST数据集,我想要观测模型对于某些数据的重建效果, 那么
我需要一个接口,调用一个模型,在数据集上执行编码解码过程`model.decode(model.encode(x))`,然后把得到的数据进行可视化. 

为了进行加速,我可能还要缓存一下结果,并且列出可供使用的模型


```
'/api/fashion_mnist_rec'
```

