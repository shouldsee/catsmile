# 9014-聚类和数据重建 Clustering And Data reconstruction

[CATSMILE-9014](http://catsmile.info/9014-cluster-rec.html)


```{toctree}
---
maxdepth: 4
---
9014-cluster-rec.md
```

## 前言

- 目标: 梳理数据重建和聚类的关系
- 背景与动机:
- 结论: 
- 备注: 
- 关键词: 
- 展望方向:
- 相关篇目
- CHANGLOG:

### 和生成模型的对比

![http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf](./9013-ddpm-p1.png)

### 其他素材

## 参考

1. MPPCA没啥现成代码能测试的
1. 能测的基本都是VAE的, GMVAE 和 BetaVAE


### VAE related

- Ruishu post on GMVAE GMVAE
  - Ruishu2016
  -  <http://ruishu.io/2016/12/25/gmvae/>
  - 一些数学推导和带高斯混合隐变量的GMVAE
  - [!code!](https://github.com/jariasf/GMVAE)
  - [!code!ruishu](https://github.com/RuiShu/vae-clustering)

- GMVAE
  - Kingma2014 
  - Semi-Supervised Learning with Deep Generative Models, 
  - <https://arxiv.org/abs/1406.5298.pdf>
  - [!code!](https://github.com/dpkingma/nips14-ssl)

- betaVAE
  - https://openreview.net/forum?id=Sy2fzU9gl
  - [!code!](https://github.com/1Konny/Beta-VAE)

- DUCMA tested some clustering metric
  - Zhang2017
  - https://arxiv.org/abs/1712.07788.pdf
  - https://github.com/icannos/mixture-autoencoder

- DAMIC 
  - Chazan2018
  - https://arxiv.org/abs/1812.06535

### Manifold, DR reviews

- Modeling the manifolds of images of handwritten digits
  Geoffrey Hinton + Peter Dayan? M. Revow

- Unsupervised dimensionality reduction: Overview and recent advances
  - Lee2010 
  - <https://www.semanticscholar.org/paper/Unsupervised-dimensionality-reduction%3A-Overview-and-Lee-Verleysen/eca48d48b9fcc5417d945f30b791d716a09d4d6d>

- Dimensionality Reduction: A Comparative Review
  - Maaten2009
  - <https://www.semanticscholar.org/paper/Dimensionality-Reduction%3A-A-Comparative-Review-Maaten-Postma/2309f7c5dad934f2adc2c5a066eba8fc2d8071ec>

### Classical PCA related

- MPPCA
  - Tipping and Bishop 1999
  - https://www.cse.iitk.ac.in/users/piyush/courses/pml_winter16/mppca.pdf
  - [(Connected Papers)](https://www.connectedpapers.com/main/276a0ed0b3ca34cda05694d72cf08c47f671053a/Mixtures-of-Probabilistic-Principal-Component-Analyzers/graph)
  - [**!code!**](https://github.com/michelbl/MPPCA)

- Model-based clustering
  - Mcnicholas2016 review
  - <https://www.semanticscholar.org/paper/Model-Based-Clustering-McNicholas/6aee33f19d324b7e5e02df4ad77f995b6e98bee3>

- Face retrieval using 1st- and 2nd-order PCA mixture model
  - Kim2002
  - no code
  - <https://www.semanticscholar.org/paper/Face-retrieval-using-1st-and-2nd-order-PCA-mixture-Kim-Kim/3124ce4aa261ef232e598e656d66e140d5a73800>

- A PCA mixture model with an efficient model selection method
  - Kim2001
  - <https://www.semanticscholar.org/paper/A-PCA-mixture-model-with-an-efficient-model-method-Kim-Kim/d8b8c039d19efb39bb67ac77662cb75fba431da9>

- Extending mixtures of multivariate t-factor analyzers
  - Andrews-McNicolas2011
  - <https://www.semanticscholar.org/paper/Extending-mixtures-of-multivariate-t-factor-Andrews-McNicholas/5249ff8e95b2698909656c7f7c7d637732f5bac8>

- Model-Based Classification via Mixtures of Multivariate t-Factor Analyzers
  - Stean-Mcnicolas-Yada2011

- Mixture of Bilateral-Projection Two-dimensional Probabilistic Principal Component Analysis
  - Ju2016
  - <https://arxiv.org/abs/1601.01431.pdf>
