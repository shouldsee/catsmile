# 9021-蛋白质接触预测 Contact Prediction in Proteins

[CATSMILE-9021](http://catsmile.info/9021-contact-preditcion.html)


```{toctree}
---
maxdepth: 4
---
9021-contact-prediction.md
```

## 前言

- 目标:
- 背景与动机:
    - 看一下ContactPrediction的发展情况
- 结论: 
- 完成度: 中
- 备注: 
- 关键词: 
- 展望方向:
- 相关篇目
- CHANGLOG:
  - 20220811 INIT

接触预测是对多序列比对进行利用的一个经典途径,基于Potts Model的经典算法有

### From ConnectedPapers Search

这边主要是传统算法

- BurgerAndVanNimwegen 的贝叶斯网络
- Jones PSICOV 
    - 假设多维高斯分布,求协方差的稀疏逆矩阵来近似推断精度矩阵
- Ekeberg plmDCA
    - 假设potts model, 做了pseudolikelihood变换,绕过了相空间上的配分函数计算,用一个局部归一化的似然替代了原始似然
- seemayer CCMPred
    - 原理上似乎跟plmDCA类似.

此处开始了基于深度学习的算法

- Wang2017 RaptorX contactMap <https://arxiv.org/abs/1609.00680>
- Jones2018 DeepCov
- ResPRE

### from CASP14 CP results [url](https://predictioncenter.org/casp14/zscores_rrc.cgi)


- dilatedConvolution看起来是比较多的预测Contact的方法

- TecentLab Tfold <https://drug.ai.tencent.com/publications/tFold_contact_prediction.pdf>

- Zhang DeepPotential <https://www.cell.com/iscience/fulltext/S2589-0042(22)00696-4?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2589004222006964%3Fshowall%3Dtrue>

- TripletRes <https://predictioncenter.org/casp13/doc/presentations/Pred_CASP13_contacts_ResTriplet_TripletRes_Redacted.pdf>

### Sequence language model 

Modeling aspects of the language of life through transfer-learning protein sequences

## 参考
