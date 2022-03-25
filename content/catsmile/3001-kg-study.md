---
title: 3001-知识图谱和NLP句子表示
---

## Traditional Knowledge Graph

Loss function: Supervised against a triplet datasets that specifies true
triplets

Tool for searching related papers:
 - https://www.connectedpapers.com/
 - Google Scholar


## Unsupervised HMM

- Ke 2016, Unsupervised Neural Hidden Markov Models https://arxiv.org/pdf/1609.09007.pdf
- PCFG: Compound Probabilistic Context-Free Grammarsfor Grammar Induction. https://aclanthology.org/P19-1228.pdf

## Unsupervised Knowledge Graph

- KG to bias LM: KELM: augment pretraining corpus with KG
  https://ai.googleblog.com/2021/05/kelm-integrating-knowledge-graphs-with.html
  Sample sentences from KG (using a random walk？)  

- KG from LM: KG extraction from BERT by evaluating attention seqs.
  https://arxiv.org/abs/2010.11967

- Visual Storytelling: Convert pictures into natural languages 看图说话.

- NotInteresting,read: Review of KG refinement http://www.semantic-web-journal.net/system/files/swj1167.pdf

- Important! Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems
  https://arxiv.org/pdf/1508.01745.pdf
 - SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient
 - https://ojs.aaai.org/index.php/AAAI/article/view/10804/10663
 - Deep Reinforcement Learning for Dialogue Generation
 - https://arxiv.org/abs/1606.01541
 - A Diversity-Promoting Objective Function for Neural Conversation Models
 - https://arxiv.org/abs/1510.03055

- Quite Weird: Neural Text Generation from Structured Data with Application to the Biography Domain
  https://arxiv.org/pdf/1603.07771.pdf

- ToRead: Controlling Linguistic Style Aspects in Neural Language Generation
  https://arxiv.org/pdf/1707.02633.pdf

- InterestingDirection: KG and Recommendation system
  https://arxiv.org/pdf/2003.00911.pdf

- Fundmental LM: Language Models 1996 https://aclanthology.org/J96-1002.pdf
  Found through WIKI https://en.wikipedia.org/wiki/Language_model
  CRF2001: https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers

- LM Review: 2019 https://arxiv.org/abs/1906.03591

- Sentence Rep: Sanjeev Arora on random walk and sentence rep:
  https://aclanthology.org/W18-3012.pdf

- NLP text Generation notes: https://zhuanlan.zhihu.com/p/162035103

- zhihu KB MemNN: https://zhuanlan.zhihu.com/p/163343976

- LSTM for drawing Deepmind: https://arxiv.org/pdf/1502.04623.pdf

- char-RNN2015: Andrej Karpathy LSTM https://karpathy.github.io/2015/05/21/rnn-effectiveness/
  http://vision.stanford.edu/pdf/KarpathyICLR2016.pdf

- Samuel R. Bowman在Generating Sentences from a Continuous Space中使用VAE进行文本生成。这种方法可以对句子间插值。
  https://arxiv.org/abs/1511.06349

- zhihu Notes on NLG: https://zhuanlan.zhihu.com/p/188446640

- conferences: ACL EMNLP NAACL

- GoolgeNN SENNA 2011: https://www.jmlr.org/papers/volume12/collobert11a/collobert11a
  Representation Learning 2012 https://arxiv.org/pdf/1206.5538.pdf
  Schizophrenia detection?? https://www.sciencedirect.com/science/article/abs/pii/S0165178121004315
