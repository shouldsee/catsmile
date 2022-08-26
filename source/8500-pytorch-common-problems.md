#! https://zhuanlan.zhihu.com/p/486933423

# 8500-PyTorch模型常用药

*号外号外，CATSMILE静态站点已经开通[8500: PyTorch模型常用药](http://catsmile.info/8500-pytorch-common-problems.html)，发zhihu强烈安利[zhihu-on-vscode](https://zhuanlan.zhihu.com/p/106057556)* 

- 前言: 建模是令人激动的旅程，但是模型很容易生病，这里列举了一些PyTorch的常见药品。**
（现象和实因是多对多关系。）
- 更新日期： 20220417

- **8501**
  - 现象：
    - loss不下降，参数不移动
    - 恢复模型后，损失函数和保存前不一样。`x = torch.load("Checkpoint.pkl");  model.load_state_dict(x['model'])`
  - 实因： `nn.Parameter`绑定不正确，造成梯度恒为0，或者模型恢复后权重重新初始化
  - 原理：当你创建`nn.Module`时，对`self`进行属性挂载从而建立计算图时，有多种可能失败。造成
  该张量从梯度计算中脱落。
  - 排查方案: 检查目标参数是否在`nn.Model.named_parameters()`中出现。例如

  ```python
  class FeedForwardLayer(nn.Module):
      def __init__(self):
          super().__init__()
          self.x1 = nn.Linear(5,10)
          self.x2 = nn.Linear(5,10).weight
          self.x3 = nn.Parameter(nn.Linear(5,10).weight)
  x = FeedForwardLayer()

  ###打印模块参数列表
  print(list(dict(x.named_parameters()).keys()))
  from pprint import pprint
  pprint(list(dict(x.named_parameters()).keys()))
  ```

- **8502**
  - 现象：训练过程中出现了`NaN`
  - 实因：进行了未定义的运算，如：`1/0 log(-0.1)  sqrt(-0.1)`
  - 排查：排查近期相关修改记录，用`git diff`或者IDE编辑器的`Ctrl+Z`工具查找近期修改过的类似函数。

- **8503**
  - 现象：TestLoss和TrainLoss相同。
  - 实因:
    - 忘记调用`Dataset.test()`和`Dataset.train()`在训练集和测试集进行切换。
    - 他们压根就是一个变量。

- **8504**
  - 现象：无法分配CUDA内存。`Unable To Allocate`
  - 实因：
    - 中间变量存在未释放的指针。如历史loss计算完毕后，需要从计算图上取下。如`test_losses.append( loss.item() )`

- **8505**
  - 现象：BERT初始化缓慢
  - 实因：
    - 有时候是hugging face transformers 内部的一些AutoModel/AutoTokenizer 的未知开销.    
  - 解决办法:
    - 把初始化好的模型存到本地
    
    ```python
    from transformers import AutoTokenizer, AutoModel
    import torch
    import os

    PKL = __file__+'.temp.pkl'
    if os.path.exists(PKL):
        tokenizer,model = torch.load(PKL)
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        torch.save((tokenizer,model),PKL)


    ```
- **8506**
   - 现象: 

   ```TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'markov_lm.util_html.Vocab'>```

   - 原因: 使用`torch.utils.data.Dataset`类,进行多条数据聚合时,使用了不能被pytorch batching自动处理的类

