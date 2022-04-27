

# 9007: BERT 结构解析

安装huggingface/transformers非常的顺利

See Also:

- [Github:huggingface/transformers](https://github.com/huggingface/transformers)

- [Riroaki的关于BERT源码的解析知乎文章](https://zhuanlan.zhihu.com/p/360988428)


## 个人阅读心得

1. 采用了独立的Config类进行传参
1. 有些forward return type采用了特化的OrderedDict作为ModelOutpu
1. dataclass看起来是python3的一个新特性,可以方便的自动定义`__init__`方法

## Api for noise injection.

因为做噪声注入的角度比较多,需要整合一下Api. 宏观来讲,考虑几个不同的方向.


- 单点噪声注入/多点噪声注入.
  - 多点噪声输入更加复杂,暂不讨论.仅考虑单点噪声输入.
  - 这意味着如果要个点都要重新注入.
- 噪声方差,在确保数值稳定的情况下尽量小,毕竟Jacobian是微分极限
- 单层噪声观察/多层噪声观察.
  - 多层噪声观察只需要一次注入
  - 单层噪声观察需要在每一层重新注入噪声.




## 核心代码,不考虑Decoding,仅仅考虑前向逻辑

`BertLayer`主要分解成`self.attention`和`self.feed_forward_chunk` 两个操作,其中feedforward是把H维的attention结果升高到4*H,加RELU,降回来H,加Dropout,再LayerNorm.这个操作是有一点迷的,不确定为啥要专门在高维做Relu


```python

class BertLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

```


## 尝试用小扰动理解不同位置之间的相关性

1. 标点符号上的扰动不会传到下游节点里
1. 不同层级的结构不一样,1-3层有很强的局域结构,6-7层附近有类似实体的表征结构



```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)


for k,v in model.named_modules(): print(k,v.__class__);#v.register_forward_hook;
```

```
embeddings <class 'transformers.models.bert.modeling_bert.BertEmbeddings'>
embeddings.word_embeddings <class 'torch.nn.modules.sparse.Embedding'>
embeddings.position_embeddings <class 'torch.nn.modules.sparse.Embedding'>
embeddings.token_type_embeddings <class 'torch.nn.modules.sparse.Embedding'>
embeddings.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
embeddings.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder <class 'transformers.models.bert.modeling_bert.BertEncoder'>
encoder.layer <class 'torch.nn.modules.container.ModuleList'>
encoder.layer.0 <class 'transformers.models.bert.modeling_bert.BertLayer'>
encoder.layer.0.attention <class 'transformers.models.bert.modeling_bert.BertAttention'>
encoder.layer.0.attention.self <class 'transformers.models.bert.modeling_bert.BertSelfAttention'>
encoder.layer.0.attention.self.query <class 'torch.nn.modules.linear.Linear'>
encoder.layer.0.attention.self.key <class 'torch.nn.modules.linear.Linear'>
encoder.layer.0.attention.self.value <class 'torch.nn.modules.linear.Linear'>
encoder.layer.0.attention.self.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.0.attention.output <class 'transformers.models.bert.modeling_bert.BertSelfOutput'>
encoder.layer.0.attention.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.0.attention.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.0.attention.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.0.intermediate <class 'transformers.models.bert.modeling_bert.BertIntermediate'>
encoder.layer.0.intermediate.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.0.intermediate.intermediate_act_fn <class 'transformers.activations.GELUActivation'>
encoder.layer.0.output <class 'transformers.models.bert.modeling_bert.BertOutput'>
encoder.layer.0.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.0.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.0.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.1 <class 'transformers.models.bert.modeling_bert.BertLayer'>
encoder.layer.1.attention <class 'transformers.models.bert.modeling_bert.BertAttention'>
encoder.layer.1.attention.self <class 'transformers.models.bert.modeling_bert.BertSelfAttention'>
encoder.layer.1.attention.self.query <class 'torch.nn.modules.linear.Linear'>
encoder.layer.1.attention.self.key <class 'torch.nn.modules.linear.Linear'>
encoder.layer.1.attention.self.value <class 'torch.nn.modules.linear.Linear'>
encoder.layer.1.attention.self.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.1.attention.output <class 'transformers.models.bert.modeling_bert.BertSelfOutput'>
encoder.layer.1.attention.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.1.attention.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.1.attention.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.1.intermediate <class 'transformers.models.bert.modeling_bert.BertIntermediate'>
encoder.layer.1.intermediate.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.1.output <class 'transformers.models.bert.modeling_bert.BertOutput'>
encoder.layer.1.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.1.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.1.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.2 <class 'transformers.models.bert.modeling_bert.BertLayer'>
encoder.layer.2.attention <class 'transformers.models.bert.modeling_bert.BertAttention'>
encoder.layer.2.attention.self <class 'transformers.models.bert.modeling_bert.BertSelfAttention'>
encoder.layer.2.attention.self.query <class 'torch.nn.modules.linear.Linear'>
encoder.layer.2.attention.self.key <class 'torch.nn.modules.linear.Linear'>
encoder.layer.2.attention.self.value <class 'torch.nn.modules.linear.Linear'>
encoder.layer.2.attention.self.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.2.attention.output <class 'transformers.models.bert.modeling_bert.BertSelfOutput'>
encoder.layer.2.attention.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.2.attention.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.2.attention.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.2.intermediate <class 'transformers.models.bert.modeling_bert.BertIntermediate'>
encoder.layer.2.intermediate.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.2.output <class 'transformers.models.bert.modeling_bert.BertOutput'>
encoder.layer.2.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.2.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.2.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.3 <class 'transformers.models.bert.modeling_bert.BertLayer'>
encoder.layer.3.attention <class 'transformers.models.bert.modeling_bert.BertAttention'>
encoder.layer.3.attention.self <class 'transformers.models.bert.modeling_bert.BertSelfAttention'>
encoder.layer.3.attention.self.query <class 'torch.nn.modules.linear.Linear'>
encoder.layer.3.attention.self.key <class 'torch.nn.modules.linear.Linear'>
encoder.layer.3.attention.self.value <class 'torch.nn.modules.linear.Linear'>
encoder.layer.3.attention.self.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.3.attention.output <class 'transformers.models.bert.modeling_bert.BertSelfOutput'>
encoder.layer.3.attention.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.3.attention.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.3.attention.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.3.intermediate <class 'transformers.models.bert.modeling_bert.BertIntermediate'>
encoder.layer.3.intermediate.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.3.output <class 'transformers.models.bert.modeling_bert.BertOutput'>
encoder.layer.3.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.3.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.3.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.4 <class 'transformers.models.bert.modeling_bert.BertLayer'>
encoder.layer.4.attention <class 'transformers.models.bert.modeling_bert.BertAttention'>
encoder.layer.4.attention.self <class 'transformers.models.bert.modeling_bert.BertSelfAttention'>
encoder.layer.4.attention.self.query <class 'torch.nn.modules.linear.Linear'>
encoder.layer.4.attention.self.key <class 'torch.nn.modules.linear.Linear'>
encoder.layer.4.attention.self.value <class 'torch.nn.modules.linear.Linear'>
encoder.layer.4.attention.self.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.4.attention.output <class 'transformers.models.bert.modeling_bert.BertSelfOutput'>
encoder.layer.4.attention.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.4.attention.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.4.attention.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.4.intermediate <class 'transformers.models.bert.modeling_bert.BertIntermediate'>
encoder.layer.4.intermediate.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.4.output <class 'transformers.models.bert.modeling_bert.BertOutput'>
encoder.layer.4.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.4.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.4.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.5 <class 'transformers.models.bert.modeling_bert.BertLayer'>
encoder.layer.5.attention <class 'transformers.models.bert.modeling_bert.BertAttention'>
encoder.layer.5.attention.self <class 'transformers.models.bert.modeling_bert.BertSelfAttention'>
encoder.layer.5.attention.self.query <class 'torch.nn.modules.linear.Linear'>
encoder.layer.5.attention.self.key <class 'torch.nn.modules.linear.Linear'>
encoder.layer.5.attention.self.value <class 'torch.nn.modules.linear.Linear'>
encoder.layer.5.attention.self.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.5.attention.output <class 'transformers.models.bert.modeling_bert.BertSelfOutput'>
encoder.layer.5.attention.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.5.attention.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.5.attention.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.5.intermediate <class 'transformers.models.bert.modeling_bert.BertIntermediate'>
encoder.layer.5.intermediate.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.5.output <class 'transformers.models.bert.modeling_bert.BertOutput'>
encoder.layer.5.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.5.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.5.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.6 <class 'transformers.models.bert.modeling_bert.BertLayer'>
encoder.layer.6.attention <class 'transformers.models.bert.modeling_bert.BertAttention'>
encoder.layer.6.attention.self <class 'transformers.models.bert.modeling_bert.BertSelfAttention'>
encoder.layer.6.attention.self.query <class 'torch.nn.modules.linear.Linear'>
encoder.layer.6.attention.self.key <class 'torch.nn.modules.linear.Linear'>
encoder.layer.6.attention.self.value <class 'torch.nn.modules.linear.Linear'>
encoder.layer.6.attention.self.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.6.attention.output <class 'transformers.models.bert.modeling_bert.BertSelfOutput'>
encoder.layer.6.attention.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.6.attention.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.6.attention.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.6.intermediate <class 'transformers.models.bert.modeling_bert.BertIntermediate'>
encoder.layer.6.intermediate.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.6.output <class 'transformers.models.bert.modeling_bert.BertOutput'>
encoder.layer.6.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.6.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.6.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.7 <class 'transformers.models.bert.modeling_bert.BertLayer'>
encoder.layer.7.attention <class 'transformers.models.bert.modeling_bert.BertAttention'>
encoder.layer.7.attention.self <class 'transformers.models.bert.modeling_bert.BertSelfAttention'>
encoder.layer.7.attention.self.query <class 'torch.nn.modules.linear.Linear'>
encoder.layer.7.attention.self.key <class 'torch.nn.modules.linear.Linear'>
encoder.layer.7.attention.self.value <class 'torch.nn.modules.linear.Linear'>
encoder.layer.7.attention.self.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.7.attention.output <class 'transformers.models.bert.modeling_bert.BertSelfOutput'>
encoder.layer.7.attention.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.7.attention.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.7.attention.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.7.intermediate <class 'transformers.models.bert.modeling_bert.BertIntermediate'>
encoder.layer.7.intermediate.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.7.output <class 'transformers.models.bert.modeling_bert.BertOutput'>
encoder.layer.7.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.7.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.7.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.8 <class 'transformers.models.bert.modeling_bert.BertLayer'>
encoder.layer.8.attention <class 'transformers.models.bert.modeling_bert.BertAttention'>
encoder.layer.8.attention.self <class 'transformers.models.bert.modeling_bert.BertSelfAttention'>
encoder.layer.8.attention.self.query <class 'torch.nn.modules.linear.Linear'>
encoder.layer.8.attention.self.key <class 'torch.nn.modules.linear.Linear'>
encoder.layer.8.attention.self.value <class 'torch.nn.modules.linear.Linear'>
encoder.layer.8.attention.self.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.8.attention.output <class 'transformers.models.bert.modeling_bert.BertSelfOutput'>
encoder.layer.8.attention.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.8.attention.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.8.attention.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.8.intermediate <class 'transformers.models.bert.modeling_bert.BertIntermediate'>
encoder.layer.8.intermediate.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.8.output <class 'transformers.models.bert.modeling_bert.BertOutput'>
encoder.layer.8.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.8.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.8.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.9 <class 'transformers.models.bert.modeling_bert.BertLayer'>
encoder.layer.9.attention <class 'transformers.models.bert.modeling_bert.BertAttention'>
encoder.layer.9.attention.self <class 'transformers.models.bert.modeling_bert.BertSelfAttention'>
encoder.layer.9.attention.self.query <class 'torch.nn.modules.linear.Linear'>
encoder.layer.9.attention.self.key <class 'torch.nn.modules.linear.Linear'>
encoder.layer.9.attention.self.value <class 'torch.nn.modules.linear.Linear'>
encoder.layer.9.attention.self.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.9.attention.output <class 'transformers.models.bert.modeling_bert.BertSelfOutput'>
encoder.layer.9.attention.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.9.attention.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.9.attention.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.9.intermediate <class 'transformers.models.bert.modeling_bert.BertIntermediate'>
encoder.layer.9.intermediate.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.9.output <class 'transformers.models.bert.modeling_bert.BertOutput'>
encoder.layer.9.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.9.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.9.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.10 <class 'transformers.models.bert.modeling_bert.BertLayer'>
encoder.layer.10.attention <class 'transformers.models.bert.modeling_bert.BertAttention'>
encoder.layer.10.attention.self <class 'transformers.models.bert.modeling_bert.BertSelfAttention'>
encoder.layer.10.attention.self.query <class 'torch.nn.modules.linear.Linear'>
encoder.layer.10.attention.self.key <class 'torch.nn.modules.linear.Linear'>
encoder.layer.10.attention.self.value <class 'torch.nn.modules.linear.Linear'>
encoder.layer.10.attention.self.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.10.attention.output <class 'transformers.models.bert.modeling_bert.BertSelfOutput'>
encoder.layer.10.attention.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.10.attention.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.10.attention.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.10.intermediate <class 'transformers.models.bert.modeling_bert.BertIntermediate'>
encoder.layer.10.intermediate.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.10.output <class 'transformers.models.bert.modeling_bert.BertOutput'>
encoder.layer.10.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.10.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.10.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.11 <class 'transformers.models.bert.modeling_bert.BertLayer'>
encoder.layer.11.attention <class 'transformers.models.bert.modeling_bert.BertAttention'>
encoder.layer.11.attention.self <class 'transformers.models.bert.modeling_bert.BertSelfAttention'>
encoder.layer.11.attention.self.query <class 'torch.nn.modules.linear.Linear'>
encoder.layer.11.attention.self.key <class 'torch.nn.modules.linear.Linear'>
encoder.layer.11.attention.self.value <class 'torch.nn.modules.linear.Linear'>
encoder.layer.11.attention.self.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.11.attention.output <class 'transformers.models.bert.modeling_bert.BertSelfOutput'>
encoder.layer.11.attention.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.11.attention.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.11.attention.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
encoder.layer.11.intermediate <class 'transformers.models.bert.modeling_bert.BertIntermediate'>
encoder.layer.11.intermediate.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.11.output <class 'transformers.models.bert.modeling_bert.BertOutput'>
encoder.layer.11.output.dense <class 'torch.nn.modules.linear.Linear'>
encoder.layer.11.output.LayerNorm <class 'torch.nn.modules.normalization.LayerNorm'>
encoder.layer.11.output.dropout <class 'torch.nn.modules.dropout.Dropout'>
pooler <class 'transformers.models.bert.modeling_bert.BertPooler'>
pooler.dense <class 'torch.nn.modules.linear.Linear'>
pooler.activation <class 'torch.nn.modules.activation.Tanh'>
```

```
embeddings.word_embeddings               torch.Size([1, 13, 768])
embeddings.token_type_embeddings         torch.Size([1, 13, 768])
embeddings.position_embeddings           torch.Size([1, 13, 768])
embeddings.LayerNorm                     torch.Size([1, 13, 768])
embeddings.dropout                       torch.Size([1, 13, 768])
embeddings                               torch.Size([1, 13, 768])
encoder.layer.0.attention/0              torch.Size([1, 13, 768])
encoder.layer.0.intermediate             torch.Size([1, 13, 3072])
encoder.layer.0.output                   torch.Size([1, 13, 768])
encoder.layer.0/0                        torch.Size([1, 13, 768])
encoder.layer.1.attention/0              torch.Size([1, 13, 768])
encoder.layer.1.intermediate             torch.Size([1, 13, 3072])
encoder.layer.1.output                   torch.Size([1, 13, 768])
encoder.layer.1/0                        torch.Size([1, 13, 768])
encoder.layer.2.attention/0              torch.Size([1, 13, 768])
encoder.layer.2.intermediate             torch.Size([1, 13, 3072])
encoder.layer.2.output                   torch.Size([1, 13, 768])
encoder.layer.2/0                        torch.Size([1, 13, 768])
encoder.layer.3.attention/0              torch.Size([1, 13, 768])
encoder.layer.3.intermediate             torch.Size([1, 13, 3072])
encoder.layer.3.output                   torch.Size([1, 13, 768])
encoder.layer.3/0                        torch.Size([1, 13, 768])
encoder.layer.4.attention/0              torch.Size([1, 13, 768])
encoder.layer.4.intermediate             torch.Size([1, 13, 3072])
encoder.layer.4.output                   torch.Size([1, 13, 768])
encoder.layer.4/0                        torch.Size([1, 13, 768])
encoder.layer.5.attention/0              torch.Size([1, 13, 768])
encoder.layer.5.intermediate             torch.Size([1, 13, 3072])
encoder.layer.5.output                   torch.Size([1, 13, 768])
encoder.layer.5/0                        torch.Size([1, 13, 768])
encoder.layer.6.attention/0              torch.Size([1, 13, 768])
encoder.layer.6.intermediate             torch.Size([1, 13, 3072])
encoder.layer.6.output                   torch.Size([1, 13, 768])
encoder.layer.6/0                        torch.Size([1, 13, 768])
encoder.layer.7.attention/0              torch.Size([1, 13, 768])
encoder.layer.7.intermediate             torch.Size([1, 13, 3072])
encoder.layer.7.output                   torch.Size([1, 13, 768])
encoder.layer.7/0                        torch.Size([1, 13, 768])
encoder.layer.8.attention/0              torch.Size([1, 13, 768])
encoder.layer.8.intermediate             torch.Size([1, 13, 3072])
encoder.layer.8.output                   torch.Size([1, 13, 768])
encoder.layer.8/0                        torch.Size([1, 13, 768])
encoder.layer.9.attention/0              torch.Size([1, 13, 768])
encoder.layer.9.intermediate             torch.Size([1, 13, 3072])
encoder.layer.9.output                   torch.Size([1, 13, 768])
encoder.layer.9/0                        torch.Size([1, 13, 768])
encoder.layer.10.attention/0             torch.Size([1, 13, 768])
encoder.layer.10.intermediate            torch.Size([1, 13, 3072])
encoder.layer.10.output                  torch.Size([1, 13, 768])
encoder.layer.10/0                       torch.Size([1, 13, 768])
encoder.layer.11.attention/0             torch.Size([1, 13, 768])
encoder.layer.11.intermediate            torch.Size([1, 13, 3072])
encoder.layer.11.output                  torch.Size([1, 13, 768])
encoder.layer.11/0                       torch.Size([1, 13, 768])
encoder/last_hidden_state                torch.Size([1, 13, 768])
pooler.dense                             torch.Size([1, 768])
pooler.activation                        torch.Size([1, 768])
pooler                                   torch.Size([1, 768])
/last_hidden_state                       torch.Size([1, 13, 768])
/pooler_output                           torch.Size([1, 768])
```
