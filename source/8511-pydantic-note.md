#! https://zhuanlan.zhihu.com/p/569104775
# 8511-pydantic学习笔记

[CATSMILE-8511](http://catsmile.info/8511-pydantic-note.html)

```{toctree}
---
maxdepth: 4
---
8511-pydantic-note.md
```

## 前言

- 目标:
- 背景与动机:
    - pydantic的文档似乎主要是基于用例的，很多写到源码里了，这里记录我探索pydantic的用法的一些经验
- 结论: 
- 完成度: 
- 备注: 
- 关键词: 
- 展望方向:
- 相关篇目
- 主要参考:
- CHANGLOG:
    - 20220928 INIT

## 简介

pydantic是一个python里做数据模型校验和转换的开源库。可以帮助我们更系统地构建复杂的类型系统，这有助于高效地建立和维护各个层级的数据模型。


### 基础阅读:基础用法

-基础用法见 Doc on Models <https://pydantic-docs.helpmanual.io/usage/models/>

### 设计理念:层级式验证

使用pydantic的一个主要目标是构造一个层级式的数据验证体系，
换句话说就是借助抽象的((对象-属性)-属性)链条自动进行验证，
这样的好处是可以将数据模型和具体数据分离开来，并且可以用较少的代码写出比较复杂的模型。

### 概念: 数据初始化与validator

pydantic.BaseModel 在初始化__init__的时候会按照fields的顺序
进行validation，也就是运行field对应的validator函数。validator是可以用来改变数据的，最后return的就是改变后的数据

```python
def check_my_value(cls, v): return v
def check_my_value(cls, v, values): return v
def check_my_value(cls, v, values, config): return v
def check_my_value(cls, v, values, config, field):
    '''
    :param cls: 是类型
    :param v: 是待校验数据
    :param values: 是(已经?)校验好的数据
    :param config: ？
    :param field: 是有关当前待校验field的信息 
    '''
     return v
```

### 概念: 动态类型和 Discriminative Union

pydantic提供了Dunion来帮助实现动态类型，如果你发现自己在手动写一个类型的registry，可以考虑用这个特性。

```python

import toml
from pydantic import BaseModel,Field
from typing import Literal, Union
class EventData(BaseModel):
    EventDataType : Literal['EventData']
    

class MarkovComapreEventData(BaseModel):
    EventDataType: Literal['MarkovComapreEventData']
    model_config_1: dict
    model_config_2: dict
    target_env: str

    @validator("model_config_1","model_config_2",pre=True)
    def parse_toml(cls,v):
        return toml.loads(v)


class BaseMessage(BaseModel):
    eid: str
    event_type: str
    target: str
    event_data: Union[EventData, MarkovComapreEventData] = Field(default={},discriminator='EventDataType')
```

### 参考:

- `Field()` gh code <https://github.com/pydantic/pydantic/blob/main/pydantic/fields.py#L222>
- Doc on Discriminative Union  <https://pydantic-docs.helpmanual.io/usage/types/#discriminated-unions-aka-tagged-unions>

