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
#! https://zhuanlan.zhihu.com/p/517570917


# 8002:构建损失函数的程序语言

静态站:<http://catsmile.info/8002-program-tools.html>

## 目标

- 梳理jax相关的扩展库

## 基于Python的框架

- python
  - jax
    - flax
    - haiku
  - tensorflow
    - sonnet
  - pytorch
    - keras


<table border=1><tbody><tr><td></td><td>Tensorflow</td><td>PyTorch</td><td>Jax</td></tr><tr><td>Developed by</td><td>Google</td><td>Facebook</td><td>Google</td></tr><tr><td>Flexible</td><td>No</td><td>Yes</td><td>Yes</td></tr><tr><td>Graph-Creation</td><td>Static/Dynamic</td><td>Dynamic</td><td>Static</td></tr><tr><td>Target Audience</td><td>Researchers,<br>Developers</td><td>Researchers,<br>Developers</td><td>Researchers<br></td></tr><tr><td>Low/High-level API</td><td>High Level</td><td>Both</td><td>Both</td></tr><tr><td>Development Stage</td><td>Mature( v2.4.1 )</td><td>Mature( v1.8.0 )</td><td>Developing( v0.1.55 )</td></tr></tbody></table>

ref:<https://www.askpython.com/python-modules/tensorflow-vs-pytorch-vs-jax>


## python-jax

```bash
python3.7 -m pip install install dm-haiku jax jaxlib keras tensorflow # tensorflow_datasets
```

评价: TBC

函数式编程,signature为王

py-jax-vae 137 lines <https://github.com/google/jax/blob/main/examples/mnist_vae.py>

## py-jax-haiku

评价：更加面向函数，传承了jax设计精神.更加细致更接近底层

### 核心组件

`haiku.transform(f, *, apply_rng=True) -> haiku.Transformed`

```python
def transform(f, *, apply_rng=True) -> Transformed:
  """Transforms a function using Haiku modules into a pair of pure functions.
  For a function ``out = f(*a, **k)`` this function returns a pair of two pure
  functions that call ``f(*a, **k)`` explicitly collecting and injecting
  parameter values::
      params = init(rng, *a, **k)
      out = apply(params, rng, *a, **k)
      """
```

```python
class Transformed(NamedTuple):
  """Holds a pair of pure functions.
  Attributes:
    init: A pure function: ``params = init(rng, *a, **k)``
    apply: A pure function: ``out = apply(params, rng, *a, **k)``
  """

  # Args: [Optional[PRNGKey], ...]
  init: Callable[..., hk.Params]

  # Args: [Params, Optional[PRNGKey], ...]
  apply: Callable[..., Any]
```

由于jax函数完全由子函数描述,因此jax的模板模块应当仅仅规定子函数的形式,而不需要规定包括device等元属性.

### 例子

<https://github.com/deepmind/dm-haiku#quickstart>

参数初始化`model.init`

```{code-cell}
:load: catsmile/c8002_haiku_example.py
:tags: [output_scroll,skip-execution]
```

```{code-cell}
:tags: [output_scroll]
! python3.7 catsmile/c8002_haiku_example.py
```

py-jax-haiku-vae ,211 lines <https://github.com/deepmind/dm-haiku/blob/main/examples/vae.py>

## py-jax-flax

评价：更加面向对象，提供适合大型工程的灵活性．处理device,flexible states之类的接口性质的杂事.

核心组件`flax.linen.Module`

```python
@dataclass_transform()
class Module:
  """Base class for all neural network modules. Layers and models should subclass this class.

  All Flax Modules are Python 3.7
  `dataclasses <https://docs.python.org/3/library/dataclasses.html>`_. Since
  dataclasses take over ``__init__``, you should instead override :meth:`setup`,
  which is automatically called to initialize the module.

  Modules can contain submodules, and in this way can be nested in a tree
  structure. Submodels can be assigned as regular attributes inside the
  :meth:`setup` method.

  You can define arbitrary "forward pass" methods on your Module subclass.
  While no methods are special-cased, ``__call__`` is a popular choice because
  it allows you to use module instances as if they are functions::

    from flax import linen as nn

    class Module(nn.Module):
      features: Tuple[int] = (16, 4)

      def setup(self):
        self.dense1 = Dense(self.features[0])
        self.dense2 = Dense(self.features[1])

      def __call__(self, x):
        return self.dense2(nn.relu(self.dense1(x)))

  Optionally, for more concise module implementations where submodules
  definitions are co-located with their usage, you can use the
  :meth:`compact` wrapper.
  """

```

例子: py-jax-flax-vae, 211 lines <https://github.com/google/flax/blob/main/examples/vae/train.py>

## Ref
