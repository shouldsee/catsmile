#! https://zhuanlan.zhihu.com/p/554166769

# 8504: Python动态方法注册

[CATSMILE-8504](http://catsmile.info/8504-python-dynamic-method.html)


```{toctree}
---
maxdepth: 4
---
8504-python-dynamic-method.md
```

## 前言

- 目标: 
- 背景与动机: 
    - 在写C的时候大家常常会用编译参数来控制宏的作用.Python这种动态语言是可以支持运行时方法生成的,当然性能不会很高,但是对于不需要做优化的简单逻辑是可以提高代码的变量隔离的
    - 写DLM的时候有太多运行时的if-else逻辑了,想把这些逻辑挪到init层,一个办法就是用注册属性的形式去模拟method的生成.搜了一圈没有看到合适的代码段,就借助decorator语法手写了一个..
    - 但是这个特定问题在本质上涉及到一个二维阵列到底按行遍历好,还是按列遍历好.我目前还没有想法...
- 结论: 
- 备注: 
- 完成度: 
- 关键词: 
- 展望方向:
- CHANGELOG:
    - 20220815 INIT




### 装饰器形式

```python
'''
Object level methods
'''
def register_object_method(self):
    '''
    将某个动态方法挂载到一个object上
    '''
    def decorator(method):
        def wrapper(*args, **kwargs):
            return method(self,*args, **kwargs)
        setattr(self, method.__name__, wrapper)
        return wrapper
    return decorator

class my_cls():
    def __init__(self,i ):
        if i==1:
            @register_object_method(self)
            def func(self,x):
                return -x
        elif i==2:
            @register_object_method(self)
            def func(self,x):
                return x**2
        else:
            assert 0
    def __call__(self, x):
        v = self.func(x)
        return v

if __name__=='__main__':
    print(my_cls(1)(2))
    print(my_cls(2)(2))

```

### 缺陷:

毕竟不是first-class method,没有考虑到做reflection的需求.也不清楚跟inheritance是否冲突


## 参考

- SO-533382 <https://stackoverflow.com/questions/533382/dynamic-runtime-method-creation-code-generation-in-python>
