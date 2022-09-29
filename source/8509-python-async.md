#! https://zhuanlan.zhihu.com/p/568645640
# 8509-python异步学习笔记

[CATSMILE-8509](http://catsmile.info/8509-python-async.html)

```{toctree}
---
maxdepth: 4
---
8509-python-async.md
```

## 前言

- 目标:
- 背景与动机:
    - 理解进程Process,线程Thread,协程Coroutine的区别和使用场景.
    - 在理解visdom的通信机制的时候发现它用（线程）在业务脚本里监听和运行callback。想要预先看看会不会出现性能瓶颈
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

### 进程间通信？

没玩过。。。multiprocessing好像提供了queue,pipe,semaphore等等不同形式，比较常见的模式是socket套接字连接

### 线程间通信,与全局解释器锁(GIL)

听说运行多线程由于GIL的存在，对于分配cpu时间到线程的效果并不是特别好。由于线程是共享内存的，所以在读写数据的时候需要加锁确保排他性，这可能是GIL的来源之一？


### 协程与asyncio

asyncio是python内置的并行框架

## 参考

- asyncio协程<https://docs.python.org/zh-cn/3/library/asyncio-task.html>
- 线程zhihu探讨: <https://zhuanlan.zhihu.com/p/20953544>
- 进程间通信模式,zhihu44637448:<https://zhuanlan.zhihu.com/p/446374478>
