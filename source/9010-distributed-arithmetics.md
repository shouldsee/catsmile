# 9010: 分布式算术测试 Distributed Arithmetics 


- 函数全称: (严格说是数据集和函数)Distributed Arithmetics 
- 函数解决的问题/不使用的后果: 用来测试各种分布式模型(Transfomer-based)的学习能力.
- 函数解决改问题的原理: 在自动生成的数据集进行Masked-Token Prediction.
- 函数可能存在的问题: 不够复杂,不能区分模型的表现力
- 函数在某一场景下的具体形式:
  - 随机生成符合约束的字符串, 满足(a-b)%10=c.
    如
    ```
    ---a5-----b4-----c1
    -b2-----a7-----c5--
    ```
  - 并随机挑选一位数字置Mask
    ```
    ---a[M]-----b4-----c1
    -b2-----a7-----c[M]--
    ```
  - 要求模型恢复出`[M]`处的令牌.

- 函数的具体计算方法
- 函数备注
  - use_dense_relu:
    - 1: 加KE->KE的dense层,relu激活,再投影恢复维度KE->E
    - 3: 用QueryMatrix的transpose进行降维KE->E
    - 7: 用QueryMatrix的transpose进行降维KE->E,再升维到KE,relu,再回到KE
    - 11: 用K个QueryVector捉全局内容,用KE->KE,relu计算偏移,
    再用K个KeyVector将偏移分配回局部.
    - 12: 用K个QueryVector捉全局内容,用KE->KE,relu计算偏移,
    再用K个同样的QueryVector将偏移分配回局部.
- 函数参考信源
