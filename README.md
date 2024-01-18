### LSTM 地震加速度时序预测

**2024年1月17日**

![image-20240117163353976](C:\Users\yjc\AppData\Roaming\Typora\typora-user-images\image-20240117163353976.png)

* 将数据送入lstm模型，预测效果良好
  但是，loss情况：测试集一直小于训练集，很奇怪
* 猜测：是因为数据过于简单又是单变量预测，LSTM模型足够成熟

![image-20240117214941833](C:\Users\yjc\AppData\Roaming\Typora\typora-user-images\image-20240117214941833.png)

* 两者模型的比较，显然，informer拟合的更好一些（informer未经过调参，且训练六轮，lstm训练20轮）

