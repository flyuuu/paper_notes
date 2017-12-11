---
title: Generative Adversarial Networks
tags: GANs, 极大似然估计, Fly_卢富毓
grammar_cjkRuby: true
---

> Yan Lecun 曾经在Quora上写道：“生成式对抗网络（GAN）及其相关的变化，是我认为的机器学习领域近十年最有趣的想法。”——这对于GAN来说无疑是非常高的评价。


## 序

---

**关于生成对抗网络主要从以下几个方面来进行介绍**：
[toc]

## 人工智能的研究层次 \*

---

随着计算能力的提升、数据的几何增加，为人工智能的飞速发展提供了基础和条件。学术界普遍认为人工智能主要有两个阶段：**感知阶段和认知阶段**。

> 在感知阶段, 机器能够接收来自外界的各种信号, 例如视觉信号、听觉信号等, 并对此作出判断, 对应的研究领域有图像识别、语音识别等。  
> 在认知阶段, 机器能够对世界的本质有一定的理解, 不再是单纯、机械地做出判断。

仔细划分的话如下图，还可以据悉分为**判断**、**生成**、**理解**和**创造以及运用**，他们相互联系相互促进，同时各层次之间又存在鸿沟，有待新的研究突破\[1\]。

![人工智能的研究层次](./assets/AI研究层次.png)



## 生成模型&判别模型

---

生成方法和判别方法是机器学习中监督学习方法的两个分支.  
具体来说，针对一个模型，对给定的输入预测相应的输出：`!$Y = f(X)$`可以表达为 `!$P(Y|X)$`的形式。


**生成方法**和判别方法说不通的地方是，生成方法是由数据学习联合概率分布`!$P(X, Y)$`,然后再来求解条件概率分布`!$P(Y|X)$`来作为预测的模型，即生成模型：

```mathjax!
$$
P(Y|X) = \frac{P(X, Y)}{P(X)}
$$
```


之所以叫生成方法就是因为模型表示了**给定输入**`!$X$`**产生输出**`!$Y$`**的生成关系**。具体的算法有： 朴素贝叶斯、隐马尔可夫模型等。  
顾名思义**判别方法**就是由数据直接学习条件概率分布`!$P(Y|X)$`,判别模型关心的是对给定的输入`!$X$`，应该预测什么样的`!$Y$`。这类算法主要有： KNN、Logisitc regression、max-entropy、SVM、adaboost、CRF。

生成方法学出联合概率，收敛速度快，存在隐变量时候（LDA）,判别方法就不能使用了。  
判别方法是直接学习条件概率，直接面对预测，准确率往往非常高\[2\]。

## 什么是GAN？\(GAN的感性认知\)

---
### 生成

![](./assets/12.png)

![](./assets/13.png)

### AutoEncoder

![](./assets/14.png)

autoencoder 不见得就是非常适合用来做生成。

![相差一个Pixel](./images/12.png)

两者都相差一个像素，但是对于自编码来说，他们可能等价的，对于我们人眼来说它就不是一样的。

### GAN的感性认识

**GAN之父在NIPS 2016上做的报告：两个竞争网**  

![](./assets/15.png)

* 简单的说就是：你要要一个强大的Generator，就需要找个强大的对手\(Discriminator\)。

* 一个经典的例子就是：制造的假钱的伪造者和验钞机博弈的故事，伪造假钱V1后，伪造者就拿去让验钞机鉴别，鉴别出是假钱后，伪造者就会进一步去改进技术，然后再制造假钱V2去让验钞机鉴别，如此往复，直到验钞机判断不出真假，这样伪造者就获取了真正的造钱技术。

* 这就是一个极小-极大博弈的过程，我们要让Discriminator去最大程度的判断出real or fake,同时又要让Generator 产生出最逼近real的data.

* 如此举一反三，当我们有一些数据，我们就可以通过训练去学习出这些数据的结构特点（`!$P_{real}(x)$` distribution）,然后我们就可以根据`!$P_{noise}(z)$` 去产生出具有相似结构的数据了。

**简洁的表达如下：**

![](./assets/16.png)

### GAN的工作流程

根据上面的图，我们现在来简单的叙述一下GAN是如何工作的。

> 生成网络采用随机输入，尝试输出数据样本。在上述图像中，我们可以看到生成器`!$G(z)$`从`!$p(z)$`获取了输入`!$z$`，其中`!$z$`是来自概率分布`!$p(z)$`的样本。生成器`!$G(z)$`会产生一个fake的数据作为`!$D(x)$`输入。  
> 判别网络的主要工作就是接受一个来自真实数据分布（`!$P_{real}(x)$`）的数据`!$x$`或者是一个来自生成器的数据`!$G(z)$`,然后判断是来自real 还是fake。`!$D(x)$`可以看作是一个Logistic regression， 0/1 分类。

根据这个思想我们就可以得到GAN的数学表达式了\[3\]：

![](./assets/17.png)

### GAN的训练过程

根据上面的公式，我们可以得到GAN的训练方式主要分为两个阶段.

> 第一阶段：训练鉴别器，固定生成器，通过采样`!$P_{real}$`得到real data 和 `!$G(z)$`生成 `!$P_{fake}$`的数据来训练。（这里固定的意思就是说，生成器不做反向传播）
>
> 第二阶段：训练生成器，固定判别器。

根据这些思想，我们就得到了GAN的整体算法：

![](./assets/18.png)

## GAN的基本思想\(理论认识\)

---

不涉及AI algorithm， computer 如何生成一个数据？
> random function --> markov chain
> probability sample
> ...

### MLE
理论上是如何去产生一个realistic的数据的，这里我们先从Maximum Likelihood Estimation讲起\[4\]。
+ 首先给定一个数据分布`!$P_{data}(x)$`  -- 这个分布式未知的
+ 我们要产生一个数据的话，我们就需要去找到一个分布`!$P_G(x; \theta)$`,这个分布受到`!$\theta$`的控制。它`!$P_{data}(x)$`相似，但是受人们控制。
+ 我们的目标就是要去找到这个`!$\theta$`,使得分布`!$P_{G}(x; \theta)$` 无限接近`!$P_{data}(x)$`。

+ Sample `!$ \{x^1, x^2, ..., x^m\}$` from `!$P_{data}(x)$`
+ 计算`!$P_{G}(x^i; \theta)$`
+ 最后就可以使用MLE来进行计算了：
```mathjax!
$$L(\theta) = \prod^m_{i=1} P_G(x^i;\theta) $$
```
+ 最终找到`!$\theta^*$` 使得`!$P_{G}(x)$` 无限接近 `!$P_{data}(x)$`

![MLE的计算 与 KL divergence的关系][2]

倒数第二步是将前面离散的转化为了连续的，而面附加了一个常数项，(只和 `!$P_{data}$`有关)

![离散转为连续][3]

通过添加常数项之后，就可以化简为KL divergence的形式

所以MLE要做的事情是要找到一个由`!$\theta$`定义的`!$P_{G}(x^i; \theta)$`与`!$P_{data}(x)$`的分布无限的接近。但是现在的主要问题是`!$P_{G}(x^i; \theta)$`是什么函数？ 高斯模型？伯努利模型？这些比较simple的模型在大多数时候是无法满足所有情况的。
这时候就需要非常复杂、一般化的`!$P_{G}$`，我们就可以使用NN来定义，如下图\[5\]。

![NN 表达 复杂分布][4]

数学公式表达的话可以如下：

![difficult compute the likelihood][5]

这个公式表示的是从`!$P_G$`中采样出`!$x$`概率`!$P_G(x)$`是由`!$P_{prior}(z)$`中所有能采样出`!$x$`的概率之和。
那么问题来了，虽然可以知道`!$P_{prior}(z)$`，但是由于`!$G(z)$`是一个NN，我们很难去求出`!$x$`的概率。那么如何解决这个问题？

> 这就是GAN成功为生成模型贡献出的地方!
> 
给定`!$x$`的时候，G是无法算出`!$P_G(x)$`的，但是这时候采用一个D，通过来评估
`!$P_{G}(x)$` 与 `!$P_{data}(x)$`之间的差距，也就是说虽然无法直接的计算`!$P_G(x)$`和`!$P_{data}(x)$`之间的KL divergence, 但是来了一个D，通过D来衡量两者之间的差异。

![导出GAN的训练模型][6]

### 求最优的Discriminator(D)
给定G， 如何让求得最优的D：

![D的求解][7]

对于`!$\forall x$`我们现在只需要积分内的值最大时，积分肯定是最大的。所以就等价于优化：

![enter description here][8]

当最优的D求出来之后，我们就可以得到：

![D的推导][9]

所以优化D就是在优化`!$P_{data}(x)$` 和 `!$P_{G}$`之间的JSD距离。
由Jensen-Shannon Divergence 的定义可以得到 JSD的上下界为\[6\]：
```mathjax!
$ 0 \leq JSD(P||Q)\leq log2$
```
可以得到以下的一个等式

![求解最优][10]

也就是说，当且仅当二者相等(`!$P_G(x) = P_{data}(x)$`)时D取得最优解：==`!$-2log2$`==

这就是original paper 里面的定理：

![Theorem 1][11]

### 如何优化Generator(G)?
在这里我们将D的部分设置为一个`!$L(G)$`,因为这一部分只与G有关。

![求解 G][12]
这样的话，我们就可以使用梯度下降法的方式来求解G的最优解了。

### 实践中
在实践中理想函数是不能用的，因为数据是无穷的。

![采样 和 binary classifier][13]

## GAN Zoo

---

这个github仓库收集了最近关于GAN的paper：[Delving deep into GANs](https://github.com/flyuuu/Delving-deep-into-GANs)

### model
下面列出几个关于GAN的模型\[1\]：

![GAN basic structure][14]

![Semi GAN][15]

![Conditional GAN][16]

![Info GAN][17]

![Seq GAN][18]

### 关于GAN的运用
![图片生成][19]

![强化学习][20]

![生成超清图片][21]

![草图生成图像][22]

![修改图片][23]

![Image translate][24]

## GAN的挑战

---

GAN虽然取得了很多成果，也被炒的很热，但是GAN还是存在很多问题：
+ GAN训练不稳地， 同时容易出现模型崩塌，梯度消失等等。
+ 在自然语言(nlp)上面临的一系列挑战
+ 图片上也还有一些问题：

![计数问题][25]

![无法适应3D][26]

![无法全局结构][27]

## 参考文献

---

1： [生成式对抗网络GAN 的研究进展与展望](http://www.aas.net.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=19012) 王跃飞等.  
2： 统计学习方法， 李航.  
3：Generative Adversarial Nets, Ian J. GoodFellow etc
4：[台湾MLDS课程](https://www.youtube.com/watch?v=0CKeqXl5IY0), 李宏毅
5：[openAI blog](https://blog.openai.com/generative-models/)
6：[Jensen–Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence), Wikipedia
7：[GAN之父在NIPS 2016上做的报告：两个竞争网](https://ask.julyedu.com/question/7664)
8：[一文看懂生成式对抗网络GANs：介绍指南及前景展望](https://mp.weixin.qq.com/s/21CN4hAA6p7ZjWsO1sT2rA)，[NPIS 2016 report](http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf), Ian goodfellow

---
其他的一些阅读资料：
http://blog.csdn.net/u014568921/article/details/52774900
https://www.zhihu.com/question/52602529/answer/158727900
http://blog.csdn.net/hjimce/article/details/55657325
http://friskit.me/2017/04/09/read-paper-generative-adversarial-text-to-image-synthesis/
http://www.sohu.com/a/145446629_473283
http://closure11.com/%E5%AF%B9%E6%8A%97%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%EF%BC%9Aadversarial-autoencoders/


## 代码

---

TensorFlow简单的GAN代码：
```python:line
#!/usr/bin/python
# encoding: utf-8

"""
@version: 1.0
@author: Fly Lu
@license: Apache Licence 
@contact: luyfuyu@gmail.com
@site: https://www.cnblogs.com/flyu6/
@software: PyCharm Community Edition
@file: gan_tf.py
@time: 2017-06-10 下午4:39
@description: simple gan 最初始的gan
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)

    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Discriminator Net Parameters
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

D_W1 = tf.Variable(xavier_init([784, 128]), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

theta_D = [D_W1, D_W2, D_b1, D_b2]

# Generator Net
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

G_W1 = tf.Variable(xavier_init([100, 128]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

G_W2 = tf.Variable(xavier_init([128, 784]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# # -------------------
D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

# Only update D(X)'s parameters, so var_list = theta_D
D_solver = tf.train.AdadeltaOptimizer().minimize(D_loss, var_list=theta_D)

# Only update G(X)'s parameters, so var_list = theta_G
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mb_size = 128
Z_dim = 100

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, _ = mnist.train.next_batch(mb_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
```


  [1]: ./images/1503555903478.jpg
  [2]: ./images/1503542082548.jpg
  [3]: ./images/1503542536405.jpg
  [4]: ./images/1503543354660.jpg
  [5]: ./images/1503543510046.jpg
  [6]: ./images/1503548514400.jpg
  [7]: ./images/1503549151229.jpg
  [8]: ./images/1503549339363.jpg
  [9]: ./images/1503550449472.jpg
  [10]: ./images/1503551601453.jpg
  [11]: ./images/1503552144978.jpg
  [12]: ./images/1503552253903.jpg
  [13]: ./images/1503552843686.jpg
  [14]: ./images/1503553196213.jpg
  [15]: ./images/1503553253849.jpg
  [16]: ./images/1503553286726.jpg
  [17]: ./images/1503553318096.jpg
  [18]: ./images/1503553340700.jpg
  [19]: ./images/1503553463332.jpg
  [20]: ./images/1503553486884.jpg
  [21]: ./images/1503553516923.jpg
  [22]: ./images/1503553603640.jpg
  [23]: ./images/1503553735322.jpg
  [24]: ./images/1503553779474.jpg
  [25]: ./images/1503554478118.jpg
  [26]: ./images/1503554520447.jpg
  [27]: ./images/1503554571797.jpg
