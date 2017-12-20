---
title: Paper Notes： The Shattered Gradients Problem:If resnets are the answer, then what is the question?
tags: Fly,www.flyuuu.cn, ResNet
grammar_cjkRuby: true
---
[toc]
---
paper link: https://arxiv.org/abs/1702.08591

## Previous Knowledge
### Whiten Noise (白噪)
白噪声是一种功率频谱密度为常数的随机信号或随机过程,此信号在各个频段上的功率是一样的。在声音定义中，是指功率谱密度在整个频域内均匀分布的噪声。运用于神经网络中表示变化差不多的的一种振荡表现。

![Whiten Noise][1]

### Brownian Motion (布朗运动)
布朗运动是微小粒子表现出的无规则运动。1827年英国植物学家布朗在花粉颗粒的水溶液中观察到花粉不停顿的无规则运动。进一步实验证实，不仅花粉颗粒，其他悬浮在流体中的微粒也表现出这种无规则运动，如悬浮在空气中的尘埃。后人就把这种微粒的运动称之为布朗运动。

![Brownian Motion][2]

在这篇论文中，布朗运动的表现如下图所示：

![Brownian Motion in paper][3]

### Highway Net and Residual Net
这两者都是skip-connection的两个标志性的模型。而且ResNet是Highway Net的特殊形式。对于两者的模型想必都十分的了解，下面给出二者的数学公式。其中HNet公式来自这篇[博客](http://blog.csdn.net/cv_family_z/article/details/50349436)。
- HighWay Net

```mathjax!
$y = H(x, W_H) \cdot T(x, W_T) + x \cdot C(x, W_C) $

```
这里T和C表示对输入的映射(transform gate) 和传送(carry gate),在论文的文献中， `!$C = 1 - T$`所以，公式又写做：
```mathjax!
$y = H(x, W_H) \cdot T(x, W_T) + x \cdot (1 - T(x, W_T)) $

```

- Residual Net

公式如下：
```mathjax!
$y = H(x, W_H) + x $

```
[网络图](http://arxiv.org/abs/1512.03385)如下：

![ResNet block][4]

### Batch Normalization
在BN层的算法中，并没有对数据进行线性求和，即：`!$W^T * x + b$`。而是对数据进行类似归一化的操作，要求最后的输出为`!$y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta{(k)}$`。主要的算法流程如下图:

![Batch Normalization Algorithm][5]

算法的具体解释见[博客](http://blog.csdn.net/elaine_bao/article/details/50890491)


## Main Purpose

下面开始介绍[这篇论文](https://arxiv.org/abs/1702.08591)的主要工作。

### Why is worth to read
- 师兄推荐看的
- 发表在PMLR期刊上
- 深度学习中理论性很强的文章

### The problem of this paper propose 
深度学习的最大的缺点就是随着神经网络层数的越来越深，神经网络存在梯度消失（vanish）和梯度爆炸（explode），针对这个问题，一些研究学者提出了不同的解决方式，有针对参数初始化的，也有针对网络结构的，例如Batch Normalization, skip-connection. 其中skip-connection的代表Residual Networks，更是让网络能够达到上百层，同时保持良好的性能。

ResNet中最让人好奇的就是它根部不需要选择良好的Initialization或者Batch Normalization，同样能够得到良好的性能。如果说ResNet是Gradient Vanish/Explode的解决方式，那么什么才是这个问题的本质原因呢？或者说梯度的爆炸和消失到底是什么原因引起的？

这篇文章定义了一个叫The shattered gradient problem的问题，来探究ResNet、Batch Normalization效果好的背后的原因，在文章的最后还给出一个“looks linear” initialization的参数初始化方式，并通过这种方式在超过200层的普通网络（指不加技巧全连接网络）上取得了很好的效果。


### 梯度与白噪声和布朗噪声的关系
文中一开始，作者就给出了一个关于梯度与噪声的论证关系。
设置了一个从实数到实数的简单网络`!$f_w : \mathbb{R} \rightarrow \mathbb{R} $`， 隐藏层包含的神经元为：`!$N=200$`。
同时为了研究反向传播中梯度的最后一层(也就是输出层)的梯度变化，作者做了如下定义：`!$\frac{df_w(x^{(i)})}{dx}\;where\; x^{i} \in [-2, 2] \;is \;in \; a\; 1-dim \;grid \;of\; M=256\; 'data\; point' \;\;\; (1)$`， 也就是说`!$x^{(i)}$`是一维网格中256个==数据点==中的一个。
> BP的使用的是链式法则来进行求导：`!$\frac{\partial f_w}{\partial w_{i, j}} = \frac{\partial f_w}{\partial n_{j}} \frac{\partial n_j}{\partial w_{i, j}}$`

作者通过输入层的梯度，计算出协方差矩阵`!$\frac{|(g - \bar{g})(g - \bar{g})^T|}{\delta^2}$`，并绘制成图形与whiten noise and Brownian noise的图形对比。见下图：

![relationship of Gradients and noise][6]。

根据图的结果，作者进一步分析得出了Gradients与Brownian noise的关系。如下：
- 浅层全连接网络的Gradient与Brownian noise 相似；
作者使用函数中心极限定理(Donsker's theorem)对布朗运动做了弱收敛（converges weakly）的证明。然后给出了当`!$N \rightarrow \infty$`时，梯度弱收敛与布朗运动。

- 深层全连接网络的Gradient与Whiten noise 相似；
作者通过自相关的方式直观的解释了随着网络层数（`!$L$`）的增加，梯度成指数级下降（`!$\frac{1}{2^L}$`）。

- 当Gradient的行为类似于white noise时， 神经网络十分难于训练；

- Deep Resnet的Gradient值介于Browian and White之间。

### Batch Normalization 的歪打正着
在发明BN的时候，纯粹是为了归一化层与层之间的数据以及防止协变量的偏移。但是，这片论文进一步发现BN还直接影响着梯度的结构的相关性。

为了做这个研究，作者引入了一个理论，无论有没有BN层作用，针对输入的值平均会有一半的神经元会被激活(active)。

针对协同激活（co-active）：在两次（two iterator）输入中，相同的神经元被激活。在**加入BN的网络**中会有`!$\frac{1}{4}$`的神经元被激活。随着网络加深，在**没有BN的网络**中，协同激活率（co-active proportion）会上升，同时被激活的神经元越来越多(Without batch normalization, the co-active proportion climbs with depth, suggesting neuronal responses are increasingly redundant.)

从图中可以看出，如果仅仅只计算输入造成的激活比率（active proportion）的话，可能会误导分析。只有计算（co-active proportion）才能看出BN的影响。

![Activation of rectifiers in deep networks][7]

### initialization 对网络的影响
这篇文章最重要的是分析了初始化参数是如何深刻影响网络中梯度结构的相关性的。
这一节中作者根据Path-weights over active pathes,定义了与梯度结构相关的协方差和相关性。

![定义][8]

然后针对不同的网络结构，给出了初始化对相关性以、协方差以及方差的计算结果。
- Feedforword networks

![][9]
- Residual networks

![][10]
- recaling in Resnets

![][11]
- Highway networks

![][12]

针对卷积神经网络，作者也分析了初始化对梯度结构有着同样的影响。

### looks linear initialization
在公式(1)中，当神经网络为线性网络时，Shattered gradients并不算是一个问题，不幸的是，线性网络缺乏对数据的表达。
作者结合linear网络和rectifer网络(ReLU)，给出了一个==looks linear initialization==初始化的方式。
其中**最关键的地方**是使用**镜像块结构**能够让产生类似线性的输出。
而作者也在文中举了两个例子：
- CReLU

```mathjax!
$$x \mapsto \dbinom{\rho(x)}{-\rho(-x)}$$
```
关键的地方在于，通过镜像块实现了*looks linear*.
```mathjax!
$$(W -W)\cdot \dbinom{\rho(x)}{-\rho(-x)} = W\rho(x)-W\rho(-x) = Wx$$
```
这里额外说明一下`!$\rho(x)$`:
```mathjax!
$$
ReLU:
\rho(x) =
\begin{cases} 
x,  & \mbox{if } x \ge 0 \\
0, & \mbox{else }.
\end{cases}
$$

```

- PReLU

和CReLU类似的操作，这里的不同之处在于PReLU的公式不同：

```mathjax!
$$
PReLU:
\rho(x) =
\begin{cases} 
x,  & \mbox{if } x \ge 0 \\
ax, & \mbox{else }.
\end{cases}
$$
```

### Conclusion
在不使用skip-connection的时候，通过LL-Initialization的方式也能够使得网络层次达到100层以上。
同时作者也讨论了将skip-connection 和LL-init两者结合起来也是未来的一个研究方向。


## reference
http://blog.csdn.net/cv_family_z/article/details/50349436
http://blog.csdn.net/elaine_bao/article/details/50890491
http://blog.csdn.net/hjimce/article/details/50866313
http://www.erogol.com/paper-notes-shattered-gradients-problem/


  [1]: ./images/1513598890576.jpg
  [2]: ./images/1513599038440.jpg
  [3]: ./images/1513599119787.jpg
  [4]: ./images/1513600712136.jpg
  [5]: ./images/1513601113626.jpg
  [6]: ./images/1513766480524.jpg
  [7]: ./images/1513769886018.jpg
  [8]: ./images/1513770218499.jpg
  [9]: ./images/1513770478782.jpg
  [10]: ./images/1513770602500.jpg
  [11]: ./images/1513770626024.jpg
  [12]: ./images/1513770680366.jpg