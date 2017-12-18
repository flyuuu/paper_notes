---
title: Double generative adversarial nets
tags: GANs, www.flyuuu.cn
grammar_cjkRuby: true
---
```mathjax!
$Double$
$generative$
$ adversarial$
$ nets$
```

[toc]

---

# Previous Work
## Auto-Encoder
auto-encoder是采用了密码学上的加密解密的思想，大概意思就是输入`!$ x $`经过编码后形成`!$z$`,`!$z$`是未知的属性，当再次经过解码后得到`!$\hat{x}$`, 理论上`!$x=\hat{x}$`。具体操作如下图所示：

![AutoEncoder 模型][1]

所以对于损失函数，我们可以简单的设计为均方误差的方式：
```mathjax!
$$loss = ||x - \hat{x} ||^2_2$$
```
通过优化`!$loss$`损失函数，我们就可以构建出优化auto-encoder 模型参数的方法了，最终`!$z$`就是我们认为可以代表`!$x$`的特征编码了。

在autoencoder的模型中已经存在着了生成模型的影子的，当我们构建一个`!$z$`
然后使用decoder模型就可以产生一张通过`!$z$`控制的图片了。


## Variantation AE
受到AutoEncoder的启发，再引入了概率模型的辩分推断方式也就得到了VAE的模型框架，如下图：

![VAE Models][2]

由模型图中可以清晰的看到，和AE的不同之处在于，AE由encoder直接产生`!$z$`然后用来生成图片，而这里做的改变是通过使用encoder去产生一个服从`!$mu$`和`!$\sigma$`的高斯分布`!$Normal(\mu, \sigma)$`，然后从高斯分布中取得隐变量`!$z$`。这样的好处是`!$z$`的变化是可控的，同时一般可以将`!$mu$`和`!$\sigma$`的高斯模型归一化为正态分布`!$N(0,1)$`，当模型`!$p_{\theta}(x|z)$`训练好之后，我们就可以使用正态分布的方式来实现图像的产生了。
VAE的损失函数是：

![loss function][3]

具体的公式推导可以看[知乎](https://zhuanlan.zhihu.com/p/25401928)上的教程。

这就是VAE生成模型的方式。当然VAE还有许多修改以及变种，这里不做过多的赘述。

VAE的方式生成的图像大多都是趋近与模糊，而没有生成很尖锐（sharp）的纹理。不会线条分明，棱角清晰。同时应为使用了概率的极大似然估计的方式，计算量以及运行效率很低。


## Generative Adversarial Nets
生成对抗网络是不同于VAE的另外一种生成模型方式，它主要的思想是通过生成的结果和真实的结果之间的真伪来评价生成的模型结果是不是好的，这种方式省去了VAE的变分推理，容易理解。具体的模型如下：

![GAN 模型结构][4]

同时GAN的损失函数如下图：

![loss function][5]

正如损失函数里面所讲解的，这是一个极小极大博弈，生成器是制钞机，判别器是验钞机，通过两者的博弈，最终制钞机生成的假钞和真钞无法被区别开，这样子，生成器就能够产生仿真的图片了。

用概率的方式来说的话就是：某一类图片服从某个位置分布`!$X$`,这个分布能够生成各种类型的图片，例如：室内布局图总是服从某一个未知的分布，这个分布只能够产生各式各样的室内布局图。生成器就是这个位置分布的代替者，通过训练，最终`!$G(z) = X$`。

这就是生成对抗网络的结构。


## AAE
paper link: https://arxiv.org/abs/1511.05644

由于GAN的成功，GAN的作者根据GAN的特性又对VAE进行了改进，在VAE的结构中，使用了`!$KL$`距离的方式来实现两个分布的逼近，在VAE中`!$KL$`距离的计算上需要使用均值和标准差的方式，在这个改进的地方将引入了一个辅助判别器去实现对两个分布的间接判断，从而省去了直接的数学公式推导以及计算。并取得了更好的结果。
AAE的模型图如下：

![AAE Model][6]


## VAE-GAN
paper link: http://proceedings.mlr.press/v48/larsen16.pdf

VAE存在很多缺点，其中一个中就是使用`!$l_2$`的方式来构建损失函数，而且这种方式的损失函数是像素级别的损失，不能得到图像语义级别的损失信息，这篇文章的作者借用了GAN的方式：生成对抗网络在产生图像后并不是直接通过`!$l_2$`的方式去判断产生的结果，而是使用了一个辅助模型：判别器去帮助判别整体的图像与真实的图像结构、纹理、内容上是否是一致的。通过对VAE的decoder模型增加一个辅助的判别器，从而去实现非像素级别的判断。VAE-GAN模型如下：

![VAE-GAN Model][7]



# My work
下面阐述的是我根据上面的情况，做出的两个改进。

## Unified Autoencoder and GAN

根据VAE和GAN的特点，我对VAE-GAN的模型进行了第一个改进尝试，统一的VAE/GAN模型，在VAE-GAN中，使用了三个模型：Encoder、Decoder以及Discriminator。这样种方式增加了一个额外的模型，是的模型参数增多。同时，我们可以看到Encoder以及Discriminator具有相近的功能，输入都是图片，中间过程都是对图片进行解码，最后产生固定的向量输出。不同的地方只是一个是`!$z$`的维度，而一个是一维`!$0$` 或者 `!$1$`。改进的地方就在于我们==能不能将Encoder和Decoder合二为一==，这样子我们就减少了训练的模型参数。
具体的模型图如下：

![UAG][8]

>当然这种当时也存在一些问题，最迫切需要解决的是，当Encoder和Decoder合二为一之后，需要证明这两者之间是否存在对抗（eg: 作为Encoder的时候增加模型的权值，而Decoder的时候减少模型的权值）。或者说绕过这个问题采用：`!$max$`的方式，只取对权值增加的部分，对权值减少的部分不取。
> 由于时间关系，这个一个没有深入的往下做。而是做了下面的模型。

## Double VAE
收到VAE-GAN和AAE这两篇论文的启发，既然VAE-GAN是用了VAE的KL距离以及增加了辅助判别器来实现的对生成模型的增强，那么我们干脆在增加一个辅助判别器，从而去除KL距离的方式（也就是引进AAE），通过这样子，我们就得到了最终的模型：

![double GAN detial][9]

经过化简后，可以得到如下的模型：

![Double GAN][10]


在这个模型中，主要针对VAE-GAN做了如下更改：
+ 通过AAE的思想，增加了一个辅助判别器来实现对`!$z$`的分布的逼近
+ 结合了VAE-GAN和AAE的损失函数优化的方式
+ 目的是为了使得AE产生的图像有良好的形状和纹理

### loss function
在整个模型中存在两对GAN：AAE，DCGAN以及一个AutoEncoder模型。针对三个模型，我们可以构建三种损失函数。

1. 针对Auto-encoder的模型，我们可以实现`!$l_2$`损失函数的构建：
    即内容损失函数：`!$reconstruction-loss = ||x-\hat{x}||^2_2$`
2. 在针对AAE的时候，可以得到如下损失函数，通过这样的方式来更新==Encoder==和==Aux_discriminator==

```mathjax!
$$\min_{G} max_{D} E_{x- p_data}[\log{D(x)}] + E_{x- p_data}[1-\log{D(G(z))}]$$
``` 

3. 在针对Decoder和Discriminator的时候，以VAE-GAN的方式来进行更新。也就是说，在Decoder和Discriminator跟新时，存在三个输入：来自Encoder生成的编码`!$z$`和来自正态分布的`!$Nz$`以及`!$x$`。这个时候考虑的损失函数为：

![][12]
或者
```mathjax!
$$\min_{G} max_{D} E_{x- p_data}[\log{D(x)}] + E_{x- p_data}[1-\log{D(G(z))}]$$
``` 

这样就完成了对真个模型的更新。


> 当然在整个过程中，应该梳理出怎么跟新内容的问题，针对generator的更新时，generator拥有两个更新权值，分别来自：reconstruction和GAN，这里就需要将两者合二为一来针对generator更新。当让也可以分别更新
> 上述中是针对分别更新来实现，下面分析如何针对每一个“组件”实现loss的整合
> 1. 针对 encoder： loss 来自两部分，一部分是reconstruction loss另外一部分是auxiliary gan loss。
> 2. 针对 auxiliary discriminator: 只有来自auxiliary gan loss的部分
> 3. 针对decoder/generator：同样来自两部分，一部分是reconstruction loss另外一部分是main gan loss。
> 4. 针对main discriminator：只有来自main gan loss 的部分

## 实验中存在的问题：
### 结果分析
在跑出来的结果中，得到的结果与AAE模型和VAE-GAN模型还是存在一些差距。


### 关于损失函数的探究
由于AAE和AVE-GAN结合的方式，去掉了KL距离来计算的形式，所以最终loss函数就变成了极大极小博弈的损失。针对这个模型做了下面的探究：
1. min-max的方式
    都使用min-max的方式以及内容损失mean-square error的方式
2. 借鉴VAE-GAN的方式，在针对解码器的地方使用`!$\gamma *L_{recon} - L_{GAN}$`的方式
3. 

## 实验分析
1. D loss 趋近与0
    这次实验中，lr = 0.0002, beta1=0.5
    训练过程中： loss_d: 0.000, loss_g: 25.306, loss_aux_d: 0.000, loss_aux_g: 28.521, loss_en: nan。
reconstruction loss 和 discriminator loss出现不正常的==0==
可能是由于lr和beta1设置有问题。继续下一步实验
    



  [1]: ./images/1512965144407.jpg
  [2]: ./images/1512969480989.jpg
  [3]: ./images/1512971630792.jpg
  [4]: ./images/1512973305487.jpg
  [5]: ./images/1512973448785.jpg
  [6]: ./images/1512975024965.jpg
  [7]: ./images/1512974464192.jpg
  [8]: ./images/1512977947249.jpg
  [9]: ./images/1512977993570.jpg
  [10]: ./images/1513567318128.jpg
  [11]: ./images/1512978037384.jpg
  [12]: ./images/1512979581438.jpg